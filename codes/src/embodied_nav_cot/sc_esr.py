"""State-conditioned evidence-slot residuals for native Qwen generation.

SC-ESR differs from the retired generic residual MLP in three important ways:

* every ability slot queries the *sequence* of native image-token states rather
  than an average-pooled visual vector;
* the decision hidden state cross-attends to the selected evidence slots once;
* the result is a zero-initialised, norm-bounded additive residual inside the
  original Transformer -> LM-head path.  It never predicts an action itself.

The module is intentionally independent from training data code.  Training may
supervise state/ability routing, contrast evidence slots against same-image
evidence embeddings, and compare correct versus counterfactual slots at the
native LM action token.  Runtime needs only the current image, public goal and
public candidates, and performs one ordinary autoregressive generation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ABILITIES = (
    "target_grounding",
    "relative_geometry",
    "spatial_relation",
    "local_affordance",
    "layout_topology",
)


@dataclass(frozen=True)
class SCESRConfig:
    hidden_size: int
    slot_rank: int = 64
    num_states: int = 2
    min_abilities: int = 1
    max_abilities: int = 3
    source_layer: int = 24
    injection_layer: int = 25
    max_residual_scale: float = 0.02
    state_dropout: float = 0.30
    candidate_conditioned: bool = False
    independent_ability_gates: bool = False
    adaptive_need_gate: bool = False
    candidate_local_masks: bool = False
    global_conditioned_local_slots: bool = False
    candidate_value_interaction: bool = False
    candidate_memory_injection: bool = False
    latent_spatial_rollout: bool = False
    relation_semantic_decision_separation: bool = False
    straight_through_sparse_route: bool = False
    soft_ability_route: bool = False
    local_evidence_control: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.slot_rank <= 0:
            raise ValueError("hidden_size and slot_rank must be positive")
        if self.num_states != 2:
            raise ValueError("SC-ESR currently requires two navigation states")
        if not 1 <= self.min_abilities <= self.max_abilities <= len(ABILITIES):
            raise ValueError("invalid sparse ability-count range")
        if self.source_layer < 1 or self.injection_layer <= self.source_layer:
            raise ValueError("injection_layer must follow source_layer")
        if not 0.0 < self.max_residual_scale <= 0.20:
            raise ValueError("SC-ESR residual budget must be in (0, 0.20]")
        if not 0.0 <= self.state_dropout <= 1.0:
            raise ValueError("state_dropout must be in [0, 1]")
        if self.candidate_local_masks and not self.candidate_conditioned:
            raise ValueError(
                "candidate-local masks require candidate-conditioned SC-ESR"
            )
        if (
            self.global_conditioned_local_slots
            and not self.candidate_local_masks
        ):
            raise ValueError(
                "global-conditioned local slots require candidate-local masks"
            )
        if self.candidate_value_interaction and not self.candidate_local_masks:
            raise ValueError(
                "candidate-value interaction requires candidate-local masks"
            )
        if self.candidate_memory_injection and not self.candidate_local_masks:
            raise ValueError(
                "candidate-memory injection requires candidate-local masks"
            )
        if (
            self.candidate_memory_injection
            and self.candidate_value_interaction
        ):
            raise ValueError(
                "candidate-memory and candidate-value paths are exclusive"
            )
        if self.candidate_memory_injection and self.adaptive_need_gate:
            raise ValueError(
                "candidate-memory injection does not use the decision need gate"
            )
        if self.latent_spatial_rollout and not (
            self.candidate_conditioned
            and self.candidate_local_masks
            and self.global_conditioned_local_slots
        ):
            raise ValueError(
                "latent spatial rollout requires candidate-local, "
                "global-conditioned candidate evidence"
            )
        if (
            self.relation_semantic_decision_separation
            and not self.latent_spatial_rollout
        ):
            raise ValueError(
                "relation semantic/decision separation requires latent "
                "spatial rollout"
            )
        if self.local_evidence_control and not self.latent_spatial_rollout:
            raise ValueError(
                "local evidence control requires latent spatial rollout"
            )


class StateConditionedEvidenceSlotResidual(nn.Module):
    """Extract five image-grounded evidence slots and fuse them once.

    ``image_hidden`` contains the complete native image-token sequence from a
    chosen Qwen decoder layer.  The public prompt representation conditions the
    ability queries, so the same image can yield different evidence for
    different goals without a hand-written goal/category rule.
    """

    def __init__(self, config: SCESRConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        rank = config.slot_rank
        abilities = len(ABILITIES)

        self.image_norm = nn.LayerNorm(hidden)
        self.context_norm = nn.LayerNorm(hidden)
        self.state_head = nn.Linear(hidden, config.num_states)
        self.prompt_query = nn.Linear(hidden, rank, bias=False)
        self.state_query = nn.Linear(config.num_states, rank, bias=False)
        self.ability_queries = nn.Parameter(torch.empty(abilities, rank))
        nn.init.normal_(self.ability_queries, std=rank**-0.5)

        self.image_key = nn.Linear(hidden, rank, bias=False)
        self.image_value = nn.Linear(hidden, rank, bias=False)
        self.slot_to_hidden = nn.Linear(rank, hidden, bias=False)
        if config.global_conditioned_local_slots:
            # A shared, ability-specific global spatial summary conditions
            # each candidate-local slot.  Candidate identity is absent here:
            # the only candidate-varying input is the local RGB region.
            self.global_local_norm = nn.LayerNorm(rank)
            self.global_local_fuse = nn.Sequential(
                nn.Linear(3 * rank, rank, bias=False),
                nn.SiLU(),
                nn.Linear(rank, rank, bias=False),
            )
            # Zero starts from the already-tested local-only V5 path.  The
            # outer residual gate still guarantees exact native-LM identity.
            self.global_local_gate = nn.Parameter(torch.tensor(0.0))
        if config.independent_ability_gates:
            # Each semantic ability needs its own decision boundary.  A shared
            # scalar scorer made frequent local-affordance/layout slots suppress
            # the rarer geometry/relation slots even when their slot contents
            # differed.  This remains data-driven: no state/category rule is
            # encoded and every score still comes from current-image evidence.
            self.ability_gate_norm = nn.LayerNorm(rank)
            self.ability_gate_weight = nn.Parameter(
                torch.empty(abilities, rank)
            )
            self.ability_gate_bias = nn.Parameter(torch.zeros(abilities))
            nn.init.normal_(self.ability_gate_weight, std=rank**-0.5)
        else:
            self.ability_gate = nn.Sequential(
                nn.LayerNorm(rank),
                nn.Linear(rank, rank),
                nn.SiLU(),
                nn.Linear(rank, 1),
            )
        self.count_head = nn.Linear(
            hidden, config.max_abilities - config.min_abilities + 1
        )

        # One gated cross-attention from native decision states to the selected
        # spatial slots.  The gate, not the branch weights, is zero at init:
        # output is exactly unchanged while the gate still receives gradient.
        self.decision_norm = nn.LayerNorm(hidden)
        self.decision_query = nn.Linear(hidden, rank, bias=False)
        self.slot_key = nn.Linear(hidden, rank, bias=False)
        self.slot_value = nn.Linear(hidden, rank, bias=False)
        if config.candidate_local_masks:
            # The key binds current-image evidence to its public action string.
            self.candidate_label_key = nn.Linear(hidden, rank, bias=False)
        if config.candidate_value_interaction:
            # Attention alone can learn which candidate slot is gold while its
            # value remains unable to tell the native LM *which action token*
            # that slot should affect.  Add a bounded visual-label interaction
            # to the value path.  It is multiplicative and bias-free: candidate
            # identity cannot produce a decision residual without current RGB
            # evidence.  The zero gate recovers the earlier path exactly.
            self.candidate_value_norm = nn.LayerNorm(rank)
            self.candidate_label_value = nn.Linear(
                hidden, rank, bias=False
            )
            self.candidate_value_fuse = nn.Linear(rank, rank, bias=False)
            self.candidate_value_gate = nn.Parameter(torch.tensor(0.0))
        if config.candidate_memory_injection:
            # Enrich the existing public candidate-label tokens with their
            # current-image spatial evidence.  The remaining native decoder
            # layers, rather than a new head, perform candidate comparison and
            # generate the action.  No candidate ID enters this value branch:
            # identity is supplied by the original token being augmented.
            if not config.latent_spatial_rollout:
                self.candidate_memory_norm = nn.LayerNorm(rank)
                self.candidate_memory_out = nn.Linear(
                    rank, hidden, bias=False
                )
            self.candidate_memory_gate = nn.Parameter(torch.tensor(0.0))
        if config.latent_spatial_rollout:
            # SC-LSR does not merely strengthen a candidate crop.  Each
            # ability has an independent relational operator that compares a
            # candidate-local spatial view with a goal/state-conditioned
            # global anchor.  Its output is the semantic value consumed by
            # V4's native decision residual (or, only in the historical
            # control, written into a candidate prompt token). Candidate
            # identity never enters the value operator.
            self.rollout_local_norm = nn.LayerNorm(rank)
            self.rollout_global_norm = nn.LayerNorm(rank)
            self.rollout_state = nn.Linear(
                config.num_states, rank, bias=False
            )
            self.rollout_context = nn.Linear(hidden, rank, bias=False)
            self.relation_operators = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(6 * rank, 2 * rank, bias=False),
                        nn.SiLU(),
                        nn.Linear(2 * rank, rank, bias=False),
                    )
                    for _ in ABILITIES
                ]
            )
            self.relation_norms = nn.ModuleList(
                [nn.LayerNorm(rank) for _ in ABILITIES]
            )
            if config.relation_semantic_decision_separation:
                # The action-blind teacher defines a semantic relation space,
                # while the native LM consumes a decision-space increment.
                # A zero residual mapper compiles between the two without
                # forcing the teacher's hidden direction directly into Qwen.
                # The identity skip makes the forward path equal to the
                # existing rollout at initialization; the outer residual gate
                # still makes the complete model exactly native-B0 at init.
                self.relation_decision_adapters = nn.ModuleList(
                    [
                        nn.Linear(rank, rank, bias=False)
                        for _ in ABILITIES
                    ]
                )
                for adapter in self.relation_decision_adapters:
                    nn.init.zeros_(adapter.weight)
        self.residual_out = nn.Linear(rank, hidden, bias=False)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        if config.adaptive_need_gate:
            # Decide whether this particular decision token needs a spatial
            # correction.  The signal compares native decision context with
            # the evidence actually retrieved from the current RGB; it cannot
            # use a gold action or simulator state at runtime.
            self.need_hidden = nn.Linear(hidden, rank, bias=False)
            self.need_evidence = nn.Linear(rank, rank, bias=False)
            self.need_bias = nn.Parameter(torch.tensor(-2.0))

    def extract_slots(
        self,
        image_hidden: Tensor,
        image_mask: Tensor,
        general_context: Tensor,
        *,
        candidate_context: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        candidate_image_mask: Tensor | None = None,
        candidate_local_valid: Tensor | None = None,
        force_state_dropout: bool | None = None,
        state_dropout_mask: Tensor | None = None,
        ablation: str = "full",
    ) -> dict[str, Tensor]:
        if image_hidden.ndim != 3:
            raise ValueError("image_hidden must have shape [batch, sequence, hidden]")
        batch, sequence, hidden = image_hidden.shape
        if hidden != self.config.hidden_size:
            raise ValueError("image hidden size does not match SC-ESR config")
        if image_mask.shape != (batch, sequence):
            raise ValueError("image_mask must align with image_hidden")
        if general_context.shape != (batch, hidden):
            raise ValueError("general_context must have shape [batch, hidden]")
        mask = image_mask.to(image_hidden.device).bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every SC-ESR sample requires at least one image token")

        context = self.context_norm(general_context)
        state_logits = self.state_head(context)
        state_soft = state_logits.softmax(-1)
        if force_state_dropout is None and self.training:
            if state_dropout_mask is None:
                drop = (
                    torch.rand(batch, 1, device=context.device)
                    < self.config.state_dropout
                )
            else:
                if state_dropout_mask.shape != (batch, 1):
                    raise ValueError(
                        "state_dropout_mask must have shape [batch, 1]"
                    )
                drop = state_dropout_mask.to(context.device).bool()
            routed_state = torch.where(
                drop, torch.zeros_like(state_soft), state_soft
            )
        elif force_state_dropout:
            routed_state = torch.zeros_like(state_soft)
        else:
            routed_state = state_soft

        base_query = self.prompt_query(context) + self.state_query(routed_state)
        use_candidates = bool(self.config.candidate_conditioned)
        if use_candidates:
            if candidate_context is None or candidate_mask is None:
                raise ValueError(
                    "candidate-conditioned SC-ESR requires candidate context"
                )
            if candidate_context.ndim != 3:
                raise ValueError(
                    "candidate_context must have shape [batch, candidate, hidden]"
                )
            if candidate_context.shape[0] != batch or candidate_context.shape[2] != hidden:
                raise ValueError("candidate_context does not match image batch/hidden")
            if candidate_mask.shape != candidate_context.shape[:2]:
                raise ValueError("candidate_mask must align with candidate_context")
            candidate_valid = candidate_mask.to(image_hidden.device).bool()
            if not bool(candidate_valid.any(dim=1).all()):
                raise ValueError("every row requires a candidate or global fallback")
            candidate_label_context = self.context_norm(candidate_context)
            if self.config.candidate_local_masks:
                # The visual query is candidate-ID blind.  Candidate-specific
                # content can enter only through the public ray/endpoint mask.
                # Label identity is introduced later as an attention key, not
                # as residual value content.
                queries = (
                    base_query[:, None, None, :]
                    + self.ability_queries[None, None, :, :]
                ).expand(
                    -1, candidate_context.shape[1], -1, -1
                )
                if candidate_image_mask is None:
                    raise ValueError(
                        "candidate-local SC-ESR requires image-token regions"
                    )
                if candidate_image_mask.shape != (
                    batch,
                    candidate_context.shape[1],
                    sequence,
                ):
                    raise ValueError(
                        "candidate image masks do not align with hidden states"
                    )
                attention_mask = (
                    candidate_image_mask.to(image_hidden.device).bool()
                    & mask[:, None, :]
                )
                if not bool(
                    attention_mask.any(-1)[candidate_valid].all()
                ):
                    raise ValueError(
                        "every valid candidate requires visual evidence tokens"
                    )
                if candidate_local_valid is None:
                    raise ValueError(
                        "candidate-local SC-ESR requires validity audit"
                    )
                if candidate_local_valid.shape != candidate_valid.shape:
                    raise ValueError(
                        "candidate local-valid mask shape drift"
                    )
                local_valid = (
                    candidate_local_valid.to(image_hidden.device).bool()
                    & candidate_valid
                )
            else:
                candidate_query = self.prompt_query(
                    candidate_label_context
                )
                queries = (
                    base_query[:, None, None, :]
                    + candidate_query[:, :, None, :]
                    + self.ability_queries[None, None, :, :]
                )
                attention_mask = mask[:, None, :].expand(
                    -1, candidate_context.shape[1], -1
                )
                local_valid = torch.zeros_like(candidate_valid)
        else:
            candidate_valid = torch.ones(
                batch, 1, dtype=torch.bool, device=image_hidden.device
            )
            candidate_label_context = torch.zeros(
                batch,
                1,
                hidden,
                dtype=image_hidden.dtype,
                device=image_hidden.device,
            )
            local_valid = torch.zeros_like(candidate_valid)
            queries = (
                base_query[:, None, None, :]
                + self.ability_queries[None, None, :, :]
            )
            attention_mask = mask[:, None, :]
        image = self.image_norm(image_hidden)
        keys = self.image_key(image)
        values = self.image_value(image)
        attention_logits = torch.einsum(
            "bcar,bsr->bcas", queries, keys
        ) / math.sqrt(self.config.slot_rank)
        attention_logits = attention_logits.masked_fill(
            ~attention_mask[:, :, None, :],
            torch.finfo(attention_logits.dtype).min,
        )
        image_attention = attention_logits.softmax(-1)
        local_candidate_slot_low = torch.einsum(
            "bcas,bsr->bcar", image_attention, values
        )
        global_image_attention: Tensor | None = None
        global_slot_low: Tensor | None = None
        global_local_scale = torch.zeros(
            (), dtype=image_hidden.dtype, device=image_hidden.device
        )
        candidate_slot_low = local_candidate_slot_low
        if self.config.global_conditioned_local_slots:
            global_queries = (
                base_query[:, None, :]
                + self.ability_queries[None, :, :]
            )
            global_logits = torch.einsum(
                "bar,bsr->bas", global_queries, keys
            ) / math.sqrt(self.config.slot_rank)
            global_logits = global_logits.masked_fill(
                ~mask[:, None, :],
                torch.finfo(global_logits.dtype).min,
            )
            global_image_attention = global_logits.softmax(-1)
            global_slot_low = torch.einsum(
                "bas,bsr->bar", global_image_attention, values
            )
            local_normalized = self.global_local_norm(
                local_candidate_slot_low
            )
            global_normalized = self.global_local_norm(
                global_slot_low
            )[:, None, :, :].expand_as(local_normalized)
            interaction = self.global_local_fuse(
                torch.cat(
                    [
                        local_normalized,
                        global_normalized,
                        local_normalized * global_normalized,
                    ],
                    dim=-1,
                )
            )
            global_local_scale = (
                0.5 * torch.tanh(self.global_local_gate)
            ).to(interaction.dtype)
            candidate_slot_low = (
                local_candidate_slot_low
                + global_local_scale * interaction
            )
        candidate_relation_low: Tensor | None = None
        candidate_relation_base_low: Tensor | None = None
        candidate_relation_semantic_low: Tensor | None = None
        candidate_relation_delta_low: Tensor | None = None
        if self.config.latent_spatial_rollout:
            if global_slot_low is None:
                raise RuntimeError(
                    "latent spatial rollout has no global spatial anchors"
                )
            local_relation = self.rollout_local_norm(
                local_candidate_slot_low
            )
            global_relation = self.rollout_global_norm(
                global_slot_low
            )[:, None, :, :].expand_as(local_relation)
            # Use the same optionally masked soft state as the visual query.
            # Otherwise the explicit relation operator could recover a fixed
            # state-to-reasoning shortcut even when state dropout is enabled.
            state_relation = self.rollout_state(routed_state)[
                :, None, None, :
            ].expand_as(local_relation)
            context_relation = self.rollout_context(context)[
                :, None, None, :
            ].expand_as(local_relation)
            if self.config.local_evidence_control:
                # Equal-parameter non-relational control.  Keep the exact
                # five operators, routing, goal/state conditioning, optimizer
                # course and native injection path, but remove the
                # global-to-local comparison.  Repeating the local evidence
                # preserves the 6r operator input and therefore parameter
                # count; it tests whether SC-LSR beats a strong
                # goal-conditioned candidate-evidence enhancer rather than a
                # smaller/disabled module.
                relation_inputs = torch.cat(
                    [
                        local_relation,
                        local_relation,
                        torch.zeros_like(local_relation),
                        local_relation * local_relation,
                        state_relation,
                        context_relation,
                    ],
                    dim=-1,
                )
            else:
                relation_inputs = torch.cat(
                    [
                        local_relation,
                        global_relation,
                        local_relation - global_relation,
                        local_relation * global_relation,
                        state_relation,
                        context_relation,
                    ],
                    dim=-1,
                )
            relation_outputs: list[Tensor] = []
            relation_semantics: list[Tensor] = []
            relation_deltas: list[Tensor] = []
            for ability, operator in enumerate(self.relation_operators):
                delta = operator(relation_inputs[:, :, ability, :])
                # The outer candidate-memory gate is the only zero gate.  A
                # second zero gate here would starve the relational operators
                # of gradient at initialization.  tanh bounds the learned
                # relation change before the final norm-bounded write.
                semantic = 0.5 * torch.tanh(delta)
                relation_semantics.append(semantic)
                if self.config.relation_semantic_decision_separation:
                    # The same runtime semantic latent is consumed by the
                    # decision mapper.  Detaching only separates optimization:
                    # semantic supervision cannot dictate the LM residual
                    # direction, and action gradients cannot erase semantics.
                    decision_source = semantic.detach()
                    decision_delta = (
                        decision_source
                        + 0.5
                        * torch.tanh(
                            self.relation_decision_adapters[ability](
                                decision_source
                            )
                        )
                    )
                else:
                    decision_delta = semantic
                relation_deltas.append(decision_delta)
                relation_outputs.append(
                    self.relation_norms[ability](
                        local_relation[:, :, ability, :]
                        + decision_delta
                    )
                )
            candidate_relation_base_low = local_relation
            candidate_relation_semantic_low = torch.stack(
                relation_semantics, dim=2
            )
            candidate_relation_delta_low = torch.stack(
                relation_deltas, dim=2
            )
            candidate_relation_low = torch.stack(
                relation_outputs, dim=2
            )
            candidate_slot_low = candidate_relation_low

        if ablation == "shuffle_relation":
            if (
                candidate_relation_base_low is None
                or candidate_relation_delta_low is None
            ):
                raise ValueError(
                    "relation shuffle requires latent spatial rollout"
                )
            # Preserve each candidate's local visual evidence and exchange only
            # the learned relation update.  The previous implementation swapped
            # the complete candidate slot (and its image attention), which
            # confounded spatial-relation reasoning with ordinary crop evidence.
            shuffled_delta = candidate_relation_delta_low.clone()
            for row in range(batch):
                permutation_mask = (
                    local_valid[row]
                    if self.config.candidate_local_masks
                    else candidate_valid[row]
                )
                indices = permutation_mask.nonzero(as_tuple=True)[0]
                if indices.numel() < 2:
                    continue
                donors = indices.roll(1)
                shuffled_delta[row, indices] = (
                    candidate_relation_delta_low[row, donors]
                )
            candidate_relation_delta_low = shuffled_delta
            candidate_relation_low = torch.stack(
                [
                    self.relation_norms[ability](
                        candidate_relation_base_low[:, :, ability, :]
                        + candidate_relation_delta_low[:, :, ability, :]
                    )
                    for ability in range(len(ABILITIES))
                ],
                dim=2,
            )
            candidate_slot_low = candidate_relation_low
        candidate_slots = self.slot_to_hidden(candidate_slot_low)

        candidate_float = candidate_valid.to(candidate_slot_low.dtype)
        candidate_denominator = candidate_float.sum(1, keepdim=True).clamp_min(1)
        slot_low = (
            candidate_slot_low * candidate_float[:, :, None, None]
        ).sum(1) / candidate_denominator[:, :, None]
        slots = self.slot_to_hidden(slot_low)
        if self.config.independent_ability_gates:
            gated = self.ability_gate_norm(candidate_slot_low)
            ability_logits_by_candidate = (
                torch.einsum(
                    "bcar,ar->bca", gated, self.ability_gate_weight
                )
                + self.ability_gate_bias[None, None, :]
            )
        else:
            ability_logits_by_candidate = self.ability_gate(
                candidate_slot_low
            ).squeeze(-1)
        ability_logits = (
            ability_logits_by_candidate
            * candidate_float[:, :, None]
        ).sum(1) / candidate_denominator
        ability_soft = ability_logits.sigmoid()
        count_logits = self.count_head(context)
        ability_count = (
            count_logits.argmax(-1) + self.config.min_abilities
        )
        ranks = ability_logits.argsort(
            dim=-1, descending=True
        ).argsort(dim=-1)
        ability_mask = ranks < ability_count[:, None]
        hard = ability_soft * ability_mask.to(ability_soft.dtype)
        hard = hard / hard.sum(-1, keepdim=True).clamp_min(1e-6)
        soft = ability_soft / ability_soft.sum(
            -1, keepdim=True
        ).clamp_min(1e-6)
        if self.config.soft_ability_route:
            # Exploration mode deliberately lets every learned ability
            # contribute.  This prevents an immature top-k router from
            # starving geometry/relation before their decision utility can be
            # established.  Once a useful solution is found, later versions
            # can tighten the same checkpoint-independent recipe back to the
            # ordinary 1--3 hard route.
            ability_weights = soft
        elif self.training and self.config.straight_through_sparse_route:
            # Match the runtime forward exactly: only the selected 1--3
            # abilities contribute spatial values.  The zero-valued
            # straight-through term preserves the previous 90% sparse / 10%
            # dense router gradient, including learning signal for abilities
            # that are not selected in this batch.  This removes a genuine
            # train/runtime mismatch without changing inference or adding a
            # second forward.
            dense_gradient = soft - hard
            ability_weights = hard + 0.1 * (
                dense_gradient - dense_gradient.detach()
            )
        else:
            ability_weights = (
                0.9 * hard + 0.1 * soft if self.training else hard
            )

        if ablation == "shuffle_ability":
            slots = slots.roll(1, dims=1)
            slot_low = slot_low.roll(1, dims=1)
            candidate_slots = candidate_slots.roll(1, dims=2)
            candidate_slot_low = candidate_slot_low.roll(1, dims=2)
            if candidate_relation_low is not None:
                candidate_relation_low = candidate_slot_low
            image_attention = image_attention.roll(1, dims=2)
        elif ablation == "shuffle_image":
            if batch > 1:
                slots = slots.roll(1, dims=0)
                slot_low = slot_low.roll(1, dims=0)
                candidate_slots = candidate_slots.roll(1, dims=0)
                candidate_slot_low = candidate_slot_low.roll(1, dims=0)
                if candidate_relation_low is not None:
                    candidate_relation_low = candidate_slot_low
                image_attention = image_attention.roll(1, dims=0)
            else:
                slots = slots.roll(1, dims=1)
                slot_low = slot_low.roll(1, dims=1)
                candidate_slots = candidate_slots.roll(1, dims=2)
                candidate_slot_low = candidate_slot_low.roll(1, dims=2)
                if candidate_relation_low is not None:
                    candidate_relation_low = candidate_slot_low
                image_attention = image_attention.roll(1, dims=2)
        elif ablation == "shuffle_candidate":
            if not use_candidates:
                raise ValueError(
                    "candidate shuffle requires candidate-conditioned slots"
                )
            shuffled_slots = candidate_slots.clone()
            shuffled_low = candidate_slot_low.clone()
            shuffled_attention = image_attention.clone()
            for row in range(batch):
                # Candidate-local counterfactuals exchange only genuinely
                # localized regions from the same RGB.  candidate_0 and
                # detector misses use the global-image fallback; permuting
                # either would test local-vs-global context instead of whether
                # the correct candidate evidence matters.
                permutation_mask = (
                    local_valid[row]
                    if self.config.candidate_local_masks
                    else candidate_valid[row]
                )
                indices = permutation_mask.nonzero(as_tuple=True)[0]
                if indices.numel() < 2:
                    continue
                donors = indices.roll(1)
                shuffled_slots[row, indices] = candidate_slots[row, donors]
                shuffled_low[row, indices] = candidate_slot_low[row, donors]
                shuffled_attention[row, indices] = image_attention[row, donors]
            candidate_slots = shuffled_slots
            candidate_slot_low = shuffled_low
            if candidate_relation_low is not None:
                candidate_relation_low = shuffled_low
            image_attention = shuffled_attention
        elif ablation not in {"full", "off", "shuffle_relation"}:
            raise ValueError(f"unknown SC-ESR ablation: {ablation}")

        return {
            "context": context,
            "state_logits": state_logits,
            "state_soft": state_soft,
            "ability_logits": ability_logits,
            "count_logits": count_logits,
            "ability_count": ability_count,
            "ability_mask": ability_mask,
            "ability_weights": ability_weights,
            "slot_low": slot_low,
            "slots": slots,
            "candidate_mask": candidate_valid,
            "candidate_local_valid": local_valid,
            "candidate_label_context": candidate_label_context,
            "local_candidate_slot_low": local_candidate_slot_low,
            "candidate_slot_low": candidate_slot_low,
            "candidate_relation_low": candidate_relation_low,
            "candidate_relation_base_low": candidate_relation_base_low,
            "candidate_relation_semantic_low": (
                candidate_relation_semantic_low
            ),
            "candidate_relation_delta_low": candidate_relation_delta_low,
            "candidate_slots": candidate_slots,
            "global_slot_low": global_slot_low,
            "global_image_attention": global_image_attention,
            "global_local_scale": global_local_scale,
            "image_attention": (
                image_attention
                if use_candidates
                else image_attention[:, 0]
            ),
        }

    def decision_residual(
        self,
        hidden_states: Tensor,
        route: dict[str, Tensor],
        *,
        ablation: str = "full",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        slots = route["slots"]
        weights = route["ability_weights"]
        if slots.shape != (
            hidden_states.shape[0],
            len(ABILITIES),
            self.config.hidden_size,
        ):
            raise ValueError("SC-ESR evidence slots do not match hidden states")
        if self.config.candidate_conditioned:
            candidate_slots = route["candidate_slots"]
            candidate_mask = route["candidate_mask"].bool()
            if candidate_slots.ndim != 4:
                raise ValueError(
                    "candidate slots must have [batch, candidate, ability, hidden]"
                )
            batch, candidates, abilities, hidden = candidate_slots.shape
            if (
                batch != hidden_states.shape[0]
                or abilities != len(ABILITIES)
                or hidden != self.config.hidden_size
            ):
                raise ValueError("candidate slots do not match hidden states")
            decision_slots = candidate_slots.reshape(
                batch, candidates * abilities, hidden
            )
            candidate_prior = candidate_mask.to(weights.dtype)
            candidate_prior = candidate_prior / candidate_prior.sum(
                -1, keepdim=True
            ).clamp_min(1)
            decision_weights = (
                candidate_prior[:, :, None] * weights[:, None, :]
            ).reshape(batch, candidates * abilities)
        else:
            decision_slots = slots
            decision_weights = weights
            candidates = 1
            abilities = len(ABILITIES)
        query = self.decision_query(self.decision_norm(hidden_states))
        keys = self.slot_key(decision_slots)
        if self.config.candidate_local_masks:
            candidate_label_context = route[
                "candidate_label_context"
            ]
            if candidate_label_context.shape != (
                batch,
                candidates,
                hidden,
            ):
                raise ValueError(
                    "candidate label context does not match visual slots"
                )
            label_keys = self.candidate_label_key(
                candidate_label_context
            )
            label_keys = label_keys[:, :, None, :].expand(
                -1, -1, abilities, -1
            ).reshape(batch, candidates * abilities, -1)
            keys = keys + label_keys
        values = self.slot_value(decision_slots)
        candidate_value_scale = torch.zeros(
            (), dtype=values.dtype, device=values.device
        )
        if self.config.candidate_value_interaction:
            label_values = self.candidate_label_value(
                candidate_label_context
            )
            label_values = label_values[:, :, None, :].expand(
                -1, -1, abilities, -1
            ).reshape(batch, candidates * abilities, -1)
            visual_label_interaction = self.candidate_value_fuse(
                self.candidate_value_norm(values)
                * torch.tanh(label_values)
            )
            candidate_value_scale = (
                0.5 * torch.tanh(self.candidate_value_gate)
            ).to(values.dtype)
            values = (
                values
                + candidate_value_scale * visual_label_interaction
            )
        logits = torch.einsum("btr,bkr->btk", query, keys)
        logits = logits / math.sqrt(self.config.slot_rank)
        logits = logits + decision_weights.clamp_min(1e-8).log()[:, None, :]
        slot_attention = logits.softmax(-1)
        fused = torch.einsum("btk,bkr->btr", slot_attention, values)
        proposed = self.residual_out(fused)
        if self.config.adaptive_need_gate:
            need_hidden = self.need_hidden(
                self.decision_norm(hidden_states)
            )
            need_evidence = self.need_evidence(fused)
            need_logits = (
                (need_hidden * need_evidence).sum(-1)
                / math.sqrt(self.config.slot_rank)
                + self.need_bias
            )
            need_probability = need_logits.sigmoid()
        else:
            need_logits = torch.zeros(
                hidden_states.shape[:2],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            need_probability = torch.ones_like(need_logits)

        hidden_norm = hidden_states.float().norm(dim=-1, keepdim=True)
        proposed_norm = proposed.float().norm(dim=-1, keepdim=True)
        # The learned branch supplies a direction; the scalar gate supplies
        # the actual, interpretable fraction of native hidden-state norm.
        # Multiplying an already-small random projection by 0.02 made earlier
        # residuals orders of magnitude below their declared budget.
        direction = proposed.float() / proposed_norm.clamp_min(1e-12)
        scale = (
            self.config.max_residual_scale
            * torch.tanh(self.residual_gate.float())
        )
        effective_scale = scale * need_probability.float()[..., None]
        residual = (direction * hidden_norm * effective_scale).to(
            hidden_states.dtype
        )
        if ablation == "off":
            residual = torch.zeros_like(hidden_states)
        return residual, {
            "slot_attention": slot_attention.reshape(
                hidden_states.shape[0],
                hidden_states.shape[1],
                candidates,
                abilities,
            ),
            "scale": scale,
            "candidate_value_scale": candidate_value_scale,
            "need_logits": need_logits,
            "need_probability": need_probability,
            "effective_scale": effective_scale,
            "residual": residual,
            "residual_to_hidden_norm": (
                residual.float().norm(dim=-1)
                / hidden_norm.squeeze(-1).clamp_min(1e-12)
            ),
        }

    def candidate_memory_residual(
        self,
        hidden_states: Tensor,
        route: dict[str, Tensor],
        candidate_positions: Tensor,
        *,
        ablation: str = "full",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Write visual evidence into existing candidate-label prompt tokens.

        This path is intentionally distinct from direct action-token
        cross-attention.  Candidate-specific evidence is bound to the native
        prompt token that names that candidate, then ordinary downstream Qwen
        layers propagate it to the action position.
        """

        if not self.config.candidate_memory_injection:
            raise RuntimeError("candidate-memory injection is not configured")
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden]"
            )
        candidate_low = route["candidate_slot_low"]
        candidate_mask = route["candidate_mask"].bool()
        weights = route["ability_weights"]
        batch, candidates, abilities, rank = candidate_low.shape
        if (
            batch != hidden_states.shape[0]
            or abilities != len(ABILITIES)
            or rank != self.config.slot_rank
        ):
            raise ValueError("candidate evidence does not match hidden states")
        if candidate_positions.shape != (batch, candidates):
            raise ValueError("candidate positions do not match evidence slots")

        proposed = self.candidate_memory_vectors(route)
        sequence = hidden_states.shape[1]
        positions = candidate_positions.to(hidden_states.device)
        valid = (
            candidate_mask
            & positions.ge(0)
            & positions.lt(sequence)
        )
        safe_positions = positions.clamp(0, max(0, sequence - 1))
        native_at_candidate = hidden_states.gather(
            1,
            safe_positions[..., None].expand(
                -1, -1, hidden_states.shape[-1]
            ),
        )
        native_norm = native_at_candidate.float().norm(
            dim=-1, keepdim=True
        )
        proposed_norm = proposed.float().norm(dim=-1, keepdim=True)
        direction = proposed.float() / proposed_norm.clamp_min(1e-12)
        scale = (
            self.config.max_residual_scale
            * torch.tanh(self.candidate_memory_gate.float())
        )
        candidate_residual = (
            direction * native_norm * scale
        ).to(hidden_states.dtype)
        candidate_residual = (
            candidate_residual * valid[..., None].to(hidden_states.dtype)
        )
        residual = torch.zeros_like(hidden_states).scatter_add(
            1,
            safe_positions[..., None].expand_as(candidate_residual),
            candidate_residual,
        )
        if ablation == "off":
            residual = torch.zeros_like(hidden_states)
        elif ablation not in {
            "full",
            "shuffle_candidate",
            "shuffle_relation",
            "shuffle_ability",
            "shuffle_image",
        }:
            raise ValueError(f"unknown SC-ESR ablation: {ablation}")
        hidden_norm = hidden_states.float().norm(
            dim=-1
        ).clamp_min(1e-12)
        return residual, {
            "scale": scale,
            "candidate_memory_scale": scale,
            "candidate_memory_valid": valid,
            "residual": residual,
            "residual_to_hidden_norm": (
                residual.float().norm(dim=-1) / hidden_norm
            ),
        }

    def candidate_memory_vectors(
        self, route: dict[str, Tensor]
    ) -> Tensor:
        """Return the exact semantic vectors used by memory injection."""

        if not self.config.candidate_memory_injection:
            raise RuntimeError("candidate-memory injection is not configured")
        candidate_low = route["candidate_slot_low"]
        weights = route["ability_weights"]
        if candidate_low.ndim != 4 or weights.shape != (
            candidate_low.shape[0],
            candidate_low.shape[2],
        ):
            raise ValueError("candidate memory route shape drift")
        if self.config.latent_spatial_rollout:
            # These are the exact per-ability relation vectors supervised by
            # all-candidate targets.  Their selected mixture is the exact value
            # written into native candidate tokens at runtime.
            candidate_relations = route["candidate_slots"]
            return (
                candidate_relations
                * weights[:, None, :, None].to(candidate_relations.dtype)
            ).sum(dim=2)
        selected = (
            candidate_low * weights[:, None, :, None]
        ).sum(dim=2)
        return self.candidate_memory_out(
            self.candidate_memory_norm(selected)
        )

    def current_global_local_scale(self) -> Tensor:
        """Return the bounded global-to-local interaction coefficient."""

        if not self.config.global_conditioned_local_slots:
            return self.residual_gate.detach() * 0.0
        return 0.5 * torch.tanh(self.global_local_gate.float())

    def current_candidate_value_scale(self) -> Tensor:
        """Return the bounded visual-label value interaction coefficient."""

        if not self.config.candidate_value_interaction:
            return self.residual_gate.detach() * 0.0
        return 0.5 * torch.tanh(self.candidate_value_gate.float())

    def current_candidate_memory_scale(self) -> Tensor:
        """Return the bounded candidate-token memory residual scale."""

        if not self.config.candidate_memory_injection:
            return self.residual_gate.detach() * 0.0
        return (
            self.config.max_residual_scale
            * torch.tanh(self.candidate_memory_gate.float())
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "sc_esr_config.json").write_text(
            json.dumps(asdict(self.config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        torch.save(self.state_dict(), output / "sc_esr.pt")

    @classmethod
    def from_pretrained(
        cls, path: str | Path, map_location: Any = "cpu"
    ) -> "StateConditionedEvidenceSlotResidual":
        root = Path(path)
        module = cls(
            SCESRConfig(
                **json.loads((root / "sc_esr_config.json").read_text())
            )
        )
        module.load_state_dict(
            torch.load(
                root / "sc_esr.pt",
                map_location=map_location,
                weights_only=True,
            )
        )
        return module


class SCESRController:
    """Install one evidence-slot residual in the native Qwen decoder stack."""

    def __init__(
        self,
        model: nn.Module,
        residual: StateConditionedEvidenceSlotResidual,
        *,
        route_seed: int = 0,
    ) -> None:
        self.model = model
        self.residual = residual
        self.enabled = True
        self.ablation = "full"
        self._image_mask: Tensor | None = None
        self._token_mask: Tensor | None = None
        self._route_positions: Tensor | None = None
        self._candidate_positions: Tensor | None = None
        self._candidate_mask: Tensor | None = None
        self._candidate_image_mask: Tensor | None = None
        self._candidate_local_valid: Tensor | None = None
        self._state_dropout_mask: Tensor | None = None
        self._route_seed = int(route_seed)
        self._route_generators: dict[str, torch.Generator] = {}
        self.last_route: dict[str, Tensor] | None = None
        self.last_residual_audit: dict[str, Tensor] | None = None
        self._handles: list[Any] = []

        name, layers = self._find_native_layers(model)
        self.native_layer_path = name
        config = residual.config
        if len(layers) < config.injection_layer:
            raise RuntimeError(
                f"native model has {len(layers)} layers but SC-ESR requests "
                f"layer {config.injection_layer}"
            )
        self._handles.append(
            layers[config.source_layer - 1].register_forward_hook(
                self._source_hook
            )
        )
        self._handles.append(
            layers[config.injection_layer - 1].register_forward_hook(
                self._injection_hook
            )
        )

    @staticmethod
    def _find_native_layers(
        model: nn.Module,
    ) -> tuple[str, nn.ModuleList]:
        candidates: list[tuple[str, nn.ModuleList]] = []
        for name, module in model.named_modules():
            layers = getattr(module, "layers", None)
            if (
                name.endswith("model.language_model")
                and isinstance(layers, nn.ModuleList)
            ):
                candidates.append((name, layers))
        if len(candidates) != 1:
            raise RuntimeError(
                "expected one native model.language_model.layers, found "
                f"{[name for name, _ in candidates]}"
            )
        return candidates[0]

    @staticmethod
    def _hidden(output: Any) -> Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output:
            if torch.is_tensor(output[0]):
                return output[0]
        raise RuntimeError("native decoder layer did not expose hidden states")

    @staticmethod
    def _replace_hidden(output: Any, hidden: Tensor) -> Any:
        if torch.is_tensor(output):
            return hidden
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        if isinstance(output, list):
            return [hidden, *output[1:]]
        raise RuntimeError("cannot replace native decoder hidden states")

    def reset(self) -> None:
        self.last_route = None
        self.last_residual_audit = None

    def _dropout_mask(self, hidden: Tensor) -> Tensor | None:
        probability = self.residual.config.state_dropout
        if not self.residual.training or probability <= 0:
            return None
        if self._state_dropout_mask is not None:
            return self._state_dropout_mask
        key = str(hidden.device)
        generator = self._route_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=hidden.device)
            generator.manual_seed(self._route_seed)
            self._route_generators[key] = generator
        # A dedicated generator prevents an auxiliary routing decision from
        # perturbing native Transformer/LoRA dropout.  The stored mask is also
        # reused by gradient-checkpoint recomputation of this same forward.
        self._state_dropout_mask = (
            torch.rand(
                hidden.shape[0],
                1,
                device=hidden.device,
                generator=generator,
            )
            < probability
        )
        return self._state_dropout_mask

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _source_hook(
        self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        if not self.enabled:
            return
        hidden = self._hidden(output)
        mask = self._image_mask
        # Cached one-token decoding reuses the evidence slots created by the
        # prefill and must not invent a text-only route.
        if mask is None or mask.shape != hidden.shape[:2]:
            return
        positions = self._route_positions
        if positions is not None and positions.shape == (hidden.shape[0],):
            positions = positions.to(hidden.device)
            valid = positions.ge(0) & positions.lt(hidden.shape[1])
            safe = positions.clamp(0, hidden.shape[1] - 1)
            selected = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), safe
            ]
            general = torch.where(valid[:, None], selected, hidden[:, -1])
        else:
            general = hidden[:, -1]
        candidate_context: Tensor | None = None
        candidate_mask = self._candidate_mask
        if self.residual.config.candidate_conditioned:
            candidate_positions = self._candidate_positions
            if candidate_positions is None or candidate_mask is None:
                raise RuntimeError(
                    "candidate-conditioned SC-ESR has no candidate positions"
                )
            if candidate_positions.shape != candidate_mask.shape:
                raise RuntimeError("candidate positions/mask shape drift")
            if candidate_positions.shape[0] != hidden.shape[0]:
                raise RuntimeError("candidate positions batch drift")
            valid_candidates = candidate_mask.to(hidden.device).bool()
            safe_candidates = candidate_positions.to(hidden.device).clamp(
                0, hidden.shape[1] - 1
            )
            candidate_context = hidden.gather(
                1,
                safe_candidates[..., None].expand(
                    -1, -1, hidden.shape[-1]
                ),
            )
            # Auxiliary spatial rows have no Move candidate list.  Their
            # single global fallback uses the prompt-terminal representation,
            # while action rows remain explicitly candidate-conditioned.
            empty = ~valid_candidates.any(1)
            if empty.any():
                candidate_context = candidate_context.clone()
                valid_candidates = valid_candidates.clone()
                candidate_context[empty, 0] = general[empty]
                valid_candidates[empty, 0] = True
            candidate_mask = valid_candidates
        self.last_route = self.residual.extract_slots(
            hidden,
            mask.to(hidden.device),
            general,
            candidate_context=candidate_context,
            candidate_mask=candidate_mask,
            candidate_image_mask=self._candidate_image_mask,
            candidate_local_valid=self._candidate_local_valid,
            state_dropout_mask=self._dropout_mask(hidden),
            ablation=self.ablation,
        )

    def _active_mask(self, hidden: Tensor) -> Tensor | None:
        mask = self._token_mask
        if mask is None:
            return None
        if mask.shape == hidden.shape[:2]:
            return mask.to(hidden.device).bool()
        if (
            mask.ndim == 2
            and mask.shape[0] == hidden.shape[0]
            and hidden.shape[1] == 1
        ):
            return mask[:, -1:].to(hidden.device).bool()
        raise RuntimeError("SC-ESR token mask does not match native hidden states")

    def _injection_hook(
        self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> Any:
        if not self.enabled:
            return output
        if self.last_route is None:
            # A cached call is valid only after prefill created the route.
            raise RuntimeError("SC-ESR injection ran before evidence extraction")
        hidden = self._hidden(output)
        if self.residual.config.candidate_memory_injection:
            if self._candidate_positions is None:
                raise RuntimeError(
                    "candidate-memory injection has no candidate positions"
                )
            residual, audit = self.residual.candidate_memory_residual(
                hidden,
                self.last_route,
                self._candidate_positions,
                ablation=self.ablation,
            )
        else:
            residual, audit = self.residual.decision_residual(
                hidden, self.last_route, ablation=self.ablation
            )
            active = self._active_mask(hidden)
            if active is not None:
                residual = residual * active[..., None].to(residual.dtype)
                audit["residual"] = residual
                audit["residual_to_hidden_norm"] = (
                    residual.float().norm(dim=-1)
                    / hidden.float().norm(dim=-1).clamp_min(1e-12)
                )
        self.last_residual_audit = audit
        if self.ablation == "off":
            return output
        return self._replace_hidden(output, hidden + residual)

    @contextmanager
    def context(
        self,
        *,
        image_mask: Tensor,
        token_mask: Tensor | None,
        route_positions: Tensor | None = None,
        candidate_positions: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        candidate_image_mask: Tensor | None = None,
        candidate_local_valid: Tensor | None = None,
        state_dropout_mask: Tensor | None = None,
        enabled: bool = True,
        ablation: str = "full",
    ) -> Iterator["SCESRController"]:
        if ablation not in {
            "full",
            "off",
            "shuffle_ability",
            "shuffle_image",
            "shuffle_candidate",
            "shuffle_relation",
        }:
            raise ValueError(f"unknown SC-ESR ablation: {ablation}")
        previous = (
            self._image_mask,
            self._token_mask,
            self._route_positions,
            self._candidate_positions,
            self._candidate_mask,
            self._candidate_image_mask,
            self._candidate_local_valid,
            self._state_dropout_mask,
            self.enabled,
            self.ablation,
            self.last_route,
            self.last_residual_audit,
        )
        self._image_mask = image_mask
        self._token_mask = token_mask
        self._route_positions = route_positions
        self._candidate_positions = candidate_positions
        self._candidate_mask = candidate_mask
        self._candidate_image_mask = candidate_image_mask
        self._candidate_local_valid = candidate_local_valid
        self._state_dropout_mask = state_dropout_mask
        self.enabled = bool(enabled)
        self.ablation = ablation
        self.reset()
        try:
            yield self
        finally:
            (
                self._image_mask,
                self._token_mask,
                self._route_positions,
                self._candidate_positions,
                self._candidate_mask,
                self._candidate_image_mask,
                self._candidate_local_valid,
                self._state_dropout_mask,
                self.enabled,
                self.ablation,
                self.last_route,
                self.last_residual_audit,
            ) = previous


def evidence_slot_contrastive_loss(
    slots: Tensor,
    targets: Tensor,
    active: Tensor,
    *,
    temperature: float = 0.07,
) -> Tensor:
    """Match each image slot to same-image evidence with hard batch negatives.

    Negatives are restricted to the same ability index.  Dataset construction
    is responsible for grouping comparable goal/state examples; this function
    prevents a trivial ability-name solution by making the row identity the
    contrastive target.
    """
    if slots.shape != targets.shape or slots.ndim != 3:
        raise ValueError("slots and targets must share [batch, ability, hidden]")
    if active.shape != slots.shape[:2]:
        raise ValueError("active evidence mask does not align with slots")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    losses: list[Tensor] = []
    for ability in range(slots.shape[1]):
        rows = active[:, ability].bool()
        if int(rows.sum()) < 2:
            continue
        query = F.normalize(slots[rows].select(1, ability).float(), dim=-1)
        target = F.normalize(
            targets[rows].select(1, ability).float(), dim=-1
        )
        logits = query @ target.transpose(0, 1) / float(temperature)
        labels = torch.arange(logits.shape[0], device=logits.device)
        losses.append(
            0.5
            * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.transpose(0, 1), labels)
            )
        )
    if not losses:
        return slots.sum() * 0.0
    return torch.stack(losses).mean()


def candidate_relation_alignment_loss(
    candidate_representations: Tensor,
    relation_targets: Tensor,
    gold_indices: Tensor,
    active: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Align the gold candidate's visual slots with an action-blind relation.

    The supervision says what the gold candidate visibly relates to, but the
    target text contains no candidate ID or action.  Gold identity is used only
    to select the corresponding current-image slot; the original LM head
    remains solely responsible for predicting the action.
    """

    if candidate_representations.ndim != 3:
        raise ValueError(
            "candidate representations must have [batch, candidate, hidden]"
        )
    batch, candidates, hidden = candidate_representations.shape
    if relation_targets.shape != (batch, hidden):
        raise ValueError(
            "relation targets do not align with candidate representations"
        )
    if gold_indices.shape != (batch,) or active.shape != (batch,):
        raise ValueError("candidate relation row metadata shape drift")
    valid = (
        active.bool()
        & gold_indices.ge(0)
        & gold_indices.lt(candidates)
    )
    safe_gold = gold_indices.clamp(0, candidates - 1)
    rows = torch.arange(
        batch, device=candidate_representations.device
    )
    pooled = candidate_representations[rows, safe_gold].float()
    cosine = (
        F.normalize(pooled, dim=-1)
        * F.normalize(relation_targets.float(), dim=-1)
    ).sum(-1)
    if not valid.any():
        return pooled.sum() * 0.0, cosine.detach(), valid
    return (1.0 - cosine[valid]).mean(), cosine.detach(), valid


def all_candidate_relation_alignment_loss(
    candidate_relations: Tensor,
    relation_targets: Tensor,
    active: Tensor,
    *,
    negative_margin: float = 0.10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Align every used ability/candidate relation with an action-blind target.

    ``candidate_relations`` is the exact tensor consumed by SC-LSR's native
    decision path. A within-image candidate roll supplies a hard negative
    without consulting the policy action.
    """

    if (
        candidate_relations.ndim != 4
        or relation_targets.shape != candidate_relations.shape
        or active.shape != candidate_relations.shape[:3]
    ):
        raise ValueError(
            "all-candidate relations require matching "
            "[batch, candidate, ability, hidden] tensors"
        )
    if negative_margin < 0:
        raise ValueError("relation negative margin must be non-negative")
    valid = active.bool()
    relation = F.normalize(candidate_relations.float(), dim=-1)
    target = F.normalize(relation_targets.float(), dim=-1)
    positive = (relation * target).sum(-1)
    if not valid.any():
        return relation.sum() * 0.0, positive.detach(), valid
    positive_loss = (1.0 - positive[valid]).mean()

    negative_target = target.roll(1, dims=1)
    negative_valid = valid & valid.roll(1, dims=1)
    negative = (relation * negative_target).sum(-1)
    if negative_valid.any():
        ranking = F.relu(
            float(negative_margin)
            - positive[negative_valid]
            + negative[negative_valid]
        ).mean()
    else:
        ranking = positive_loss * 0.0
    return positive_loss + ranking, positive.detach(), valid


def residual_utility_margin_loss(
    correct_margin: Tensor,
    counterfactual_margin: Tensor,
    *,
    margin: float = 0.05,
) -> Tensor:
    """Require current-image evidence to help more than mismatched evidence."""
    if correct_margin.shape != counterfactual_margin.shape:
        raise ValueError("utility margins must have equal shape")
    return F.relu(
        float(margin) - (correct_margin - counterfactual_margin)
    ).mean()


def residual_preservation_loss(
    full_margin: Tensor,
    off_margin: Tensor,
    *,
    tolerance: float = 0.02,
) -> Tensor:
    """Keep the spatial increment from damaging reliable navigation anchors."""
    if full_margin.shape != off_margin.shape:
        raise ValueError("preservation margins must have equal shape")
    return F.relu(off_margin.detach() - float(tolerance) - full_margin).mean()
