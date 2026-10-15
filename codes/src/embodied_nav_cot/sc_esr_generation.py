"""Single-forward native Qwen generation with an internal SC-ESR residual."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import torch

from embodied_nav_cot.sc_esr import (
    ABILITIES,
    SCESRController,
    StateConditionedEvidenceSlotResidual,
)
from embodied_nav_cot.move_sc_rle_protocol import (
    public_move_inputs_from_messages,
)
from embodied_nav_cot.candidate_overlay_regions import (
    candidate_overlay_token_masks,
)


def prompt_terminal_positions(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    positions: list[int] = []
    for row in range(attention_mask.shape[0]):
        active = attention_mask[row].nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            raise ValueError("generation request has no active prompt tokens")
        positions.append(int(active[-1]))
    return torch.tensor(
        positions, dtype=torch.long, device=attention_mask.device
    )


def generation_token_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Activate the prompt terminal and every subsequent cached token."""
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    positions = prompt_terminal_positions(attention_mask)
    mask[
        torch.arange(mask.shape[0], device=mask.device),
        positions,
    ] = True
    return mask


def candidate_terminal_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    legal_actions: Sequence[Sequence[str]],
    *,
    prefix_end_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate public candidate labels in the prompt token sequence.

    Positions point at the final token of each full ``candidate_k`` string.
    Matching is restricted to the active prompt prefix, so an assistant target
    cannot leak the gold action into the evidence query during training.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids/attention_mask shape mismatch")
    if len(legal_actions) != input_ids.shape[0]:
        raise ValueError("legal action batch mismatch")
    if prefix_end_positions is None:
        prefix_end_positions = prompt_terminal_positions(attention_mask)
    if prefix_end_positions.shape != (input_ids.shape[0],):
        raise ValueError("prefix_end_positions must have shape [batch]")
    width = max([len(actions) for actions in legal_actions] or [1])
    width = max(1, width)
    positions = torch.full(
        (input_ids.shape[0], width),
        -1,
        dtype=torch.long,
        device=input_ids.device,
    )
    valid = torch.zeros_like(positions, dtype=torch.bool)
    for row, actions in enumerate(legal_actions):
        end = int(prefix_end_positions[row]) + 1
        active_start = int(
            attention_mask[row, :end].nonzero(as_tuple=True)[0][0]
        )
        haystack = input_ids[row, active_start:end].tolist()
        for column, action in enumerate(actions):
            if not str(action).startswith("candidate_"):
                raise ValueError(f"invalid public Move action: {action}")
            patterns = []
            for text in (str(action), f" {action}"):
                pattern = tokenizer(
                    text, add_special_tokens=False
                )["input_ids"]
                if pattern and pattern not in patterns:
                    patterns.append(pattern)
            matches: list[tuple[int, int]] = []
            for pattern in patterns:
                size = len(pattern)
                for offset in range(0, len(haystack) - size + 1):
                    if haystack[offset : offset + size] == pattern:
                        matches.append((offset, size))
            if not matches:
                raise ValueError(
                    f"candidate label missing from prompt tokens: {action}; "
                    f"row={row}, prefix_tokens={len(haystack)}, "
                    f"decoded_tail={tokenizer.decode(haystack[-160:])!r}"
                )
            # The canonical prompt lists each concrete action once.  If a
            # tokenizer yields both plain/space matches, choose the earliest
            # occurrence and longest exact label representation.
            offset, size = sorted(matches, key=lambda item: (item[0], -item[1]))[0]
            positions[row, column] = active_start + offset + size - 1
            valid[row, column] = True
    return positions, valid


@dataclass(frozen=True)
class SCESRGenerationResult:
    text: str
    navigation_state: int | None
    abilities: tuple[str, ...]
    ability_scores: tuple[float, ...]
    ablation: str
    batch_size: int
    effective_ablation: str
    full_request_passes: int = 1
    local_candidates: int = 0
    public_candidates: int = 0


class NativeSCESRGenerator:
    """Batchable direct-decision runtime; no evidence cache or visible CoT."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: Any,
        residual: StateConditionedEvidenceSlotResidual,
        image_token_id: int = 151655,
    ) -> None:
        self.model = model
        self.processor = processor
        self.residual = residual
        self.image_token_id = int(image_token_id)
        self.controller = SCESRController(model, residual)
        self.model.eval()
        self.residual.eval()
        self.processor.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        *,
        base_model: str | Path,
        adapter: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        image_token_id: int = 151655,
    ) -> "NativeSCESRGenerator":
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        adapter = Path(adapter)
        processor = AutoProcessor.from_pretrained(
            str(base_model), trust_remote_code=True, use_fast=False
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(base_model),
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        model = PeftModel.from_pretrained(
            model, str(adapter), is_trainable=False
        )
        residual = StateConditionedEvidenceSlotResidual.from_pretrained(
            adapter / "sc_esr", map_location="cpu"
        ).to(device=device, dtype=dtype)
        model.add_module("_sc_esr_residual", residual)
        return cls(
            model=model,
            processor=processor,
            residual=residual,
            image_token_id=image_token_id,
        )

    def close(self) -> None:
        self.controller.close()

    @torch.inference_mode()
    def generate(
        self,
        *,
        messages: Sequence[Sequence[dict[str, Any]]],
        images: Sequence[Sequence[Image.Image]],
        max_new_tokens: int,
        temperature: float = 0.0,
        ablation: str = "full",
    ) -> list[SCESRGenerationResult]:
        if len(messages) != len(images) or not messages:
            raise ValueError("messages/images batch must be non-empty and aligned")
        rendered = [
            self.processor.apply_chat_template(
                list(item), tokenize=False, add_generation_prompt=True
            )
            for item in messages
        ]
        inputs = self.processor(
            text=rendered,
            images=[list(item) for item in images],
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_mask = input_ids.eq(self.image_token_id)
        if not bool(image_mask.any(1).all()):
            raise ValueError("every SC-ESR request must contain current RGB")
        route_positions = prompt_terminal_positions(attention_mask)
        token_mask = generation_token_mask(attention_mask)
        legal_actions = [
            public_move_inputs_from_messages(item)[1] for item in messages
        ]
        candidate_positions, candidate_mask = candidate_terminal_positions(
            input_ids,
            attention_mask,
            self.processor.tokenizer,
            legal_actions,
            prefix_end_positions=route_positions,
        )
        candidate_image_mask: torch.Tensor | None = None
        candidate_local_valid: torch.Tensor | None = None
        if self.residual.config.candidate_local_masks:
            if any(len(item) != 1 for item in images):
                raise ValueError(
                    "candidate-local Move requires one current overlay RGB"
                )
            (
                candidate_image_mask,
                candidate_local_valid,
                _detections,
            ) = candidate_overlay_token_masks(
                images=[item[0] for item in images],
                legal_actions=legal_actions,
                input_ids=input_ids.detach().cpu(),
                image_grid_thw=inputs["image_grid_thw"].detach().cpu(),
                image_token_id=self.image_token_id,
                candidate_width=candidate_positions.shape[1],
            )
            candidate_image_mask = candidate_image_mask.to(device)
            candidate_local_valid = candidate_local_valid.to(device)
        generation_args: dict[str, Any] = {
            "max_new_tokens": min(24, int(max_new_tokens)),
            "do_sample": temperature > 0,
            "use_cache": True,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_args["temperature"] = float(temperature)
        with self.controller.context(
            image_mask=image_mask,
            token_mask=token_mask,
            route_positions=route_positions,
            candidate_positions=candidate_positions,
            candidate_mask=candidate_mask,
            candidate_image_mask=candidate_image_mask,
            candidate_local_valid=candidate_local_valid,
            ablation=ablation,
        ):
            outputs = self.model.generate(**inputs, **generation_args)
            route = self.controller.last_route
        if route is None and ablation != "off":
            raise RuntimeError("SC-ESR generation did not create a route")
        generated = outputs[:, input_ids.shape[1] :]
        texts = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )
        results: list[SCESRGenerationResult] = []
        batch_size = len(messages)
        effective_ablation = (
            "shuffle_ability_fallback"
            if ablation == "shuffle_image" and batch_size == 1
            else ablation
        )
        for row, text in enumerate(texts):
            if route is None:
                state = None
                scores = (0.0,) * len(ABILITIES)
                selected: tuple[str, ...] = ()
            else:
                state = int(route["state_logits"][row].argmax().item())
                scores = tuple(
                    float(value)
                    for value in route["ability_logits"][row]
                    .float()
                    .sigmoid()
                    .tolist()
                )
                selected = tuple(
                    ability
                    for ability, enabled in zip(
                        ABILITIES, route["ability_mask"][row].tolist()
                    )
                    if enabled
                )
            local_candidates = (
                int(route["candidate_local_valid"][row].sum().item())
                if route is not None
                else 0
            )
            public_candidates = (
                int(route["candidate_mask"][row].sum().item())
                if route is not None
                else 0
            )
            results.append(
                SCESRGenerationResult(
                    text=text,
                    navigation_state=state,
                    abilities=selected,
                    ability_scores=scores,
                    ablation=ablation,
                    batch_size=batch_size,
                    effective_ablation=effective_ablation,
                    local_candidates=local_candidates,
                    public_candidates=public_candidates,
                )
            )
        return results
