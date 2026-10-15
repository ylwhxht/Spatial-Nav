"""Frozen navigation-anchor LoRA plus a separately trainable delta LoRA."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from peft import LoraConfig, PeftMixedModel, PeftModel
from torch import nn


ANCHOR_ADAPTER = "anchor"
DELTA_ADAPTER = "default"


def load_trainable_adapter_continuation(
    base: nn.Module,
    adapter: str | Path,
) -> PeftModel:
    """Load one existing LoRA as the only trainable navigation adapter.

    This is the simple continuation path: the input adapter is updated in
    place conceptually and the saved result remains one normal PEFT adapter,
    rather than an anchor-plus-delta runtime stack.
    """

    model = PeftModel.from_pretrained(
        base,
        str(adapter),
        adapter_name=DELTA_ADAPTER,
        is_trainable=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    matched = 0
    marker = f".{DELTA_ADAPTER}."
    for name, parameter in model.named_parameters():
        if marker in name and ("lora_A" in name or "lora_B" in name):
            parameter.requires_grad_(True)
            matched += parameter.numel()
    if matched == 0:
        raise RuntimeError("continued LoRA has no trainable parameters")
    return model


def load_frozen_anchor_with_delta(
    base: nn.Module,
    anchor_adapter: str | Path,
    *,
    delta_adapter: str | Path | None = None,
    delta_config: LoraConfig | None = None,
    train_delta: bool = False,
) -> PeftMixedModel:
    """Load two additive LoRAs without merging or bypassing Qwen."""
    if (delta_adapter is None) == (delta_config is None):
        raise ValueError("supply exactly one of delta_adapter or delta_config")
    if train_delta and delta_config is None:
        raise ValueError("train_delta requires a freshly-created delta_config")

    model = PeftMixedModel.from_pretrained(
        base, str(anchor_adapter), adapter_name=ANCHOR_ADAPTER,
        is_trainable=False,
    )
    if delta_config is not None:
        model.add_adapter(DELTA_ADAPTER, delta_config)
    else:
        model.load_adapter(
            str(delta_adapter), adapter_name=DELTA_ADAPTER,
            is_trainable=False,
        )
    model.set_adapter([ANCHOR_ADAPTER, DELTA_ADAPTER])

    # set_adapter marks active adapters trainable. Restore the stronger
    # invariant explicitly: only the fresh delta may receive gradients.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if train_delta:
        matched = 0
        marker = f".{DELTA_ADAPTER}."
        for name, parameter in model.named_parameters():
            if marker in name and ("lora_A" in name or "lora_B" in name):
                parameter.requires_grad_(True)
                matched += parameter.numel()
        if matched == 0:
            raise RuntimeError("delta LoRA has no trainable parameters")
    return model


def load_frozen_anchor_with_named_deltas(
    base: nn.Module,
    anchor_adapter: str | Path,
    delta_adapters: Mapping[str, str | Path],
) -> PeftMixedModel:
    """Load one immutable anchor and multiple immutable named LoRA deltas.

    Exactly one named delta is activated together with the anchor at a time.
    This supports Move/Stop routing in one resident base-model process without
    merging adapters or bypassing any native Qwen component.
    """

    if not delta_adapters:
        raise ValueError("at least one named delta adapter is required")
    invalid = {
        name
        for name in delta_adapters
        if not name or name == ANCHOR_ADAPTER
    }
    if invalid:
        raise ValueError(f"invalid or reserved delta adapter names: {invalid}")

    model = PeftMixedModel.from_pretrained(
        base,
        str(anchor_adapter),
        adapter_name=ANCHOR_ADAPTER,
        is_trainable=False,
    )
    for name, path in delta_adapters.items():
        model.load_adapter(
            str(path),
            adapter_name=name,
            is_trainable=False,
        )
    first = next(iter(delta_adapters))
    model.set_adapter([ANCHOR_ADAPTER, first])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def named_adapter_parameter_counts(
    model: nn.Module,
    adapter_names: list[str] | tuple[str, ...],
) -> dict[str, int]:
    """Count total/trainable LoRA parameters for explicitly named adapters."""

    result: dict[str, int] = {}
    for adapter in adapter_names:
        marker = f".{adapter}."
        params = [
            parameter
            for name, parameter in model.named_parameters()
            if marker in name and ("lora_A" in name or "lora_B" in name)
        ]
        result[f"{adapter}_total"] = sum(p.numel() for p in params)
        result[f"{adapter}_trainable"] = sum(
            p.numel() for p in params if p.requires_grad
        )
    return result


def activate_frozen_adapter_stack(
    model: PeftMixedModel,
    adapter_names: list[str] | tuple[str, ...],
) -> None:
    """Activate an adapter stack while preserving inference-only freezing.

    PEFT's public ``set_adapter`` intentionally marks active adapters
    trainable.  Runtime adapter routing must undo that side effect after every
    switch so no branch can silently become a mutable or graph-building path.
    """

    model.set_adapter(list(adapter_names))
    for name, parameter in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.requires_grad_(False)


def adapter_parameter_counts(model: nn.Module) -> dict[str, int]:
    result: dict[str, int] = {}
    for adapter in (ANCHOR_ADAPTER, DELTA_ADAPTER):
        marker = f".{adapter}."
        params = [
            parameter for name, parameter in model.named_parameters()
            if marker in name and ("lora_A" in name or "lora_B" in name)
        ]
        result[f"{adapter}_total"] = sum(p.numel() for p in params)
        result[f"{adapter}_trainable"] = sum(
            p.numel() for p in params if p.requires_grad
        )
    return result
