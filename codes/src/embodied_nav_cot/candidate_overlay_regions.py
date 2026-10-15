"""Public candidate-overlay regions for Qwen image-token grounding.

The ObjectNav Move image already contains the only geometry used here: red
candidate rays and numbered white endpoint circles.  This module detects those
visible endpoint circles and converts each ray corridor into a mask over the
native Qwen image-token sequence.  It never consumes simulator poses, depth,
landing positions, oracle actions, or trajectory metadata.

Detection is deliberately fail-safe.  A candidate whose endpoint cannot be
identified receives the ordinary global image mask and is marked as not
locally grounded; callers can then disable spatial-gain supervision for that
candidate while the native LM policy remains fully available.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
from torch import Tensor


_CANDIDATE_RE = re.compile(r"candidate_(\d+)$")


@dataclass(frozen=True)
class CandidateEndpoint:
    """One endpoint inferred only from the rendered public overlay."""

    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class CandidateOverlayDetection:
    """Normalized endpoint coordinates keyed by public candidate number."""

    endpoints: dict[int, CandidateEndpoint]
    forward_components: int
    zero_marker_detected: bool
    image_width: int
    image_height: int


def _candidate_number(value: str) -> int:
    match = _CANDIDATE_RE.fullmatch(str(value))
    if match is None:
        raise ValueError(f"invalid public Move candidate: {value!r}")
    return int(match.group(1))


def _resize_for_detection(image: Image.Image, max_dimension: int = 512) -> np.ndarray:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        rgb = rgb.resize(
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(rgb)


def detect_public_candidate_endpoints(
    image: Image.Image,
    legal_actions: Sequence[str],
) -> CandidateOverlayDetection:
    """Detect high-confidence visible candidate endpoints.

    Candidate labels are assigned from left to right because VLMNav renders
    forward candidates after sorting their egocentric bearing.  The special
    candidate_0 marker is detected separately near the left image edge, but
    its useful evidence remains global rather than the marker crop itself.
    """

    action_numbers = [_candidate_number(action) for action in legal_actions]
    if len(action_numbers) != len(set(action_numbers)):
        raise ValueError("duplicate public Move candidates")
    positive_numbers = sorted(number for number in action_numbers if number > 0)

    array = _resize_for_detection(image)
    height, width = array.shape[:2]
    red_channel = array[..., 0].astype(np.int16)
    green_channel = array[..., 1].astype(np.int16)
    blue_channel = array[..., 2].astype(np.int16)

    maximum = np.maximum.reduce((red_channel, green_channel, blue_channel))
    minimum = np.minimum.reduce((red_channel, green_channel, blue_channel))
    # These thresholds cover both lossless training PNGs and deployment's
    # 512-pixel JPEG-85 transport.
    white = (
        (minimum > 205)
        & ((maximum - minimum) < 45)
    ).astype(np.uint8)
    red = (
        (red_channel > 135)
        & ((red_channel - green_channel) > 45)
        & ((red_channel - blue_channel) > 45)
        & (red_channel > 1.35 * np.maximum(green_channel, blue_channel))
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white, connectivity=8
    )
    diameter_min = max(8.0, 0.045 * height)
    diameter_max = max(24.0, 0.19 * height)
    kernel_size = max(3, int(round(0.017 * height)))
    kernel_size |= 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    components: list[tuple[float, float, float]] = []
    for component in range(1, count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[component]
        )
        if not (
            diameter_min <= box_width <= diameter_max
            and diameter_min <= box_height <= diameter_max
        ):
            continue
        aspect = box_width / max(1, box_height)
        fill = area / max(1, box_width * box_height)
        if not 0.48 <= aspect <= 2.05 or fill < 0.22:
            continue
        component_mask = (labels == component).astype(np.uint8)
        annulus = cv2.dilate(component_mask, kernel).astype(bool)
        annulus &= ~component_mask.astype(bool)
        if not annulus.any():
            continue
        red_ratio = float(red[annulus].mean())
        if red_ratio < 0.11:
            continue
        center_x, center_y = (float(value) for value in centroids[component])
        # A rendered endpoint has a dense white interior and red annulus.
        confidence = min(1.0, 0.65 * red_ratio / 0.35 + 0.35 * fill / 0.70)
        components.append((center_x, center_y, confidence))

    zero_components = [
        item
        for item in components
        if abs(item[0] / width - 0.05) <= 0.045
        and abs(item[1] / height - 0.50) <= 0.13
    ]
    forward = [
        item
        for item in components
        if item not in zero_components and item[1] / height >= 0.48
    ]
    # Scene objects can occasionally satisfy the color test.  Retain at most
    # the number of public positive candidates, preferring the strongest ring
    # evidence, then restore left-to-right bearing order.
    if len(forward) > len(positive_numbers):
        forward = sorted(
            forward, key=lambda item: item[2], reverse=True
        )[: len(positive_numbers)]
    forward.sort(key=lambda item: item[0])

    endpoints: dict[int, CandidateEndpoint] = {}
    if 0 in action_numbers and zero_components:
        center_x, center_y, confidence = max(
            zero_components, key=lambda item: item[2]
        )
        endpoints[0] = CandidateEndpoint(
            x=center_x / width,
            y=center_y / height,
            confidence=confidence,
        )
    for number, (center_x, center_y, confidence) in zip(
        positive_numbers, forward
    ):
        endpoints[number] = CandidateEndpoint(
            x=center_x / width,
            y=center_y / height,
            confidence=confidence,
        )

    return CandidateOverlayDetection(
        endpoints=endpoints,
        forward_components=len(forward),
        zero_marker_detected=bool(zero_components),
        image_width=width,
        image_height=height,
    )


def candidate_corridor_grid_mask(
    endpoint: CandidateEndpoint,
    *,
    grid_height: int,
    grid_width: int,
    corridor_radius: float = 1.75,
    endpoint_radius: float = 2.75,
) -> Tensor:
    """Return a row-major mask around a public ray and its destination."""

    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("candidate grid dimensions must be positive")
    rows = torch.arange(grid_height, dtype=torch.float32) + 0.5
    columns = torch.arange(grid_width, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    end = torch.tensor(
        [endpoint.x * grid_width, endpoint.y * grid_height],
        dtype=torch.float32,
    )
    # VLMNav projects the camera origin just below the visible image.  A small
    # extension below the bottom border captures the visible floor corridor.
    start = torch.tensor(
        [0.5 * grid_width, 1.06 * grid_height], dtype=torch.float32
    )
    segment = end - start
    denominator = float((segment * segment).sum().clamp_min(1e-6))
    projection = (
        (xx - start[0]) * segment[0]
        + (yy - start[1]) * segment[1]
    ) / denominator
    projection = projection.clamp(0.0, 1.0)
    closest_x = start[0] + projection * segment[0]
    closest_y = start[1] + projection * segment[1]
    distance_to_ray = torch.sqrt(
        (xx - closest_x).square() + (yy - closest_y).square()
    )
    distance_to_endpoint = torch.sqrt(
        (xx - end[0]).square() + (yy - end[1]).square()
    )
    mask = (
        (distance_to_ray <= corridor_radius)
        | (distance_to_endpoint <= endpoint_radius)
    )
    if int(mask.sum()) < 5:
        nearest = distance_to_endpoint.flatten().topk(
            k=min(5, grid_height * grid_width), largest=False
        ).indices
        mask = mask.flatten()
        mask[nearest] = True
        mask = mask.view(grid_height, grid_width)
    return mask.flatten()


def candidate_overlay_token_masks(
    *,
    images: Sequence[Image.Image],
    legal_actions: Sequence[Sequence[str]],
    input_ids: Tensor,
    image_grid_thw: Tensor,
    image_token_id: int,
    candidate_width: int | None = None,
    spatial_merge_size: int = 2,
) -> tuple[Tensor, Tensor, list[CandidateOverlayDetection]]:
    """Map public overlay regions to padded native image-token masks.

    Returns:
        token_masks: ``[batch, candidate, sequence]``.  Undetected candidates
            receive the global image mask, preserving the native visual path.
        local_valid: ``[batch, candidate]``.  True only for a detected positive
            candidate with an explicit ray corridor.
        detections: CPU audit metadata for each row.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    batch, sequence = input_ids.shape
    if len(images) != batch or len(legal_actions) != batch:
        raise ValueError("overlay images/actions must align with input batch")
    if image_grid_thw.shape != (batch, 3):
        raise ValueError(
            "candidate-local Move currently requires exactly one image per row"
        )
    width = candidate_width or max(
        [len(actions) for actions in legal_actions] or [1]
    )
    width = max(1, int(width))
    token_masks = torch.zeros(
        (batch, width, sequence), dtype=torch.bool
    )
    local_valid = torch.zeros((batch, width), dtype=torch.bool)
    detections: list[CandidateOverlayDetection] = []

    for row, (image, actions) in enumerate(zip(images, legal_actions)):
        detection = detect_public_candidate_endpoints(image, actions)
        detections.append(detection)
        image_positions = input_ids[row].eq(int(image_token_id)).nonzero(
            as_tuple=True
        )[0]
        grid_t, grid_h, grid_w = (
            int(value) for value in image_grid_thw[row].tolist()
        )
        if grid_t != 1:
            raise ValueError("candidate-local Move accepts still RGB only")
        if (
            grid_h % spatial_merge_size
            or grid_w % spatial_merge_size
        ):
            raise ValueError("Qwen image grid is not merge-size aligned")
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size
        expected_tokens = grid_t * merged_h * merged_w
        if image_positions.numel() != expected_tokens:
            raise ValueError(
                "Qwen image-token/grid order drift: "
                f"observed={image_positions.numel()} expected={expected_tokens}"
            )
        global_mask = torch.zeros(sequence, dtype=torch.bool)
        global_mask[image_positions] = True

        if not actions:
            # Auxiliary rows use one explicit global fallback.
            token_masks[row, 0] = global_mask
            continue
        if len(actions) > width:
            raise ValueError("candidate mask width is too small")
        for column, action in enumerate(actions):
            number = _candidate_number(action)
            endpoint = detection.endpoints.get(number)
            # Reorientation and failed detections stay global.  The residual
            # is never allowed to erase the base model's image access.
            if number == 0 or endpoint is None:
                token_masks[row, column] = global_mask
                continue
            flat = candidate_corridor_grid_mask(
                endpoint,
                grid_height=merged_h,
                grid_width=merged_w,
            )
            row_mask = torch.zeros(sequence, dtype=torch.bool)
            row_mask[image_positions] = flat
            token_masks[row, column] = row_mask
            local_valid[row, column] = True
    return token_masks, local_valid, detections


def local_gold_candidate_mask(
    local_valid: Tensor,
    gold_indices: Tensor,
    surface_valid: Tensor,
) -> Tensor:
    """Rows whose gold action has a detected local region and a comparator."""

    if local_valid.ndim != 2:
        raise ValueError("local_valid must have shape [batch, candidate]")
    if gold_indices.shape != (local_valid.shape[0],):
        raise ValueError("gold indices do not align with local candidate masks")
    safe = gold_indices.clamp(0, local_valid.shape[1] - 1)
    rows = torch.arange(local_valid.shape[0], device=local_valid.device)
    gold_local = local_valid[rows, safe]
    return (
        surface_valid.bool()
        & gold_indices.ge(0)
        & gold_indices.lt(local_valid.shape[1])
        & gold_local
        & local_valid.sum(-1).ge(2)
    )
