"""Exact, opt-in storage contract for visual-arrival certificate inputs."""

from __future__ import annotations

import hashlib
import io
from typing import Mapping

import numpy as np
from PIL import Image


DEPLOYMENT_JPEG_STORAGE_CONTRACT = "openai_vlm_runtime_jpeg_transport_v1"
DEPLOYMENT_JPEG_FILENAME = "terminal_certificate_input_rgb.jpg"
DEPLOYMENT_JPEG_FORMAT = "jpeg"
DEPLOYMENT_JPEG_MIME_TYPE = "image/jpeg"
DEPLOYMENT_JPEG_MAX_DIMENSION = 512
DEPLOYMENT_JPEG_QUALITY = 85
DEPLOYMENT_JPEG_RESAMPLE = "PIL_LANCZOS"
DEPLOYMENT_JPEG_PIXEL_SEMANTICS = "decoded_saved_transport_pixels"

STORAGE_CONTRACT_ENV = "VLMNAV_TERMINAL_CERTIFICATE_INPUT_STORAGE_CONTRACT"
CERTIFICATE_ONLY_IMAGE_LOGGING_ENV = (
    "VLMNAV_TERMINAL_CERTIFICATE_ONLY_IMAGE_LOGGING"
)


def _strict_optional_bool(value: object, *, name: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "0", "false", "no", "n", "off"}:
        return False
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def resolve_terminal_certificate_storage_contract(
    environment: Mapping[str, str],
) -> dict[str, object] | None:
    """Resolve the opt-in contract; an empty environment preserves legacy PNG."""
    contract = str(environment.get(STORAGE_CONTRACT_ENV, "") or "").strip()
    certificate_only = _strict_optional_bool(
        environment.get(CERTIFICATE_ONLY_IMAGE_LOGGING_ENV, ""),
        name=CERTIFICATE_ONLY_IMAGE_LOGGING_ENV,
    )
    if not contract:
        if certificate_only:
            raise ValueError(
                f"{CERTIFICATE_ONLY_IMAGE_LOGGING_ENV}=1 requires an explicit "
                f"{STORAGE_CONTRACT_ENV}"
            )
        return None
    if contract != DEPLOYMENT_JPEG_STORAGE_CONTRACT:
        raise ValueError(f"unsupported terminal-certificate storage contract: {contract!r}")
    if not certificate_only:
        raise ValueError(
            f"{DEPLOYMENT_JPEG_STORAGE_CONTRACT} requires "
            f"{CERTIFICATE_ONLY_IMAGE_LOGGING_ENV}=1"
        )
    return {
        "storage_contract": DEPLOYMENT_JPEG_STORAGE_CONTRACT,
        "filename": DEPLOYMENT_JPEG_FILENAME,
        "format": DEPLOYMENT_JPEG_FORMAT,
        "mime_type": DEPLOYMENT_JPEG_MIME_TYPE,
        "max_dimension": DEPLOYMENT_JPEG_MAX_DIMENSION,
        "resize_resample": DEPLOYMENT_JPEG_RESAMPLE,
        "jpeg_quality": DEPLOYMENT_JPEG_QUALITY,
        "pixel_semantics": DEPLOYMENT_JPEG_PIXEL_SEMANTICS,
        "certificate_only_image_logging": True,
    }


def image_name_allowed_by_storage_contract(
    image_name: str,
    storage_contract: Mapping[str, object] | None,
) -> bool:
    """Legacy logging keeps all images; deployment logging keeps only RGB input."""
    return storage_contract is None or image_name == "terminal_certificate_input_rgb"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prepare_terminal_certificate_deployment_jpeg(
    image: np.ndarray,
) -> tuple[bytes, dict[str, object]]:
    """Apply the exact OpenAI stopping-image transport and return saved bytes.

    The JPEG is encoded exactly once.  Dataset training must decode this saved
    payload directly; it must not JPEG-encode the decoded pixels a second time.
    """
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("terminal-certificate image must be HxWx3 RGB")
    if source.dtype != np.uint8:
        raise ValueError("terminal-certificate deployment JPEG requires uint8 RGB")
    source = np.ascontiguousarray(source[:, :, :3])
    source_shape = [int(value) for value in source.shape]
    source_pixel_sha256 = _sha256_bytes(source.tobytes())

    pil_image = Image.fromarray(source)
    width, height = pil_image.size
    if width > DEPLOYMENT_JPEG_MAX_DIMENSION or height > DEPLOYMENT_JPEG_MAX_DIMENSION:
        if width > height:
            new_width = DEPLOYMENT_JPEG_MAX_DIMENSION
            new_height = int(height * (DEPLOYMENT_JPEG_MAX_DIMENSION / width))
        else:
            new_height = DEPLOYMENT_JPEG_MAX_DIMENSION
            new_width = int(width * (DEPLOYMENT_JPEG_MAX_DIMENSION / height))
        if new_width <= 0 or new_height <= 0:
            raise ValueError("terminal-certificate resize produced an empty dimension")
        lanczos = getattr(Image, "Resampling", Image).LANCZOS
        pil_image = pil_image.resize((new_width, new_height), lanczos)

    preencode = np.ascontiguousarray(np.asarray(pil_image.convert("RGB")))
    with io.BytesIO() as output:
        pil_image.save(output, format="JPEG", quality=DEPLOYMENT_JPEG_QUALITY)
        encoded = output.getvalue()
    with Image.open(io.BytesIO(encoded)) as decoded_image:
        decoded = np.ascontiguousarray(np.asarray(decoded_image.convert("RGB")))
    metadata = {
        "terminal_certificate_input_rgb_storage_contract": (
            DEPLOYMENT_JPEG_STORAGE_CONTRACT
        ),
        "terminal_certificate_input_rgb_filename": DEPLOYMENT_JPEG_FILENAME,
        "terminal_certificate_input_rgb_storage_format": DEPLOYMENT_JPEG_FORMAT,
        "terminal_certificate_input_rgb_storage_mime_type": (
            DEPLOYMENT_JPEG_MIME_TYPE
        ),
        "terminal_certificate_input_rgb_storage_max_dimension": (
            DEPLOYMENT_JPEG_MAX_DIMENSION
        ),
        "terminal_certificate_input_rgb_storage_resize_resample": (
            DEPLOYMENT_JPEG_RESAMPLE
        ),
        "terminal_certificate_input_rgb_storage_jpeg_quality": (
            DEPLOYMENT_JPEG_QUALITY
        ),
        "terminal_certificate_input_rgb_pixel_semantics": (
            DEPLOYMENT_JPEG_PIXEL_SEMANTICS
        ),
        "terminal_certificate_input_rgb_source_pixel_sha256": source_pixel_sha256,
        "terminal_certificate_input_rgb_source_shape": source_shape,
        "terminal_certificate_input_rgb_source_dtype": str(source.dtype),
        "terminal_certificate_input_rgb_preencode_pixel_sha256": (
            _sha256_bytes(preencode.tobytes())
        ),
        "terminal_certificate_input_rgb_preencode_shape": [
            int(value) for value in preencode.shape
        ],
        "terminal_certificate_input_rgb_encoded_file_sha256": (
            _sha256_bytes(encoded)
        ),
        "terminal_certificate_input_rgb_encoded_bytes": len(encoded),
        # These established keys now describe the actual pixels recovered by
        # decoding the stored deployment payload, which are the training input.
        "terminal_certificate_input_rgb_pixel_sha256": (
            _sha256_bytes(decoded.tobytes())
        ),
        "terminal_certificate_input_rgb_shape": [int(value) for value in decoded.shape],
        "terminal_certificate_input_rgb_dtype": str(decoded.dtype),
        "terminal_certificate_only_image_logging": True,
    }
    return encoded, metadata
