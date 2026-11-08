"""Strict opt-in loader for selected HM3Dv2 ObjectNav train episodes.

The legacy ObjectNav loader intentionally remains the default.  This module is
entered only when all sparse-runtime environment variables are present.  Its
bundle contains exact full episode objects and only the goal objects needed by
those episodes, while retaining the canonical global episode indices used by
the evaluator.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


PANEL_SCHEMA = "hm3dv2_train_stop_certificate_rollout_panel_v2"
PANEL_PURPOSE = "train_only_closed_loop_stop_certificate_collection"
NONACCESS_SCHEMA = "hm3dv2_validation_data_nonaccess_declaration_v1"
SPARSE_RUNTIME_BUNDLE_SCHEMA = "hm3dv2_objectnav_sparse_runtime_bundle_v1"
SPARSE_RUNTIME_BUNDLE_PURPOSE = "objectnav_train_selected_episode_runtime"

ENV_BUNDLE = "VLMNAV_OBJECTNAV_SPARSE_RUNTIME_BUNDLE"
ENV_MANIFEST = "VLMNAV_OBJECTNAV_SPARSE_SOURCE_MANIFEST"
ENV_MANIFEST_SHA256 = "VLMNAV_OBJECTNAV_SPARSE_SOURCE_MANIFEST_SHA256"
ENV_CONTENT_SHA256 = "VLMNAV_OBJECTNAV_SPARSE_SOURCE_CONTENT_SHA256"
ENV_EPISODE_LIST = "VLMNAV_EPISODE_INDEX_LIST_PATH"
_OPT_IN_ENV_KEYS = (
    ENV_BUNDLE,
    ENV_MANIFEST,
    ENV_MANIFEST_SHA256,
    ENV_CONTENT_SHA256,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SparseObjectNavRuntime:
    episodes_by_global_index: dict[int, dict[str, Any]]
    goal_objects_by_global_index: dict[int, list[dict[str, Any]]]
    selected_indices: tuple[int, ...]
    canonical_episode_count: int
    source_manifest_sha256: str
    source_content_sha256: str
    bundle_sha256: str


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, where: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise RuntimeError(f"{where} must be a lowercase SHA-256 digest")
    return normalized


def _require_explicit_train_content_dir(path: Path, where: str) -> Path:
    resolved = path.resolve()
    parts = [part.casefold() for part in resolved.parts]
    forbidden = {"val", "validation", "eval", "evaluation", "test", "testing"}
    if any(part in forbidden for part in parts):
        raise RuntimeError(f"{where} resolves through a non-train split: {resolved}")
    if resolved.name.casefold() != "content" or resolved.parent.name.casefold() != "train":
        raise RuntimeError(f"{where} must resolve to .../train/content: {resolved}")
    return resolved


def _require_nonaccess_declaration(payload: Mapping[str, Any], where: str) -> None:
    if payload.get("validation_data_accessed") is not False:
        raise RuntimeError(f"{where} must explicitly declare validation_data_accessed=false")
    declaration = payload.get("validation_data_access_declaration")
    expected_keys = {
        "schema",
        "validation_data_accessed",
        "scope",
        "validation_content_paths_read",
        "statement",
    }
    if not isinstance(declaration, dict) or set(declaration) != expected_keys:
        raise RuntimeError(f"{where} has no exact validation non-access declaration")
    if (
        declaration.get("schema") != NONACCESS_SCHEMA
        or declaration.get("validation_data_accessed") is not False
        or declaration.get("scope") != "terminal_certificate_rollout_panel_creation"
        or declaration.get("validation_content_paths_read") != []
        or not isinstance(declaration.get("statement"), str)
        or not declaration["statement"].strip()
    ):
        raise RuntimeError(f"{where} validation non-access declaration is invalid")
    forbidden_fields = {
        "val_content_dir_audit_only",
        "val_content_sha256_audit_only",
        "val_dataset_episodes_audit_only",
        "validation_scene_overlap_audit_only",
        "validation_episode_overlap_audit_only",
    }
    if forbidden_fields.intersection(payload):
        raise RuntimeError(f"{where} contains validation-derived audit fields")


def _require_index_list(value: Any, where: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{where} must be a non-empty JSON index list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise RuntimeError(f"{where} contains an invalid global episode index")
    if len(value) != len(set(value)):
        raise RuntimeError(f"{where} contains duplicate global episode indices")
    return list(value)


def _goal_lookup(episode: Mapping[str, Any]) -> tuple[str, str]:
    scene_parts = str(episode.get("scene_id") or "").split("/")[1:]
    if len(scene_parts) < 3:
        raise RuntimeError(f"unexpected ObjectNav scene_id: {episode.get('scene_id')!r}")
    object_category = episode.get("object_category")
    if not isinstance(object_category, str) or not object_category:
        raise RuntimeError("sparse episode has no object_category")
    return scene_parts[1][6:], f"{scene_parts[-1]}_{object_category}"


def _load_json(path: Path, where: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to read {where}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{where} must be a JSON object: {path}")
    return payload


def load_sparse_objectnav_runtime_from_environment(
    *,
    cfg_split: str,
    dataset_name: str,
    content_dir: str,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[SparseObjectNavRuntime]:
    """Return a validated sparse runtime, or ``None`` for the legacy default."""
    environment = os.environ if environ is None else environ
    values = {key: str(environment.get(key, "") or "").strip() for key in _OPT_IN_ENV_KEYS}
    configured = [key for key, value in values.items() if value]
    if not configured:
        return None
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "partial ObjectNav sparse-runtime opt-in is forbidden; missing " + ", ".join(missing)
        )
    if str(cfg_split).strip().casefold() != "train":
        raise RuntimeError("ObjectNav sparse runtime is train-only")
    if str(dataset_name).strip().casefold() != "hm3d_v2":
        raise RuntimeError("ObjectNav sparse runtime is restricted to HM3Dv2")

    actual_content_dir = _require_explicit_train_content_dir(
        Path(content_dir), "configured ObjectNav content directory"
    )
    expected_manifest_sha256 = _require_sha256(
        values[ENV_MANIFEST_SHA256], ENV_MANIFEST_SHA256
    )
    expected_content_sha256 = _require_sha256(values[ENV_CONTENT_SHA256], ENV_CONTENT_SHA256)
    manifest_path = Path(values[ENV_MANIFEST]).resolve()
    bundle_path = Path(values[ENV_BUNDLE]).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"missing sparse source manifest: {manifest_path}")
    if not bundle_path.is_file():
        raise RuntimeError(f"missing sparse runtime bundle: {bundle_path}")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError("ObjectNav sparse source manifest SHA-256 mismatch")

    manifest = _load_json(manifest_path, "sparse source manifest")
    if manifest.get("schema") != PANEL_SCHEMA or manifest.get("purpose") != PANEL_PURPOSE:
        raise RuntimeError("unexpected ObjectNav sparse source panel contract")
    if manifest.get("split") != "train":
        raise RuntimeError("ObjectNav sparse source panel is not train split")
    _require_nonaccess_declaration(manifest, "ObjectNav sparse source panel")
    manifest_content_dir = _require_explicit_train_content_dir(
        Path(str(manifest.get("train_content_dir") or "")), "manifest train_content_dir"
    )
    if manifest_content_dir != actual_content_dir:
        raise RuntimeError("manifest train content path does not match configured ObjectNav root")
    manifest_content_sha256 = _require_sha256(
        manifest.get("train_content_sha256"), "manifest train_content_sha256"
    )
    if manifest_content_sha256 != expected_content_sha256:
        raise RuntimeError("preflight train-content SHA-256 does not match sparse manifest")
    canonical_episode_count = manifest.get("train_dataset_episodes")
    if (
        isinstance(canonical_episode_count, bool)
        or not isinstance(canonical_episode_count, int)
        or canonical_episode_count < 1
    ):
        raise RuntimeError("manifest train_dataset_episodes is invalid")
    selected_indices = _require_index_list(manifest.get("selected_indices"), "manifest selected_indices")
    if manifest.get("selected_count") != len(selected_indices):
        raise RuntimeError("manifest selected_count mismatch")
    if manifest.get("selected_unique_count") != len(selected_indices):
        raise RuntimeError("manifest selected_unique_count mismatch")
    if max(selected_indices) >= canonical_episode_count:
        raise RuntimeError("manifest selected index exceeds canonical train dataset")

    contract_sha256 = _require_sha256(
        manifest.get("source_panel_contract_sha256"),
        "manifest source_panel_contract_sha256",
    )
    contract_payload = dict(manifest)
    contract_payload.pop("source_panel_contract_sha256", None)
    contract_payload.pop("sparse_runtime_bundle", None)
    if canonical_json_sha256(contract_payload) != contract_sha256:
        raise RuntimeError("ObjectNav source panel contract SHA-256 mismatch")

    bundle_metadata = manifest.get("sparse_runtime_bundle")
    expected_bundle_metadata_keys = {
        "schema",
        "filename",
        "sha256",
        "payload_sha256",
        "selected_count",
        "source_panel_contract_sha256",
    }
    if not isinstance(bundle_metadata, dict) or set(bundle_metadata) != expected_bundle_metadata_keys:
        raise RuntimeError("manifest sparse_runtime_bundle metadata is invalid")
    if bundle_metadata.get("schema") != SPARSE_RUNTIME_BUNDLE_SCHEMA:
        raise RuntimeError("manifest sparse runtime bundle schema mismatch")
    if bundle_metadata.get("source_panel_contract_sha256") != contract_sha256:
        raise RuntimeError("manifest sparse bundle is not bound to source panel contract")
    if bundle_metadata.get("selected_count") != len(selected_indices):
        raise RuntimeError("manifest sparse bundle selected_count mismatch")
    bundle_filename = str(bundle_metadata.get("filename") or "")
    if not bundle_filename or Path(bundle_filename).name != bundle_filename:
        raise RuntimeError("manifest sparse bundle filename is unsafe")
    expected_bundle_path = (manifest_path.parent / bundle_filename).resolve()
    if bundle_path != expected_bundle_path:
        raise RuntimeError("sparse bundle path does not match source manifest")
    expected_bundle_sha256 = _require_sha256(
        bundle_metadata.get("sha256"), "manifest sparse bundle sha256"
    )
    actual_bundle_sha256 = sha256_file(bundle_path)
    if actual_bundle_sha256 != expected_bundle_sha256:
        raise RuntimeError("ObjectNav sparse runtime bundle SHA-256 mismatch")
    try:
        with gzip.open(bundle_path, "rb") as handle:
            bundle_payload_bytes = handle.read()
    except Exception as exc:
        raise RuntimeError(f"failed to decompress sparse runtime bundle: {bundle_path}") from exc
    expected_payload_sha256 = _require_sha256(
        bundle_metadata.get("payload_sha256"), "manifest sparse bundle payload_sha256"
    )
    if hashlib.sha256(bundle_payload_bytes).hexdigest() != expected_payload_sha256:
        raise RuntimeError("ObjectNav sparse runtime bundle payload SHA-256 mismatch")
    try:
        bundle = json.loads(bundle_payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("ObjectNav sparse runtime bundle payload is invalid JSON") from exc
    if not isinstance(bundle, dict):
        raise RuntimeError("ObjectNav sparse runtime bundle must be a JSON object")
    if (
        bundle.get("schema") != SPARSE_RUNTIME_BUNDLE_SCHEMA
        or bundle.get("purpose") != SPARSE_RUNTIME_BUNDLE_PURPOSE
        or bundle.get("source_panel_schema") != PANEL_SCHEMA
        or bundle.get("source_panel_contract_sha256") != contract_sha256
        or bundle.get("source_split") != "train"
    ):
        raise RuntimeError("ObjectNav sparse runtime bundle contract mismatch")
    _require_nonaccess_declaration(bundle, "ObjectNav sparse runtime bundle")
    bundle_content_dir = _require_explicit_train_content_dir(
        Path(str(bundle.get("train_content_dir") or "")), "bundle train_content_dir"
    )
    if bundle_content_dir != actual_content_dir:
        raise RuntimeError("sparse bundle train content path mismatch")
    if bundle.get("train_content_sha256") != manifest_content_sha256:
        raise RuntimeError("sparse bundle train content SHA-256 mismatch")
    if bundle.get("train_dataset_episodes") != canonical_episode_count:
        raise RuntimeError("sparse bundle canonical episode count mismatch")
    if bundle.get("selected_indices") != selected_indices:
        raise RuntimeError("sparse bundle selected index sequence mismatch")
    if bundle.get("selected_count") != len(selected_indices):
        raise RuntimeError("sparse bundle selected_count mismatch")

    episode_list_path_value = str(environment.get(ENV_EPISODE_LIST, "") or "").strip()
    if not episode_list_path_value:
        raise RuntimeError(f"{ENV_EPISODE_LIST} is required by sparse runtime")
    episode_list_path = Path(episode_list_path_value)
    try:
        worker_indices = json.loads(episode_list_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to read sparse worker episode list: {episode_list_path}") from exc
    worker_indices = _require_index_list(worker_indices, "sparse worker episode list")
    if worker_indices != selected_indices:
        raise RuntimeError("worker episode list does not exactly match sparse selected indices")

    selected_rows = manifest.get("selected_episodes")
    if not isinstance(selected_rows, list) or len(selected_rows) != len(selected_indices):
        raise RuntimeError("manifest selected_episodes does not match selected indices")
    selected_rows_by_index: dict[int, dict[str, Any]] = {}
    for row in selected_rows:
        if not isinstance(row, dict):
            raise RuntimeError("manifest selected episode row is invalid")
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index in selected_rows_by_index:
            raise RuntimeError("manifest selected episode row index is invalid")
        selected_rows_by_index[index] = row
    if set(selected_rows_by_index) != set(selected_indices):
        raise RuntimeError("manifest selected episode row set mismatch")

    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != len(selected_indices):
        raise RuntimeError("sparse bundle entry count mismatch")
    episodes_by_index: dict[int, dict[str, Any]] = {}
    goals_by_index: dict[int, list[dict[str, Any]]] = {}
    entry_indices: list[int] = []
    expected_entry_keys = {
        "global_index",
        "source_shard",
        "source_local_index",
        "source_shard_sha256",
        "episode_sha256",
        "goal_group_id",
        "goal_lookup_key",
        "goal_objects_sha256",
        "episode",
        "goal_objects",
    }
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != expected_entry_keys:
            raise RuntimeError("sparse runtime entry schema mismatch")
        global_index = entry.get("global_index")
        if (
            isinstance(global_index, bool)
            or not isinstance(global_index, int)
            or global_index in episodes_by_index
        ):
            raise RuntimeError("sparse runtime entry global index is invalid")
        source_local_index = entry.get("source_local_index")
        if (
            isinstance(source_local_index, bool)
            or not isinstance(source_local_index, int)
            or source_local_index < 0
        ):
            raise RuntimeError("sparse runtime entry source local index is invalid")
        source_shard = str(entry.get("source_shard") or "")
        if not source_shard or Path(source_shard).name != source_shard:
            raise RuntimeError("sparse runtime entry source shard is unsafe")
        _require_sha256(entry.get("source_shard_sha256"), "entry source_shard_sha256")
        episode = entry.get("episode")
        goal_objects = entry.get("goal_objects")
        if not isinstance(episode, dict) or not isinstance(goal_objects, list):
            raise RuntimeError("sparse runtime entry episode/goals payload is invalid")
        episode_sha256 = _require_sha256(entry.get("episode_sha256"), "entry episode_sha256")
        goals_sha256 = _require_sha256(
            entry.get("goal_objects_sha256"), "entry goal_objects_sha256"
        )
        if canonical_json_sha256(episode) != episode_sha256:
            raise RuntimeError("sparse runtime embedded episode SHA-256 mismatch")
        if canonical_json_sha256(goal_objects) != goals_sha256:
            raise RuntimeError("sparse runtime embedded goal objects SHA-256 mismatch")
        goal_group_id, goal_lookup_key = _goal_lookup(episode)
        if (
            entry.get("goal_group_id") != goal_group_id
            or entry.get("goal_lookup_key") != goal_lookup_key
            or source_shard.split(".", 1)[0] != goal_group_id
        ):
            raise RuntimeError("sparse runtime episode/goal source linkage mismatch")
        panel_row = selected_rows_by_index.get(global_index)
        if panel_row is None:
            raise RuntimeError("sparse runtime entry is absent from source panel")
        expected_panel_binding = {
            "source_shard": source_shard,
            "source_local_index": source_local_index,
            "source_shard_sha256": entry["source_shard_sha256"],
            "episode_sha256": episode_sha256,
            "goal_lookup_key": goal_lookup_key,
            "goal_objects_sha256": goals_sha256,
        }
        for key, expected in expected_panel_binding.items():
            if panel_row.get(key) != expected:
                raise RuntimeError(f"source panel sparse binding mismatch for {key}")
        normalized_goal = str(episode.get("object_category") or "").strip().lower()
        if normalized_goal == "tv_monitor":
            normalized_goal = "tv screen"
        scene_parts = [part for part in str(episode.get("scene_id") or "").split("/") if part]
        scene_id = scene_parts[-2] if len(scene_parts) >= 2 else str(episode.get("scene_id") or "")
        if panel_row.get("goal") != normalized_goal or panel_row.get("scene_id") != scene_id:
            raise RuntimeError("source panel episode identity mismatch")
        episodes_by_index[global_index] = episode
        goals_by_index[global_index] = goal_objects
        entry_indices.append(global_index)
    if entry_indices != selected_indices:
        raise RuntimeError("sparse runtime entry sequence mismatch")

    return SparseObjectNavRuntime(
        episodes_by_global_index=episodes_by_index,
        goal_objects_by_global_index=goals_by_index,
        selected_indices=tuple(selected_indices),
        canonical_episode_count=canonical_episode_count,
        source_manifest_sha256=actual_manifest_sha256,
        source_content_sha256=manifest_content_sha256,
        bundle_sha256=actual_bundle_sha256,
    )

