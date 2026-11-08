"""MP3D lightweight multi-floor signal helper (env-gated).

Per-step depth slope + climb latch. Cross-floor search swaps 2D maps on
*landing* (Δy + vertical settle), not on navigability-flatness.

t1x lesson: keeping suppress_stop for the whole latch (timeout=40 + peak-stair
stale-protect) left Oracle intact but blocked Stop after reaching the goal.
Landing-swap now decouples Stop suppression from latch lifetime: suppress only
while still changing floors. Map swap no longer requires a stair visual cue
once Δy and settle are satisfied.

LANDING_V2 (t1z, env-gated): do not treat a mid-stair pause as a landing, and
do not force-swap on climb timeout unless landing_ready. t1/t1y stay
reproducible when VLMNAV_MULTIFLOOR_LANDING_V2=0.

t1s (env-gated, default off):
- KEEP_ANCHOR: incomplete climbs do not reset y_anchor, so segmented
  stairs can still land/hard-swap.
- SATURATED_SEEK / SEEK_LATCH: live t1s showed mouth-latch without Δy
  destroys same-floor SR (sat 34.6% vs t1z 50% on n=26). Default latch off.

Enabled when VLMNAV_MP3D_MULTIFLOOR=1. Default off (HM3D unaffected).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    from utils import depth_to_height
except ImportError:  # pragma: no cover
    depth_to_height = None  # type: ignore


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def multifloor_enabled() -> bool:
    return _env_bool("VLMNAV_MP3D_MULTIFLOOR", "0")


@dataclass
class FloorMapBundle:
    voxel_map: Any = None
    explored_map: Any = None


class MultiFloorController:
    """Ephemeral climb latch + depth stair signal + floor map copy buckets."""

    def __init__(self) -> None:
        self.floor_pitch = _env_float("VLMNAV_MULTIFLOOR_PITCH", 2.5)
        self.dy_threshold = _env_float("VLMNAV_MULTIFLOOR_DY_THRESH", 1.2)
        self.dy_confirm_frames = _env_int("VLMNAV_MULTIFLOOR_DY_FRAMES", 3)
        # Latch defaults match mflight2 (42.3% SR full2195).
        self.climb_start_dy = _env_float("VLMNAV_MULTIFLOOR_CLIMB_START_DY", 0.6)
        self.climb_timeout = _env_int("VLMNAV_MULTIFLOOR_CLIMB_TIMEOUT", 20)
        self.climb_step_dy = _env_float("VLMNAV_MULTIFLOOR_CLIMB_STEP_DY", 0.12)
        self.climb_engage_frames = _env_int("VLMNAV_MULTIFLOOR_CLIMB_ENGAGE_FRAMES", 2)
        self.stale_release_frames = _env_int("VLMNAV_MULTIFLOOR_STALE_FRAMES", 5)
        self.stale_dy_eps = _env_float("VLMNAV_MULTIFLOOR_STALE_DY_EPS", 0.05)
        self.stale_dy_max = _env_float("VLMNAV_MULTIFLOOR_STALE_DY_MAX", 0.8)
        self.nav_thresh_stairs = _env_float(
            "VLMNAV_MULTIFLOOR_NAV_HEIGHT_THRESH", 0.55
        )
        self.stair_conf_thresh = _env_float(
            "VLMNAV_MULTIFLOOR_STAIR_CONF", 0.65
        )
        self.nav_stair_conf_thresh = _env_float(
            "VLMNAV_MULTIFLOOR_NAV_STAIR_CONF", 0.85
        )
        self.move_context_prompt = _env_bool(
            "VLMNAV_MULTIFLOOR_MOVE_CONTEXT_PROMPT", "0"
        )
        self.raise_nav_bias = _env_bool("VLMNAV_MULTIFLOOR_RAISE_NAV_BIAS", "0")
        self.bias_min_dy = _env_float("VLMNAV_MULTIFLOOR_BIAS_MIN_DY", 0.45)
        self.climb_forward_r_weight = _env_float(
            "VLMNAV_MULTIFLOOR_CLIMB_FORWARD_R_WEIGHT", 0.4
        )
        self.post_swap_prompt_steps = _env_int(
            "VLMNAV_MULTIFLOOR_POST_SWAP_PROMPT_STEPS", 3
        )
        # mflight2t2: stair-mouth engagement + relaxed swap + anti false-latch.
        self.stair_mouth_engage = _env_bool(
            "VLMNAV_MULTIFLOOR_STAIR_MOUTH_ENGAGE", "0"
        )
        self.stair_mouth_conf_thresh = _env_float(
            "VLMNAV_MULTIFLOOR_STAIR_MOUTH_CONF", 0.85
        )
        self.stair_mouth_frames = _env_int(
            "VLMNAV_MULTIFLOOR_STAIR_MOUTH_FRAMES", 3
        )
        self.stair_mouth_min_dy = _env_float(
            "VLMNAV_MULTIFLOOR_STAIR_MOUTH_MIN_DY", 0.12
        )
        self.stair_mouth_require_flat = _env_bool(
            "VLMNAV_MULTIFLOOR_STAIR_MOUTH_REQUIRE_FLAT", "0"
        )
        self.stair_mouth_max_flat = _env_float(
            "VLMNAV_MULTIFLOOR_STAIR_MOUTH_MAX_FLAT", 0.75
        )
        self.stair_mouth_bias = _env_bool(
            "VLMNAV_MULTIFLOOR_STAIR_MOUTH_BIAS", "0"
        )
        self.swap_flat_min = _env_float(
            "VLMNAV_MULTIFLOOR_SWAP_FLAT_MIN", 0.08
        )
        self.soft_swap_on_timeout = _env_bool(
            "VLMNAV_MULTIFLOOR_SOFT_SWAP_ON_TIMEOUT", "0"
        )
        self.soft_swap_min_dy = _env_float(
            "VLMNAV_MULTIFLOOR_SOFT_SWAP_MIN_DY", 0.9
        )
        self.soft_swap_min_climb_steps = _env_int(
            "VLMNAV_MULTIFLOOR_SOFT_SWAP_MIN_CLIMB_STEPS", 8
        )
        self.soft_swap_alt_min_dy = _env_float(
            "VLMNAV_MULTIFLOOR_SOFT_SWAP_ALT_MIN_DY", 0.6
        )
        self.flat_stale_dy_max = _env_float(
            "VLMNAV_MULTIFLOOR_FLAT_STALE_DY_MAX", 0.3
        )
        self.flat_stale_frames = _env_int(
            "VLMNAV_MULTIFLOOR_FLAT_STALE_FRAMES", 3
        )
        self.require_stair_for_anchor_climb = _env_bool(
            "VLMNAV_MULTIFLOOR_REQUIRE_STAIR_FOR_ANCHOR_CLIMB", "0"
        )
        # Cross-floor landing swap: settle (vertical speed ~0) after a real Δy.
        # Do not use navigability-flatness; mid-stair floors are rarely "flat".
        self.landing_swap = _env_bool("VLMNAV_MULTIFLOOR_LANDING_SWAP", "0")
        self.landing_min_dy = _env_float("VLMNAV_MULTIFLOOR_LANDING_MIN_DY", 0.85)
        self.landing_settle_frames = _env_int(
            "VLMNAV_MULTIFLOOR_LANDING_SETTLE_FRAMES", 4
        )
        self.landing_settle_eps = _env_float(
            "VLMNAV_MULTIFLOOR_LANDING_SETTLE_EPS", 0.04
        )
        self.landing_min_climb_steps = _env_int(
            "VLMNAV_MULTIFLOOR_LANDING_MIN_CLIMB_STEPS", 8
        )
        self.stale_protect_dy = _env_float(
            "VLMNAV_MULTIFLOOR_STALE_PROTECT_DY", 0.45
        )
        self.stale_protect_stair = _env_float(
            "VLMNAV_MULTIFLOOR_STALE_PROTECT_STAIR", 0.50
        )
        self.post_swap_forward_bias = _env_bool(
            "VLMNAV_MULTIFLOOR_POST_SWAP_FORWARD_BIAS", "0"
        )
        # t1z: landing_v2. Off keeps t1y timeout-force-swap + no left-stairs gate.
        self.landing_v2 = _env_bool("VLMNAV_MULTIFLOOR_LANDING_V2", "0")
        # t1s: accumulate height across failed latches; late stair-seek.
        self.keep_anchor = _env_bool("VLMNAV_MULTIFLOOR_KEEP_ANCHOR", "0")
        self.keep_anchor_min_dy = _env_float(
            "VLMNAV_MULTIFLOOR_KEEP_ANCHOR_MIN_DY", 0.25
        )
        self.keep_anchor_cooloff = _env_int(
            "VLMNAV_MULTIFLOOR_KEEP_ANCHOR_COOLOFF", 8
        )
        self.saturated_seek = _env_bool("VLMNAV_MULTIFLOOR_SATURATED_SEEK", "0")
        self.seek_min_step = _env_int("VLMNAV_MULTIFLOOR_SEEK_MIN_STEP", 16)
        self.seek_hard_step = _env_int("VLMNAV_MULTIFLOOR_SEEK_HARD_STEP", 28)
        self.seek_revisit = _env_int("VLMNAV_MULTIFLOOR_SEEK_REVISIT", 3)
        self.seek_max_frontier = _env_int(
            "VLMNAV_MULTIFLOOR_SEEK_MAX_FRONTIER", 2
        )
        self.seek_stair_conf = _env_float(
            "VLMNAV_MULTIFLOOR_SEEK_STAIR_CONF", 0.85
        )
        self.seek_frames = _env_int("VLMNAV_MULTIFLOOR_SEEK_FRAMES", 3)
        # t1s live: mouth-latch without Δy killed same-floor cabinet SR.
        # Default off; SATURATED_SEEK may still bias turn-around only.
        self.seek_latch = _env_bool("VLMNAV_MULTIFLOOR_SEEK_LATCH", "0")
        # t1n: leave the start room without using the stair detector.
        # t1z must-cross never-climb (n=215) had room_transition=0 and SR 0;
        # stair_conf saturates on furniture so SEEK-by-stair is unsafe.
        self.escape_seek = _env_bool("VLMNAV_MULTIFLOOR_ESCAPE_SEEK", "0")
        # step_ndx is VLM decisions, not Habitat steps. 500 Habitat ≈ 60 VLM
        # decisions, so 80/150 never fired (t1n). 16/40 is late-enough same-floor
        # Stop (p50 success ≈ 6 VLM steps) but inside a max-step episode.
        self.escape_min_step = _env_int("VLMNAV_MULTIFLOOR_ESCAPE_MIN_STEP", 16)
        self.escape_hard_step = _env_int("VLMNAV_MULTIFLOOR_ESCAPE_HARD_STEP", 40)
        self.escape_revisit = _env_int("VLMNAV_MULTIFLOOR_ESCAPE_REVISIT", 4)
        # t1n2: 180-reorient at d<0.2 with a 1.7m ray yanked off the object.
        # 0 disables. 1.15 keeps closet unstick (max r ~0.9) and skips rooms.
        self.escape_tight_r = _env_float("VLMNAV_MULTIFLOOR_ESCAPE_TIGHT_R", "0")
        # t1p: looping on a short ray while a long hallway exists → take it.
        # Never 180, never Stop, never rewrite turn-around (0). Default off.
        self.hallway_commit = _env_bool("VLMNAV_MULTIFLOOR_HALLWAY_COMMIT", "0")
        self.hallway_min_step = _env_int("VLMNAV_MULTIFLOOR_HALLWAY_MIN_STEP", 16)
        self.hallway_revisit = _env_int("VLMNAV_MULTIFLOOR_HALLWAY_REVISIT", 4)
        self.hallway_short_r = _env_float("VLMNAV_MULTIFLOOR_HALLWAY_SHORT_R", "0.95")
        self.hallway_long_r = _env_float("VLMNAV_MULTIFLOOR_HALLWAY_LONG_R", "1.4")
        # t1q: looping already on the longest ray, Stop logit says object is
        # not here → 180. t1n2 yanked when d≈0 (margin not this negative).
        self.far_reorient = _env_bool("VLMNAV_MULTIFLOOR_FAR_REORIENT", "0")
        self.far_min_step = _env_int("VLMNAV_MULTIFLOOR_FAR_REORIENT_MIN_STEP", 16)
        self.far_revisit = _env_int("VLMNAV_MULTIFLOOR_FAR_REORIENT_REVISIT", 4)
        self.far_margin = _env_float("VLMNAV_MULTIFLOOR_FAR_REORIENT_MARGIN", "-5.0")
        self.far_longest_eps = _env_float("VLMNAV_MULTIFLOOR_FAR_REORIENT_LONGEST_EPS", "0.08")
        self.reset()

    def reset(self) -> None:
        self.enabled = multifloor_enabled()
        self.floor_id = 0
        self.floor_maps: Dict[int, FloorMapBundle] = {}
        self.climbing = False
        self.climb_steps = 0
        self.entry_y: Optional[float] = None
        self.y_anchor: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.dy_streak = 0
        self.vertical_progress_streak = 0
        self.stale_dy_streak = 0
        self.last_dy_from_entry: Optional[float] = None
        self.raise_nav_thresh = False
        self.last_stair_conf = 0.0
        self.last_stair_yaw = 0.0
        self.last_metadata: Dict[str, Any] = {}
        self._prev_climbing = False
        self._post_swap_steps = 0
        self.stair_mouth_streak = 0
        self.stair_mouth_dy_streak = 0
        self.flat_stale_streak = 0
        self.landing_settle_streak = 0
        self.peak_stair_conf = 0.0
        self._still_changing_floor = False
        self._saturated = False
        self.seek_mouth_streak = 0
        self._keep_cooloff = 0
        self._escape = False
        self._hallway = False
        self._far = False

    def _clear_climb_tracks(self, y: float, *, keep_anchor: bool = False) -> None:
        self.climbing = False
        self.climb_steps = 0
        self.entry_y = None
        if not keep_anchor:
            self.y_anchor = y
        self.dy_streak = 0
        self.stale_dy_streak = 0
        self.flat_stale_streak = 0
        self.landing_settle_streak = 0
        self.peak_stair_conf = 0.0
        self.last_dy_from_entry = None
        self.vertical_progress_streak = 0
        self.stair_mouth_streak = 0
        self.stair_mouth_dy_streak = 0
        self.seek_mouth_streak = 0
        self._still_changing_floor = False
        if keep_anchor:
            self._keep_cooloff = max(1, self.keep_anchor_cooloff)

    def _effective_dy(
        self,
        dy_from_entry: Optional[float],
        dy_from_anchor: float,
    ) -> Optional[float]:
        # Only accumulate from y_anchor when this latch started there.
        # Otherwise same-floor ramps inflate Δy and pin Stop / force-swap.
        if (
            self.keep_anchor
            and self.entry_y is not None
            and self.y_anchor is not None
            and abs(float(self.entry_y) - float(self.y_anchor)) < 1e-4
        ):
            vals = []
            if dy_from_entry is not None:
                vals.append(float(dy_from_entry))
            vals.append(float(dy_from_anchor))
            return max(vals)
        return dy_from_entry

    def _should_keep_anchor(
        self,
        dy_from_entry: Optional[float],
        dy_from_anchor: float,
    ) -> bool:
        if not self.keep_anchor:
            return False
        dy = self._effective_dy(dy_from_entry, dy_from_anchor)
        if dy is None:
            dy = dy_from_entry
        if dy is None or dy < self.keep_anchor_min_dy:
            return False
        # t1s2 ep 99: timeout at Δy≈0.93 kept the anchor, then anchor_ok
        # re-latched immediately and suppressed Stop for 57 frames.
        if dy >= self.landing_min_dy:
            return False
        return float(self.peak_stair_conf) >= self.stair_conf_thresh

    def _entry_y_for_engage(self, y: float, dy_from_anchor: float) -> float:
        if (
            self.keep_anchor
            and self.y_anchor is not None
            and dy_from_anchor >= self.keep_anchor_min_dy
        ):
            return float(self.y_anchor)
        return float(y)

    def _compute_saturated(
        self,
        step_ndx: int,
        frontier_count: Optional[int],
        revisit_count: int,
        loop_risk: bool,
        visit_count: int,
    ) -> bool:
        if not self.saturated_seek:
            return False
        if int(step_ndx) < self.seek_min_step:
            return False
        if loop_risk or int(revisit_count) >= self.seek_revisit:
            return True
        if int(visit_count) >= (self.seek_revisit + 1):
            return True
        if (
            int(step_ndx) >= self.seek_hard_step
            and frontier_count is not None
            and int(frontier_count) <= self.seek_max_frontier
        ):
            return True
        return False

    def _compute_escape(
        self,
        step_ndx: int,
        revisit_count: int,
        loop_risk: bool,
    ) -> bool:
        if not self.escape_seek or self.climbing:
            return False
        if int(step_ndx) < self.escape_min_step:
            return False
        if loop_risk or int(revisit_count) >= self.escape_revisit:
            return True
        return int(step_ndx) >= self.escape_hard_step

    def _compute_hallway(
        self,
        step_ndx: int,
        revisit_count: int,
        loop_risk: bool,
    ) -> bool:
        if not self.hallway_commit or self.climbing:
            return False
        if int(step_ndx) < self.hallway_min_step:
            return False
        return bool(loop_risk) or int(revisit_count) >= self.hallway_revisit

    def _compute_far(
        self,
        step_ndx: int,
        revisit_count: int,
        loop_risk: bool,
    ) -> bool:
        if not self.far_reorient or self.climbing:
            return False
        if int(step_ndx) < self.far_min_step:
            return False
        return bool(loop_risk) or int(revisit_count) >= self.far_revisit

    def _current_is_longest(
        self, a_final: list, cur: Optional[int], eps: Optional[float] = None
    ) -> bool:
        if not a_final:
            return False
        mx = self.max_candidate_range(list(a_final))
        if mx is None:
            return False
        tol = float(self.far_longest_eps if eps is None else eps)
        if cur in (0, None):
            return True
        try:
            cur_r = float(a_final[int(cur) - 1][0])
        except Exception:
            return False
        return abs(cur_r - float(mx)) <= tol

    def choose_longest_action(self, a_final: list) -> Optional[int]:
        if not a_final:
            return None
        best_idx = None
        best = (-1.0, 0.0)
        for i, action in enumerate(a_final, start=1):
            try:
                r, theta = action
            except Exception:
                continue
            r = float(r)
            theta = abs(float(theta))
            key = (r, -theta)
            if best_idx is None or key > best:
                best = key
                best_idx = i
        return best_idx

    def choose_reorient_action(
        self, a_final: list, current: Optional[int] = None
    ) -> Optional[int]:
        """Prefer a large heading change. Longest-ray often is the wall."""
        if not a_final:
            return None
        best_idx = None
        best = (-1.0, -1.0)
        for i, action in enumerate(a_final, start=1):
            if current is not None and i == int(current):
                continue
            try:
                r, theta = action
            except Exception:
                continue
            key = (abs(float(theta)), float(r))
            if best_idx is None or key > best:
                best = key
                best_idx = i
        return best_idx

    def max_candidate_range(self, a_final: list) -> float:
        best = 0.0
        for action in a_final or []:
            try:
                r, _theta = action
            except Exception:
                continue
            best = max(best, float(r))
        return best

    # ------------------------------------------------------------------ prompt
    def prompt_suffix(self) -> str:
        if not self.enabled:
            return ""
        base = (
            "\n\nMulti-floor note:\n"
            "- Going up or down stairs is allowed when the goal is on another floor.\n"
            "- Closed doors remain forbidden.\n"
            "- Do not Stop while you are clearly on a staircase; finish the floor change first.\n"
        )
        ctx = self.move_context_suffix()
        return base + ctx if ctx else base

    def move_context_suffix(self) -> str:
        """Step-aware Move prompt hints (no Stop-side changes)."""
        if not self.enabled or not self.move_context_prompt:
            return ""
        parts = []
        if self._post_swap_steps > 0:
            parts.append(
                "- You just changed floors; explore this floor for the goal."
            )
        if self.climbing:
            parts.append(
                "- You are on stairs: pick a forward arrow that continues up/down; "
                "avoid turn-around."
            )
        if not parts:
            return ""
        return "\n\nMove context:\n" + "\n".join(parts)

    def strip_no_stairs_clause(self, prompt: str) -> str:
        if not self.enabled or not isinstance(prompt, str):
            return prompt
        return (
            prompt.replace(
                ", and you DO NOT NEED TO GO UP OR DOWN STAIRS",
                ", and you MAY go up or down stairs when needed to reach the goal",
            )
            .replace(
                " and you DO NOT NEED TO GO UP OR DOWN STAIRS",
                " and you MAY go up or down stairs when needed to reach the goal",
            )
            .replace(
                "DO NOT NEED TO GO UP OR DOWN STAIRS",
                "MAY go up or down stairs when needed to reach the goal",
            )
        )

    # ------------------------------------------------------------------ nav / maps
    def navigability_height_threshold(self, default: float) -> float:
        if self.enabled and self.raise_nav_thresh:
            return max(float(default), self.nav_thresh_stairs)
        return float(default)

    def should_skip_explored_update(self) -> bool:
        # Freeze the 2D map only while the agent is still changing floors.
        # A stale same-floor latch must not stop exploration updates (t1x).
        if not (self.enabled and self.climbing):
            return False
        if self.landing_swap:
            return bool(self._still_changing_floor)
        return True

    def floor_id_from_y(self, y: float) -> int:
        pitch = max(0.5, self.floor_pitch)
        return int(round(float(y) / pitch))

    def save_current_maps(
        self, floor_id: int, voxel_map: Any, explored_map: Any
    ) -> None:
        bundle = self.floor_maps.get(floor_id) or FloorMapBundle()
        bundle.voxel_map = voxel_map.copy() if voxel_map is not None else None
        bundle.explored_map = (
            explored_map.copy() if explored_map is not None else None
        )
        self.floor_maps[floor_id] = bundle

    def load_or_create_maps(
        self,
        floor_id: int,
        blank_factory: Callable[[], Tuple[Any, Any]],
    ) -> Tuple[Any, Any]:
        bundle = self.floor_maps.get(floor_id)
        if (
            bundle is not None
            and bundle.voxel_map is not None
            and bundle.explored_map is not None
        ):
            return bundle.voxel_map.copy(), bundle.explored_map.copy()
        voxel, explored = blank_factory()
        self.floor_maps[floor_id] = FloorMapBundle(
            voxel_map=voxel.copy(),
            explored_map=explored.copy(),
        )
        return voxel, explored

    # ------------------------------------------------------------------ signals
    def detect_stair_slope(
        self,
        depth_image: Optional[np.ndarray],
        agent_state: Any,
        sensor_state: Any,
        fov: float,
    ) -> Tuple[float, float]:
        """Return (confidence, preferred_relative_yaw)."""
        if depth_image is None or depth_to_height is None or sensor_state is None:
            return 0.0, 0.0
        try:
            depth = np.asarray(depth_image, dtype=np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth, 0.0, 12.0)
            h, w = depth.shape[:2]
            step_h = max(1, h // 64)
            step_w = max(1, w // 64)
            depth_s = depth[::step_h, ::step_w]
            height_map = depth_to_height(
                depth_s, fov, sensor_state.position, sensor_state.rotation
            )
            agent_y = float(agent_state.position[1])
            mid = depth_s.shape[1] // 2
            half = max(2, depth_s.shape[1] // 8)
            strip_h = height_map[:, mid - half : mid + half]
            strip_d = depth_s[:, mid - half : mid + half]
            valid = (strip_d > 0.4) & (strip_d < 8.0)
            if int(valid.sum()) < 30:
                return 0.0, 0.0
            dy = strip_h - agent_y
            row_means_dy = []
            row_means_d = []
            for r in range(strip_h.shape[0]):
                m = valid[r]
                if int(m.sum()) < 3:
                    continue
                row_means_dy.append(float(np.mean(dy[r][m])))
                row_means_d.append(float(np.mean(strip_d[r][m])))
            if len(row_means_dy) < 8:
                return 0.0, 0.0
            ys = np.asarray(row_means_dy, dtype=np.float32)
            ds = np.asarray(row_means_d, dtype=np.float32)
            order = np.argsort(ds)
            ys_s = ys[order]
            ds_s = ds[order]
            if float(ds_s[-1] - ds_s[0]) < 0.8:
                return 0.0, 0.0
            b = float(np.polyfit(ds_s, ys_s, 1)[0])
            abs_b = abs(b)
            if abs_b < 0.12:
                return 0.0, 0.0
            conf = float(np.clip((abs_b - 0.12) / 0.35, 0.0, 1.0))
            left = height_map[:, :mid]
            right = height_map[:, mid:]
            left_dy = float(np.nanmean(np.abs(left - agent_y)))
            right_dy = float(np.nanmean(np.abs(right - agent_y)))
            yaw = 0.0
            if right_dy > left_dy + 0.05:
                yaw = 0.25
            elif left_dy > right_dy + 0.05:
                yaw = -0.25
            return conf, yaw
        except Exception:
            logging.debug("stair slope detect failed", exc_info=True)
            return 0.0, 0.0

    def choose_forward_action(
        self,
        a_final: list,
        preferred_yaw: float = 0.0,
        *,
        climbing: bool = False,
    ) -> Optional[int]:
        if not a_final:
            return None
        r_weight = self.climb_forward_r_weight if climbing else 0.4
        best_idx = None
        best_score = -1e9
        for i, action in enumerate(a_final, start=1):
            try:
                r, theta = action
            except Exception:
                continue
            r = float(r)
            theta = float(theta)
            align = math.cos(theta - preferred_yaw)
            score = align * 2.0 + min(r, 1.5) * r_weight - abs(theta) * 0.15
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def flatness_ratio(self, navigability_mask: Optional[np.ndarray]) -> float:
        if navigability_mask is None:
            return 0.0
        try:
            return float(np.mean(navigability_mask))
        except Exception:
            return 0.0

    def _confirm_map_swap(
        self,
        y: float,
        dy_from_entry: Optional[float],
        *,
        soft: bool = False,
        min_dy: Optional[float] = None,
    ) -> Optional[int]:
        """Return new floor_id when a floor change is confirmed."""
        if min_dy is None:
            min_dy = self.soft_swap_min_dy if soft else self.dy_threshold
        if dy_from_entry is None or dy_from_entry <= min_dy:
            return None
        new_floor = self.floor_id_from_y(y)
        if new_floor == self.floor_id:
            sign = 1 if y > float(self.entry_y or y) else -1
            new_floor = self.floor_id + sign
        return int(new_floor)

    def _apply_map_swap(self, y: float, new_floor: int) -> None:
        self.floor_id = int(new_floor)
        self._clear_climb_tracks(y)
        self._post_swap_steps = max(1, self.post_swap_prompt_steps)

    def _engage_climb_latch(self, y: float, *, entry_y: Optional[float] = None) -> None:
        self.climbing = True
        self.climb_steps = 0
        self.entry_y = float(self.y_anchor if entry_y is None else entry_y)
        self.dy_streak = 0
        self.stale_dy_streak = 0
        self.flat_stale_streak = 0
        self.landing_settle_streak = 0
        self.peak_stair_conf = 0.0
        self.last_dy_from_entry = abs(y - float(self.entry_y))
        self.stair_mouth_streak = 0
        self.stair_mouth_dy_streak = 0

    def _landing_ready(
        self, dy_from_entry: Optional[float], stair_conf: float = 0.0
    ) -> bool:
        if not self.landing_swap or dy_from_entry is None:
            return False
        if self.climb_steps < self.landing_min_climb_steps:
            return False
        if dy_from_entry < self.landing_min_dy:
            return False
        if self.landing_settle_streak < self.landing_settle_frames:
            return False
        # Height change plus settle is the floor-change evidence. Do not also
        # require a stair visual; slope detection misses ramps and landings.
        if self.landing_v2:
            # A mid-stair pause (Δy in [landing_min, hard) + still seeing stairs)
            # is not a landing. Ramps with no stair visual still swap. The
            # dy>1.1 × N hard-swap path does not use this gate.
            if (
                dy_from_entry < self.dy_threshold
                and float(stair_conf) >= self.stair_conf_thresh
            ):
                return False
        return True

    def _still_changing_floor_now(
        self,
        dy_from_entry: Optional[float],
        y_delta: float,
        stair_conf: float,
    ) -> bool:
        """True while the agent is mid floor-change, not after landing / false latch."""
        if not self.climbing:
            return False
        vertical_active = y_delta >= self.landing_settle_eps
        on_stairs = stair_conf >= self.stair_conf_thresh
        if dy_from_entry is None:
            return True
        if dy_from_entry < self.landing_min_dy:
            # t1s: a saturated mouth latch with no height gain must not
            # block Stop the way t1x did. Real climbs (Δy past protect)
            # still suppress while on stairs.
            if self.saturated_seek and dy_from_entry < self.stale_protect_dy:
                return vertical_active
            return vertical_active or on_stairs
        return vertical_active

    # ------------------------------------------------------------------ step
    def on_step(
        self,
        *,
        step_ndx: int,
        agent_state: Any,
        depth_image: Optional[np.ndarray],
        sensor_state: Any,
        fov: float,
        navigability_mask: Optional[np.ndarray],
        a_final: Optional[list],
        action_number: Optional[int],
        frontier_count: Optional[int] = None,
        revisit_count: int = 0,
        loop_risk: bool = False,
        visit_count: int = 0,
        underexecuted: bool = False,
        stop_generated: bool = False,
        stop_logit_margin: Optional[float] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "raise_nav_thresh": False,
            "suppress_stop": False,
            "bias_action": False,
            "action_number": action_number,
            "map_swap_floor_id": None,
            "save_maps_on_climb_start": False,
            "floor_id": None,
            "climbing": False,
            "stair_slope_conf": 0.0,
            "dy": None,
            "multifloor": {},
        }
        if not self.enabled:
            return result

        if self._post_swap_steps > 0:
            self._post_swap_steps -= 1

        pos = np.asarray(agent_state.position, dtype=np.float32)
        y = float(pos[1])
        if step_ndx == 0 or self.y_anchor is None:
            self.y_anchor = y
            self.floor_id = self.floor_id_from_y(y)
            self.prev_y = y

        stair_conf, stair_yaw = self.detect_stair_slope(
            depth_image, agent_state, sensor_state, fov
        )
        self.last_stair_conf = stair_conf
        self.last_stair_yaw = stair_yaw
        # Raise nav while climbing OR on a strong slope cue (mflight2 Move path).
        self.raise_nav_thresh = self.climbing or (
            stair_conf >= self.nav_stair_conf_thresh
        )
        result["raise_nav_thresh"] = bool(self.raise_nav_thresh)
        result["stair_slope_conf"] = float(stair_conf)

        dy_from_anchor = abs(y - float(self.y_anchor))
        dy_from_entry = (
            abs(y - float(self.entry_y)) if self.entry_y is not None else None
        )
        dy_eff = self._effective_dy(dy_from_entry, dy_from_anchor)
        self._saturated = self._compute_saturated(
            step_ndx, frontier_count, revisit_count, loop_risk, visit_count
        )
        self._escape = self._compute_escape(step_ndx, revisit_count, loop_risk)
        self._hallway = self._compute_hallway(step_ndx, revisit_count, loop_risk)
        self._far = self._compute_far(step_ndx, revisit_count, loop_risk)
        if (not self.climbing) and self._keep_cooloff > 0:
            self._keep_cooloff -= 1
        result["dy"] = dy_from_entry if dy_from_entry is not None else dy_from_anchor

        y_delta = 0.0 if self.prev_y is None else abs(y - float(self.prev_y))
        if (
            stair_conf >= self.stair_conf_thresh
            and y_delta >= self.climb_step_dy
        ):
            self.vertical_progress_streak += 1
        else:
            self.vertical_progress_streak = 0

        flat_pre = self.flatness_ratio(navigability_mask)

        # Engage climb latch.
        if not self.climbing:
            engaged = False
            engage_entry = self._entry_y_for_engage(y, dy_from_anchor)
            if self.stair_mouth_engage:
                flat_gate_ok = (
                    not self.stair_mouth_require_flat
                    or flat_pre < self.stair_mouth_max_flat
                )
                if y_delta >= self.climb_step_dy:
                    self.stair_mouth_dy_streak += 1
                else:
                    self.stair_mouth_dy_streak = 0
                if (
                    stair_conf >= self.stair_mouth_conf_thresh
                    and flat_gate_ok
                    and self.stair_mouth_dy_streak >= self.stair_mouth_frames
                ):
                    self.stair_mouth_streak += 1
                else:
                    self.stair_mouth_streak = 0
                if self.stair_mouth_streak >= self.stair_mouth_frames:
                    self._engage_climb_latch(y, entry_y=engage_entry)
                    engaged = True
            if (
                not engaged
                and self.saturated_seek
                and self.seek_latch
                and self._saturated
                and stair_conf >= self.seek_stair_conf
            ):
                self.seek_mouth_streak += 1
            else:
                if not engaged:
                    self.seek_mouth_streak = 0
            if (
                not engaged
                and self.saturated_seek
                and self.seek_latch
                and self._saturated
                and self.seek_mouth_streak >= self.seek_frames
            ):
                self._engage_climb_latch(y, entry_y=engage_entry)
                engaged = True
            if not engaged and self.vertical_progress_streak >= self.climb_engage_frames:
                self._engage_climb_latch(y, entry_y=engage_entry)
                engaged = True
            # KEEP_ANCHOR must not re-latch from residual Δy alone (t1s2
            # chair 99: timeout → keep → anchor_ok same frame).
            skip_anchor_ok = bool(
                self.keep_anchor
                and dy_from_anchor >= self.keep_anchor_min_dy
            )
            if not engaged and self._keep_cooloff <= 0 and not skip_anchor_ok:
                anchor_ok = dy_from_anchor >= self.climb_start_dy
                if self.require_stair_for_anchor_climb:
                    anchor_ok = anchor_ok and stair_conf >= self.stair_conf_thresh
                if anchor_ok:
                    self._engage_climb_latch(y, entry_y=float(self.y_anchor))

        map_swap_floor_id = None
        soft_swap = False
        landing_swap_hit = False
        if self.climbing:
            if self.entry_y is not None:
                dy_from_entry = abs(y - float(self.entry_y))
            dy_eff = self._effective_dy(dy_from_entry, dy_from_anchor)
            self.climb_steps += 1
            flat = self.flatness_ratio(navigability_mask)
            self.peak_stair_conf = max(self.peak_stair_conf, float(stair_conf))
            if y_delta < self.landing_settle_eps:
                self.landing_settle_streak += 1
            else:
                self.landing_settle_streak = 0
            if dy_eff is not None and dy_eff > self.dy_threshold:
                self.dy_streak += 1
            else:
                self.dy_streak = 0

            # Stale latch: dy not progressing. Real climbs pause on stairs;
            # only count stale when we have not already gained protect_dy and
            # the slope cue is gone.
            if dy_from_entry is not None:
                prev_dy = (
                    self.last_dy_from_entry
                    if self.last_dy_from_entry is not None
                    else dy_from_entry
                )
                if (
                    dy_from_entry < self.stale_dy_max
                    and (dy_from_entry - prev_dy) < self.stale_dy_eps
                ):
                    self.stale_dy_streak += 1
                else:
                    self.stale_dy_streak = 0
                self.last_dy_from_entry = dy_from_entry

            # Flat false-latch: almost no vertical progress on same-floor slope noise.
            if (
                dy_from_entry is not None
                and dy_from_entry < self.flat_stale_dy_max
                and flat >= self.swap_flat_min
            ):
                self.flat_stale_streak += 1
            else:
                self.flat_stale_streak = 0

            # Protect only after a real height gain. Peak/current stair alone
            # used to pin false latches until timeout=40 (t1x same-floor Stop miss).
            real_climb_protect = bool(
                self.landing_swap
                and dy_from_entry is not None
                and dy_from_entry >= self.stale_protect_dy
            )

            false_latch_release = (
                dy_from_entry is not None
                and dy_from_entry < self.flat_stale_dy_max
                and self.flat_stale_streak >= self.flat_stale_frames
            )
            stale_pause_release = (
                not real_climb_protect
                and dy_from_entry is not None
                and dy_from_entry < self.stale_dy_max
                and self.stale_dy_streak >= self.stale_release_frames
            )
            keep_on_release = self._should_keep_anchor(dy_from_entry, dy_from_anchor)

            if (
                self.dy_streak >= self.dy_confirm_frames
                and dy_eff is not None
                and dy_eff > self.dy_threshold
            ):
                new_floor = self._confirm_map_swap(y, dy_eff)
                if new_floor is not None:
                    map_swap_floor_id = int(new_floor)
                    self._apply_map_swap(y, map_swap_floor_id)
            elif self._landing_ready(dy_eff, stair_conf):
                new_floor = self._confirm_map_swap(
                    y, dy_eff, min_dy=self.landing_min_dy
                )
                if new_floor is not None:
                    map_swap_floor_id = int(new_floor)
                    landing_swap_hit = True
                    self._apply_map_swap(y, map_swap_floor_id)
            elif false_latch_release or stale_pause_release:
                self._clear_climb_tracks(y, keep_anchor=keep_on_release)
            elif self.climb_steps >= self.climb_timeout:
                timeout_swap = False
                if self.landing_swap and dy_eff is not None:
                    # t1y: timeout force-swaps at Δy>=landing_min even mid-stair.
                    # t1z landing_v2: only swap if the landing predicate holds.
                    timeout_ready = dy_eff >= self.landing_min_dy
                    if self.landing_v2:
                        timeout_ready = self._landing_ready(
                            dy_eff, stair_conf
                        )
                    if timeout_ready:
                        new_floor = self._confirm_map_swap(
                            y, dy_eff, min_dy=self.landing_min_dy
                        )
                        if new_floor is not None:
                            map_swap_floor_id = int(new_floor)
                            landing_swap_hit = True
                            timeout_swap = True
                            self._apply_map_swap(y, map_swap_floor_id)
                elif (
                    self.soft_swap_on_timeout
                    and dy_eff is not None
                    and flat >= self.swap_flat_min
                ):
                    soft_ok = False
                    if dy_eff >= self.soft_swap_min_dy:
                        soft_ok = True
                    elif (
                        self.climb_steps >= self.soft_swap_min_climb_steps
                        and dy_eff >= self.soft_swap_alt_min_dy
                    ):
                        soft_ok = True
                    if soft_ok:
                        new_floor = self._confirm_map_swap(
                            y, dy_eff, soft=True
                        )
                        if new_floor is not None:
                            map_swap_floor_id = int(new_floor)
                            soft_swap = True
                            timeout_swap = True
                            self._apply_map_swap(y, map_swap_floor_id)
                if self.climbing and not timeout_swap:
                    self._clear_climb_tracks(y, keep_anchor=keep_on_release)

        self.prev_y = y
        # Recompute raise flag after possible latch clear.
        self.raise_nav_thresh = self.climbing or (
            stair_conf >= self.nav_stair_conf_thresh
        )
        result["raise_nav_thresh"] = bool(self.raise_nav_thresh)
        result["climbing"] = bool(self.climbing)
        if self.landing_swap:
            self._still_changing_floor = self._still_changing_floor_now(
                dy_from_entry, y_delta, stair_conf
            )
            suppress_stop = bool(self._still_changing_floor)
        else:
            self._still_changing_floor = bool(self.climbing)
            suppress_stop = bool(self.climbing)
        result["suppress_stop"] = suppress_stop
        result["map_swap_floor_id"] = map_swap_floor_id
        result["floor_id"] = int(self.floor_id)

        if self.climbing and not self._prev_climbing:
            result["save_maps_on_climb_start"] = True

        # Weak bias ONLY while changing floors. After landing, keep Stop.
        self._escape = self._compute_escape(step_ndx, revisit_count, loop_risk)
        self._hallway = self._compute_hallway(step_ndx, revisit_count, loop_risk)
        self._far = self._compute_far(step_ndx, revisit_count, loop_risk)
        override_num = action_number
        did_bias = False
        if self.climbing and a_final:
            try:
                cur = int(action_number) if action_number is not None else None
            except (TypeError, ValueError):
                cur = None
            if cur == -1 and not suppress_stop:
                pass
            elif cur in (-1, 0, None):
                pick = self.choose_forward_action(
                    list(a_final),
                    preferred_yaw=stair_yaw if stair_conf >= 0.3 else 0.0,
                    climbing=True,
                )
                if pick is not None and pick != cur:
                    override_num = pick
                    did_bias = True
        elif (
            self.post_swap_forward_bias
            and self._post_swap_steps > 0
            and a_final
        ):
            try:
                cur = int(action_number) if action_number is not None else None
            except (TypeError, ValueError):
                cur = None
            if cur in (0, None):
                pick = self.choose_forward_action(
                    list(a_final),
                    preferred_yaw=float(self.last_stair_yaw or 0.0),
                    climbing=False,
                )
                if pick is not None and pick != cur:
                    override_num = pick
                    did_bias = True
        elif (
            self._escape
            and not self.climbing
            and a_final
        ):
            # Leave a dead-end room via longest-range candidate.
            # Never rewrite Stop; never latch.
            try:
                cur = int(action_number) if action_number is not None else None
            except (TypeError, ValueError):
                cur = None
            if cur == -1 or stop_generated:
                pass
            elif cur in (0, None):
                pick = self.choose_longest_action(list(a_final))
                if pick is not None and pick != cur:
                    override_num = pick
                    did_bias = True
            elif loop_risk or int(revisit_count) >= self.escape_revisit:
                tight = float(self.escape_tight_r or 0.0)
                if tight > 0 and self.max_candidate_range(list(a_final)) > tight:
                    pick = None
                else:
                    pick = self.choose_reorient_action(list(a_final), current=cur)
                if pick is not None and pick != cur:
                    override_num = pick
                    did_bias = True
        elif (
            self._hallway
            and not self.climbing
            and a_final
        ):
            # Looping on a short ray with a long hallway elsewhere → take it.
            # Never Stop, never action 0, never 180-for-its-own-sake.
            try:
                cur = int(action_number) if action_number is not None else None
            except (TypeError, ValueError):
                cur = None
            if cur in (-1, 0, None) or stop_generated:
                pass
            else:
                cur_r = None
                try:
                    cur_r = float(a_final[int(cur) - 1][0])
                except Exception:
                    cur_r = None
                mx = self.max_candidate_range(list(a_final))
                if (
                    cur_r is not None
                    and cur_r < float(self.hallway_short_r)
                    and mx >= float(self.hallway_long_r)
                ):
                    pick = self.choose_longest_action(list(a_final))
                    if pick is not None and pick != cur:
                        override_num = pick
                        did_bias = True
        elif (
            self._far
            and not self.climbing
            and a_final
        ):
            # Looping on the longest ray while Stop says the object is not
            # here → 180. Never rewrite Stop. Skip missing/weak margins.
            try:
                cur = int(action_number) if action_number is not None else None
            except (TypeError, ValueError):
                cur = None
            margin = None
            try:
                if stop_logit_margin is not None:
                    margin = float(stop_logit_margin)
            except (TypeError, ValueError):
                margin = None
            if cur == -1 or stop_generated:
                pass
            elif margin is None or margin >= float(self.far_margin):
                pass
            elif self._current_is_longest(list(a_final), cur):
                pick = self.choose_reorient_action(list(a_final), current=cur)
                if pick is not None and pick != cur:
                    override_num = pick
                    did_bias = True
        elif (
            self.saturated_seek
            and self._saturated
            and not self.climbing
            and stair_conf >= self.seek_stair_conf
            and a_final
        ):
            # Late stair-seek: rewrite turn-around only. Never touch Stop.
            try:
                cur = int(action_number) if action_number is not None else None
            except (TypeError, ValueError):
                cur = None
            if cur in (0, None):
                pick = self.choose_forward_action(
                    list(a_final),
                    preferred_yaw=stair_yaw if stair_conf >= 0.3 else 0.0,
                    climbing=False,
                )
                if pick is not None and pick != cur:
                    override_num = pick
                    did_bias = True
        elif (
            (self.raise_nav_bias or self.stair_mouth_bias)
            and not self.climbing
            and self.raise_nav_thresh
            and stair_conf >= self.nav_stair_conf_thresh
            and a_final
        ):
            min_dy = 0.2 if self.stair_mouth_bias else self.bias_min_dy
            if dy_from_anchor >= min_dy:
                try:
                    cur = int(action_number) if action_number is not None else None
                except (TypeError, ValueError):
                    cur = None
                if cur in (0, None):
                    pick = self.choose_forward_action(
                        list(a_final),
                        preferred_yaw=stair_yaw if stair_conf >= 0.3 else 0.0,
                        climbing=False,
                    )
                    if pick is not None and pick != cur:
                        override_num = pick
                        did_bias = True

        result["bias_action"] = did_bias
        result["action_number"] = override_num
        result["multifloor"] = {
            "climbing": bool(self.climbing),
            "climb_steps": int(self.climb_steps),
            "floor_id": int(self.floor_id),
            "stair_slope_conf": float(stair_conf),
            "stair_yaw": float(stair_yaw),
            "raise_nav_thresh": bool(self.raise_nav_thresh),
            "suppress_stop": bool(suppress_stop),
            "still_changing_floor": bool(self._still_changing_floor),
            "dy_from_anchor": float(dy_from_anchor),
            "dy_from_entry": (
                None if dy_from_entry is None else float(dy_from_entry)
            ),
            "dy_streak": int(self.dy_streak),
            "stale_dy_streak": int(self.stale_dy_streak),
            "vertical_progress_streak": int(self.vertical_progress_streak),
            "map_swapped_to": map_swap_floor_id,
            "soft_swap": bool(soft_swap),
            "landing_swap": bool(landing_swap_hit),
            "landing_v2": bool(self.landing_v2),
            "keep_anchor": bool(self.keep_anchor),
            "keep_cooloff": int(self._keep_cooloff),
            "saturated_seek": bool(self.saturated_seek),
            "seek_latch": bool(self.seek_latch),
            "saturated": bool(self._saturated),
            "escape_seek": bool(self.escape_seek),
            "escape_active": bool(self._escape),
            "hallway_commit": bool(self.hallway_commit),
            "hallway_active": bool(self._hallway),
            "far_reorient": bool(self.far_reorient),
            "far_active": bool(self._far),
            "seek_mouth_streak": int(self.seek_mouth_streak),
            "dy_eff": (None if dy_eff is None else float(dy_eff)),
            "landing_settle_streak": int(self.landing_settle_streak),
            "peak_stair_conf": float(self.peak_stair_conf),
            "stair_mouth_streak": int(self.stair_mouth_streak),
            "stair_mouth_dy_streak": int(self.stair_mouth_dy_streak),
            "stair_mouth_min_dy": float(self.stair_mouth_min_dy),
            "stair_mouth_max_flat": float(self.stair_mouth_max_flat),
            "flat_stale_streak": int(self.flat_stale_streak),
            "swap_flat_min": float(self.swap_flat_min),
            "save_maps_on_climb_start": bool(result["save_maps_on_climb_start"]),
            "post_swap_prompt_steps": int(self._post_swap_steps),
            "raise_nav_bias": bool(
                did_bias and not result["suppress_stop"]
            ),
            "bias_action": did_bias,
        }
        self.last_metadata = result["multifloor"]
        self._prev_climbing = bool(self.climbing)
        return result
