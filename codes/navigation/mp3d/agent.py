import logging
import math
import random
import os
import habitat_sim
import numpy as np
import cv2
import ast
import hashlib
import concurrent.futures
import time
import sys
import re
import json

from simWrapper import PolarAction
from utils import *
from vlm import *
from pivot import PIVOT
from multifloor_controller import MultiFloorController, multifloor_enabled
from navigation_runtime import (
    PRIVATE_INFERENCE_FIELDS,
    action_requests_stop_confirmation,
    blind_stop_crosscheck_prompt_suffix,
    candidate_model_theta_to_yaw_command,
    cap_stop_confirmation_actions,
    point_segment_distance_xz,
    public_inference_observation,
    response_navigation_mode,
    navigation_mode_allows_stop,
    factorized_evidence_allows_stop,
    terminal_certificate_allows_stop,
    state_conditioned_spatial_stop_allows_stop,
    state_conditioned_evidence_stop_allows_stop,
    state_conditioned_causal_stop_bottleneck_allows_stop,
    parse_state_conditioned_causal_stop_bottleneck_response,
    parse_visual_arrival_certificate_v2_runtime_response,
    visual_arrival_certificate_v2_allows_stop,
    temporal_reference_contract_valid,
    temporal_reference_confirmation_eligible,
    temporal_relation_allows_stop,
    select_public_memory_candidate_indices,
    should_stop_after_same_pose_crosscheck,
    use_consecutive_stop_fallback,
    use_single_stop_direct_execute,
)

EMBODIED_NAV_COT_SRC = os.environ.get(
    "EMBODIED_NAV_COT_SRC",
    str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"),
)


def _load_objectnav_cot_protocol():
    if EMBODIED_NAV_COT_SRC and EMBODIED_NAV_COT_SRC not in sys.path:
        sys.path.insert(0, EMBODIED_NAV_COT_SRC)
    from embodied_nav_cot.vlmnav_protocol import (  # type: ignore
        ABILITYMODE_PROTOCOL_ID,
        ABILITYMODE_715_PROTOCOL_ID,
        AUX_SPATIAL_STATE_PROTOCOL_ID,
        DIRECT_ACTION_WITH_STATE_PROTOCOL_ID,
        DIRECT_ACTION_PROTOCOL_ID,
        DYNAMIC_STATE_ROUTED_PROTOCOL_ID,
        EVIDENCE_BRIDGE_PROTOCOL_ID,
        FACTORIZED_BRIDGE_PROTOCOL_ID,
        TEMPORAL_REFERENCE_BRIDGE_PROTOCOL_ID,
        STABLE_TEMPORAL_REFERENCE_BRIDGE_PROTOCOL_ID,
        PUBLIC_MEMORY_STATE_BRIDGE_PROTOCOL_ID,
        PROTOCOL_ID,
        STATE_ROUTED_PROTOCOL_ID,
        V4_INTEGRATED_PROTOCOL_ID,
        render_abilitymode715_action_prompt,
        render_abilitymode_action_prompt,
        render_action_prompt,
        render_aux_spatial_state_prompt,
        render_direct_action_prompt,
        render_direct_action_with_state_prompt,
        render_direct_stopping_prompt,
        render_dynamic_state_routed_action_prompt,
        render_evidence_bridge_action_prompt,
        render_factorized_bridge_action_prompt,
        render_temporal_reference_bridge_action_prompt,
        render_stable_temporal_reference_bridge_action_prompt,
        render_public_memory_state_bridge_action_prompt,
        render_stopping_prompt,
        render_stopping715_prompt,
        render_state_routed_action_prompt,
        render_v4_integrated_action_prompt,
    )

    return {
        PROTOCOL_ID: (render_action_prompt, render_stopping_prompt),
        ABILITYMODE_PROTOCOL_ID: (render_abilitymode_action_prompt, render_stopping_prompt),
        ABILITYMODE_715_PROTOCOL_ID: (render_abilitymode715_action_prompt, render_stopping715_prompt),
        STATE_ROUTED_PROTOCOL_ID: (render_state_routed_action_prompt, render_stopping715_prompt),
        DYNAMIC_STATE_ROUTED_PROTOCOL_ID: (render_dynamic_state_routed_action_prompt, render_stopping715_prompt),
        EVIDENCE_BRIDGE_PROTOCOL_ID: (render_evidence_bridge_action_prompt, render_stopping715_prompt),
        FACTORIZED_BRIDGE_PROTOCOL_ID: (render_factorized_bridge_action_prompt, render_stopping715_prompt),
        TEMPORAL_REFERENCE_BRIDGE_PROTOCOL_ID: (render_temporal_reference_bridge_action_prompt, render_stopping715_prompt),
        STABLE_TEMPORAL_REFERENCE_BRIDGE_PROTOCOL_ID: (render_stable_temporal_reference_bridge_action_prompt, render_stopping715_prompt),
        PUBLIC_MEMORY_STATE_BRIDGE_PROTOCOL_ID: (render_public_memory_state_bridge_action_prompt, render_stopping715_prompt),
        DIRECT_ACTION_PROTOCOL_ID: (render_direct_action_prompt, render_direct_stopping_prompt),
        AUX_SPATIAL_STATE_PROTOCOL_ID: (render_aux_spatial_state_prompt, render_direct_stopping_prompt),
        DIRECT_ACTION_WITH_STATE_PROTOCOL_ID: (render_direct_action_with_state_prompt, render_direct_stopping_prompt),
        V4_INTEGRATED_PROTOCOL_ID: (render_v4_integrated_action_prompt, render_direct_stopping_prompt),
    }


class Agent:
    def __init__(self, cfg: dict):
        pass

    def step(self, obs: dict):
        """Primary agent loop to map observations to the agent's action and returns metadata."""
        raise NotImplementedError

    def get_spend(self):
        """Returns the dollar amount spent by the agent on API calls."""
        return 0

    def reset(self):
        """To be called after each episode."""
        pass


class RandomAgent(Agent):
    """Example implementation of a random agent."""
    
    def step(self, obs):
        rotate = random.uniform(-0.2, 0.2)
        forward = random.uniform(0, 1)

        agent_action = PolarAction(forward, rotate)
        metadata = {
            'step_metadata': {'success': 1}, # indicating the VLM succesfully selected an action
            'logging_data': {}, # to be logged in the txt file
            'images': {'color_sensor': obs['color_sensor']} # to be visualized in the GIF
        }
        return agent_action, metadata


class VLMNavAgent(Agent):
    """
    Primary class for the VLMNav agent. Four primary components: navigability, action proposer, projection, and prompting. Runs seperate threads for stopping and preprocessing. This class steps by taking in an observation and returning a PolarAction, along with metadata for logging and visulization.
    """
    explored_color = GREY
    unexplored_color = GREEN
    map_size = 5000
    explore_threshold = 3
    voxel_ray_size = 60
    e_i_scaling = 0.8
    private_inference_fields = PRIVATE_INFERENCE_FIELDS
    circled_digit_map = {
        "⓪": "0", "①": "1", "②": "2", "③": "3", "④": "4",
        "⑤": "5", "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9",
        "⑩": "10", "➀": "1", "➁": "2", "➂": "3", "➃": "4",
        "➄": "5", "➅": "6", "➆": "7", "➇": "8", "➈": "9",
        "➉": "10",
    }

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fov = cfg['sensor_cfg']['fov']
        self.resolution = (
            1080 // cfg['sensor_cfg']['res_factor'],
            1920 // cfg['sensor_cfg']['res_factor']
        )

        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])       
        self.pivot = PIVOT(self.actionVLM, self.fov, self.resolution, max_action_length=cfg['max_action_dist']) if cfg['pivot'] else None

        assert cfg['navigability_mode'] in ['none', 'depth_estimate', 'segmentation', 'depth_sensor']
        self.depth_estimator = DepthEstimator() if cfg['navigability_mode'] == 'depth_estimate' else None
        self.segmentor = Segmentor() if cfg['navigability_mode'] == 'segmentation' else None
        self.reset()

    def step(self, obs: dict):
        agent_state: habitat_sim.AgentState = obs['agent_state']
        obs = self._public_inference_observation(obs)
        agent_state = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position

        agent_action, metadata = self._choose_action(obs)
        metadata['step_metadata'].update(self.cfg)

        if metadata['step_metadata']['action_number'] == 0:
            self.turned = self.step_ndx

        # Visualize the chosen action
        chosen_action_image = obs['color_sensor'].copy()
        self._project_onto_image(
            metadata['a_final'], chosen_action_image, agent_state,
            agent_state.sensor_states['color_sensor'], 
            chosen_action=metadata['step_metadata']['action_number']
        )
        metadata['images']['color_sensor_chosen'] = chosen_action_image

        try:
            self.prompt_action_history.append(int(metadata['step_metadata']['action_number']))
            max_history = int(os.environ.get("VLMNAV_WMNAV_LITE_ACTION_HISTORY_MAX", "32"))
            max_history = max(1, max_history)
            if len(self.prompt_action_history) > max_history:
                self.prompt_action_history = self.prompt_action_history[-max_history:]
        except Exception:
            pass

        self.step_ndx += 1
        return agent_action, metadata
    
    def get_spend(self):
        return self.actionVLM.get_spend() + self.stoppingVLM.get_spend()

    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.actionVLM.reset()
        self.last_candidate_action_scores = []
        self.prompt_position_counts = {}
        self.prompt_action_history = []
        self.online_position_counts = {}
        self.online_blocked_points = []
        self._online_motion_sequence_id = None
        self._latest_motion_feedback = {}
        self._online_candidate_memory_by_pose = {}
        self._last_private_fields_removed = []
        self._temporal_target_reference_image = None
        self._temporal_target_reference_candidate_image = None
        self._temporal_target_reference_confirmation_streak = 0
        self._temporal_target_reference_mismatch_streak = 0
        self._temporal_target_reference_unconfirmed_streak = 0
        self._temporal_target_reference_age = 0
        self._last_stopping_prompt = ""
        self._last_stopping_prompt_sha256 = ""
        self._last_navigability_mask = None
        if not hasattr(self, "multifloor") or self.multifloor is None:
            self.multifloor = MultiFloorController()
        else:
            self.multifloor.reset()

    def _public_inference_observation(self, obs: dict) -> dict:
        """Return the agent-visible observation with privileged labels removed."""
        if not isinstance(obs, dict):
            return obs
        clean, removed = public_inference_observation(obs)
        self._last_private_fields_removed = removed
        return clean

    def _online_nav_memory_enabled(self) -> bool:
        return os.environ.get(
            'VLMNAV_ONLINE_NAV_MEMORY',
            '1',
        ).strip().lower() not in ('0', 'false', 'no', 'off')

    def _online_position_key(self, position):
        try:
            cell_size = float(os.environ.get('VLMNAV_ONLINE_NAV_CELL_SIZE', '0.35'))
        except ValueError:
            cell_size = 0.35
        cell_size = max(0.1, cell_size)
        pos = np.asarray(position, dtype=float)
        return tuple(np.round(pos[[0, 2]] / cell_size).astype(int).tolist())

    @staticmethod
    def _point_segment_distance_xz(point, start, end) -> float:
        return point_segment_distance_xz(point, start, end)

    def _ingest_online_motion_feedback(self, obs: dict) -> None:
        feedback = obs.get('motion_feedback') if isinstance(obs, dict) else None
        if not isinstance(feedback, dict):
            return
        try:
            sequence_id = int(feedback.get('sequence_id'))
        except (TypeError, ValueError):
            return
        if sequence_id == self._online_motion_sequence_id:
            return
        self._online_motion_sequence_id = sequence_id
        self._latest_motion_feedback = dict(feedback)
        if not self._online_nav_memory_enabled():
            return

        end_position = feedback.get('end_position')
        if end_position is not None:
            key = self._online_position_key(end_position)
            self.online_position_counts[key] = int(self.online_position_counts.get(key, 0)) + 1

        try:
            requested = float(feedback.get('requested_distance') or 0.0)
            ratio = float(feedback.get('execution_ratio') or 0.0)
            step_id = int(self.step_ndx)
        except (TypeError, ValueError):
            return
        if requested <= 0.05:
            return

        try:
            ttl = int(os.environ.get('VLMNAV_ONLINE_BLOCK_TTL_STEPS', '24'))
        except ValueError:
            ttl = 24
        ttl = max(1, ttl)
        self.online_blocked_points = [
            item for item in self.online_blocked_points
            if step_id - int(item.get('last_step', step_id)) <= ttl
        ]

        start = np.asarray(feedback.get('start_position'), dtype=float)
        expected = np.asarray(feedback.get('expected_end_position'), dtype=float)
        end = np.asarray(feedback.get('end_position'), dtype=float)
        if start.shape != (3,) or expected.shape != (3,) or end.shape != (3,):
            return
        direction = expected - start
        direction[1] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return
        direction /= norm

        if ratio < 0.5:
            block_point = end + direction * 0.20
            weight = 2 if ratio < 0.2 else 1
            merged = False
            for item in self.online_blocked_points:
                if np.linalg.norm(np.asarray(item['position'])[[0, 2]] - block_point[[0, 2]]) <= 0.35:
                    item['count'] = int(item.get('count', 0)) + weight
                    item['last_step'] = step_id
                    item['position'] = block_point.tolist()
                    merged = True
                    break
            if not merged:
                self.online_blocked_points.append({
                    'position': block_point.tolist(),
                    'count': int(weight),
                    'last_step': step_id,
                })
        elif ratio >= 0.8:
            self.online_blocked_points = [
                item for item in self.online_blocked_points
                if self._point_segment_distance_xz(item['position'], start, end) > 0.20
            ]

    def _candidate_online_memory(self, action, agent_state) -> dict:
        r, theta = float(action[0]), float(action[1])
        local_goal = np.array([r * np.sin(theta), 0.0, -r * np.cos(theta)], dtype=float)
        world_goal = local_to_global(agent_state.position, agent_state.rotation, local_goal)
        visit_count = int(self.online_position_counts.get(self._online_position_key(world_goal), 0))
        blocked_count = 0
        for item in self.online_blocked_points:
            if self._point_segment_distance_xz(
                item.get('position'), agent_state.position, world_goal
            ) <= 0.28:
                blocked_count += int(item.get('count', 1))
        return {
            'landing_position_public': np.asarray(world_goal, dtype=float).tolist(),
            'online_visit_count': visit_count,
            'online_blocked_count': blocked_count,
            'online_blocked': bool(blocked_count > 0),
        }

    def _filter_online_navigation_candidates(self, candidates: list, agent_state):
        self._online_candidate_memory_by_pose = {}
        filter_enabled = os.environ.get(
            'VLMNAV_ONLINE_CANDIDATE_FILTER',
            '1',
        ).strip().lower() not in ('0', 'false', 'no', 'off')
        if (
            not self._online_nav_memory_enabled()
            or not filter_enabled
            or not candidates
        ):
            return candidates
        annotated = []
        for item in candidates:
            memory = self._candidate_online_memory(item, agent_state)
            pose_key = (round(float(item[0]), 3), round(float(item[1]), 4))
            self._online_candidate_memory_by_pose[pose_key] = memory
            annotated.append((item, memory))

        clean_indices = select_public_memory_candidate_indices(
            [memory for _, memory in annotated]
        )
        clean = [annotated[index][0] for index in clean_indices]
        if clean:
            return clean

        turn_available = (
            self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']
        )
        if turn_available:
            return []
        least_risky, _ = min(
            annotated,
            key=lambda pair: (
                pair[1]['online_blocked_count'],
                pair[1]['online_visit_count'],
                float(pair[0][0]),
            ),
        )
        return [least_risky]

    def _motion_feedback_metadata(self) -> dict:
        feedback = self._latest_motion_feedback or {}
        return {
            'public_inference_only': True,
            'private_observation_fields_removed': list(self._last_private_fields_removed),
            'previous_execution_sequence_id': feedback.get('sequence_id'),
            'previous_execution_requested_distance': feedback.get('requested_distance'),
            'previous_execution_net_displacement': feedback.get('net_displacement'),
            'previous_execution_traveled_distance': feedback.get('traveled_distance'),
            'previous_execution_ratio': feedback.get('execution_ratio'),
            'previous_execution_collision_substeps': feedback.get('collision_substeps'),
            'previous_execution_slide_rejected_substeps': feedback.get('slide_rejected_substeps'),
            'previous_execution_underexecuted': bool(feedback.get('underexecuted', False)),
            'online_visited_cell_count': int(len(self.online_position_counts)),
            'online_blocked_point_count': int(len(self.online_blocked_points)),
        }

    def _construct_prompt(self, **kwargs):
        raise NotImplementedError
    
    def _choose_action(self, obs):
        raise NotImplementedError

    @staticmethod
    def _apply_openai_role_sampling_kwargs(model_kwargs: dict, role: str) -> None:
        """Apply an explicitly requested per-role OpenAI sampling contract.

        The historical VLMNav surface uses only the generic
        ``VLMNAV_OPENAI_SEND_SAMPLING_ARGS`` /
        ``VLMNAV_OPENAI_SEND_MAX_TOKENS`` / ``OPENAI_*`` controls.  Do not
        replace that behavior merely because action and stopping models now
        have distinct endpoints: role controls take effect only when a
        non-empty role-specific environment variable is present.  This lets a
        terminal-certificate run freeze the action surface (no explicit
        sampling kwargs) while making the certificate call deterministic.
        """
        if role not in ("action", "stopping"):
            raise ValueError("unsupported OpenAI VLM role: {}".format(role))

        prefix = "VLMNAV_{}_OPENAI".format(role.upper())
        send_env = "{}_SEND_SAMPLING_ARGS".format(prefix)
        send_max_tokens_env = "{}_SEND_MAX_TOKENS".format(prefix)
        temperature_env = "{}_TEMPERATURE".format(prefix)
        max_tokens_env = "{}_MAX_TOKENS".format(prefix)

        raw_send = os.environ.get(send_env)
        raw_send_max_tokens = os.environ.get(send_max_tokens_env)
        raw_temperature = os.environ.get(temperature_env)
        raw_max_tokens = os.environ.get(max_tokens_env)
        has_role_override = any(
            value is not None and str(value).strip() != ""
            for value in (
                raw_send,
                raw_send_max_tokens,
                raw_temperature,
                raw_max_tokens,
            )
        )
        if not has_role_override:
            return

        # OpenAIVLM falls back to the generic sampling switch when this role
        # switch is not set, preserving legacy launches.  Once a role contract
        # is explicitly supplied, make its switch authoritative.
        model_kwargs["sampling_args_env_name"] = send_env
        model_kwargs["send_max_tokens_env_name"] = send_max_tokens_env
        if raw_send is not None and str(raw_send).strip() != "":
            model_kwargs["send_sampling_args"] = raw_send
        if raw_send_max_tokens is not None and str(raw_send_max_tokens).strip() != "":
            model_kwargs["send_max_tokens"] = raw_send_max_tokens
        if raw_temperature is not None and str(raw_temperature).strip() != "":
            try:
                temperature = float(raw_temperature)
            except ValueError as exc:
                raise ValueError(
                    "{} must be a finite numeric temperature, got {!r}".format(
                        temperature_env, raw_temperature
                    )
                ) from exc
            if not math.isfinite(temperature):
                raise ValueError(
                    "{} must be a finite numeric temperature, got {!r}".format(
                        temperature_env, raw_temperature
                    )
                )
            model_kwargs["temperature"] = temperature
        if raw_max_tokens is not None and str(raw_max_tokens).strip() != "":
            try:
                max_tokens = int(raw_max_tokens)
            except ValueError as exc:
                raise ValueError(
                    "{} must be a positive integer, got {!r}".format(
                        max_tokens_env, raw_max_tokens
                    )
                ) from exc
            if max_tokens <= 0:
                raise ValueError(
                    "{} must be a positive integer, got {!r}".format(
                        max_tokens_env, raw_max_tokens
                    )
                )
            model_kwargs["max_tokens"] = max_tokens

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        action_kwargs = dict(cfg['model_kwargs'])
        action_model = os.environ.get("VLMNAV_ACTION_MODEL")
        action_base_url = os.environ.get("VLMNAV_ACTION_OPENAI_BASE_URL")
        action_api_key = os.environ.get("VLMNAV_ACTION_OPENAI_API_KEY")
        action_timeout_s = os.environ.get("VLMNAV_ACTION_OPENAI_TIMEOUT")
        if action_model:
            action_kwargs['model'] = action_model
        if action_base_url:
            action_kwargs['base_url'] = action_base_url
        if action_api_key:
            action_kwargs['api_key'] = action_api_key
        if action_timeout_s:
            action_kwargs['timeout_s'] = float(action_timeout_s)
        if isinstance(vlm_cls, type) and issubclass(vlm_cls, OpenAIVLM):
            self._apply_openai_role_sampling_kwargs(action_kwargs, "action")
        self.actionVLM: VLM = vlm_cls(**action_kwargs, system_instruction=system_instruction)
        stopping_kwargs = dict(cfg.get('stopping_model_kwargs') or cfg['model_kwargs'])
        stopping_model = os.environ.get("VLMNAV_STOPPING_MODEL")
        stopping_base_url = os.environ.get("VLMNAV_STOPPING_OPENAI_BASE_URL")
        stopping_api_key = os.environ.get("VLMNAV_STOPPING_OPENAI_API_KEY")
        stopping_timeout_s = os.environ.get("VLMNAV_STOPPING_OPENAI_TIMEOUT")
        stopping_max_image_res = os.environ.get("VLMNAV_STOPPING_MAX_IMAGE_RES")
        stopping_image_encode_format = os.environ.get(
            "VLMNAV_STOPPING_IMAGE_ENCODE_FORMAT"
        )
        stopping_jpeg_quality = os.environ.get("VLMNAV_STOPPING_JPEG_QUALITY")
        if stopping_model:
            stopping_kwargs['model'] = stopping_model
        if stopping_base_url:
            stopping_kwargs['base_url'] = stopping_base_url
        if stopping_api_key:
            stopping_kwargs['api_key'] = stopping_api_key
        if stopping_timeout_s:
            stopping_kwargs['timeout_s'] = float(stopping_timeout_s)
        if stopping_max_image_res:
            try:
                parsed_max_image_res = int(stopping_max_image_res)
            except ValueError as exc:
                raise ValueError(
                    "VLMNAV_STOPPING_MAX_IMAGE_RES must be a positive integer, "
                    f"got {stopping_max_image_res!r}"
                ) from exc
            if parsed_max_image_res <= 0:
                raise ValueError(
                    "VLMNAV_STOPPING_MAX_IMAGE_RES must be a positive integer, "
                    f"got {stopping_max_image_res!r}"
                )
            stopping_kwargs['max_image_res'] = parsed_max_image_res
        if stopping_image_encode_format is not None and stopping_image_encode_format.strip():
            parsed_image_format = stopping_image_encode_format.strip().lower()
            if parsed_image_format not in {'png', 'jpeg', 'jpg'}:
                raise ValueError(
                    "VLMNAV_STOPPING_IMAGE_ENCODE_FORMAT must be png, jpeg, or jpg, "
                    f"got {stopping_image_encode_format!r}"
                )
            # Canonicalize the alias so runtime provenance has one stable value.
            stopping_kwargs['image_encode_format'] = (
                'jpeg' if parsed_image_format == 'jpg' else parsed_image_format
            )
        if stopping_jpeg_quality is not None and stopping_jpeg_quality.strip():
            try:
                parsed_jpeg_quality = int(stopping_jpeg_quality)
            except ValueError as exc:
                raise ValueError(
                    "VLMNAV_STOPPING_JPEG_QUALITY must be an integer in [1, 100], "
                    f"got {stopping_jpeg_quality!r}"
                ) from exc
            if not 1 <= parsed_jpeg_quality <= 100:
                raise ValueError(
                    "VLMNAV_STOPPING_JPEG_QUALITY must be an integer in [1, 100], "
                    f"got {stopping_jpeg_quality!r}"
                )
            effective_format = str(
                stopping_kwargs.get('image_encode_format', 'png')
            ).strip().lower()
            if effective_format not in {'jpeg', 'jpg'}:
                raise ValueError(
                    "VLMNAV_STOPPING_JPEG_QUALITY requires the stopping image "
                    "encode format to be jpeg"
                )
            stopping_kwargs['jpeg_quality'] = parsed_jpeg_quality
        if isinstance(vlm_cls, type) and issubclass(vlm_cls, OpenAIVLM):
            self._apply_openai_role_sampling_kwargs(stopping_kwargs, "stopping")
        self.stoppingVLM: VLM = vlm_cls(**stopping_kwargs)

    @classmethod
    def _action_token_to_int(cls, token):
        token = str(token).strip().strip("\"'`.,;:()[]{}")
        if token.lower().startswith('candidate_'):
            token = token.split('_', 1)[1]
        if token in cls.circled_digit_map:
            return int(cls.circled_digit_map[token])
        return int(token)

    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        # IMPORTANT:
        # Do not use `with ThreadPoolExecutor(...)` here.
        # On timeout, cancelled futures that are already running cannot be force-killed.
        # Exiting a `with` block calls shutdown(wait=True), which can still block forever
        # if a worker thread is hung in C/CUDA code. We must explicitly shutdown(wait=False)
        # to keep the main episode loop responsive.
        trace_timing = os.environ.get("VLMNAV_TRACE_STAGE_TIMING", "").lower() in ("1", "true", "yes")
        def _trace(message: str):
            if trace_timing:
                logging.info(message)

        preprocess_timeout_s = float(os.environ.get("VLMNAV_PREPROCESS_TIMEOUT", "120"))
        stopping_timeout_s = float(os.environ.get("VLMNAV_STOPPING_TIMEOUT", "60"))
        sync_stop_first = os.environ.get("VLMNAV_SYNC_STOP_BEFORE_PREPROCESS", "").lower() in ("1", "true", "yes")
        disable_stopping_module = os.environ.get("VLMNAV_DISABLE_STOPPING_MODULE", "").lower() in ("1", "true", "yes")
        try:
            helper_threads = int(os.environ.get("VLMNAV_HELPER_THREADS", "2"))
        except ValueError:
            helper_threads = 2
        # The normal path submits stop and preprocessing together, so fewer
        # than two workers would serialize or deadlock the helper stage.
        helper_threads = max(2, min(8, helper_threads))
        defer_stopping_until_action_stop = (
            isinstance(obs, dict)
            and bool(obs.get("defer_stop_verification_until_action_stop", False))
        )
        force_stop_on_timeout = False

        if disable_stopping_module:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=helper_threads)
            try:
                _trace(f"stage step={self.step_ndx} submit_preprocessing")
                preprocessing_thread = executor.submit(self._preprocessing_module, obs)
                _trace(f"stage step={self.step_ndx} wait_preprocessing")
                a_final, images = preprocessing_thread.result(timeout=preprocess_timeout_s)
                _trace(f"stage step={self.step_ndx} done_preprocessing")
            except concurrent.futures.TimeoutError:
                logging.error(
                    f"Preprocessing thread timeout after {preprocess_timeout_s}s "
                    f"(episode_step={self.step_ndx}, instance may be stuck). Falling back to default arrows."
                )
                preprocessing_thread.cancel()
                a_final = self._get_default_arrows() if hasattr(self, "_get_default_arrows") else []
                images = {"color_sensor": obs["color_sensor"].copy()}
                force_stop_on_timeout = True
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            called_stop, stopping_response = False, ""
        elif defer_stopping_until_action_stop:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=helper_threads)
            try:
                _trace(f"stage step={self.step_ndx} submit_preprocessing_deferred_stop")
                preprocessing_thread = executor.submit(self._preprocessing_module, obs)
                _trace(f"stage step={self.step_ndx} wait_preprocessing_deferred_stop")
                a_final, images = preprocessing_thread.result(timeout=preprocess_timeout_s)
                _trace(f"stage step={self.step_ndx} done_preprocessing_deferred_stop")
            except concurrent.futures.TimeoutError:
                logging.error(
                    f"Preprocessing thread timeout after {preprocess_timeout_s}s "
                    f"(episode_step={self.step_ndx}, deferred stop path). Falling back to default arrows."
                )
                preprocessing_thread.cancel()
                a_final = self._get_default_arrows() if hasattr(self, "_get_default_arrows") else []
                images = {"color_sensor": obs["color_sensor"].copy()}
                force_stop_on_timeout = True
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            called_stop, stopping_response = False, ""
        elif sync_stop_first:
            # In some scenes the local depth/navigability path can monopolize the
            # process before the OpenAI client actually sends its request.  For
            # long evals, make the stop request first so every step reaches vLLM
            # before expensive local geometry begins.
            try:
                _trace(f"stage step={self.step_ndx} submit_stopping")
                called_stop, stopping_response = self._stopping_module(stopping_images, goal)
                _trace(f"stage step={self.step_ndx} done_stopping")
            except Exception as e:
                logging.error(f"Stopping module failed before preprocessing: {e}")
                called_stop, stopping_response = False, ""
                force_stop_on_timeout = True

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=helper_threads)
            try:
                _trace(f"stage step={self.step_ndx} submit_preprocessing")
                preprocessing_thread = executor.submit(self._preprocessing_module, obs)
                _trace(f"stage step={self.step_ndx} wait_preprocessing")
                a_final, images = preprocessing_thread.result(timeout=preprocess_timeout_s)
                _trace(f"stage step={self.step_ndx} done_preprocessing")
            except concurrent.futures.TimeoutError:
                logging.error(
                    f"Preprocessing thread timeout after {preprocess_timeout_s}s "
                    f"(episode_step={self.step_ndx}, instance may be stuck). Falling back to default arrows."
                )
                preprocessing_thread.cancel()
                a_final = self._get_default_arrows() if hasattr(self, "_get_default_arrows") else []
                images = {"color_sensor": obs["color_sensor"].copy()}
                force_stop_on_timeout = True
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=helper_threads)
            # Submit stopping first so local navigation/preprocessing cannot monopolize
            # the worker process before the VLM stop request is even encoded/sent.
            _trace(f"stage step={self.step_ndx} submit_stopping")
            stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)
            _trace(f"stage step={self.step_ndx} submit_preprocessing")
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)

            try:
                _trace(f"stage step={self.step_ndx} wait_preprocessing")
                a_final, images = preprocessing_thread.result(timeout=preprocess_timeout_s)
                _trace(f"stage step={self.step_ndx} done_preprocessing")
            except concurrent.futures.TimeoutError:
                logging.error(
                    f"Preprocessing thread timeout after {preprocess_timeout_s}s "
                    f"(episode_step={self.step_ndx}, instance may be stuck). Falling back to default arrows."
                )
                preprocessing_thread.cancel()
                # Fallback: still return something valid for prompting/action selection.
                a_final = self._get_default_arrows() if hasattr(self, "_get_default_arrows") else []
                images = {"color_sensor": obs["color_sensor"].copy()}
                force_stop_on_timeout = True

            try:
                _trace(f"stage step={self.step_ndx} wait_stopping")
                called_stop, stopping_response = stopping_thread.result(timeout=stopping_timeout_s)
                _trace(f"stage step={self.step_ndx} done_stopping")
            except concurrent.futures.TimeoutError:
                logging.error(
                    f"Stopping thread timeout after {stopping_timeout_s}s "
                    f"(episode_step={self.step_ndx}). Proceeding without stopping signal."
                )
                stopping_thread.cancel()
                called_stop, stopping_response = False, ""
                force_stop_on_timeout = True
            finally:
                # Never block the control thread on potentially stuck worker futures.
                executor.shutdown(wait=False, cancel_futures=True)

        if force_stop_on_timeout:
            # A stalled preprocessing/VLM request is an infrastructure error,
            # not visual evidence that the goal has been reached.  Treating it
            # as a Stop used to turn timeouts (and occasionally budget-adjacent
            # stalls) into artificial successes.  Keep the historical behavior
            # behind an explicit legacy opt-in only; clean evaluation fails
            # closed and continues with the fallback navigation action.
            legacy_force_stop_on_timeout = os.environ.get(
                "VLMNAV_FORCE_STOP_ON_STAGE_TIMEOUT",
                "0",
            ).strip().lower() in ("1", "true", "yes", "y", "on")
            if legacy_force_stop_on_timeout:
                logging.warning(
                    "Legacy VLMNAV_FORCE_STOP_ON_STAGE_TIMEOUT is enabled; "
                    "forcing Stop after a stage timeout."
                )
                called_stop = True
            else:
                logging.error(
                    "Stage timeout occurred; refusing to synthesize a Stop signal."
                )
                called_stop = False

        stop_signal_reproject_default_arrows = os.environ.get(
            "VLMNAV_STOP_SIGNAL_REPROJECT_DEFAULT_ARROWS",
            "1",
        ).strip().lower() in ("1", "true", "yes", "y", "on")
        try:
            model_name = str(getattr(self.actionVLM, "name", "")).lower()
            reproject_min_step_default = "3" if "goalbalanced_stop_v13" in model_name else "0"
            stop_signal_reproject_min_step = int(os.environ.get(
                "VLMNAV_STOP_SIGNAL_REPROJECT_MIN_STEP",
                reproject_min_step_default,
            ))
        except ValueError:
            stop_signal_reproject_min_step = 0

        stop_signal_reproject_mode = 'none'
        stop_confirmation_action_cap = None
        blind_same_pose_crosscheck = self._blind_same_pose_stop_crosscheck_enabled()
        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            if (
                stop_signal_reproject_default_arrows
                and not blind_same_pose_crosscheck
                and self.step_ndx >= stop_signal_reproject_min_step
                and self.cfg['navigability_mode'] != 'none'
                and self.cfg['project']
            ):
                new_image = obs['color_sensor'].copy()
                confirmation_actions = self._get_stop_confirmation_arrows(list(a_final))
                a_final = self._project_onto_image(
                    confirmation_actions, new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor']
                )
                stop_signal_reproject_mode = 'short_public_candidates'
                stop_confirmation_action_cap = self._stop_confirmation_action_distance()
                self.last_candidate_action_scores = [
                    {
                        'index': idx,
                        'r': float(r),
                        'theta_deg_model_frame': float(np.degrees(theta)),
                        'frontier_like': False,
                        'exploration_score_raw': None,
                        'source': 'stop_confirmation_short_action',
                    }
                    for idx, (r, theta) in enumerate(a_final, start=1)
                ]
                images['color_sensor'] = new_image
            elif blind_same_pose_crosscheck:
                # Keep the ordinary public candidates and image so the action
                # policy receives no stop-conditioned visual cue.
                stop_signal_reproject_mode = 'blind_standard_candidates'

        stop_confidence = None
        parsed_stop = {}
        try:
            parsed_stop = self._eval_response(stopping_response) if stopping_response else {}
            if isinstance(parsed_stop, dict):
                for key in ('confidence', 'score', 'prob', 'probability'):
                    if key in parsed_stop:
                        stop_confidence = float(parsed_stop[key])
                        break
                if stop_confidence is None and 'stop' in parsed_stop:
                    value = parsed_stop.get('stop')
                    if isinstance(value, bool):
                        stop_confidence = 1.0 if value else 0.0
                    elif isinstance(value, (int, float)):
                        stop_confidence = 1.0 if int(value) == 1 else 0.0
                    elif isinstance(value, str):
                        stop_confidence = 1.0 if value.strip().lower() in {'true', '1', 'yes', 'stop'} else 0.0
                if stop_confidence is None and 'done' in parsed_stop:
                    stop_confidence = 1.0 if int(parsed_stop['done']) == 1 else 0.0
        except Exception:
            parsed_stop = {}
            stop_confidence = None

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'pivot': 1 if self.pivot is not None else 0,
            'model': self.actionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop,
            'stop_confidence': stop_confidence,
            'stop_decision_source': (
                'disabled'
                if disable_stopping_module
                else ('deferred_until_action_stop' if defer_stopping_until_action_stop else 'vlm_stopping')
            ),
            'stop_deferred_until_action_stop': bool(defer_stopping_until_action_stop),
            'stop_signal_reproject_default_arrows': bool(stop_signal_reproject_default_arrows),
            'stop_signal_reproject_min_step': int(stop_signal_reproject_min_step),
            'stop_signal_reproject_mode': stop_signal_reproject_mode,
            'stop_confirmation_action_cap': stop_confirmation_action_cap,
            'blind_same_pose_stop_crosscheck_enabled': bool(blind_same_pose_crosscheck),
            'helper_threads': int(helper_threads),
        }
        stop_logit_payload = getattr(self, '_last_stop_logit_payload', None)
        if isinstance(stop_logit_payload, dict):
            stop_scores = stop_logit_payload.get('scores') or {}
            step_metadata['stop_logit_threshold'] = stop_logit_payload.get('threshold')
            step_metadata['stop_logit_status'] = stop_logit_payload.get('status')
            step_metadata['stop_logit_valid'] = bool(stop_logit_payload.get('valid', False))
            step_metadata['stop_logit_score_0'] = stop_scores.get('0')
            step_metadata['stop_logit_score_1'] = stop_scores.get('1')
            step_metadata['stop_logit_margin'] = stop_logit_payload.get('margin_stop_minus_continue')
            step_metadata['stop_logit_generated_stop'] = stop_logit_payload.get('baseline_generated_stop')
            step_metadata['stop_logit_accepted_stop'] = stop_logit_payload.get('accepted_stop')
            step_metadata['stop_logit_fallback'] = bool(stop_logit_payload.get('fallback_to_generated', False))
            step_metadata['stop_logit_zero_agreement'] = stop_logit_payload.get('threshold_zero_agrees_with_generated')
            if stop_logit_payload.get('category_min_step_veto'):
                step_metadata['stop_logit_category_min_step_veto'] = True
                step_metadata['stop_logit_category_min_step'] = stop_logit_payload.get('category_min_step')
        step_metadata.update(self._motion_feedback_metadata())
        if isinstance(parsed_stop, dict):
            for key in (
                'ability',
                'spatial_reasoning',
                'stop',
                'target_visibility',
                'distance_measurement',
                'reason',
                'target_visible',
                'target_identity_clear',
                'distance_judgment',
                'blocking_or_gap',
                'stop_reason',
                'done',
            ):
                if key in parsed_stop:
                    step_metadata[f'stop_cot_{key}'] = parsed_stop.get(key)
        self._update_prompt_history_fields(obs['agent_state'], step_metadata)
        return a_final, images, step_metadata, stopping_response

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        obs = self._public_inference_observation(obs)
        self._ingest_online_motion_feedback(obs)
        trace_timing = os.environ.get("VLMNAV_TRACE_STAGE_TIMING", "").lower() in ("1", "true", "yes")
        stage_t0 = time.time()
        if trace_timing:
            logging.info(f"stage step={self.step_ndx} preprocessing_start")
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        if not self.cfg['project']:
            # Actions for the w/o proj baseline
            a_final = {
                (self.cfg['max_action_dist'], -0.28 * np.pi): 1,
                (self.cfg['max_action_dist'], 0): 2,
                (self.cfg['max_action_dist'], 0.28 * np.pi): 3,
            }
            return a_final, images

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            if trace_timing:
                logging.info(f"stage step={self.step_ndx} navigability_start")
            a_initial = self._navigability(obs)
            if trace_timing:
                logging.info(f"stage step={self.step_ndx} navigability_done n={len(a_initial)} dt={time.time() - stage_t0:.2f}s")
            if trace_timing:
                logging.info(f"stage step={self.step_ndx} action_proposer_start")
            a_final = self._action_proposer(a_initial, agent_state)
            if trace_timing:
                logging.info(f"stage step={self.step_ndx} action_proposer_done n={len(a_final)} dt={time.time() - stage_t0:.2f}s")

        a_final = self._permute_candidate_labels_for_audit(a_final, agent_state)

        if trace_timing:
            logging.info(f"stage step={self.step_ndx} projection_start")
        a_final_projected = self._projection(a_final, images, agent_state)
        if trace_timing:
            logging.info(f"stage step={self.step_ndx} projection_done n={len(a_final_projected)} dt={time.time() - stage_t0:.2f}s")
        if os.environ.get("VLMNAV_SKIP_VOXEL_MAP", "").lower() not in ("1", "true", "yes"):
            if trace_timing:
                logging.info(f"stage step={self.step_ndx} voxel_start")
            images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state)
            if trace_timing:
                logging.info(f"stage step={self.step_ndx} voxel_done dt={time.time() - stage_t0:.2f}s")
        return a_final_projected, images

    def _permute_candidate_labels_for_audit(self, actions: list, agent_state) -> list:
        """Randomize candidate IDs without changing the visible/executed geometry.

        Candidate IDs are normally assigned after sorting by bearing, which
        permits an ID-only directional shortcut.  This opt-in audit happens
        before projection, keeping the rendered label, parsed action, and
        executed candidate aligned.  It is disabled by default.
        """
        enabled = os.environ.get(
            "VLMNAV_AUDIT_PERMUTE_CANDIDATE_LABELS", ""
        ).strip().lower() in ("1", "true", "yes", "y", "on")
        if not enabled or len(actions) < 2:
            return actions

        try:
            seed = int(os.environ.get(
                "VLMNAV_AUDIT_CANDIDATE_PERMUTATION_SEED", "20260717"
            ))
        except ValueError:
            seed = 20260717
        position = getattr(agent_state, "position", (0.0, 0.0, 0.0))
        material = ":".join([
            str(seed),
            str(int(self.step_ndx)),
            *(f"{float(value):.3f}" for value in position),
            str(len(actions)),
        ]).encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        order = [int(index) for index in rng.permutation(len(actions))]

        score_by_pose = {}
        for item in getattr(self, "last_candidate_action_scores", []) or []:
            try:
                score_by_pose[(
                    round(float(item.get("r")), 3),
                    round(float(np.radians(item.get("theta_deg_model_frame"))), 4),
                )] = dict(item)
            except (TypeError, ValueError):
                continue
        permuted = [actions[index] for index in order]
        permuted_scores = []
        for new_index, action in enumerate(permuted, start=1):
            mag, theta = action[:2]
            item = dict(score_by_pose.get(
                (round(float(mag), 3), round(float(theta), 4)), {}
            ))
            item.update({
                "index": new_index,
                "r": float(mag),
                "theta_deg_model_frame": float(np.degrees(theta)),
                "candidate_label_permuted_for_audit": True,
                "candidate_label_original_index": int(order[new_index - 1] + 1),
            })
            permuted_scores.append(item)
        self.last_candidate_action_scores = permuted_scores
        return permuted

    @staticmethod
    def _normalize_goal_category(goal) -> str:
        if isinstance(goal, dict):
            goal = (
                goal.get("name")
                or goal.get("object_category")
                or goal.get("object_category_name")
                or ""
            )
        text = str(goal or "").strip().lower()
        return text.replace(" ", "_").replace("-", "_")

    def _stop_min_step_for_goal(self, goal):
        """Env ``VLMNAV_STOP_MIN_STEP_BY_GOAL=picture:4`` → min step before Stop."""
        raw = os.environ.get("VLMNAV_STOP_MIN_STEP_BY_GOAL", "").strip()
        if not raw:
            return None
        key = self._normalize_goal_category(goal)
        if not key:
            return None
        for part in raw.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            name, _, value = part.partition(":")
            name = self._normalize_goal_category(name)
            if name != key:
                continue
            try:
                return max(0, int(value.strip()))
            except ValueError:
                return None
        return None

    def _stop_logit_threshold_for_goal(self, goal):
        """Per-goal Stop margin threshold; falls back to ``VLMNAV_STOP_LOGIT_THRESHOLD``."""
        default = float(os.environ.get("VLMNAV_STOP_LOGIT_THRESHOLD", "0.0"))
        raw = os.environ.get("VLMNAV_STOP_LOGIT_THRESHOLD_BY_GOAL", "").strip()
        if not raw:
            return default
        key = self._normalize_goal_category(goal)
        if not key:
            return default
        for part in raw.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            name, _, value = part.partition(":")
            name = self._normalize_goal_category(name)
            if name != key:
                continue
            try:
                thr = float(value.strip())
            except ValueError:
                return default
            if np.isfinite(thr):
                return thr
            return default
        return default

    def _append_fn_category_stop_suffix(self, prompt: str, goal) -> str:
        """Encourage Stop on high-FN bathroom/small-object goals (env-gated)."""
        if not os.environ.get("VLMNAV_STOP_FN_PROMPT_ENABLE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        ):
            return prompt
        raw_goals = os.environ.get(
            "VLMNAV_STOP_FN_PROMPT_GOALS",
            "shower,towel,toilet,plant,chest_of_drawers,stool",
        )
        allowed = {
            self._normalize_goal_category(g)
            for g in raw_goals.split(",")
            if g.strip()
        }
        key = self._normalize_goal_category(goal)
        if key not in allowed:
            return prompt
        pretty = key.replace("_", " ")
        suffix = os.environ.get(
            "VLMNAV_STOP_FN_PROMPT_TEXT",
            (
                "\n\nCategory stop prior:\n"
                f"- If a real {pretty} is clearly visible and already nearby in the same room, "
                "output done=1 now.\n"
                "- Do not wait for perfect centering, a frontal view, or a nicer next frame.\n"
                "- Off-center, partially cropped, or beside other fixtures is still done=1 when close."
            ),
        )
        return str(prompt) + str(suffix)

    def _append_gym_move_prior(self, prompt: str, goal) -> str:
        """Light Move prior for gym_equipment (Oracle≈SR; Stop cannot recover)."""
        if not os.environ.get("VLMNAV_MOVE_GYM_PROMPT_ENABLE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        ):
            return prompt
        key = self._normalize_goal_category(goal)
        if key != "gym_equipment":
            return prompt
        suffix = os.environ.get(
            "VLMNAV_MOVE_GYM_PROMPT_TEXT",
            (
                "\n\nGym equipment prior:\n"
                "- Fitness machines are often in open rooms, basements, or side wings.\n"
                "- Prefer doorway/open-space exploration over hugging walls or early Stop.\n"
                "- If the machine is visible but still far, approach through free space first."
            ),
        )
        return str(prompt) + str(suffix)

    def _stopping_module(self, stopping_images: list[np.array], goal):
        """Determines if the agent should stop."""
        trace_timing = os.environ.get("VLMNAV_TRACE_STAGE_TIMING", "").lower() in ("1", "true", "yes")
        stage_t0 = time.time()
        if trace_timing:
            logging.info(f"stage step={self.step_ndx} stopping_prompt_start")
        stopping_prompt = self._construct_prompt(goal, 'stopping')
        stopping_prompt = self._append_fn_category_stop_suffix(stopping_prompt, goal)
        # The prompt is public model input.  Retain an exact per-call audit
        # record so a later train-only calibration set can prove that its SFT
        # surface matches deployment rather than reconstructing it from an
        # evolving set of environment flags.
        self._last_stopping_prompt = str(stopping_prompt)
        self._last_stopping_prompt_sha256 = hashlib.sha256(
            self._last_stopping_prompt.encode('utf-8')
        ).hexdigest()
        if trace_timing:
            logging.info(f"stage step={self.step_ndx} stopping_call_start dt={time.time() - stage_t0:.2f}s")
        stop_logit_threshold_enabled = os.environ.get(
            "VLMNAV_STOP_LOGIT_THRESHOLD_ENABLE", "0"
        ).strip().lower() in ("1", "true", "yes", "y", "on")
        self._last_stop_logit_payload = {
            "enabled": bool(stop_logit_threshold_enabled),
            "valid": False,
            "status": "disabled",
        }
        if stop_logit_threshold_enabled:
            if not hasattr(self.stoppingVLM, "call_stop_with_logprobs"):
                raise RuntimeError(
                    "Stop logit threshold requires call_stop_with_logprobs support"
                )
            stopping_response, stop_logit_payload = (
                self.stoppingVLM.call_stop_with_logprobs(
                    stopping_images, stopping_prompt
                )
            )
            self._last_stop_logit_payload = dict(stop_logit_payload or {})
            self._last_stop_logit_payload["enabled"] = True
        else:
            stopping_response = self.stoppingVLM.call(stopping_images, stopping_prompt)
        if trace_timing:
            logging.info(f"stage step={self.step_ndx} stopping_call_done dt={time.time() - stage_t0:.2f}s")
        if (
            self._terminal_certificate_evidence_required()
            and self._terminal_certificate_protocol_id()
            == 'objectnav_state_conditioned_causal_stop_bottleneck_v3'
        ):
            dct = parse_state_conditioned_causal_stop_bottleneck_response(stopping_response)
        else:
            dct = self._eval_response(stopping_response)
        # A strict terminal-certificate protocol is authoritative.  In
        # particular, do this before any legacy category/model compatibility
        # branch below: those branches may accept an action-policy proposal or
        # synthesize a legacy ``done`` response, neither of which is a valid V2
        # visual-arrival certificate.
        if self._terminal_certificate_evidence_required():
            if (
                self._terminal_certificate_protocol_id()
                == 'objectnav_visual_arrival_certificate_v2'
            ):
                dct = parse_visual_arrival_certificate_v2_runtime_response(
                    stopping_response
                )
            return self._terminal_certificate_response_allows_stop(dct), stopping_response

        def _structured_should_stop(parsed):
            if not isinstance(parsed, dict):
                return False
            if self._terminal_certificate_evidence_required():
                return self._terminal_certificate_response_allows_stop(parsed)
            if 'stop' in parsed:
                value = parsed.get('stop')
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return int(value) == 1
                if isinstance(value, str):
                    return value.strip().lower() in {'true', '1', 'yes', 'stop'}
            if 'done' in parsed:
                try:
                    return int(parsed['done']) == 1
                except Exception:
                    pass
            decision = parsed.get('decision')
            if not isinstance(decision, dict):
                return False
            action = str(decision.get('action') or '').strip().lower()
            mode = str(decision.get('mode') or '').strip().lower()
            should_stop = bool(decision.get('should_stop', False))
            return bool(should_stop or action in {'candidate_-1', '-1', 'stop'} or mode == 'stop')

        def _apply_stop_logit_threshold(baseline_stop):
            payload = self._last_stop_logit_payload
            payload["baseline_generated_stop"] = bool(baseline_stop)
            accepted_stop = bool(baseline_stop)
            if not stop_logit_threshold_enabled:
                payload["accepted_stop"] = bool(accepted_stop)
            else:
                try:
                    threshold = float(self._stop_logit_threshold_for_goal(goal))
                except ValueError as exc:
                    raise ValueError("stop logit threshold must be a float") from exc
                if not np.isfinite(threshold):
                    raise ValueError("stop logit threshold must be finite")
                payload["threshold"] = threshold
                margin = payload.get("margin_stop_minus_continue")
                if not payload.get("valid") or margin is None:
                    payload["fallback_to_generated"] = True
                    accepted_stop = bool(baseline_stop)
                else:
                    # A zero-margin tie is generated as done=0 by the baseline decoder.
                    # Use a strict boundary so threshold=0 is exactly behavior-preserving.
                    accepted_stop = float(margin) > threshold
                    payload["fallback_to_generated"] = False
                    payload["threshold_zero_agrees_with_generated"] = bool(
                        (float(margin) > 0.0) == bool(baseline_stop)
                    )
            # Category early-stop veto (e.g. picture:4): reject Stop before min step.
            if accepted_stop:
                min_step = self._stop_min_step_for_goal(goal)
                if min_step is not None and int(self.step_ndx) < int(min_step):
                    payload["category_min_step_veto"] = True
                    payload["category_min_step"] = int(min_step)
                    accepted_stop = False
            payload["accepted_stop"] = bool(accepted_stop)
            return bool(accepted_stop)

        model_name = str(getattr(self.actionVLM, "name", "")).lower()
        # Keep legacy behavior for all models except SenseNova-Qwen2.5.
        # NOTE: SenseNova-Qwen3 should follow legacy policy for fair comparison/perf.
        is_sensenova_qwen25 = ("sensenova" in model_name) and ("qwen2.5" in model_name)
        if not is_sensenova_qwen25:
            baseline_stop = _structured_should_stop(dct)
            return _apply_stop_logit_threshold(baseline_stop), stopping_response

        # SenseNova-Qwen2.5 compatibility path.
        min_stop_step = int(os.environ.get("VLMNAV_MIN_STOP_STEP", "2"))
        if self.step_ndx < min_stop_step:
            return False, stopping_response

        def _to_int(v):
            try:
                return int(v)
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return None

        # Expected schemas: {"done": 0/1} or structured ObjectNav CoT decision.
        if _structured_should_stop(dct):
            return _apply_stop_logit_threshold(True), stopping_response

        # Observed schema (ObjectNav w/ some VLMs): {"<goal_name>": "1", "distance": "0"}
        goal_key = goal if isinstance(goal, str) else (goal.get('name') if isinstance(goal, dict) else None)
        if goal_key and goal_key in dct:
            gv = _to_int(dct.get(goal_key))
            if gv == 1:
                return _apply_stop_logit_threshold(True), stopping_response
        
        return _apply_stop_logit_threshold(False), stopping_response

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        return a_initial

    def _action_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        """Refines the initial set of actions, ensuring spacing and adding a bias towards exploration."""
        min_angle = self.fov/self.cfg['spacing_ratio']
        explore_bias = self.cfg['explore_bias']
        clip_frac = self.cfg['clip_frac']
        clip_mag = self.cfg['max_action_dist']
        explore = explore_bias > 0
        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]
        arrowData = []

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        legacy_four_cell_score = os.getenv(
            'VLMNAV_LEGACY_FOUR_CELL_ACTION_PROPOSER_SCORE', '0'
        ).strip().lower() in {'1', 'true', 'yes', 'on'}
        historical_theta_set_bug = os.getenv(
            'VLMNAV_HISTORICAL_ACTION_PROPOSER_THETA_SET_BUG', '0'
        ).strip().lower() in {'1', 'true', 'yes', 'on'}
        for theta, mags in unique.items():
            # Reference the map to classify actions as explored or unexplored
            mag = min(mags)
            cart = [self.e_i_scaling*mag*np.sin(theta), 0, -self.e_i_scaling*mag*np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self._global_to_grid(global_coords)
            grid_x, grid_y = int(grid_coords[0]), int(grid_coords[1])
            if legacy_four_cell_score:
                # Pre-Selected3 legacy frontier classification.  Selected3 itself
                # already used the bounds-safe five-cell score below.
                score = (
                    int(np.all(
                        topdown_map[grid_y - 2:grid_y + 2, grid_x]
                        == self.explored_color,
                        axis=-1,
                    ).sum())
                    + int(np.all(
                        topdown_map[grid_y, grid_x - 2:grid_x + 2]
                        == self.explored_color,
                        axis=-1,
                    ).sum())
                )
            else:
                map_h, map_w = topdown_map.shape[:2]
                if 0 <= grid_x < map_w and 0 <= grid_y < map_h:
                    y0 = max(0, grid_y - 2)
                    y1 = min(map_h, grid_y + 3)
                    x0 = max(0, grid_x - 2)
                    x1 = min(map_w, grid_x + 3)
                    vertical = topdown_map[y0:y1, grid_x]
                    horizontal = topdown_map[grid_y, x0:x1]
                    score = (
                        int(np.all(vertical == self.explored_color, axis=-1).sum())
                        + int(np.all(horizontal == self.explored_color, axis=-1).sum())
                    )
                else:
                    # Candidate projected outside the finite local map. Keep it eligible
                    # instead of crashing the episode; downstream navigation can reject it.
                    score = 0
            arrowData.append([clip_frac*mag, theta, score < 3, int(score)])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        filtered.sort(key=lambda x: x[1])
        if filtered == []:
            return []
        if explore:
            # Add unexplored actions with spacing, starting with the longest one
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)
            
                out.append([min(longest[0], clip_mag), longest[1], longest[2], longest[3]])
                thetas.add(longest[1])
                for i in range(longest_ndx+1, len(f)):
                    if f[i][1] - longest_theta > (min_angle*0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2], f[i][3]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]
                for i in range(longest_ndx-1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle*0.9):
                        
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2], f[i][3]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                for r_i, theta_i, e_i, raw_score in filtered:
                    if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle*explore_bias:
                        out.append((min(r_i, clip_mag), theta_i, e_i, raw_score))
                        # Selected3's 2026-07-14 runtime accidentally inserted the
                        # stale loop variable ``theta`` here.  That bug retained a
                        # denser set of explored candidates and materially changes
                        # the model input, so reproduce it only for baseline replay.
                        thetas.add(theta if historical_theta_set_bug else theta_i)

        if len(out) == 0:
            # if no explored actions or no explore bias
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2], longest[3]])
            
            for i in range(longest_ndx+1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2], filtered[i][3]])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx-1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2], filtered[i][3]])
                    smallest_theta = filtered[i][1]


        out = self._filter_online_navigation_candidates(out, agent_state)

        if (out == [] or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            default_actions = self._get_default_arrows()
            self.last_candidate_action_scores = [
                {
                    'index': idx,
                    'r': float(mag),
                    'theta_deg_model_frame': float(np.degrees(theta)),
                    'frontier_like': False,
                    'exploration_score_raw': None,
                    'source': 'default_arrows',
                }
                for idx, (mag, theta) in enumerate(default_actions, start=1)
            ]
            return default_actions
        
        out.sort(key=lambda x: x[1])
        self.last_candidate_action_scores = [
            {
                'index': idx,
                'r': float(mag),
                'theta_deg_model_frame': float(np.degrees(theta)),
                'frontier_like': bool(frontier_like),
                'exploration_score_raw': raw_score,
                'source': 'action_proposer',
                **self._online_candidate_memory_by_pose.get(
                    (round(float(mag), 3), round(float(theta), 4)),
                    {},
                ),
            }
            for idx, (mag, theta, frontier_like, raw_score) in enumerate(out, start=1)
        ]
        return [(mag, theta) for mag, theta, _, _ in out]

    def _projection(self, a_final: list, images: dict, agent_state: habitat_sim.AgentState):
        """
        Projection component of VLMnav. Projects the arrows onto the image, annotating them with action numbers.
        Note actions that are too close together or too close to the boundaries of the image will not get projected.
        """
        a_final_projected = self._project_onto_image(
            a_final, images['color_sensor'], agent_state,
            agent_state.sensor_states['color_sensor']
        )

        if not a_final_projected and (self.step_ndx - self.turned < self.cfg['turn_around_cooldown']):
            logging.info('No actions projected and cannot turn around')
            a_final = self._get_default_arrows()
            a_final_projected = self._project_onto_image(
                a_final, images['color_sensor'], agent_state,
                agent_state.sensor_states['color_sensor']
            )

        score_by_pose = {}
        for item in self.last_candidate_action_scores:
            try:
                score_by_pose[(
                    round(float(item.get('r')), 3),
                    round(float(np.radians(item.get('theta_deg_model_frame'))), 4),
                )] = item
            except (TypeError, ValueError):
                continue
        aligned_scores = []
        for idx, action in enumerate(a_final_projected, start=1):
            r, theta = action
            item = dict(score_by_pose.get((round(float(r), 3), round(float(theta), 4)), {}))
            item.update({
                'index': idx,
                'r': float(r),
                'theta_deg_model_frame': float(np.degrees(theta)),
            })
            aligned_scores.append(item)
        self.last_candidate_action_scores = aligned_scores

        return a_final_projected

    def _candidate_prompt_features(self, a_final: list) -> list[str]:
        lines: list[str] = []
        compact_features = os.environ.get(
            "VLMNAV_COT_COMPACT_CANDIDATE_FEATURES",
            "",
        ).strip().lower() in ("1", "true", "yes", "y", "on")
        score_by_index = {}
        score_by_pose = {}
        for item in getattr(self, "last_candidate_action_scores", []) or []:
            try:
                score_by_index[int(item.get("index"))] = item
            except Exception:
                pass
            try:
                pose_key = (
                    round(float(item.get("r")), 2),
                    round(float(np.radians(float(item.get("theta_deg_model_frame")))), 3),
                )
                score_by_pose[pose_key] = item
            except Exception:
                pass

        for idx, action in enumerate(a_final, start=1):
            try:
                r, theta = action[:2]
                distance_m = float(r)
                heading_deg = float(np.degrees(theta))
            except Exception:
                distance_m = 0.0
                heading_deg = 0.0
            turn_direction = "left" if heading_deg < -10 else "right" if heading_deg > 10 else "forward"
            score_info = score_by_index.get(idx)
            if score_info is None:
                score_info = score_by_pose.get((round(distance_m, 2), round(float(theta), 3)), {})
            frontier_like = bool(score_info.get("frontier_like", False))
            source = str(score_info.get("source") or "projected_arrow")
            raw_score = score_info.get("exploration_score_raw")
            online_visit_count = int(score_info.get("online_visit_count") or 0)
            online_blocked_count = int(score_info.get("online_blocked_count") or 0)
            role = "frontier_transition_option" if frontier_like else "local_movement_option"
            if abs(heading_deg) <= 12 and distance_m >= 1.0:
                role = "direct_route_option" if not frontier_like else "frontier_transition_option"
            try:
                explored_overlap = int(raw_score) if raw_score is not None else None
            except Exception:
                explored_overlap = None
            reliable_forward = (
                distance_m >= float(os.environ.get("VLMNAV_WMNAV_LITE_RELIABLE_FORWARD_MIN_DIST", "0.90"))
                and source != "default_arrows"
            )
            route_score = 0.0
            route_score += min(distance_m, 3.0) * 0.30
            route_score += 0.45 if abs(heading_deg) <= 18 else 0.0
            route_score += 0.65 if frontier_like else 0.0
            if explored_overlap is not None:
                route_score -= max(0, explored_overlap) * 0.05
            if source == "default_arrows":
                route_score -= 0.25
            route_score -= min(online_visit_count, 4) * 0.15
            route_score -= min(online_blocked_count, 4) * 0.75
            route_role_hint = (
                "prefer_when_target_visible" if role == "direct_route_option"
                else "prefer_for_hidden_target_search" if role == "frontier_transition_option"
                else "neutral_local_motion"
            )
            if compact_features:
                lines.append(
                    f"- candidate_{idx}: executable action {idx}; visible red-arrow label {idx}; "
                    f"distance_m={distance_m:.2f}; heading_deg={heading_deg:.1f}; "
                    f"turn_direction={turn_direction}; frontier_like={str(frontier_like).lower()}; "
                    f"reliable_forward={str(reliable_forward).lower()}; "
                    f"visited_count={online_visit_count}; blocked_count={online_blocked_count}"
                )
                continue
            raw_score_text = "" if raw_score is None else f"; explored_overlap_score={raw_score}"
            lines.append(
                f"- candidate_{idx}: executable action {idx}; visible red-arrow label {idx}; "
                f"distance_m={distance_m:.2f}; heading_deg={heading_deg:.1f}; turn_direction={turn_direction}; "
                f"route_role={role}; route_role_hint={route_role_hint}; "
                f"frontier_like={str(frontier_like).lower()}; reliable_forward={str(reliable_forward).lower()}; "
                f"visited_count={online_visit_count}; blocked_count={online_blocked_count}; "
                f"route_prior_score={route_score:.2f}; source={source}{raw_score_text}"
            )
        return lines

    def _wmnav_lite_best_forward_candidate(self, a_final: list):
        """Pick a non-GT forward fallback when the model wants candidate_0.

        This uses only online candidate geometry/exploration metadata already
        produced by VLMNav.  It is intentionally simple so the behavior can be
        mirrored in SFT/RFT data as prompt features instead of a WMNav-style
        extra VLM planning module.
        """
        try:
            min_dist = float(os.environ.get("VLMNAV_WMNAV_LITE_RELIABLE_FORWARD_MIN_DIST", "0.90"))
        except ValueError:
            min_dist = 0.90
        score_by_index = {}
        for item in getattr(self, "last_candidate_action_scores", []) or []:
            try:
                score_by_index[int(item.get("index"))] = item
            except Exception:
                continue

        best = None
        best_score = None
        for idx, action in enumerate(a_final, start=1):
            try:
                distance_m = float(action[0])
                theta = float(action[1])
                heading_abs = abs(float(np.degrees(theta)))
            except Exception:
                continue
            if distance_m < min_dist:
                continue
            score_info = score_by_index.get(idx, {})
            source = str(score_info.get("source") or "")
            if source == "default_arrows":
                continue
            frontier_like = bool(score_info.get("frontier_like", False))
            raw_score = score_info.get("exploration_score_raw")
            try:
                explored_overlap = float(raw_score) if raw_score is not None else 3.0
            except Exception:
                explored_overlap = 3.0
            score = 0.0
            score += min(distance_m, 3.0) * 0.30
            score += 0.55 if heading_abs <= 18 else 0.0
            score += 0.70 if frontier_like else 0.0
            score -= max(0.0, explored_overlap) * 0.05
            # Stable tie-break toward lower heading change, then lower label.
            key = (score, -heading_abs * 0.001, -idx * 0.0001)
            if best_score is None or key > best_score:
                best_score = key
                best = {
                    "candidate": idx,
                    "distance_m": distance_m,
                    "heading_deg": float(np.degrees(theta)),
                    "frontier_like": frontier_like,
                    "explored_overlap_score": explored_overlap,
                    "route_prior_score": score,
                    "source": source or "projected_arrow",
                }
        return best

    def _update_prompt_history_fields(self, agent_state: habitat_sim.AgentState, step_metadata: dict) -> None:
        """Track coarse online revisit state for optional prompt-only loop recovery."""
        try:
            position = np.asarray(agent_state.position, dtype=float)
            key = tuple(np.round(position[[0, 2]], 1).tolist())
        except Exception:
            step_metadata.update({
                "prompt_position_key": None,
                "prompt_visit_count_current_cell": None,
                "prompt_revisit_count": 0,
                "prompt_loop_risk": False,
            })
            return
        count = int(self.prompt_position_counts.get(key, 0)) + 1
        self.prompt_position_counts[key] = count
        step_metadata.update({
            "prompt_position_key": list(key),
            "prompt_visit_count_current_cell": count,
            "prompt_revisit_count": max(count - 1, 0),
            "prompt_loop_risk": bool(count >= 4),
        })

    def _prompt_history_cue(self, step_metadata: dict) -> str:
        enabled = os.environ.get("VLMNAV_WMNAV_LITE_HISTORY_CUE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        if not enabled:
            return ""
        try:
            min_revisit = int(os.environ.get("VLMNAV_WMNAV_LITE_HISTORY_MIN_REVISIT", "1"))
        except ValueError:
            min_revisit = 1
        min_revisit = max(1, min_revisit)
        revisit_count = int(step_metadata.get("prompt_revisit_count") or 0)
        loop_risk = bool(step_metadata.get("prompt_loop_risk"))
        if revisit_count < min_revisit and not loop_risk:
            return ""
        return (
            "Online navigation history cue:\n"
            f"- current_cell_revisit_count: {revisit_count}\n"
            f"- cue_min_revisit_count: {min_revisit}\n"
            f"- loop_risk: {'true' if loop_risk else 'false'}\n"
            "- If the target is not clearly confirmed close, avoid actions that keep circling in this already visited cell. "
            "Prefer a valid candidate that makes progress into new free space, a doorway, hallway, room transition, or frontier. "
            "Use candidate_0 only if reorientation is needed for recovery; do not stop merely because of loop risk.\n"
        )

    def _prompt_action_streak_cue(self, step_metadata: dict) -> str:
        enabled = os.environ.get("VLMNAV_WMNAV_LITE_ACTION_STREAK_CUE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        if not enabled:
            return ""
        try:
            min_streak = int(os.environ.get("VLMNAV_WMNAV_LITE_ACTION_STREAK_MIN", "8"))
        except ValueError:
            min_streak = 8
        min_streak = max(2, min_streak)
        history = []
        for action in getattr(self, "prompt_action_history", []) or []:
            try:
                action_number = int(action)
            except Exception:
                continue
            if action_number >= 0:
                history.append(action_number)
        if not history:
            step_metadata.update({
                "prompt_repeated_action": None,
                "prompt_repeated_action_streak": 0,
                "prompt_action_streak_cue_min": int(min_streak),
            })
            return ""
        last_action = history[-1]
        streak = 0
        for action in reversed(history):
            if action != last_action:
                break
            streak += 1
        step_metadata.update({
            "prompt_repeated_action": int(last_action),
            "prompt_repeated_action_streak": int(streak),
            "prompt_action_streak_cue_min": int(min_streak),
        })
        if streak < min_streak:
            return ""
        return (
            "Online repeated-action cue:\n"
            f"- previous_action: candidate_{last_action}\n"
            f"- consecutive_repeat_count: {streak}\n"
            f"- cue_min_repeat_count: {min_streak}\n"
            "- If the target is not clearly confirmed close, do not keep repeating the same candidate by inertia. "
            "Compare other valid red-arrow candidates for new free space, doorway/hallway/room transition, or recovery. "
            "Repeat the same candidate only if it is still visibly the safest and most goal-directed route.\n"
        )

    def _prompt_memory_cue(self, step_metadata: dict) -> str:
        cues = [
            self._prompt_history_cue(step_metadata),
            self._prompt_action_streak_cue(step_metadata),
        ]
        pano_cue = self._prompt_lightweight_panoramic_cue(step_metadata)
        if pano_cue:
            cues.append(pano_cue)
        return "\n".join(cue for cue in cues if cue)

    def _prompt_lightweight_panoramic_cue(self, step_metadata: dict) -> str:
        if not bool(step_metadata.get("lightweight_panoramic_enabled", False)):
            return ""
        if not bool(step_metadata.get("lightweight_panoramic_success", False)):
            return ""
        if os.environ.get(
            "VLMNAV_LIGHTWEIGHT_PANORAMIC_ACTION_CUE",
            "1",
        ).strip().lower() not in ("1", "true", "yes", "y", "on"):
            return ""
        angle = step_metadata.get("lightweight_panoramic_selected_angle")
        score = step_metadata.get("lightweight_panoramic_selected_score")
        reason = str(step_metadata.get("lightweight_panoramic_selected_reason") or "").strip()
        depth_score = step_metadata.get("lightweight_panoramic_selected_depth_score")
        global_state = step_metadata.get("lightweight_panoramic_global_state")
        view_direction = step_metadata.get("lightweight_panoramic_view_direction_deg")
        lock_remaining = step_metadata.get("lightweight_panoramic_lock_steps_remaining")
        context_only = bool(step_metadata.get("lightweight_panoramic_context_only", False))
        parts = [
            "Global navigation state:",
            f"- panoramic_state: {global_state}",
            f"- current_view_direction_deg: {view_direction}",
            f"- direction_lock_steps_remaining: {lock_remaining}",
            f"- panorama_selected_relative_angle_deg: {angle}",
        ]
        if score is not None:
            parts.append(f"- selected_direction_score: {score}")
        if depth_score is not None:
            parts.append(f"- selected_direction_open_space_score: {depth_score}")
        if reason:
            parts.append(f"- selected_direction_reason: {reason}")
        if context_only:
            parts.extend([
                "- The panorama selector is only a weak direction hint from a scan; the agent is still facing the current action image.",
                "- Use the hint only to break recovery ambiguity. Choose among the visible red-arrow candidates in the current image.",
                "- Do not stop unless the target is immediate foreground/within reach.",
            ])
        else:
            parts.extend([
                "- The panorama selector only sets the current facing direction; it does not choose the final local action.",
                "- While direction_lock_steps_remaining is positive, avoid reinterpreting the same scene as a new global turn unless the visible red-arrow action is blocked or recovery is needed.",
                "- Choose among the visible red-arrow candidates using the current image; do not stop unless the target is immediate foreground/within reach.",
            ])
        return "\n".join(parts)

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict):
        """
        Prompting component of VLMNav. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        removed_private_metadata = []
        for key in self.private_inference_fields:
            if key in step_metadata:
                step_metadata.pop(key, None)
                removed_private_metadata.append(key)
        step_metadata['inference_private_metadata_removed'] = sorted(
            removed_private_metadata
        )
        prompt_type = 'action' if self.cfg['project'] else 'no_project'
        use_candidate_features = os.environ.get(
            'VLMNAV_COT_CANDIDATE_FEATURES',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        candidate_features = self._candidate_prompt_features(a_final) if use_candidate_features else None
        memory = self._prompt_memory_cue(step_metadata)
        temporal_reference_enabled = os.environ.get(
            'VLMNAV_TEMPORAL_TARGET_REFERENCE', ''
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        temporal_reference = (
            getattr(self, '_temporal_target_reference_image', None)
            if temporal_reference_enabled else None
        )
        if temporal_reference is None:
            prompt_images = [images['color_sensor']]
        else:
            prompt_images = [temporal_reference, images['color_sensor']]
        step_metadata['temporal_target_reference_enabled'] = bool(temporal_reference_enabled)
        step_metadata['temporal_target_reference_available'] = bool(temporal_reference is not None)
        step_metadata['temporal_target_reference_input_images'] = len(prompt_images)
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        aux_spatial_state_response = ""
        aux_spatial_state_enabled = os.environ.get(
            'VLMNAV_AUX_STATE_BEFORE_ACTION',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        if aux_spatial_state_enabled:
            aux_protocol_id = os.environ.get(
                'VLMNAV_AUX_STATE_PROTOCOL_ID',
                'objectnav_aux_spatial_state_v3',
            ).strip()
            aux_prompt = self._construct_prompt(
                goal,
                prompt_type,
                num_actions=len(a_final),
                candidate_features=candidate_features,
                memory=memory,
                protocol_id_override=aux_protocol_id,
            )
            aux_spatial_state_response = self.actionVLM.call_chat(
                self.cfg['context_history'],
                prompt_images,
                aux_prompt,
            )
            # The auxiliary response is passed explicitly to the action prompt.
            # Rewind keeps the two-pass helper from silently becoming an extra
            # chat-history conditioning channel.
            try:
                self.actionVLM.rewind()
            except Exception:
                logging.warning("Could not rewind action VLM after auxiliary spatial-state call.", exc_info=True)
            step_metadata['aux_spatial_state_protocol_id'] = aux_protocol_id
            step_metadata['aux_spatial_state_response'] = aux_spatial_state_response
        action_prompt = self._construct_prompt(
            goal,
            prompt_type,
            num_actions=len(a_final),
            candidate_features=candidate_features,
            memory=memory,
            spatial_state=aux_spatial_state_response,
            temporal_reference_available=temporal_reference is not None,
        )
        
        response = self.actionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

        logging_data = {}
        original_response = response
        repaired_contract = False
        repair_enabled = os.environ.get('VLMNAV_ACTION_CONTRACT_REPAIR', '').strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        validation_repair_enabled = os.environ.get(
            'VLMNAV_ACTION_VALIDATION_REPAIR',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = self._action_token_to_int(response_dict['action'])
            decision = response_dict.get('decision') if isinstance(response_dict, dict) else None
            if isinstance(response_dict, dict):
                # Newer state-routed protocols emit navigation_state directly.
                # Treat it as the authoritative mode when the legacy duplicate
                # navigation_mode field is absent, so Stop checks fail closed.
                nav_mode = response_navigation_mode(response_dict)
                stop_state = response_dict.get('stop_state')
                comparison = response_dict.get('candidate_comparison')
                identity_evidence = response_dict.get('identity_evidence')
                proximity_evidence = response_dict.get('proximity_evidence')
                candidate_evidence = response_dict.get('candidate_evidence')
                temporal_relation = response_dict.get('temporal_relation')
                step_metadata['action_navigation_mode'] = str(nav_mode) if nav_mode is not None else None
                step_metadata['action_stop_state'] = str(stop_state) if stop_state is not None else None
                step_metadata['action_candidate_comparison'] = str(comparison) if comparison is not None else None
                step_metadata['action_identity_evidence'] = str(identity_evidence) if identity_evidence is not None else None
                step_metadata['action_proximity_evidence'] = str(proximity_evidence) if proximity_evidence is not None else None
                step_metadata['action_candidate_evidence'] = candidate_evidence if isinstance(candidate_evidence, dict) else None
                step_metadata['action_temporal_relation'] = str(temporal_relation) if temporal_relation is not None else None
            else:
                step_metadata['action_navigation_mode'] = None
                step_metadata['action_stop_state'] = None
                step_metadata['action_candidate_comparison'] = None
                step_metadata['action_identity_evidence'] = None
                step_metadata['action_proximity_evidence'] = None
                step_metadata['action_candidate_evidence'] = None
                step_metadata['action_temporal_relation'] = None
            if isinstance(decision, dict):
                decision_action = decision.get('action')
                step_metadata['action_decision_action'] = str(decision_action) if decision_action is not None else None
                step_metadata['action_decision_mode'] = str(decision.get('mode')) if decision.get('mode') is not None else None
                step_metadata['action_should_stop'] = bool(decision.get('should_stop', False))
            else:
                step_metadata['action_decision_action'] = None
                step_metadata['action_decision_mode'] = None
                step_metadata['action_should_stop'] = step_metadata.get('action_number') == -1
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            if repair_enabled:
                step_metadata['action_number'] = 0
                step_metadata['success'] = 1
                step_metadata['action_contract_repair_reason'] = 'parse_failure_default_turnaround'
                repaired_contract = True
            else:
                step_metadata['success'] = 0
            step_metadata['action_should_stop'] = False

        # Build a target reference only after stable, protocol-valid public
        # confirmations.  A single hallucinated identity (or a malformed
        # no-reference relation) must never become evidence on the next step.
        relation = step_metadata.get('action_temporal_relation')
        identity = str(step_metadata.get('action_identity_evidence') or '').strip().lower()
        temporal_contract_valid = temporal_reference_contract_valid(
            relation,
            temporal_reference is not None,
        )
        step_metadata['temporal_reference_contract_format_valid'] = bool(
            temporal_contract_valid
        )
        step_metadata['temporal_target_reference_candidate_staged'] = False
        step_metadata['temporal_target_reference_promoted'] = False
        step_metadata['temporal_target_reference_cleared_reason'] = ''
        if temporal_reference_enabled and temporal_reference is not None:
            self._temporal_target_reference_age += 1
            if temporal_contract_valid and str(relation or '').strip().lower() == 'target_mismatch':
                self._temporal_target_reference_mismatch_streak += 1
            else:
                self._temporal_target_reference_mismatch_streak = 0
            if identity == 'unconfirmed':
                self._temporal_target_reference_unconfirmed_streak += 1
            else:
                self._temporal_target_reference_unconfirmed_streak = 0
            try:
                mismatch_limit = max(1, int(os.environ.get(
                    'VLMNAV_TEMPORAL_REFERENCE_MISMATCH_CLEAR_STREAK', '2'
                )))
            except ValueError:
                mismatch_limit = 2
            try:
                unconfirmed_limit = max(1, int(os.environ.get(
                    'VLMNAV_TEMPORAL_REFERENCE_UNCONFIRMED_CLEAR_STREAK', '4'
                )))
            except ValueError:
                unconfirmed_limit = 4
            clear_reason = ''
            if self._temporal_target_reference_mismatch_streak >= mismatch_limit:
                clear_reason = 'stable_target_mismatch'
            elif self._temporal_target_reference_unconfirmed_streak >= unconfirmed_limit:
                clear_reason = 'stale_unconfirmed_target'
            if clear_reason:
                self._temporal_target_reference_image = None
                self._temporal_target_reference_candidate_image = None
                self._temporal_target_reference_confirmation_streak = 0
                self._temporal_target_reference_mismatch_streak = 0
                self._temporal_target_reference_unconfirmed_streak = 0
                self._temporal_target_reference_age = 0
                step_metadata['temporal_target_reference_cleared_reason'] = clear_reason
        elif temporal_reference_enabled:
            eligible = temporal_reference_confirmation_eligible(
                navigation_state=step_metadata.get('action_navigation_mode'),
                identity_evidence=identity,
                proximity_evidence=step_metadata.get('action_proximity_evidence'),
                temporal_relation=relation,
                reference_available=False,
            )
            try:
                confirmation_min = max(1, int(os.environ.get(
                    'VLMNAV_TEMPORAL_REFERENCE_CONFIRMATION_STREAK', '2'
                )))
            except ValueError:
                confirmation_min = 2
            if eligible:
                self._temporal_target_reference_candidate_image = images['color_sensor'].copy()
                self._temporal_target_reference_confirmation_streak += 1
                step_metadata['temporal_target_reference_candidate_staged'] = True
                if self._temporal_target_reference_confirmation_streak >= confirmation_min:
                    self._temporal_target_reference_image = images['color_sensor'].copy()
                    self._temporal_target_reference_candidate_image = None
                    self._temporal_target_reference_confirmation_streak = 0
                    self._temporal_target_reference_mismatch_streak = 0
                    self._temporal_target_reference_unconfirmed_streak = 0
                    self._temporal_target_reference_age = 0
                    step_metadata['temporal_target_reference_promoted'] = True
            else:
                self._temporal_target_reference_candidate_image = None
                self._temporal_target_reference_confirmation_streak = 0
        step_metadata['temporal_target_reference_updated'] = bool(
            step_metadata['temporal_target_reference_promoted']
        )
        step_metadata['temporal_target_reference_confirmation_streak'] = int(
            self._temporal_target_reference_confirmation_streak
        )
        step_metadata['temporal_target_reference_mismatch_streak'] = int(
            self._temporal_target_reference_mismatch_streak
        )
        step_metadata['temporal_target_reference_unconfirmed_streak'] = int(
            self._temporal_target_reference_unconfirmed_streak
        )
        step_metadata['temporal_target_reference_age'] = int(
            self._temporal_target_reference_age
        )

        guided_shadow = os.environ.get(
            'VLMNAV_ACTION_GUIDED_CHOICE_SHADOW',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        guided_execute = os.environ.get(
            'VLMNAV_ACTION_GUIDED_CHOICE_EXECUTE',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        if guided_shadow or guided_execute:
            valid_action_numbers = sorted({-1, 0, *range(1, len(a_final) + 1)})
            valid_choice_labels = [f"candidate_{num}" for num in valid_action_numbers]
            guided_prompt = self._guided_action_choice_prompt(action_prompt, valid_choice_labels)
            guided_response = ""
            guided_number = None
            guided_valid = False
            try:
                if hasattr(self.actionVLM, 'call_guided_choice'):
                    guided_response = self.actionVLM.call_guided_choice(
                        prompt_images,
                        guided_prompt,
                        valid_choice_labels,
                        history=self.cfg['context_history'],
                    )
                else:
                    guided_response = self.actionVLM.call(prompt_images, guided_prompt)
                guided_number = self._parse_guided_action_choice(guided_response, valid_action_numbers)
                guided_valid = guided_number in valid_action_numbers
            except Exception as exc:  # noqa: BLE001
                logging.warning("Guided action choice failed.", exc_info=True)
                guided_response = f"GUIDED_CHOICE_ERROR: {exc!r}"
            step_metadata['action_guided_choice_enabled'] = True
            step_metadata['action_guided_choice_mode'] = 'execute' if guided_execute else 'shadow'
            step_metadata['action_guided_choice_response'] = guided_response
            step_metadata['action_guided_choice_action_number'] = guided_number
            step_metadata['action_guided_choice_action'] = (
                f"candidate_{guided_number}" if guided_number is not None else None
            )
            step_metadata['action_guided_choice_valid'] = bool(guided_valid)
            step_metadata['action_guided_choice_original_action_number'] = step_metadata.get('action_number')
            if guided_execute and guided_valid:
                step_metadata['action_guided_choice_applied'] = True
                step_metadata['action_number'] = int(guided_number)
                step_metadata['action_decision_action'] = f"candidate_{int(guided_number)}"
                step_metadata['action_decision_mode'] = (
                    'stop' if int(guided_number) == -1 else step_metadata.get('action_decision_mode')
                )
                step_metadata['action_should_stop'] = int(guided_number) == -1
                response = (
                    '{"decision":{"action":"candidate_'
                    + str(int(guided_number))
                    + '","should_stop":'
                    + ('true' if int(guided_number) == -1 else 'false')
                    + '}}'
                )
                response, appended = self._ensure_action_contract_line(response, int(guided_number), force=True)
                repaired_contract = repaired_contract or appended
            else:
                step_metadata['action_guided_choice_applied'] = False
        label_scorer_shadow = os.environ.get(
            'VLMNAV_ACTION_LABEL_SCORER_SHADOW',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        label_scorer_execute = os.environ.get(
            'VLMNAV_ACTION_LABEL_SCORER_EXECUTE',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        if label_scorer_shadow or label_scorer_execute:
            valid_action_numbers = sorted({-1, 0, *range(1, len(a_final) + 1)})
            valid_choice_labels = [f"candidate_{num}" for num in valid_action_numbers]
            scorer_payload = {
                "status": "unavailable",
                "error": "actionVLM has no score_action_labels method",
                "valid": False,
            }
            try:
                if hasattr(self.actionVLM, 'score_action_labels'):
                    scorer_payload = self.actionVLM.score_action_labels(
                        prompt_images,
                        action_prompt,
                        valid_choice_labels,
                        history=self.cfg['context_history'],
                    )
            except Exception as exc:  # noqa: BLE001
                logging.warning("Action label scorer failed.", exc_info=True)
                scorer_payload = {
                    "status": "error",
                    "error": repr(exc),
                    "valid": False,
                }
            scorer_number = scorer_payload.get("action_number") if isinstance(scorer_payload, dict) else None
            try:
                scorer_number = int(scorer_number)
            except (TypeError, ValueError):
                scorer_number = None
            scorer_valid = scorer_number in set(valid_action_numbers)
            step_metadata['action_label_scorer_enabled'] = True
            step_metadata['action_label_scorer_mode'] = 'execute' if label_scorer_execute else 'shadow'
            step_metadata['action_label_scorer_status'] = (
                scorer_payload.get("status") if isinstance(scorer_payload, dict) else "invalid_payload"
            )
            step_metadata['action_label_scorer_action_number'] = scorer_number
            step_metadata['action_label_scorer_action'] = (
                f"candidate_{scorer_number}" if scorer_number is not None else None
            )
            step_metadata['action_label_scorer_valid'] = bool(scorer_valid)
            step_metadata['action_label_scorer_applied'] = False
            step_metadata['action_label_scorer_original_action_number'] = step_metadata.get('action_number')
            if isinstance(scorer_payload, dict):
                step_metadata['action_label_scorer_score'] = scorer_payload.get("score")
                step_metadata['action_label_scorer_score_margin'] = scorer_payload.get("score_margin")
                step_metadata['action_label_scorer_scores'] = scorer_payload.get("scores")
                step_metadata['action_label_scorer_missing_labels'] = scorer_payload.get("missing_labels")
                step_metadata['action_label_scorer_generated'] = scorer_payload.get("generated")
                step_metadata['action_label_scorer_top_logprobs'] = scorer_payload.get("top_logprobs")
                step_metadata['action_label_scorer_error'] = scorer_payload.get("error")
            try:
                scorer_execute_min_margin = float(os.environ.get(
                    'VLMNAV_ACTION_LABEL_SCORER_EXECUTE_MIN_MARGIN',
                    '0',
                ))
            except ValueError:
                scorer_execute_min_margin = 0.0
            scorer_execute_min_margin = max(0.0, scorer_execute_min_margin)
            try:
                scorer_margin_value = float(step_metadata.get('action_label_scorer_score_margin'))
            except (TypeError, ValueError):
                scorer_margin_value = float('nan')
            scorer_margin_passed = (
                scorer_execute_min_margin <= 0
                or (
                    not math.isnan(scorer_margin_value)
                    and scorer_margin_value >= scorer_execute_min_margin
                )
            )
            step_metadata['action_label_scorer_execute_min_margin'] = float(
                scorer_execute_min_margin
            )
            step_metadata['action_label_scorer_margin_passed'] = bool(
                scorer_margin_passed
            )
            preserve_stop = os.environ.get(
                'VLMNAV_ACTION_LABEL_SCORER_PRESERVE_STOP',
                '1',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            try:
                original_action_number = int(step_metadata.get('action_number'))
            except (TypeError, ValueError):
                original_action_number = None
            preserved_stop = bool(
                preserve_stop
                and (
                    original_action_number == -1
                    or step_metadata.get('action_should_stop')
                    or step_metadata.get('stop_gate_passed')
                )
            )
            if preserved_stop:
                hybrid_number = -1
            elif scorer_valid and scorer_margin_passed:
                hybrid_number = int(scorer_number)
            else:
                hybrid_number = original_action_number
            step_metadata['action_label_scorer_preserve_stop'] = bool(preserve_stop)
            step_metadata['action_label_scorer_preserved_stop'] = bool(preserved_stop)
            step_metadata['action_label_scorer_hybrid_action_number'] = hybrid_number
            step_metadata['action_label_scorer_hybrid_action'] = (
                f"candidate_{hybrid_number}" if hybrid_number is not None else None
            )
            if (
                label_scorer_execute
                and scorer_valid
                and scorer_margin_passed
                and not preserved_stop
            ):
                step_metadata['action_label_scorer_applied'] = True
                step_metadata['action_number'] = int(scorer_number)
                step_metadata['action_decision_action'] = f"candidate_{int(scorer_number)}"
                step_metadata['action_decision_mode'] = (
                    'stop' if int(scorer_number) == -1 else step_metadata.get('action_decision_mode')
                )
                step_metadata['action_should_stop'] = int(scorer_number) == -1
                response = (
                    '{"decision":{"action":"candidate_'
                    + str(int(scorer_number))
                    + '","should_stop":'
                    + ('true' if int(scorer_number) == -1 else 'false')
                    + '}}'
                )
                response, appended = self._ensure_action_contract_line(response, int(scorer_number), force=True)
                repaired_contract = repaired_contract or appended
            elif label_scorer_execute and preserved_stop:
                step_metadata['action_label_scorer_applied'] = False
                step_metadata['action_label_scorer_execute_skip_reason'] = 'preserve_stop'
            elif label_scorer_execute and scorer_valid and not scorer_margin_passed:
                step_metadata['action_label_scorer_applied'] = False
                step_metadata['action_label_scorer_execute_skip_reason'] = 'low_margin'
        stop_binary_shadow = os.environ.get(
            'VLMNAV_ACTION_STOP_BINARY_SCORER_SHADOW',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        stop_binary_execute = os.environ.get(
            'VLMNAV_ACTION_STOP_BINARY_SCORER_EXECUTE',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        if stop_binary_shadow or stop_binary_execute:
            stop_payload = {
                "status": "unavailable",
                "error": "actionVLM has no score_action_stop_binary method",
                "valid": False,
            }
            try:
                if hasattr(self.actionVLM, 'score_action_stop_binary'):
                    stop_payload = self.actionVLM.score_action_stop_binary(
                        prompt_images,
                        action_prompt,
                        history=self.cfg['context_history'],
                    )
            except Exception as exc:  # noqa: BLE001
                logging.warning("Action stop-binary scorer failed.", exc_info=True)
                stop_payload = {
                    "status": "error",
                    "error": repr(exc),
                    "valid": False,
                }
            stop_label = stop_payload.get("stop_label") if isinstance(stop_payload, dict) else None
            try:
                stop_label = int(stop_label)
            except (TypeError, ValueError):
                stop_label = None
            stop_binary_valid = stop_label in (0, 1) and bool(
                stop_payload.get("valid") if isinstance(stop_payload, dict) else False
            )
            step_metadata['action_stop_binary_scorer_enabled'] = True
            step_metadata['action_stop_binary_scorer_mode'] = 'execute' if stop_binary_execute else 'shadow'
            step_metadata['action_stop_binary_scorer_status'] = (
                stop_payload.get("status") if isinstance(stop_payload, dict) else "invalid_payload"
            )
            step_metadata['action_stop_binary_scorer_label'] = stop_label
            step_metadata['action_stop_binary_scorer_valid'] = bool(stop_binary_valid)
            step_metadata['action_stop_binary_scorer_predicted_stop'] = bool(
                stop_binary_valid and stop_label == 1
            )
            step_metadata['action_stop_binary_scorer_applied'] = False
            if isinstance(stop_payload, dict):
                step_metadata['action_stop_binary_scorer_score'] = stop_payload.get("score")
                step_metadata['action_stop_binary_scorer_score_margin'] = stop_payload.get("score_margin")
                step_metadata['action_stop_binary_scorer_scores'] = stop_payload.get("scores")
                step_metadata['action_stop_binary_scorer_missing_labels'] = stop_payload.get("missing_labels")
                step_metadata['action_stop_binary_scorer_generated'] = stop_payload.get("generated")
                step_metadata['action_stop_binary_scorer_top_logprobs'] = stop_payload.get("top_logprobs")
                step_metadata['action_stop_binary_scorer_error'] = stop_payload.get("error")
            try:
                stop_binary_min_margin = float(os.environ.get(
                    'VLMNAV_ACTION_STOP_BINARY_SCORER_EXECUTE_MIN_MARGIN',
                    '0',
                ))
            except ValueError:
                stop_binary_min_margin = 0.0
            stop_binary_min_margin = max(0.0, stop_binary_min_margin)
            try:
                stop_binary_margin = float(step_metadata.get('action_stop_binary_scorer_score_margin'))
            except (TypeError, ValueError):
                stop_binary_margin = float('nan')
            stop_binary_margin_passed = (
                stop_binary_min_margin <= 0
                or (
                    not math.isnan(stop_binary_margin)
                    and stop_binary_margin >= stop_binary_min_margin
                )
            )
            step_metadata['action_stop_binary_scorer_execute_min_margin'] = float(
                stop_binary_min_margin
            )
            step_metadata['action_stop_binary_scorer_margin_passed'] = bool(
                stop_binary_margin_passed
            )
            stop_binary_runtime_wants_stop = bool(
                stop_binary_execute
                and stop_binary_valid
                and stop_label == 1
                and stop_binary_margin_passed
            )
            step_metadata['action_stop_binary_wants_stop'] = bool(stop_binary_runtime_wants_stop)
            if stop_binary_runtime_wants_stop:
                step_metadata['action_stop_binary_scorer_applied'] = True
            elif stop_binary_execute and stop_binary_valid and stop_label == 1 and not stop_binary_margin_passed:
                step_metadata['action_stop_binary_scorer_execute_skip_reason'] = 'low_margin'
            elif stop_binary_execute and not stop_binary_valid:
                step_metadata['action_stop_binary_scorer_execute_skip_reason'] = 'invalid'
        if (
            validation_repair_enabled
            and step_metadata.get('action_number') not in (None, -10)
        ):
            valid_actions = {-1, 0, *range(1, len(a_final) + 1)}
            try:
                parsed_action_number = int(step_metadata.get('action_number'))
            except (TypeError, ValueError):
                parsed_action_number = None
            if parsed_action_number not in valid_actions:
                step_metadata['action_invalid_original_number'] = parsed_action_number
                step_metadata['action_invalid_valid_max'] = len(a_final)
                step_metadata['action_contract_repair_reason'] = 'invalid_action_default_turnaround'
                step_metadata['action_number'] = 0
                step_metadata['action_should_stop'] = False
                repaired_contract = True
        if repair_enabled and step_metadata.get('action_number') not in (None, -10):
            response, appended = self._ensure_action_contract_line(
                response,
                step_metadata['action_number'],
                force=bool(step_metadata.get('action_invalid_original_number') is not None),
            )
            repaired_contract = repaired_contract or appended
            if repaired_contract:
                step_metadata['action_contract_repaired'] = True
        candidate0_veto = os.environ.get(
            'VLMNAV_WMNAV_LITE_CANDIDATE0_VETO',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        veto_require_history = os.environ.get(
            'VLMNAV_WMNAV_LITE_CANDIDATE0_VETO_REQUIRE_HISTORY',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        veto_history_ready = True
        if veto_require_history:
            try:
                veto_min_revisit = int(os.environ.get("VLMNAV_WMNAV_LITE_HISTORY_MIN_REVISIT", "1"))
            except ValueError:
                veto_min_revisit = 1
            veto_min_revisit = max(1, veto_min_revisit)
            revisit_count = int(step_metadata.get("prompt_revisit_count") or 0)
            loop_risk = bool(step_metadata.get("prompt_loop_risk"))
            veto_history_ready = bool(loop_risk or revisit_count >= veto_min_revisit)
            step_metadata['wmnav_lite_candidate0_veto_require_history'] = True
            step_metadata['wmnav_lite_candidate0_veto_history_ready'] = veto_history_ready
            step_metadata['wmnav_lite_candidate0_veto_history_min_revisit'] = veto_min_revisit
        else:
            step_metadata['wmnav_lite_candidate0_veto_require_history'] = False
            step_metadata['wmnav_lite_candidate0_veto_history_ready'] = True
        if candidate0_veto and veto_history_ready and step_metadata.get('action_number') == 0:
            replacement = self._wmnav_lite_best_forward_candidate(a_final)
            step_metadata['wmnav_lite_candidate0_veto_enabled'] = True
            step_metadata['wmnav_lite_candidate0_veto_replacement'] = replacement
            if replacement is not None:
                step_metadata['wmnav_lite_candidate0_veto_applied'] = True
                step_metadata['wmnav_lite_original_action_number'] = 0
                step_metadata['action_number'] = int(replacement['candidate'])
                step_metadata['action_should_stop'] = False
                response, appended = self._ensure_action_contract_line(
                    response,
                    step_metadata['action_number'],
                    force=True,
                )
                repaired_contract = repaired_contract or appended
            else:
                step_metadata['wmnav_lite_candidate0_veto_applied'] = False
        else:
            step_metadata['wmnav_lite_candidate0_veto_enabled'] = bool(candidate0_veto)
            step_metadata['wmnav_lite_candidate0_veto_applied'] = False
        loop_guard_enabled = os.environ.get(
            'VLMNAV_RUNTIME_LOOP_GUARD',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        try:
            loop_guard_min_streak = int(os.environ.get(
                'VLMNAV_RUNTIME_LOOP_GUARD_MIN_STREAK',
                '8',
            ))
        except ValueError:
            loop_guard_min_streak = 8
        try:
            loop_guard_min_revisit = int(os.environ.get(
                'VLMNAV_RUNTIME_LOOP_GUARD_MIN_REVISIT',
                '4',
            ))
        except ValueError:
            loop_guard_min_revisit = 4
        try:
            loop_guard_min_step = int(os.environ.get(
                'VLMNAV_RUNTIME_LOOP_GUARD_MIN_STEP',
                '16',
            ))
        except ValueError:
            loop_guard_min_step = 16
        try:
            loop_guard_fallback_action = int(os.environ.get(
                'VLMNAV_RUNTIME_LOOP_GUARD_FALLBACK_ACTION',
                '0',
            ))
        except ValueError:
            loop_guard_fallback_action = 0
        loop_guard_min_streak = max(2, loop_guard_min_streak)
        loop_guard_min_revisit = max(0, loop_guard_min_revisit)
        loop_guard_min_step = max(0, loop_guard_min_step)
        if loop_guard_fallback_action < 0 or loop_guard_fallback_action > len(a_final):
            loop_guard_fallback_action = 0
        try:
            loop_guard_current_action = int(step_metadata.get('action_number'))
        except (TypeError, ValueError):
            loop_guard_current_action = None
        loop_guard_streak = 0
        if loop_guard_current_action is not None and loop_guard_current_action >= 0:
            for history_action in reversed(getattr(self, 'prompt_action_history', []) or []):
                try:
                    history_action = int(history_action)
                except (TypeError, ValueError):
                    continue
                if history_action != loop_guard_current_action:
                    break
                loop_guard_streak += 1
        try:
            loop_guard_revisit = int(step_metadata.get('prompt_revisit_count') or 0)
        except (TypeError, ValueError):
            loop_guard_revisit = 0
        loop_guard_execution_failed = bool(
            step_metadata.get('previous_execution_underexecuted', False)
        )
        loop_guard_public_history_ready = bool(
            loop_guard_revisit >= loop_guard_min_revisit
            or loop_guard_execution_failed
        )
        loop_guard_stop_active = bool(
            step_metadata.get('action_should_stop')
            or step_metadata.get('called_stopping')
            or loop_guard_current_action == -1
        )
        loop_guard_ready = bool(
            loop_guard_enabled
            and loop_guard_current_action is not None
            and loop_guard_current_action > 0
            and loop_guard_fallback_action != loop_guard_current_action
            and self.step_ndx >= loop_guard_min_step
            and (loop_guard_streak + 1) >= loop_guard_min_streak
            and loop_guard_public_history_ready
            and not loop_guard_stop_active
        )
        step_metadata['runtime_loop_guard_enabled'] = bool(loop_guard_enabled)
        step_metadata['runtime_loop_guard_applied'] = False
        step_metadata['runtime_loop_guard_current_action_number'] = loop_guard_current_action
        step_metadata['runtime_loop_guard_previous_streak'] = int(loop_guard_streak)
        step_metadata['runtime_loop_guard_candidate_streak'] = int(loop_guard_streak + 1)
        step_metadata['runtime_loop_guard_min_streak'] = int(loop_guard_min_streak)
        step_metadata['runtime_loop_guard_revisit_count'] = int(loop_guard_revisit)
        step_metadata['runtime_loop_guard_min_revisit'] = int(loop_guard_min_revisit)
        step_metadata['runtime_loop_guard_step'] = int(self.step_ndx)
        step_metadata['runtime_loop_guard_min_step'] = int(loop_guard_min_step)
        step_metadata['runtime_loop_guard_target_relative_distance'] = None
        step_metadata['runtime_loop_guard_min_target_relative_distance'] = None
        step_metadata['runtime_loop_guard_target_relative_distance_ready'] = False
        step_metadata['runtime_loop_guard_public_history_ready'] = bool(
            loop_guard_public_history_ready
        )
        step_metadata['runtime_loop_guard_execution_failed'] = bool(
            loop_guard_execution_failed
        )
        step_metadata['runtime_loop_guard_stop_active'] = bool(loop_guard_stop_active)
        step_metadata['runtime_loop_guard_fallback_action_number'] = int(
            loop_guard_fallback_action
        )
        step_metadata['runtime_loop_guard_ready'] = bool(loop_guard_ready)
        if loop_guard_ready:
            step_metadata['runtime_loop_guard_applied'] = True
            step_metadata['runtime_loop_guard_original_action_number'] = loop_guard_current_action
            step_metadata['action_number'] = int(loop_guard_fallback_action)
            step_metadata['action_decision_action'] = f"candidate_{int(loop_guard_fallback_action)}"
            step_metadata['action_decision_mode'] = 'runtime_loop_recovery'
            step_metadata['action_should_stop'] = False
            response, appended = self._ensure_action_contract_line(
                response,
                step_metadata['action_number'],
                force=True,
            )
            repaired_contract = repaired_contract or appended
        logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
        logging_data['PROMPT'] = action_prompt
        if guided_shadow or guided_execute:
            logging_data['GUIDED_CHOICE_ORIGINAL_RESPONSE'] = original_response
            logging_data['GUIDED_CHOICE_RESPONSE'] = step_metadata.get('action_guided_choice_response')
        if label_scorer_shadow or label_scorer_execute:
            logging_data['LABEL_SCORER_ORIGINAL_RESPONSE'] = original_response
            logging_data['LABEL_SCORER_RESPONSE'] = json.dumps({
                "status": step_metadata.get('action_label_scorer_status'),
                "action": step_metadata.get('action_label_scorer_action'),
                "action_number": step_metadata.get('action_label_scorer_action_number'),
                "valid": step_metadata.get('action_label_scorer_valid'),
                "score": step_metadata.get('action_label_scorer_score'),
                "score_margin": step_metadata.get('action_label_scorer_score_margin'),
                "scores": step_metadata.get('action_label_scorer_scores'),
                "missing_labels": step_metadata.get('action_label_scorer_missing_labels'),
                "generated": step_metadata.get('action_label_scorer_generated'),
                "hybrid_action": step_metadata.get('action_label_scorer_hybrid_action'),
                "hybrid_action_number": step_metadata.get('action_label_scorer_hybrid_action_number'),
                "preserved_stop": step_metadata.get('action_label_scorer_preserved_stop'),
                "error": step_metadata.get('action_label_scorer_error'),
            }, ensure_ascii=False, sort_keys=True)
        if stop_binary_shadow or stop_binary_execute:
            logging_data['ACTION_STOP_BINARY_SCORER_RESPONSE'] = json.dumps({
                "status": step_metadata.get('action_stop_binary_scorer_status'),
                "label": step_metadata.get('action_stop_binary_scorer_label'),
                "valid": step_metadata.get('action_stop_binary_scorer_valid'),
                "predicted_stop": step_metadata.get('action_stop_binary_scorer_predicted_stop'),
                "runtime_wants_stop": step_metadata.get('action_stop_binary_wants_stop'),
                "score": step_metadata.get('action_stop_binary_scorer_score'),
                "score_margin": step_metadata.get('action_stop_binary_scorer_score_margin'),
                "scores": step_metadata.get('action_stop_binary_scorer_scores'),
                "missing_labels": step_metadata.get('action_stop_binary_scorer_missing_labels'),
                "generated": step_metadata.get('action_stop_binary_scorer_generated'),
                "margin_passed": step_metadata.get('action_stop_binary_scorer_margin_passed'),
                "error": step_metadata.get('action_stop_binary_scorer_error'),
            }, ensure_ascii=False, sort_keys=True)
        logging_data['RESPONSE'] = response

        return step_metadata, logging_data, response

    def _guided_action_choice_prompt(self, action_prompt: str, valid_choice_labels: list[str]) -> str:
        return (
            action_prompt.rstrip()
            + "\n\nConstrained action-choice pass:\n"
            "- Ignore the JSON response format above for this pass.\n"
            "- Choose the single best next action label from the valid list.\n"
            "- Candidate numbers are identifiers, not preferences.\n"
            "- Return exactly one label and no other text.\n"
            f"Valid labels: {', '.join(valid_choice_labels)}\n"
        )

    def _parse_guided_action_choice(self, response: str, valid_action_numbers: list[int]):
        text = str(response or "").strip()
        valid_set = {int(x) for x in valid_action_numbers}
        matches = re.findall(r"\bcandidate_(-?\d+)\b", text)
        if matches:
            try:
                action = int(matches[-1])
                if action in valid_set:
                    return action
            except ValueError:
                pass
        try:
            parsed = self._eval_response(text)
            action = parsed.get('action') if isinstance(parsed, dict) else None
            if action is not None:
                action = int(action)
                if action in valid_set:
                    return action
        except Exception:
            pass
        nums = re.findall(r"(?<![\w-])-?\d+(?![\w-])", text)
        if len(nums) == 1:
            try:
                action = int(nums[0])
                if action in valid_set:
                    return action
            except ValueError:
                pass
        return None

    def _ensure_action_contract_line(self, response: str, action_number: int, force: bool = False):
        """
        Ensure logs contain the executable action dict required by the original
        VLMNav policy contract. This is intentionally opt-in from the launcher.
        """
        response = response or ''
        import re

        if re.search(r"(?m)^\s*\{\s*['\"]action['\"]\s*:\s*-?\d+\s*\}\s*$", response):
            return response, False
        try:
            action_number = int(action_number)
        except (TypeError, ValueError):
            action_number = 0
        action_line_re = r"(?m)^\s*\{\s*['\"]action['\"]\s*:\s*-?\d+\s*\}\s*$"
        replacement = f"{{'action': {action_number}}}"
        if re.search(action_line_re, response):
            if force:
                return re.sub(action_line_re, replacement, response).rstrip(), True
            return response, False
        return response.rstrip() + f"\n\n{replacement}", True

    def _get_navigability_mask(self, rgb_image: np.array, depth_image: np.array, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Get the navigability mask for the current state, according to the configured navigability mode.
        """
        if self.cfg['navigability_mode'] == 'segmentation':
            navigability_mask = self.segmentor.get_navigability_mask(rgb_image)
        else:
            base_thresh = 1 if self.cfg['navigability_mode'] == 'depth_estimate' else self.cfg['navigability_height_threshold']
            if hasattr(self, "multifloor") and self.multifloor is not None:
                thresh = self.multifloor.navigability_height_threshold(base_thresh)
            else:
                thresh = base_thresh
            historical_mask = os.environ.get(
                'VLMNAV_HISTORICAL_RAW_DEPTH_NAV_MASK',
                '0',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            if historical_mask:
                # Exact pre-sanitization path used by the 2026-07-14
                # Selected3 baseline. This compatibility branch is isolated
                # because later NaN/Inf handling changes some candidate sets.
                height_map = depth_to_height(
                    depth_image,
                    self.fov,
                    sensor_state.position,
                    sensor_state.rotation,
                )
                navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh
                self._last_navigability_mask = navigability_mask
                return navigability_mask
            # Some episodes/sensor drops can produce NaNs/Infs in depth. That can poison downstream
            # geometry (quaternion transforms) and cause hangs or extremely slow computations.
            if depth_image is None:
                navigability_mask = np.ones(rgb_image.shape[:2], dtype=bool)
                self._last_navigability_mask = navigability_mask
                return navigability_mask

            depth = np.asarray(depth_image, dtype=np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_clip = float(os.environ.get("VLMNAV_DEPTH_CLIP", "20.0"))
            depth = np.clip(depth, 0.0, depth_clip)

            downsample = max(1, int(os.environ.get("VLMNAV_NAV_MASK_DOWNSAMPLE", "1")))
            if downsample > 1:
                h, w = depth.shape[:2]
                small_w = max(1, w // downsample)
                small_h = max(1, h // downsample)
                depth_small = cv2.resize(depth, (small_w, small_h), interpolation=cv2.INTER_AREA)
                height_map = depth_to_height(depth_small, self.fov, sensor_state.position, sensor_state.rotation)
                mask_small = abs(height_map - (agent_state.position[1] - 0.04)) < thresh
                navigability_mask = cv2.resize(
                    mask_small.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                height_map = depth_to_height(depth, self.fov, sensor_state.position, sensor_state.rotation)
                navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh

        self._last_navigability_mask = navigability_mask
        return navigability_mask

    def _get_default_arrows(self):
        """
        Get fallback action options when no navigable actions are found.
        """
        angle = np.deg2rad(self.fov / 2) * 0.7
        
        default_actions = [
            (self.cfg['stopping_action_dist'], -angle),
            (self.cfg['stopping_action_dist'], -angle / 4),
            (self.cfg['stopping_action_dist'], angle / 4),
            (self.cfg['stopping_action_dist'], angle)
        ]
        
        default_actions.sort(key=lambda x: x[1])
        return default_actions

    def _stop_confirmation_action_distance(self) -> float:
        try:
            value = float(os.environ.get(
                'VLMNAV_STOP_CONFIRMATION_ACTION_DIST',
                str(self.cfg.get('stop_confirmation_action_dist', 0.5)),
            ))
        except (TypeError, ValueError):
            value = 0.5
        return max(0.1, min(0.75, value))

    def _same_pose_stop_crosscheck_enabled(self) -> bool:
        return os.environ.get(
            'VLMNAV_SAME_POSE_STOP_ACTION_CROSSCHECK',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _blind_same_pose_stop_crosscheck_enabled(self) -> bool:
        return os.environ.get(
            'VLMNAV_BLIND_SAME_POSE_STOP_ACTION_CROSSCHECK',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _blind_consecutive_stop_fallback_enabled(self) -> bool:
        return os.environ.get(
            'VLMNAV_BLIND_CONSECUTIVE_STOP_FALLBACK',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _consecutive_stop_fallback_enabled(self) -> bool:
        return os.environ.get(
            'VLMNAV_CONSECUTIVE_STOP_FALLBACK',
            '1',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _strict_dual_model_stop_enabled(self) -> bool:
        return os.environ.get(
            'VLMNAV_STRICT_DUAL_MODEL_STOP',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _action_first_stop_confirmation_enabled(self) -> bool:
        return os.environ.get(
            'VLMNAV_ACTION_FIRST_STOP_CONFIRMATION',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _action_first_requires_final_stop_action(self) -> bool:
        """Require the executable action field, rather than auxiliary text, to propose Stop.

        This is deliberately opt-in: older action-first experiments treated a
        structured ``should_stop`` field as a proposal even when the final
        executable action was a move.  A clean action-preserving verifier needs
        the narrower semantics, but changing the default would alter historical
        configurations.
        """
        value = os.environ.get(
            'VLMNAV_ACTION_FIRST_REQUIRE_FINAL_ACTION_STOP',
            os.environ.get(
                'VLMNAV_ACTION_FIRST_REQUIRE_FINAL_STOP_ACTION',
                '0',
            ),
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        return value

    def _action_first_append_crosscheck_prompt(self) -> bool:
        """Whether action-first also changes the action model's prompt surface.

        Kept enabled by default for backwards compatibility.  New clean
        verifier ablations disable it so the action policy is bit-for-bit on its
        normal prompt path and only the downstream Stop decision is gated.
        """
        return os.environ.get(
            'VLMNAV_ACTION_FIRST_STOP_APPEND_CROSSCHECK_PROMPT',
            '1',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _terminal_certificate_every_step_enabled(self) -> bool:
        """Run one independent visual terminal certificate per decision state."""
        return os.environ.get(
            'VLMNAV_TERMINAL_CERTIFICATE_EVERY_STEP',
            '0',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _terminal_certificate_prompt_audit_enabled(self) -> bool:
        """Persist the exact public certificate prompt during train-only collection."""
        return os.environ.get(
            'VLMNAV_TERMINAL_CERTIFICATE_PROMPT_AUDIT',
            '0',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _terminal_certificate_prompt_enabled(self) -> bool:
        return os.environ.get(
            'VLMNAV_TERMINAL_CERTIFICATE_PROMPT',
            '0',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _terminal_certificate_evidence_required(self) -> bool:
        return os.environ.get(
            'VLMNAV_REQUIRE_TERMINAL_CERTIFICATE_EVIDENCE',
            '0',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _terminal_certificate_protocol_id(self) -> str:
        """Return the explicit terminal-certificate response contract.

        Preserve V1 as the default for historical runs.  New visual-arrival
        work must opt into V2 by name, which makes an accidental schema mix
        fail closed rather than silently interpreting an old ``done`` field.
        """
        return os.environ.get(
            'VLMNAV_TERMINAL_CERTIFICATE_PROTOCOL',
            'objectnav_terminal_visual_certificate_v1',
        ).strip()

    def _terminal_certificate_protocol_supported(self) -> bool:
        return self._terminal_certificate_protocol_id() in {
            'objectnav_terminal_visual_certificate_v1',
            'objectnav_visual_arrival_certificate_v2',
            'objectnav_state_conditioned_spatial_stop_v1',
            'objectnav_state_conditioned_evidence_stop_v2',
            'objectnav_state_conditioned_causal_stop_bottleneck_v3',
        }

    def _terminal_certificate_response_allows_stop(self, response) -> bool:
        protocol = self._terminal_certificate_protocol_id()
        if protocol == 'objectnav_visual_arrival_certificate_v2':
            return visual_arrival_certificate_v2_allows_stop(response)
        if protocol == 'objectnav_state_conditioned_spatial_stop_v1':
            return state_conditioned_spatial_stop_allows_stop(response)
        if protocol == 'objectnav_state_conditioned_evidence_stop_v2':
            return state_conditioned_evidence_stop_allows_stop(response)
        if protocol == 'objectnav_state_conditioned_causal_stop_bottleneck_v3':
            return state_conditioned_causal_stop_bottleneck_allows_stop(response)
        if protocol == 'objectnav_terminal_visual_certificate_v1':
            return terminal_certificate_allows_stop(response)
        return False

    def _current_stop_signal(self) -> bool:
        return bool(
            self.stopping_calls
            and self.stopping_calls[-1] == self.step_ndx
        )

    def _get_stop_confirmation_arrows(self, current_actions: list):
        """Keep public visual headings but cap the first-stop approach distance."""
        return cap_stop_confirmation_actions(
            current_actions,
            cap=self._stop_confirmation_action_distance(),
            fov_deg=self.fov,
        )

    def _get_radial_distance(self, start_pxl: tuple, theta_i: float, navigability_mask: np.ndarray, 
                             agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, 
                             depth_image: np.ndarray):
        """
        Calculates the distance r_i that the agent can move in the direction theta_i, according to the navigability mask.
        """
        agent_point = [2 * np.sin(theta_i), 0, -2 * np.cos(theta_i)]
        end_pxl = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_pxl is None or end_pxl[1] >= self.resolution[0]:
            return None, None

        H, W = navigability_mask.shape

        # Find intersections of the theoretical line with the image boundaries
        intersections = find_intersections(start_pxl[0], start_pxl[1], end_pxl[0], end_pxl[1], W, H)
        if intersections is None:
            return None, None

        (x1, y1), (x2, y2) = intersections
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)

        out = (int(x_coords[-1]), int(y_coords[-1]))
        if not navigability_mask[int(y_coords[0]), int(x_coords[0])]:
            return 0, theta_i

        for i in range(num_points - 4):
            # Trace pixels until they are not navigable
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)]) <= 2:
                out = (x, y)
                break

        if i < 5:
            return 0, theta_i

        if self.cfg['navigability_mode'] == 'segmentation':
            #Simple estimation of distance based on number of pixels
            r_i = 0.0794 * np.exp(0.006590 * i) + 0.616

        else:
            #use depth to get distance
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            camera_coords = unproject_2d(
                *out, depth_image[out[1], out[0]], resolution=self.resolution, focal_length=self.focal_length
            )
            local_coords = global_to_local(
                agent_state.position, agent_state.rotation,
                local_to_global(sensor_state.position, sensor_state.rotation, camera_coords)
            )
            r_i = np.linalg.norm([local_coords[0], local_coords[2]])

        return r_i, theta_i

    def _can_project(self, r_i: float, theta_i: float, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Checks whether the specified polar action can be projected onto the image, i.e., not too close to the boundaries of the image.
        """
        agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]
        end_px = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_px is None:
            return None

        if (
            self.cfg['image_edge_threshold'] * self.resolution[1] <= end_px[0] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[1] and
            self.cfg['image_edge_threshold'] * self.resolution[0] <= end_px[1] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[0]
        ):
            return end_px
        return None

    def _project_onto_image(self, a_final: list, rgb_image: np.ndarray, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, chosen_action: int=None):
        """
        Projects a set of actions onto a single image. Keeps track of action-to-number mapping.
        """
        scale_factor = rgb_image.shape[0] / 1080
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        circle_color = WHITE
        projected = {}
        if chosen_action == -1:
            put_text_on_image(
                rgb_image, 'TERMINATING EPISODE', text_color=GREEN, text_size=4 * scale_factor,
                location='center', text_thickness=math.ceil(3 * scale_factor), highlight=False
            )
            return projected

        start_px = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        for _, (r_i, theta_i) in enumerate(a_final):
            text_size = 2.4 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)

            end_px = self._can_project(r_i, theta_i, agent_state, sensor_state)
            if end_px is not None:
                action_name = len(projected) + 1
                projected[(r_i, theta_i)] = action_name

                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), RED, math.ceil(5 * scale_factor), tipLength=0.0)
                text = str(action_name)
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

                if chosen_action is not None and action_name == chosen_action:
                    cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
                else:
                    cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
                cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

        if (self.step_ndx - self.turned) >= self.cfg['turn_around_cooldown'] or self.step_ndx == self.turned or (chosen_action == 0):
            text = '0'
            text_size = 3.1 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

            if chosen_action is not None and chosen_action == 0:
                cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
            cv2.putText(rgb_image, 'TURN AROUND', (text_position[0] // 2, text_position[1] + math.ceil(80 * scale_factor)), font, text_size * 0.75, RED, text_thickness)

        return projected


    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        skip_explored = (
            hasattr(self, "multifloor")
            and self.multifloor is not None
            and self.multifloor.should_skip_explored_update()
        )
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r - 0.5, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        if not skip_explored:
            cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)

        # Mark explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        if not skip_explored:
            cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _multifloor_blank_maps(self):
        voxel = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        explored = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        return voxel, explored

    def _apply_multifloor_step(
        self,
        obs: dict,
        a_final: list,
        agent_action,
        step_metadata: dict,
    ):
        """Lightweight climb latch: nav thresh, Stop suppress, weak bias, map copy."""
        if not hasattr(self, "multifloor") or self.multifloor is None:
            self.multifloor = MultiFloorController()
        if not self.multifloor.enabled and not multifloor_enabled():
            return agent_action, step_metadata
        self.multifloor.enabled = multifloor_enabled()
        if not self.multifloor.enabled:
            return agent_action, step_metadata

        agent_state = obs.get("agent_state")
        depth_image = obs.get("depth_sensor")
        sensor_state = None
        try:
            sensor_state = agent_state.sensor_states["color_sensor"]
        except Exception:
            try:
                sensor_state = agent_state.sensor_states["depth_sensor"]
            except Exception:
                sensor_state = None

        prev_floor = int(getattr(self.multifloor, "floor_id", 0))
        mf = self.multifloor.on_step(
            step_ndx=int(self.step_ndx),
            agent_state=agent_state,
            depth_image=depth_image,
            sensor_state=sensor_state,
            fov=float(self.fov),
            navigability_mask=getattr(self, "_last_navigability_mask", None),
            a_final=list(a_final) if a_final is not None else [],
            action_number=step_metadata.get("action_number"),
            frontier_count=len(a_final) if a_final is not None else 0,
            revisit_count=int(step_metadata.get("prompt_revisit_count") or 0),
            loop_risk=bool(step_metadata.get("prompt_loop_risk")),
            visit_count=int(
                step_metadata.get("prompt_visit_count_current_cell") or 0
            ),
            underexecuted=bool(
                step_metadata.get("previous_execution_underexecuted")
            ),
            stop_generated=bool(step_metadata.get("stop_logit_generated_stop")),
            stop_logit_margin=step_metadata.get("stop_logit_margin"),
        )
        step_metadata["multifloor_enabled"] = True
        step_metadata["multifloor_climbing"] = bool(mf.get("climbing"))
        step_metadata["floor_id"] = mf.get("floor_id")
        step_metadata["stair_slope_conf"] = mf.get("stair_slope_conf")
        step_metadata["multifloor_dy"] = mf.get("dy")
        step_metadata["multifloor"] = mf.get("multifloor") or {}

        if mf.get("save_maps_on_climb_start"):
            try:
                self.multifloor.save_current_maps(
                    int(prev_floor), self.voxel_map, self.explored_map
                )
                step_metadata["multifloor_maps_saved_at_climb"] = int(prev_floor)
            except Exception:
                pass

        swap_to = mf.get("map_swap_floor_id")
        if swap_to is not None:
            try:
                self.multifloor.save_current_maps(
                    int(prev_floor), self.voxel_map, self.explored_map
                )
            except Exception:
                pass
            voxel, explored = self.multifloor.load_or_create_maps(
                int(swap_to), self._multifloor_blank_maps
            )
            self.voxel_map = voxel
            self.explored_map = explored
            step_metadata["multifloor_map_swapped_to"] = int(swap_to)

        # High-confidence Stop overrides climb latch (false latch was eating FN@1m).
        stop_override = False
        if (
            os.environ.get("VLMNAV_MULTIFLOOR_STOP_OVERRIDE_ENABLE", "0")
            .strip()
            .lower()
            in ("1", "true", "yes", "y", "on")
            and mf.get("suppress_stop")
            and (
                agent_action == PolarAction.stop
                or step_metadata.get("action_number") == -1
            )
        ):
            payload = getattr(self, "_last_stop_logit_payload", None) or {}
            margin = payload.get("margin_stop_minus_continue")
            try:
                thr = float(
                    payload.get(
                        "threshold",
                        os.environ.get("VLMNAV_STOP_LOGIT_THRESHOLD", "2.0"),
                    )
                )
            except (TypeError, ValueError):
                thr = 2.0
            if (
                payload.get("accepted_stop")
                and margin is not None
                and float(margin) > thr
            ):
                stop_override = True
                step_metadata["multifloor_stop_override"] = True

        if (
            mf.get("suppress_stop")
            and step_metadata.get("action_number") == -1
            and not stop_override
        ):
            step_metadata["multifloor_stop_suppressed"] = True

        # Preserve an already accepted direct Stop while the multifloor
        # controller is active; do not replace it with a stair bias action.
        direct_stop_accepted = bool(
            agent_action == PolarAction.stop
            and step_metadata.get("action_number") == -1
            and step_metadata.get("same_pose_stop_crosscheck_outcome")
            == "model_single_stop_direct"
        )
        if direct_stop_accepted:
            step_metadata["multifloor_direct_stop_preserved"] = True
            return agent_action, step_metadata

        if mf.get("bias_action") and mf.get("action_number") is not None:
            new_num = int(mf["action_number"])
            step_metadata["action_number"] = new_num
            step_metadata["multifloor_action_bias"] = True
            if new_num == -1:
                agent_action = PolarAction.stop
            else:
                agent_action = self._action_number_to_polar(
                    new_num, list(a_final or [])
                )
        elif (
            mf.get("suppress_stop")
            and agent_action == PolarAction.stop
            and not stop_override
        ):
            pick = self.multifloor.choose_forward_action(
                list(a_final or []),
                preferred_yaw=float(
                    getattr(self.multifloor, "last_stair_yaw", 0.0) or 0.0
                ),
                climbing=True,
            )
            if pick is not None:
                step_metadata["action_number"] = int(pick)
                step_metadata["multifloor_stop_suppressed"] = True
                agent_action = self._action_number_to_polar(
                    int(pick), list(a_final or [])
                )

        return agent_action, step_metadata

    def _global_to_grid(self, position: np.ndarray, rotation=None):
        """Convert global coordinates to grid coordinates in the agent's voxel map"""
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.voxel_map.shape
        x = int(resolution[1] // 2 + dx * self.scale)
        y = int(resolution[0] // 2 + dz * self.scale)

        if rotation is not None:
            original_coords = np.array([x, y, 1])
            new_coords = np.dot(rotation, original_coords)
            new_x, new_y = new_coords[0], new_coords[1]
            return (int(new_x), int(new_y))

        return (x, y)

    def _generate_voxel(self, a_final: dict, zoom: int=9, agent_state: habitat_sim.AgentState=None, chosen_action: int=None):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        text_size = 1.25
        text_thickness = 1
        rotation_matrix = None
        agent_coords = self._global_to_grid(agent_state.position, rotation=rotation_matrix)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']:
            a_final[(0.75, np.pi)] = 0

        for (r, theta), action in a_final.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt, rotation=rotation_matrix)

            # Draw action arrows and labels
            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), RED, 5, tipLength=0.05)
            text = str(action)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15

            if chosen_action is not None and action == chosen_action:
                cv2.circle(topdown_map, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
            cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)

        # Draw agent's current position
        cv2.circle(topdown_map, agent_coords, radius=15, color=RED, thickness=-1)

        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]
        return zoomed_map

    def _action_number_to_polar(self, action_number: int, a_final: list):
        """Converts the chosen action number to its PolarAction instance"""
        try:
            action_number = int(action_number)
            if action_number <= len(a_final) and action_number > 0:
                r, theta = a_final[action_number - 1]
                return PolarAction(r, candidate_model_theta_to_yaw_command(theta))
            if action_number == 0:
                return PolarAction(0, np.pi)
        except ValueError:
            pass

        logging.info("Bad action number: " + str(action_number))
        return PolarAction.default

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible.

        This implementation is more robust than the previous naive slice: it
        extracts a balanced {...} block, tries json.loads and ast.literal_eval,
        and falls back to regex extraction of an 'action' integer.
        """
        if not response:
            logging.error('Empty response to parse')
            return {}

        action_token = r"(candidate_[+-]?\d+|[+-]?\d+|[⓪①②③④⑤⑥⑦⑧⑨⑩➀➁➂➃➄➅➆➇➈➉])"

        # CoT-style responses may contain a decision-free JSON block followed by
        # the executable action dict on the final line. Prefer the explicit
        # action dict instead of returning the first non-action JSON block.
        try:
            import re

            action_line_matches = list(
                re.finditer(
                    r"(?m)^\s*\{\s*['\"]action['\"]\s*:\s*['\"]?" + action_token + r"['\"]?\s*\}\s*$",
                    response,
                    flags=re.I,
                )
            )
            if action_line_matches:
                return {'action': self._action_token_to_int(action_line_matches[-1].group(1))}
        except Exception:
            pass

        def _extract_balanced_braces(s: str):
            start = s.find('{')
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(s)):
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
            return None

        def _normalize_parsed_action(parsed):
            if not isinstance(parsed, dict):
                return parsed
            if 'action' in parsed:
                if str(parsed['action']).strip().strip("\"'`.,;:()[]{}").lower() in ('none', 'null', 'n/a', 'na'):
                    parsed['action'] = 0
                    return parsed
                try:
                    parsed['action'] = self._action_token_to_int(parsed['action'])
                except (TypeError, ValueError):
                    return parsed
                return parsed
            decision = parsed.get('decision')
            if not isinstance(decision, dict):
                def _find_nested_decision(obj):
                    if isinstance(obj, dict):
                        nested = obj.get('decision')
                        if isinstance(nested, dict) and 'action' in nested:
                            return nested
                        for value in obj.values():
                            found = _find_nested_decision(value)
                            if found is not None:
                                return found
                    elif isinstance(obj, list):
                        for value in obj:
                            found = _find_nested_decision(value)
                            if found is not None:
                                return found
                    return None

                decision = _find_nested_decision(parsed)
                if isinstance(decision, dict):
                    parsed['decision'] = decision
            if isinstance(decision, dict) and 'action' in decision:
                try:
                    parsed['action'] = self._action_token_to_int(decision.get('action'))
                except (TypeError, ValueError):
                    pass
            return parsed

        # 1) try to extract a balanced {...} candidate
        try:
            candidate = _extract_balanced_braces(response)
            if candidate:
                # normalize common artifacts: multiple leading '{{' or trailing '}}}'
                # shrink multiple leading braces to one, and trim trailing braces until balanced
                while candidate.startswith('{{'):
                    candidate = candidate[1:]
                # trim trailing braces until counts match
                while candidate.count('{') < candidate.count('}'):
                    candidate = candidate[:-1]

                # try JSON first
                try:
                    import json

                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        parsed = _normalize_parsed_action(parsed)
                        if isinstance(parsed, dict) and 'action' in parsed:
                            return parsed
                        if isinstance(parsed, dict) and (
                            'done' in parsed
                            or 'stop' in parsed
                            or 'decision' in parsed
                            or 'arrival_certified' in parsed
                        ):
                            return parsed
                except Exception:
                    pass

                # try Python literal
                try:
                    eval_resp = ast.literal_eval(candidate)
                    if isinstance(eval_resp, dict):
                        eval_resp = _normalize_parsed_action(eval_resp)
                        if isinstance(eval_resp, dict) and 'action' in eval_resp:
                            return eval_resp
                        if isinstance(eval_resp, dict) and (
                            'done' in eval_resp
                            or 'stop' in eval_resp
                            or 'decision' in eval_resp
                            or 'arrival_certified' in eval_resp
                        ):
                            return eval_resp
                except Exception:
                    pass
        except Exception as e:
            logging.error(f'Error extracting candidate from response: {e}')

        # 2) fallback: try to find an "action": number pattern
        try:
            import re

            m = re.search(r"[\"']?action[\"']?\s*[:=]\s*[\"']?" + action_token, response, flags=re.I)
            if m:
                return {'action': self._action_token_to_int(m.group(1))}

            choice_patterns = [
                r"\b(?:choose|select|pick|take|return|use|go with)\s+(?:the\s+)?(?:action\s*)?(?:labeled\s*)?[\"']?" + action_token + r"[\"']?",
                r"\b(?:must|should|would|will)\s+(?:choose|select|pick|take|return|use|go with)\s+(?:the\s+)?(?:action\s*)?(?:labeled\s*)?[\"']?" + action_token + r"[\"']?",
                r"\b(?:final answer|therefore|so|thus)[^\\n.]{0,120}?\b(?:action\s*)?[\"']?" + action_token + r"[\"']?",
                r"\b(?:best choice|best action|best option|final choice)\s+(?:is|would be|should be)\s+(?:to\s+)?(?:choose\s+)?(?:the\s+)?(?:action\s*)?(?:labeled\s*)?[\"']?" + action_token + r"[\"']?",
                r"\b(?:I\s+)?(?:will|would|should)\s+(?:choose|select|pick|take)\s+(?:the\s+)?(?:action\s*)?(?:labeled\s*)?[\"']?" + action_token + r"[\"']?",
                r"\b(?:only|best)\s+(?:available\s+)?(?:forward\s+movement|action|option|choice)\s+(?:available\s+)?(?:is|would be)\s+(?:the\s+)?(?:action\s*)?(?:labeled\s*)?[\"']?" + action_token + r"[\"']?",
                r"\b(?:the\s+)?(?:only|best)\s+(?:available\s+)?(?:forward\s+movement|action|option|choice)[^.\n]{0,80}?\b(?:which\s+is|is)\s+(?:the\s+)?(?:action\s*)?(?:labeled\s*)?[\"']?" + action_token + r"[\"']?",
                r"\b(?:action\s*)?(?:labeled\s*)?[\"']?" + action_token + r"[\"']?(?:\s*\([^)]*\))?\s+is\s+(?:the\s+)?only\s+(?:available\s+)?(?:forward\s+movement|action|option|choice)\b",
            ]
            explicit_choices = []
            for pattern in choice_patterns:
                explicit_choices.extend(re.finditer(pattern, response, flags=re.I))
            if explicit_choices:
                # Use the last explicit choice because models often discuss rejected
                # alternatives first, then state the final action near the end.
                explicit_choices.sort(key=lambda match: match.start())
                return {'action': self._action_token_to_int(explicit_choices[-1].group(1))}

            turn_patterns = [
                r"\b(?:best choice|best action|best option|final choice)\s+(?:is|would be|should be)\s+(?:to\s+)?turn around\b",
                r"\b(?:I\s+)?(?:will|would|should|must)\s+(?:choose|select|pick|take)?\s*(?:to\s+)?turn around\b",
                r"\bchoose\s+action\s+0\s+if\s+you\s+want\s+to\s+turn around\b",
                r"\b(?:no|not any)\s+(?:available|valid|good|correct|suitable)\s+action\b",
                r"\bno\s+available\s+action\s+that\s+(?:leads|goes|takes|points)\b",
                r"\b(?:do not|don't|cannot|can't)\s+see\s+any\s+good\s+actions?\b",
            ]
            turn_choices = []
            for pattern in turn_patterns:
                turn_choices.extend(re.finditer(pattern, response, flags=re.I))
            if turn_choices:
                return {'action': 0}

            # if there's a single small integer in response, treat it as action
            nums = re.findall(r'\b([0-9]{1,3})\b', response)
            if len(nums) == 1:
                return {'action': int(nums[0])}
        except Exception as e:
            logging.error(f'Fallback parse error: {e}')

        # 3) give up and log raw response for debugging
        logging.error(f'Error parsing response {response!r}')
        return {}



class GOATAgent(VLMNavAgent):
 
    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        if goal['mode'] == 'image':
            stopping_images = [obs['color_sensor'], goal['goal_image']]
        else:
            stopping_images = [obs['color_sensor']]

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, stopping_images, goal)
        if goal['mode'] == 'image':
            images['goal_image'] = goal['goal_image']

        step_metadata.update({
            'goal': goal['name'],
            'goal_mode': goal['mode']
        })

        # If model calls stop two times in a row, we return the stop action and terminate the episode
        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if self.pivot is not None:
                pivot_instruction = self._construct_prompt(goal, 'pivot')
                agent_action, pivot_images = self.pivot.run(
                    obs['color_sensor'], pivot_instruction,
                    agent_state, agent_state.sensor_states['color_sensor'],
                    goal_image=goal['goal_image'] if goal['mode'] == 'image' else None
                )
                images.update(pivot_images)
                logging_data = {}
                step_metadata['action_number'] = -100
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'candidate_action_scores': list(self.last_candidate_action_scores),
            'images': images
        }
        return agent_action, metadata
    
    def _construct_prompt(
        self,
        goal: dict,
        prompt_type: str,
        num_actions=0,
        candidate_features=None,
    ):
        """Constructs the prompt, depending on the goal modality. """
        if goal['mode'] == 'object':
            task = f'Navigate to the nearest {goal["name"]}'
            first_instruction = f'Find the nearest {goal["name"]} and navigate as close as you can to it. '
        if goal['mode'] == 'description':
            first_instruction = f"Find and navigate to the {goal['lang_desc']}. Navigate as close as you can to it. "
            task = first_instruction
        if goal['mode'] == 'image':
            task = f'Navigate to the specific {goal["name"]} shown in the image labeled GOAL IMAGE. Pay close attention to the details, and note you may see the object from a different angle than in the goal image. Navigate as close as you can to it '
            first_instruction = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate as close as you can to it. "

        if prompt_type == 'stopping':        
            stopping_prompt = (f"The agent has the following navigation task: \n{task}\n. The agent has sent you an image taken from its current location{' as well as the goal image. ' if goal['mode'] == 'image' else '. '} "
                                f'Your job is to determine whether the agent is close to the specified {goal["name"].upper()}'
                                f"First, tell me what you see in the image, and tell me if there is a {goal['name']} that matches the description. Then, return 1 if the agent is close to the {goal['name']}, and 0 if it isn't. Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt

        if prompt_type == 'pivot':
            return f'{first_instruction} Use your prior knowledge about where items are typically located within a home. '
        
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: {first_instruction} use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see, and if you have any leads on finding the {goal['name']}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return baseline_prompt
        
        if prompt_type == 'action':
            action_prompt = (f"TASK: {first_instruction} use your prior knowledge about where items are typically located within a home. "
            f"There are {num_actions-1} red arrow(s) superimposed onto your observation, which represent potential actions. " 
            f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS.' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
            f"First, tell me what you see, and if you have any leads on finding the {goal['name']}. Second, tell me which general direction you should go in. "
            f"Lastly, explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

    def reset_goal(self):
        """Called after every subtask of GOAT. Notably does not reset the voxel map, only resets all areas to be unexplored"""
        self.stopping_calls = [self.step_ndx-2]
        self.explored_map = np.zeros_like(self.explored_map)
        self.turned = self.step_ndx - self.cfg['turn_around_cooldown']


class ObjectNavAgent(VLMNavAgent):

    def _choose_action(self, obs: dict):
        agent_action, metadata = self._objectnav_choose_action_body(obs)
        step_metadata = metadata.get("step_metadata") or {}
        a_final = metadata.get("a_final") or []
        agent_action, step_metadata = self._apply_multifloor_step(
            obs, a_final, agent_action, step_metadata
        )
        metadata["step_metadata"] = step_metadata
        return agent_action, metadata

    def _objectnav_choose_action_body(self, obs: dict):
        obs = self._public_inference_observation(obs)
        action_first_stop_confirmation = self._action_first_stop_confirmation_enabled()
        if action_first_stop_confirmation:
            obs = dict(obs)
            obs['defer_stop_verification_until_action_stop'] = True
        agent_state = obs['agent_state']
        goal = obs['goal']

        stopping_images = obs.get('stopping_images_override') if isinstance(obs, dict) else None
        if not isinstance(stopping_images, list) or not stopping_images:
            stopping_images = [obs['color_sensor']]
        # The action surface below intentionally projects red candidate arrows
        # onto ``images['color_sensor']`` in-place. The stopping VLM consumes
        # the unprojected observation above. An audit snapshot is captured only
        # immediately before an actual verifier call below.
        save_stopping_input_image = os.environ.get(
            'VLMNAV_SAVE_STOPPING_INPUT_IMAGE',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        stopping_input_rgb = None
        stopping_input_rgb_capture_error = ''
        save_terminal_certificate_input_image = os.environ.get(
            'VLMNAV_SAVE_TERMINAL_CERTIFICATE_INPUT_IMAGE',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        terminal_certificate_input_rgb = None
        terminal_certificate_input_rgb_capture_error = ''
        if save_terminal_certificate_input_image:
            if len(stopping_images) != 1:
                terminal_certificate_input_rgb_capture_error = (
                    f'expected_single_view_got_{len(stopping_images)}'
                )
            else:
                try:
                    terminal_certificate_input_rgb = np.ascontiguousarray(
                        np.asarray(stopping_images[0])[:, :, :3]
                    ).copy()
                except Exception as exc:
                    terminal_certificate_input_rgb_capture_error = (
                        f'{type(exc).__name__}: {exc}'
                    )
        terminal_certificate_prompt_audit = ''
        terminal_certificate_prompt_audit_hash = ''
        if self._terminal_certificate_prompt_audit_enabled():
            # Collection needs the exact future certificate prompt, while the
            # frozen action-policy rollout must retain its legacy Stop prompt.
            terminal_certificate_prompt_audit = self._construct_terminal_certificate_prompt(goal)
            terminal_certificate_prompt_audit_hash = hashlib.sha256(
                terminal_certificate_prompt_audit.encode('utf-8')
            ).hexdigest()
        stop_prompt_context = obs.get('stop_multiview_prompt_context') if isinstance(obs, dict) else None
        self._stopping_prompt_context = stop_prompt_context if isinstance(stop_prompt_context, dict) else None
        try:
            a_final, images, step_metadata, stopping_response = self._run_threads(obs, stopping_images, goal)
        finally:
            self._stopping_prompt_context = None
        # ``step`` already passed ``obs`` through
        # ``_public_inference_observation`` and every legacy privileged runtime
        # gate below is hard-disabled.  Record that invariant before any early
        # return (notably the original ObjectNav branch) so formal evaluation
        # can prove the public-only inference contract from every model-I/O row.
        step_metadata['privileged_runtime_gates_disabled'] = True
        step_metadata['object'] = goal
        step_metadata['stop_multiview_input_images'] = int(len(stopping_images))
        pano_context = obs.get('lightweight_panoramic_context') if isinstance(obs, dict) else None
        if isinstance(pano_context, dict):
            selected_angle = pano_context.get("selected_angle")
            scores = pano_context.get("scores", {}) or {}
            depth_scores = pano_context.get("depth_scores", {}) or {}
            step_metadata['lightweight_panoramic_enabled'] = True
            step_metadata['lightweight_panoramic_success'] = selected_angle is not None
            step_metadata['lightweight_panoramic_selected_angle'] = selected_angle
            step_metadata['lightweight_panoramic_selected_score'] = scores.get(selected_angle)
            step_metadata['lightweight_panoramic_selected_depth_score'] = depth_scores.get(selected_angle)
            step_metadata['lightweight_panoramic_selected_reason'] = pano_context.get("selected_reason", "")
            step_metadata['lightweight_panoramic_gate_reason'] = pano_context.get("gate_reason", "")
            step_metadata['lightweight_panoramic_global_state'] = pano_context.get("global_state")
            step_metadata['lightweight_panoramic_view_direction_deg'] = pano_context.get("view_direction_deg")
            step_metadata['lightweight_panoramic_lock_active'] = pano_context.get("lock_active")
            step_metadata['lightweight_panoramic_lock_until_step'] = pano_context.get("lock_until_step")
            step_metadata['lightweight_panoramic_lock_steps_remaining'] = pano_context.get("lock_steps_remaining")
            step_metadata['lightweight_panoramic_context_only'] = bool(pano_context.get("context_only", False))

        if action_first_stop_confirmation:
            step_metadata['original_objectnav_logic'] = True
            step_metadata['action_first_stop_confirmation_enabled'] = True
            strict_final_action_stop = self._action_first_requires_final_stop_action()
            step_metadata['action_first_stop_confirmation_require_final_action'] = bool(
                strict_final_action_stop
            )
            step_metadata['action_first_stop_confirmation_crosscheck_prompt'] = bool(
                self._action_first_append_crosscheck_prompt()
            )
            step_metadata['action_first_stop_confirmation_verifier_model'] = str(
                getattr(self.stoppingVLM, 'name', '')
            )
            step_metadata['action_first_stop_confirmation_input_images'] = int(
                len(stopping_images)
            )
            step_metadata['action_first_stop_confirmation_prompt_sha256'] = None
            step_metadata['action_first_stop_confirmation_called'] = False
            step_metadata['action_first_stop_confirmation_outcome'] = 'action_move'
            step_metadata['terminal_certificate_every_step_enabled'] = bool(
                self._terminal_certificate_every_step_enabled()
            )
            step_metadata['terminal_certificate_prompt_audit_enabled'] = bool(
                self._terminal_certificate_prompt_audit_enabled()
            )
            if terminal_certificate_input_rgb is not None:
                step_metadata['terminal_certificate_input_rgb_log_key'] = (
                    'terminal_certificate_input_rgb'
                )
                step_metadata['terminal_certificate_input_rgb_pixel_sha256'] = (
                    hashlib.sha256(terminal_certificate_input_rgb.tobytes()).hexdigest()
                )
                step_metadata['terminal_certificate_input_rgb_shape'] = list(
                    terminal_certificate_input_rgb.shape
                )
                step_metadata['terminal_certificate_input_rgb_dtype'] = str(
                    terminal_certificate_input_rgb.dtype
                )
            elif save_terminal_certificate_input_image:
                step_metadata['terminal_certificate_input_rgb_capture_error'] = (
                    terminal_certificate_input_rgb_capture_error or 'unknown_capture_error'
                )
            if terminal_certificate_prompt_audit:
                step_metadata['terminal_certificate_prompt_sha256'] = (
                    terminal_certificate_prompt_audit_hash
                )
            step_metadata['action_wants_stop'] = False
            if self.pivot is not None:
                pivot_instruction = self._construct_prompt(goal, 'pivot')
                agent_action, pivot_images = self.pivot.run(
                    obs['color_sensor'], pivot_instruction,
                    agent_state, agent_state.sensor_states['color_sensor']
                )
                images.update(pivot_images)
                logging_data = {}
                step_metadata['action_number'] = -100
            else:
                step_metadata, logging_data, _ = self._prompting(
                    goal,
                    a_final,
                    images,
                    step_metadata,
                )
                if terminal_certificate_prompt_audit:
                    logging_data['TERMINAL CERTIFICATE PROMPT'] = (
                        terminal_certificate_prompt_audit
                    )
                legacy_action_wants_stop = action_requests_stop_confirmation(
                    step_metadata.get('action_number'),
                    action_should_stop=bool(step_metadata.get('action_should_stop', False)),
                    decision_action=step_metadata.get('action_decision_action'),
                )
                step_metadata['action_first_stop_proposal_action_number'] = (
                    step_metadata.get('action_number')
                )
                step_metadata['action_first_stop_proposal_legacy_wants_stop'] = bool(
                    legacy_action_wants_stop
                )
                if strict_final_action_stop:
                    action_wants_stop = action_requests_stop_confirmation(
                        step_metadata.get('action_number'),
                        require_final_action_stop=True,
                    )
                    step_metadata['action_first_stop_confirmation_trigger'] = (
                        'final_executable_action'
                    )
                else:
                    action_wants_stop = action_requests_stop_confirmation(
                        step_metadata.get('action_number'),
                        action_should_stop=bool(step_metadata.get('action_should_stop', False)),
                        decision_action=step_metadata.get('action_decision_action'),
                    )
                    step_metadata['action_first_stop_confirmation_trigger'] = (
                        'action_or_structured_stop_field'
                    )
                step_metadata['action_first_stop_proposal_suppressed_inconsistent'] = bool(
                    strict_final_action_stop
                    and legacy_action_wants_stop
                    and not action_wants_stop
                )
                step_metadata['action_wants_stop'] = bool(action_wants_stop)
                # The terminal certificate is deliberately independent of the
                # action model's ``-1`` token.  A frozen Move policy can reach a
                # visually valid terminal pose without ever proposing Stop, so
                # gating this model behind ``action_wants_stop`` makes the
                # certificate incapable of recovering those cases.  It is one
                # discrete call per decision state -- never a confidence
                # accumulator or a repeated readiness signal.
                terminal_certificate_enabled = self._terminal_certificate_every_step_enabled()
                terminal_certificate_single_rgb = len(stopping_images) == 1
                terminal_certificate_ready = bool(
                    terminal_certificate_enabled
                    and self._terminal_certificate_prompt_enabled()
                    and self._terminal_certificate_evidence_required()
                    and self._terminal_certificate_protocol_supported()
                    and terminal_certificate_single_rgb
                )
                terminal_certificate_called = False
                terminal_certificate_stop = False
                terminal_certificate_response = ''
                terminal_certificate_error = ''
                step_metadata['terminal_certificate_policy_action_number'] = (
                    step_metadata.get('action_number')
                )
                step_metadata['terminal_certificate_overrode_move'] = False
                step_metadata['terminal_certificate_called'] = False
                step_metadata['terminal_certificate_stop'] = False
                step_metadata['terminal_certificate_outcome'] = 'disabled'

                # Enabling the runtime switch without the strict prompt and
                # conjunction would silently turn this into a legacy done-only
                # Stop model.  Fail closed instead.
                if terminal_certificate_enabled and not terminal_certificate_ready:
                    missing_requirements = []
                    if not self._terminal_certificate_prompt_enabled():
                        missing_requirements.append('terminal_certificate_prompt')
                    if not self._terminal_certificate_evidence_required():
                        missing_requirements.append('terminal_certificate_evidence_gate')
                    if not self._terminal_certificate_protocol_supported():
                        missing_requirements.append('supported_terminal_certificate_protocol')
                    if not terminal_certificate_single_rgb:
                        missing_requirements.append(
                            f'single_raw_rgb_required_got_{len(stopping_images)}'
                        )
                    terminal_certificate_error = 'requires_' + '_and_'.join(missing_requirements)
                    step_metadata['terminal_certificate_outcome'] = 'configuration_error'
                    step_metadata['terminal_certificate_error'] = terminal_certificate_error
                    logging.error(
                        'Terminal certificate enabled but strict prompt/evidence gate is missing; '
                        'will not call legacy stopping model.'
                    )
                elif terminal_certificate_ready:
                    self._stopping_prompt_context = (
                        stop_prompt_context if isinstance(stop_prompt_context, dict) else None
                    )
                    try:
                        terminal_certificate_stop, terminal_certificate_response = self._stopping_module(
                            stopping_images,
                            goal,
                        )
                        terminal_certificate_called = True
                    except Exception as exc:
                        logging.error('Terminal visual certificate failed: %s', exc)
                        terminal_certificate_error = str(exc)
                    finally:
                        self._stopping_prompt_context = None

                    step_metadata['terminal_certificate_called'] = bool(terminal_certificate_called)
                    step_metadata['terminal_certificate_stop'] = bool(terminal_certificate_stop)
                    step_metadata['terminal_certificate_prompt_sha256'] = (
                        self._last_stopping_prompt_sha256 or None
                    )
                    logging_data['TERMINAL CERTIFICATE PROMPT'] = self._last_stopping_prompt
                    logging_data['TERMINAL CERTIFICATE RESPONSE'] = terminal_certificate_response
                    if terminal_certificate_error:
                        step_metadata['terminal_certificate_error'] = terminal_certificate_error
                        step_metadata['terminal_certificate_outcome'] = 'error'
                    elif terminal_certificate_stop:
                        step_metadata['terminal_certificate_outcome'] = 'confirmed_stop'
                    else:
                        step_metadata['terminal_certificate_outcome'] = 'rejected_continue'

                    certificate_protocol = self._terminal_certificate_protocol_id()
                    if certificate_protocol == 'objectnav_state_conditioned_causal_stop_bottleneck_v3':
                        parsed_certificate = parse_state_conditioned_causal_stop_bottleneck_response(
                            terminal_certificate_response
                        )
                    else:
                        parsed_certificate = self._eval_response(terminal_certificate_response)
                    if isinstance(parsed_certificate, dict):
                        if certificate_protocol == 'objectnav_visual_arrival_certificate_v2':
                            certificate_keys = ('arrival_certified',)
                        elif certificate_protocol == 'objectnav_state_conditioned_spatial_stop_v1':
                            certificate_keys = (
                                'navigation_state',
                                'identity_evidence',
                                'proximity_evidence',
                                'stop',
                            )
                        elif certificate_protocol == 'objectnav_state_conditioned_evidence_stop_v2':
                            certificate_keys = (
                                'navigation_state',
                                'ability',
                                'spatial_reasoning',
                                'decision_bridge',
                                'stop',
                            )
                        elif certificate_protocol == 'objectnav_state_conditioned_causal_stop_bottleneck_v3':
                            certificate_keys = (
                                'navigation_state',
                                'ability',
                                'spatial_reasoning',
                            )
                        else:
                            certificate_keys = (
                                'terminal_state', 'target_visible', 'near_enough', 'done'
                            )
                        for key in certificate_keys:
                            if key in parsed_certificate:
                                step_metadata[f'terminal_certificate_{key}'] = (
                                    parsed_certificate.get(key)
                                )

                if terminal_certificate_called and terminal_certificate_stop:
                    # A same-frame, visually grounded certificate is sufficient
                    # to end the episode even when the action policy proposes a
                    # move.  It does not consume simulator geometry or prior
                    # Stop signals.
                    self.stopping_calls.append(self.step_ndx)
                    step_metadata['called_stopping'] = True
                    step_metadata['stop_signal'] = True
                    step_metadata['stop_response_present'] = bool(terminal_certificate_response)
                    step_metadata['stop_decision_source'] = 'terminal_visual_certificate'
                    step_metadata['stop_gate_passed'] = True
                    step_metadata['action_number'] = -1
                    step_metadata['action_should_stop'] = True
                    step_metadata['action_wants_stop'] = bool(action_wants_stop)
                    step_metadata['terminal_certificate_overrode_move'] = bool(
                        not action_wants_stop
                    )
                    if action_wants_stop:
                        step_metadata['action_first_stop_confirmation_called'] = True
                        step_metadata['action_first_stop_confirmation_outcome'] = (
                            'confirmed_stop_reused_terminal_certificate'
                        )
                        step_metadata['action_first_stop_confirmation_prompt_sha256'] = (
                            self._last_stopping_prompt_sha256 or None
                        )
                    agent_action = PolarAction.stop
                    stopping_response = terminal_certificate_response
                elif action_wants_stop:
                    step_metadata['action_first_stop_confirmation_called'] = True
                    # Reuse the one per-step certificate rather than making a
                    # second Stop call for action ``-1``.  This guarantees one
                    # certificate decision per state and keeps its prompt/image
                    # surface auditable.
                    if terminal_certificate_enabled:
                        stop_confirmed = bool(terminal_certificate_stop)
                        stopping_response = terminal_certificate_response
                        step_metadata['stop_decision_source'] = (
                            'terminal_certificate_reused_for_action_stop'
                        )
                    else:
                        if save_stopping_input_image:
                            if len(stopping_images) != 1:
                                stopping_input_rgb_capture_error = (
                                    f'expected_single_view_got_{len(stopping_images)}'
                                )
                            else:
                                try:
                                    # Match VLM.call's visual surface: contiguous
                                    # RGB only, before the action-image projection.
                                    stopping_input_rgb = np.ascontiguousarray(
                                        np.asarray(stopping_images[0])[:, :, :3]
                                    ).copy()
                                except Exception as exc:
                                    stopping_input_rgb_capture_error = (
                                        f'{type(exc).__name__}: {exc}'
                                    )
                        self._stopping_prompt_context = (
                            stop_prompt_context if isinstance(stop_prompt_context, dict) else None
                        )
                        try:
                            stop_confirmed, stopping_response = self._stopping_module(
                                stopping_images,
                                goal,
                            )
                        except Exception as exc:
                            logging.error('Action-first stop confirmation failed: %s', exc)
                            stop_confirmed = False
                            stopping_response = ''
                            step_metadata['action_first_stop_confirmation_error'] = str(exc)
                        finally:
                            self._stopping_prompt_context = None
                        step_metadata['stop_decision_source'] = 'action_first_confirmation'

                    # ``called_stopping`` means an accepted terminal decision
                    # to the rest of VLMNav.  Do not set it merely because the
                    # certificate was queried: several navigation gates treat
                    # it as an active Stop signal.
                    step_metadata['called_stopping'] = bool(
                        stop_confirmed and not terminal_certificate_enabled
                    )
                    step_metadata['stop_signal'] = bool(stop_confirmed)
                    step_metadata['stop_response_present'] = bool(stopping_response)
                    step_metadata['stop_gate_passed'] = bool(stop_confirmed)
                    step_metadata['action_first_stop_confirmation_prompt_sha256'] = (
                        self._last_stopping_prompt_sha256 or None
                    )
                    logging_data['STOPPING PROMPT'] = self._last_stopping_prompt
                    if stopping_input_rgb is not None:
                        step_metadata['action_first_stop_confirmation_rgb_log_key'] = (
                            'stopping_input_rgb'
                        )
                        step_metadata['action_first_stop_confirmation_rgb_pixel_sha256'] = (
                            hashlib.sha256(stopping_input_rgb.tobytes()).hexdigest()
                        )
                        step_metadata['action_first_stop_confirmation_rgb_shape'] = list(
                            stopping_input_rgb.shape
                        )
                        step_metadata['action_first_stop_confirmation_rgb_dtype'] = str(
                            stopping_input_rgb.dtype
                        )
                    elif save_stopping_input_image and not terminal_certificate_enabled:
                        step_metadata['action_first_stop_confirmation_rgb_capture_error'] = (
                            stopping_input_rgb_capture_error or 'unknown_capture_error'
                        )
                    if stop_confirmed:
                        self.stopping_calls.append(self.step_ndx)
                        step_metadata['action_number'] = -1
                        step_metadata['action_should_stop'] = True
                        step_metadata['action_first_stop_confirmation_outcome'] = 'confirmed_stop'
                        agent_action = PolarAction.stop
                    else:
                        fallback_action = int(os.environ.get(
                            'VLMNAV_STOP_GATE_FALLBACK_ACTION',
                            '1',
                        ))
                        if fallback_action < 0 or fallback_action > len(a_final):
                            fallback_action = 0
                        step_metadata['stop_gate_blocked_action'] = -1
                        step_metadata['action_number'] = int(fallback_action)
                        step_metadata['action_should_stop'] = False
                        step_metadata['action_first_stop_confirmation_outcome'] = 'rejected_fallback'
                        agent_action = self._action_number_to_polar(
                            step_metadata['action_number'],
                            list(a_final),
                        )
                else:
                    step_metadata['called_stopping'] = False
                    step_metadata['stop_signal'] = bool(terminal_certificate_stop)
                    step_metadata['stop_response_present'] = bool(terminal_certificate_response)
                    step_metadata['stop_decision_source'] = (
                        'terminal_certificate_rejected_continue'
                        if terminal_certificate_called
                        else 'not_called_action_move'
                    )
                    step_metadata['stop_gate_passed'] = False
                    if terminal_certificate_called:
                        stopping_response = terminal_certificate_response
                    agent_action = self._action_number_to_polar(
                        step_metadata['action_number'],
                        list(a_final),
                    )

            logging_data['STOPPING RESPONSE'] = stopping_response
            if stopping_input_rgb is not None:
                images['stopping_input_rgb'] = stopping_input_rgb
            if terminal_certificate_input_rgb is not None:
                images['terminal_certificate_input_rgb'] = terminal_certificate_input_rgb
            metadata = {
                'step_metadata': step_metadata,
                'logging_data': logging_data,
                'a_final': a_final,
                'candidate_action_scores': list(self.last_candidate_action_scores),
                'images': images,
            }
            return agent_action, metadata

        original_objectnav_logic = os.environ.get(
            'VLMNAV_ORIGINAL_OBJECTNAV_LOGIC',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        if original_objectnav_logic:
            step_metadata['original_objectnav_logic'] = True
            current_stop_signal = self._current_stop_signal()
            blind_same_pose_crosscheck = bool(
                self._blind_same_pose_stop_crosscheck_enabled()
                and current_stop_signal
            )
            same_pose_crosscheck = bool(
                (
                    self._same_pose_stop_crosscheck_enabled()
                    or self._blind_same_pose_stop_crosscheck_enabled()
                )
                and current_stop_signal
            )
            step_metadata['same_pose_stop_crosscheck_enabled'] = bool(
                self._same_pose_stop_crosscheck_enabled()
            )
            step_metadata['same_pose_stop_crosscheck_requested'] = bool(
                same_pose_crosscheck
            )
            step_metadata['blind_same_pose_stop_crosscheck_requested'] = bool(
                blind_same_pose_crosscheck
            )
            step_metadata['same_pose_stop_crosscheck_agreed'] = False
            step_metadata['same_pose_stop_crosscheck_outcome'] = 'not_requested'
            # Match the original VLMNav ObjectNav control flow: the stopping
            # module terminates only after consecutive stop calls; otherwise the
            # action policy directly selects the executable action.
            consecutive_stop_signals = bool(
                len(self.stopping_calls) >= 2
                and self.stopping_calls[-2] == self.step_ndx - 1
            )
            blind_consecutive_fallback = bool(
                blind_same_pose_crosscheck
                and self._blind_consecutive_stop_fallback_enabled()
            )
            consecutive_stop_fallback_enabled = bool(
                self._consecutive_stop_fallback_enabled()
            )
            single_stop_direct_execute_enabled = os.environ.get(
                'VLMNAV_SINGLE_STOP_DIRECT_EXECUTE',
                '0',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            strict_dual_model_stop = bool(
                self._strict_dual_model_stop_enabled()
            )
            step_metadata['blind_consecutive_stop_fallback_enabled'] = bool(
                blind_consecutive_fallback
            )
            step_metadata['consecutive_stop_fallback_enabled'] = bool(
                consecutive_stop_fallback_enabled
            )
            step_metadata['single_stop_direct_execute_enabled'] = bool(
                single_stop_direct_execute_enabled
            )
            step_metadata['strict_dual_model_stop'] = bool(
                strict_dual_model_stop
            )
            if use_single_stop_direct_execute(
                current_stop_signal,
                enabled=single_stop_direct_execute_enabled,
            ):
                step_metadata['action_number'] = -1
                step_metadata['action_wants_stop'] = True
                step_metadata['same_pose_stop_crosscheck_outcome'] = (
                    'model_single_stop_direct'
                )
                agent_action = PolarAction.stop
                logging_data = {}
            elif use_consecutive_stop_fallback(
                consecutive_stop_signals,
                blind_same_pose_crosscheck,
                blind_consecutive_fallback,
                enabled=consecutive_stop_fallback_enabled,
            ):
                step_metadata['action_number'] = -1
                step_metadata['action_wants_stop'] = True
                step_metadata['same_pose_stop_crosscheck_outcome'] = 'consecutive_stop_fallback'
                agent_action = PolarAction.stop
                logging_data = {}
            else:
                if self.pivot is not None:
                    pivot_instruction = self._construct_prompt(goal, 'pivot')
                    agent_action, pivot_images = self.pivot.run(
                        obs['color_sensor'], pivot_instruction,
                        agent_state, agent_state.sensor_states['color_sensor']
                    )
                    images.update(pivot_images)
                    logging_data = {}
                    step_metadata['action_number'] = -100
                else:
                    step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                    crosscheck_agreed = should_stop_after_same_pose_crosscheck(
                        same_pose_crosscheck,
                        step_metadata.get('action_number'),
                    )
                    step_metadata['same_pose_stop_crosscheck_action_number'] = (
                        step_metadata.get('action_number')
                    )
                    step_metadata['same_pose_stop_crosscheck_agreed'] = bool(
                        crosscheck_agreed
                    )
                    step_metadata['action_wants_stop'] = bool(crosscheck_agreed)
                    if crosscheck_agreed:
                        step_metadata['action_number'] = -1
                        step_metadata['action_should_stop'] = True
                        step_metadata['action_decision_action'] = 'candidate_-1'
                        step_metadata['same_pose_stop_crosscheck_outcome'] = (
                            'blind_stop_now'
                            if blind_same_pose_crosscheck
                            else 'stop_now'
                        )
                        agent_action = PolarAction.stop
                    else:
                        step_metadata['same_pose_stop_crosscheck_outcome'] = (
                            'blind_continue'
                            if blind_same_pose_crosscheck
                            else 'short_approach'
                            if same_pose_crosscheck
                            else 'not_requested'
                        )
                        if (
                            strict_dual_model_stop
                            and action_requests_stop_confirmation(
                                step_metadata.get('action_number'),
                                action_should_stop=bool(
                                    step_metadata.get('action_should_stop')
                                ),
                                decision_action=step_metadata.get(
                                    'action_decision_action'
                                ),
                            )
                            and not current_stop_signal
                        ):
                            fallback_action = int(os.environ.get(
                                'VLMNAV_STOP_GATE_FALLBACK_ACTION',
                                '1',
                            ))
                            if fallback_action < 0 or fallback_action > len(a_final):
                                fallback_action = 0
                            step_metadata['stop_gate_blocked_action'] = -1
                            step_metadata['action_number'] = int(fallback_action)
                            step_metadata['action_should_stop'] = False
                            step_metadata['action_wants_stop'] = False
                            step_metadata['same_pose_stop_crosscheck_outcome'] = (
                                'action_stop_rejected_no_verifier'
                            )
                        agent_action = self._action_number_to_polar(
                            step_metadata['action_number'],
                            list(a_final),
                        )

            logging_data['STOPPING RESPONSE'] = stopping_response
            metadata = {
                'step_metadata': step_metadata,
                'logging_data': logging_data,
                'a_final': a_final,
                'candidate_action_scores': list(self.last_candidate_action_scores),
                'images': images
            }
            return agent_action, metadata

        model_name = str(getattr(self.actionVLM, "name", "")).lower()
        is_sensenova_qwen25 = ("sensenova" in model_name) and ("qwen2.5" in model_name)

        # Stopping VLM is only a stop candidate.  For structured ObjectNav evals,
        # the action policy must also explicitly choose candidate_-1 / should_stop.
        current_stop_signal = (
            len(self.stopping_calls) >= 1 and self.stopping_calls[-1] == self.step_ndx
        )
        stop_signal_ready = (
            (len(self.stopping_calls) >= 1 and self.stopping_calls[-1] == self.step_ndx)
            if is_sensenova_qwen25
            else (len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1)
        )
        step_metadata['stop_signal'] = bool(current_stop_signal)
        step_metadata['stop_signal_ready'] = bool(stop_signal_ready)
        stop_ready_streak = 0
        expected_stop_step = self.step_ndx
        for stop_step in reversed(self.stopping_calls):
            if stop_step == expected_stop_step:
                stop_ready_streak += 1
                expected_stop_step -= 1
                continue
            if stop_step < expected_stop_step:
                break
        step_metadata['stop_ready_streak'] = int(stop_ready_streak)

        if self.pivot is not None:
            pivot_instruction = self._construct_prompt(goal, 'pivot')
            agent_action, pivot_images = self.pivot.run(
                obs['color_sensor'], pivot_instruction,
                agent_state, agent_state.sensor_states['color_sensor']
            )
            images.update(pivot_images)
            logging_data = {}
            step_metadata['action_number'] = -100
        else:
            step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
            action_wants_stop = (
                step_metadata.get('action_number') == -1
                or bool(step_metadata.get('action_should_stop', False))
                or bool(step_metadata.get('action_stop_binary_wants_stop', False))
                or str(step_metadata.get('action_decision_action') or '').strip().lower() == 'candidate_-1'
            )
            require_factorized_stop_evidence = os.environ.get(
                'VLMNAV_REQUIRE_FACTORIZED_STOP_EVIDENCE',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            factorized_stop_gate_passed = True
            factorized_stop_gate_reason = ''
            if action_wants_stop and require_factorized_stop_evidence:
                factorized_stop_gate_passed = factorized_evidence_allows_stop(
                    step_metadata.get('action_identity_evidence'),
                    step_metadata.get('action_proximity_evidence'),
                )
                if not factorized_stop_gate_passed:
                    factorized_stop_gate_reason = (
                        f"identity:{step_metadata.get('action_identity_evidence') or 'missing'};"
                        f"proximity:{step_metadata.get('action_proximity_evidence') or 'missing'}"
                    )
                    action_wants_stop = False
                    if step_metadata.get('action_number') == -1:
                        fallback_action = int(os.environ.get('VLMNAV_STOP_GATE_FALLBACK_ACTION', '1'))
                        max_action = len(a_final)
                        if fallback_action < 0 or fallback_action > max_action:
                            fallback_action = 0
                        step_metadata['action_number'] = fallback_action
                        step_metadata['stop_gate_blocked_action'] = -1
            step_metadata['factorized_stop_gate_required'] = bool(require_factorized_stop_evidence)
            step_metadata['factorized_stop_gate_passed'] = bool(factorized_stop_gate_passed)
            step_metadata['factorized_stop_gate_reason'] = factorized_stop_gate_reason
            require_temporal_reference_contract = os.environ.get(
                'VLMNAV_REQUIRE_TEMPORAL_REFERENCE_CONTRACT', ''
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            temporal_reference_contract_format_valid = temporal_reference_contract_valid(
                step_metadata.get('action_temporal_relation'),
                step_metadata.get('temporal_target_reference_available'),
            )
            temporal_reference_gate_passed = temporal_relation_allows_stop(
                step_metadata.get('action_temporal_relation'),
                step_metadata.get('temporal_target_reference_available'),
            )
            if (
                action_wants_stop
                and require_temporal_reference_contract
                and not temporal_reference_gate_passed
            ):
                action_wants_stop = False
                if step_metadata.get('action_number') == -1:
                    fallback_action = int(os.environ.get('VLMNAV_STOP_GATE_FALLBACK_ACTION', '1'))
                    max_action = len(a_final)
                    if fallback_action < 0 or fallback_action > max_action:
                        fallback_action = 0
                    step_metadata['action_number'] = fallback_action
                    step_metadata['stop_gate_blocked_action'] = -1
            step_metadata['temporal_reference_gate_required'] = bool(
                require_temporal_reference_contract
            )
            step_metadata['temporal_reference_contract_format_valid'] = bool(
                temporal_reference_contract_format_valid
            )
            step_metadata['temporal_reference_gate_passed'] = bool(
                temporal_reference_gate_passed
            )
            require_stop_mode_for_action_stop = os.environ.get(
                'VLMNAV_REQUIRE_STOP_MODE_FOR_ACTION_STOP',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            action_stop_mode_gate_passed = True
            action_stop_mode_gate_reason = ''
            if action_wants_stop and require_stop_mode_for_action_stop:
                nav_mode = str(step_metadata.get('action_navigation_mode') or '').strip().lower()
                comparison = str(step_metadata.get('action_candidate_comparison') or '').strip().lower()
                movement_candidate_preferred = re.search(
                    r'\bcandidate[_ ]?[1-9]\b.{0,40}\b(best|optimal|superior|preferred|better)\b'
                    r'|\b(best|optimal|superior|preferred)\b.{0,40}\bcandidate[_ ]?[1-9]\b',
                    comparison,
                ) is not None
                if not navigation_mode_allows_stop(nav_mode):
                    action_stop_mode_gate_passed = False
                    action_stop_mode_gate_reason = (
                        f'navigation_mode:{nav_mode}' if nav_mode else 'navigation_mode:missing'
                    )
                elif movement_candidate_preferred:
                    action_stop_mode_gate_passed = False
                    action_stop_mode_gate_reason = 'candidate_comparison_prefers_movement'
                if not action_stop_mode_gate_passed:
                    action_wants_stop = False
                    if step_metadata.get('action_number') == -1:
                        fallback_action = int(os.environ.get('VLMNAV_STOP_GATE_FALLBACK_ACTION', '1'))
                        max_action = len(a_final)
                        if fallback_action < 0 or fallback_action > max_action:
                            fallback_action = 0
                        step_metadata['action_number'] = fallback_action
                        step_metadata['stop_gate_blocked_action'] = -1
            step_metadata['action_stop_mode_gate_required'] = bool(require_stop_mode_for_action_stop)
            step_metadata['action_stop_mode_gate_passed'] = bool(action_stop_mode_gate_passed)
            step_metadata['action_stop_mode_gate_reason'] = action_stop_mode_gate_reason
            step_metadata['action_wants_stop'] = bool(action_wants_stop)
            require_action_stop_default = "0"
            require_action_stop = os.environ.get(
                'VLMNAV_REQUIRE_ACTION_STOP_FOR_STOP',
                require_action_stop_default,
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            disable_stopping_module = os.environ.get(
                'VLMNAV_DISABLE_STOPPING_MODULE',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            try:
                stop_ready_streak_override = int(os.environ.get(
                    'VLMNAV_STOP_READY_STREAK_OVERRIDE',
                    '0',
                ))
            except ValueError:
                stop_ready_streak_override = 0
            try:
                stop_ready_streak_min = int(os.environ.get(
                    'VLMNAV_STOP_READY_STREAK_MIN',
                    '0',
                ))
            except ValueError:
                stop_ready_streak_min = 0
            stop_ready_streak_min = max(0, stop_ready_streak_min)
            stop_signal_ready_for_gate = bool(
                stop_signal_ready
                and (
                    stop_ready_streak_min <= 0
                    or stop_ready_streak >= stop_ready_streak_min
                )
            )
            require_stable_action_near = os.environ.get(
                'VLMNAV_STOP_READY_STREAK_REQUIRE_ACTION_NEAR',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            require_stable_action_stop = os.environ.get(
                'VLMNAV_STOP_READY_STREAK_REQUIRE_ACTION_STOP',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            action_response_text = str(logging_data.get('RESPONSE', '') or '').lower()
            action_near_stop_terms = (
                'close enough',
                'confirmed close',
                'confirmed-close',
                'near-goal',
                'stopping region',
                'stop range',
                'success distance',
                'within success',
                'immediate interaction',
                'already reached',
                'no further navigation',
                'no further approach',
                'no movement is needed',
            )
            action_not_near_terms = (
                'not visible',
                'hidden',
                'suspected',
                'uncertain',
                'ambiguous',
                'far',
                'distant',
                'moderate distance',
                'across the room',
                'not close',
                'not close enough',
                'not yet close enough',
                'not yet at',
                'not confirmed close enough',
                'not confirmed reachable',
                'still needs approach',
                'still needs to approach',
                'requires further approach',
                'requires a small final gap',
                'requires a short final gap',
                'requires approach',
                'requiring approach',
                'must approach',
                'needs further approach',
                'need further approach',
                'continue approaching',
                'continue to approach',
                'final gap before stop',
                'final gap before stopping',
                'before stopping',
                'clear navigable gap',
                'walking gap',
            )
            action_has_not_near = any(term in action_response_text for term in action_not_near_terms)
            action_has_positive_near = (
                any(term in action_response_text for term in action_near_stop_terms)
                and not action_has_not_near
            )
            stable_action_near = (
                action_has_positive_near
            )
            stable_stop_override = (
                stop_ready_streak_override > 0
                and stop_ready_streak >= stop_ready_streak_override
                and (stable_action_near or not require_stable_action_near)
                and (action_wants_stop or not require_stable_action_stop)
            )
            stop_veto_action_far_default = "1" if "goalbalanced_stop_v13" in model_name else ""
            stop_veto_action_far = os.environ.get(
                'VLMNAV_STOP_VETO_ACTION_FAR',
                stop_veto_action_far_default,
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            stop_veto_action_approach_conflict = os.environ.get(
                'VLMNAV_STOP_VETO_ACTION_APPROACH_CONFLICT',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            stop_veto_action_stop_mode_conflict = os.environ.get(
                'VLMNAV_STOP_VETO_ACTION_STOP_MODE_CONFLICT',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            try:
                stop_veto_max_ready_streak = int(os.environ.get(
                    'VLMNAV_STOP_VETO_MAX_READY_STREAK',
                    '0',
                ))
            except ValueError:
                stop_veto_max_ready_streak = 0
            stop_veto_max_ready_streak = max(0, stop_veto_max_ready_streak)
            # Privileged target geometry is never available to runtime gates.
            stop_veto_target_rel_dist_thresh = 0.0
            target_relative_distance_value = float('nan')
            distance_to_goal_value = float('nan')
            step_metadata['privileged_runtime_gates_disabled'] = True
            stop_veto_target_runtime_far = (
                stop_veto_target_rel_dist_thresh > 0
                and not math.isnan(target_relative_distance_value)
                and target_relative_distance_value > stop_veto_target_rel_dist_thresh
            )
            stop_veto_runtime_gate_passed = (
                stop_veto_target_rel_dist_thresh <= 0
                or stop_veto_target_runtime_far
            )
            stop_signal_target_rel_dist_override = 0.0
            stop_signal_runtime_near_override = (
                stop_signal_target_rel_dist_override > 0
                and bool(step_metadata.get('stop_signal', False))
                and not math.isnan(target_relative_distance_value)
                and target_relative_distance_value <= stop_signal_target_rel_dist_override
            )
            stop_veto_runtime_far_direct = False
            try:
                stop_veto_runtime_far_thresh = float(os.environ.get(
                    'VLMNAV_STOP_VETO_RUNTIME_FAR_THRESH',
                    str(stop_veto_target_rel_dist_thresh if stop_veto_target_rel_dist_thresh > 0 else 0.0),
                ))
            except ValueError:
                stop_veto_runtime_far_thresh = 0.0
            try:
                stop_veto_runtime_far_allow_goal_dist = float(os.environ.get(
                    'VLMNAV_STOP_VETO_RUNTIME_FAR_ALLOW_GOAL_DIST',
                    '0',
                ))
            except ValueError:
                stop_veto_runtime_far_allow_goal_dist = 0.0
            try:
                stop_veto_runtime_far_allow_ready_streak = int(os.environ.get(
                    'VLMNAV_STOP_VETO_RUNTIME_FAR_ALLOW_READY_STREAK',
                    '0',
                ))
            except ValueError:
                stop_veto_runtime_far_allow_ready_streak = 0
            stop_veto_runtime_far_allow_ready_streak = max(
                0,
                stop_veto_runtime_far_allow_ready_streak,
            )
            stop_veto_runtime_far_allow_ready_require_action_stop = os.environ.get(
                'VLMNAV_STOP_VETO_RUNTIME_FAR_ALLOW_READY_REQUIRE_ACTION_STOP',
                '1',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            try:
                stop_veto_runtime_far_allow_ready_max_target_rel = float(os.environ.get(
                    'VLMNAV_STOP_VETO_RUNTIME_FAR_ALLOW_READY_MAX_TARGET_REL_DIST',
                    '0',
                ))
            except ValueError:
                stop_veto_runtime_far_allow_ready_max_target_rel = 0.0
            stop_veto_runtime_far_allow_ready_target_rel_ok = (
                stop_veto_runtime_far_allow_ready_max_target_rel <= 0
                or (
                    not math.isnan(target_relative_distance_value)
                    and target_relative_distance_value
                    <= stop_veto_runtime_far_allow_ready_max_target_rel
                )
            )
            stop_veto_runtime_far_goal_near = (
                stop_veto_runtime_far_allow_goal_dist > 0
                and not math.isnan(distance_to_goal_value)
                and distance_to_goal_value <= stop_veto_runtime_far_allow_goal_dist
            )
            stop_veto_runtime_far_ready_near = (
                stop_veto_runtime_far_allow_ready_streak > 0
                and stop_ready_streak >= stop_veto_runtime_far_allow_ready_streak
                and (
                    not stop_veto_runtime_far_allow_ready_require_action_stop
                    or action_wants_stop
                )
                and stop_veto_runtime_far_allow_ready_target_rel_ok
            )
            stop_veto_runtime_far_direct_block = (
                stop_veto_runtime_far_direct
                and stop_veto_runtime_far_thresh > 0
                and not math.isnan(target_relative_distance_value)
                and target_relative_distance_value > stop_veto_runtime_far_thresh
                and not stop_veto_runtime_far_goal_near
                and not stop_veto_runtime_far_ready_near
            )
            stop_binary_override_enabled = os.environ.get(
                'VLMNAV_ACTION_STOP_BINARY_STOP_OVERRIDE',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            try:
                stop_binary_override_max_target_rel = float(os.environ.get(
                    'VLMNAV_ACTION_STOP_BINARY_OVERRIDE_MAX_TARGET_REL_DIST',
                    '0.6',
                ))
            except ValueError:
                stop_binary_override_max_target_rel = 0.6
            stop_binary_override_max_target_rel = max(0.0, stop_binary_override_max_target_rel)
            stop_binary_override_target_ready = (
                stop_binary_override_max_target_rel > 0
                and not math.isnan(target_relative_distance_value)
                and target_relative_distance_value <= stop_binary_override_max_target_rel
            )
            stop_binary_runtime_override_ready = bool(
                stop_binary_override_enabled
                and step_metadata.get('action_stop_binary_wants_stop', False)
                and action_wants_stop
                and stop_binary_override_target_ready
            )
            action_far_or_approach = (
                bool(step_metadata.get('action_number', 0) not in (-1, None))
                and not bool(action_wants_stop)
                and action_has_not_near
                and not action_has_positive_near
                and stop_veto_runtime_gate_passed
            )
            action_approach_conflict = (
                bool(step_metadata.get('action_number', 0) not in (-1, None))
                and not bool(action_wants_stop)
                and action_has_not_near
                and stop_veto_runtime_gate_passed
            )
            action_stop_mode_conflict = (
                bool(action_stop_mode_gate_reason)
                and not bool(action_stop_mode_gate_passed)
                and stop_veto_runtime_gate_passed
            )
            stop_veto_ready_streak_gate_passed = (
                stop_veto_max_ready_streak <= 0
                or stop_ready_streak <= stop_veto_max_ready_streak
            )
            step_metadata['stop_gate_require_action_stop'] = bool(require_action_stop)
            step_metadata['disable_stopping_module'] = bool(disable_stopping_module)
            step_metadata['stop_gate_veto_action_far_enabled'] = bool(stop_veto_action_far)
            step_metadata['stop_gate_veto_action_approach_conflict_enabled'] = bool(
                stop_veto_action_approach_conflict
            )
            step_metadata['stop_gate_veto_action_stop_mode_conflict_enabled'] = bool(
                stop_veto_action_stop_mode_conflict
            )
            step_metadata['stop_gate_veto_target_rel_dist_thresh'] = float(
                stop_veto_target_rel_dist_thresh
            )
            step_metadata['stop_gate_veto_max_ready_streak'] = int(stop_veto_max_ready_streak)
            step_metadata['stop_gate_veto_ready_streak_gate_passed'] = bool(
                stop_veto_ready_streak_gate_passed
            )
            step_metadata['stop_gate_target_relative_distance'] = (
                None
                if math.isnan(target_relative_distance_value)
                else float(target_relative_distance_value)
            )
            step_metadata['stop_gate_veto_target_runtime_far'] = bool(
                stop_veto_target_runtime_far
            )
            step_metadata['stop_signal_target_rel_dist_override'] = float(
                stop_signal_target_rel_dist_override
            )
            step_metadata['stop_signal_runtime_near_override'] = bool(
                stop_signal_runtime_near_override
            )
            step_metadata['stop_gate_veto_runtime_far_enabled'] = bool(
                stop_veto_runtime_far_direct
            )
            step_metadata['stop_gate_veto_runtime_far_thresh'] = float(
                stop_veto_runtime_far_thresh
            )
            step_metadata['stop_gate_veto_runtime_far_allow_goal_dist'] = float(
                stop_veto_runtime_far_allow_goal_dist
            )
            step_metadata['stop_gate_veto_runtime_far_allow_ready_streak'] = int(
                stop_veto_runtime_far_allow_ready_streak
            )
            step_metadata['stop_gate_veto_runtime_far_allow_ready_require_action_stop'] = bool(
                stop_veto_runtime_far_allow_ready_require_action_stop
            )
            step_metadata['stop_gate_veto_runtime_far_allow_ready_max_target_rel_dist'] = float(
                stop_veto_runtime_far_allow_ready_max_target_rel
            )
            step_metadata['stop_gate_veto_runtime_far_allow_ready_target_rel_ok'] = bool(
                stop_veto_runtime_far_allow_ready_target_rel_ok
            )
            step_metadata['stop_gate_veto_runtime_far_distance_to_goal'] = (
                None
                if math.isnan(distance_to_goal_value)
                else float(distance_to_goal_value)
            )
            step_metadata['stop_gate_veto_runtime_far_goal_near'] = bool(
                stop_veto_runtime_far_goal_near
            )
            step_metadata['stop_gate_veto_runtime_far_ready_near'] = bool(
                stop_veto_runtime_far_ready_near
            )
            step_metadata['stop_gate_veto_runtime_far_direct_block'] = bool(
                stop_veto_runtime_far_direct_block
            )
            step_metadata['action_stop_binary_stop_override_enabled'] = bool(
                stop_binary_override_enabled
            )
            step_metadata['action_stop_binary_override_max_target_rel_dist'] = float(
                stop_binary_override_max_target_rel
            )
            step_metadata['action_stop_binary_override_target_ready'] = bool(
                stop_binary_override_target_ready
            )
            step_metadata['action_stop_binary_runtime_override_ready'] = bool(
                stop_binary_runtime_override_ready
            )
            step_metadata['stop_gate_action_far_or_approach'] = bool(action_far_or_approach)
            step_metadata['stop_gate_action_approach_conflict'] = bool(action_approach_conflict)
            step_metadata['stop_gate_action_stop_mode_conflict'] = bool(action_stop_mode_conflict)
            step_metadata['stop_ready_streak_override'] = int(stop_ready_streak_override)
            step_metadata['stop_ready_streak_min'] = int(stop_ready_streak_min)
            step_metadata['stop_signal_ready_for_gate'] = bool(stop_signal_ready_for_gate)
            step_metadata['stop_ready_streak_require_action_near'] = bool(require_stable_action_near)
            step_metadata['stop_ready_streak_require_action_stop'] = bool(require_stable_action_stop)
            step_metadata['stop_gate_stable_action_near'] = bool(stable_action_near)
            step_metadata['stop_gate_stable_override'] = bool(stable_stop_override)
            should_stop = (
                bool(action_wants_stop)
                if disable_stopping_module
                else bool(
                    (
                        stop_signal_ready_for_gate
                        or stop_signal_runtime_near_override
                        or stable_stop_override
                        or stop_binary_runtime_override_ready
                    )
                    and (
                        action_wants_stop
                        or not require_action_stop
                        or stable_stop_override
                    )
                )
            )
            pano_stop_veto_enabled = os.environ.get(
                'VLMNAV_LIGHTWEIGHT_PANORAMIC_STOP_VETO',
                '0',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            pano_stop_conflict = (
                pano_stop_veto_enabled
                and bool(step_metadata.get('lightweight_panoramic_enabled', False))
                and bool(step_metadata.get('lightweight_panoramic_success', False))
                and not bool(action_wants_stop)
                and not bool(stable_action_near)
            )
            step_metadata['lightweight_panoramic_stop_veto_enabled'] = bool(pano_stop_veto_enabled)
            step_metadata['lightweight_panoramic_stop_conflict'] = bool(pano_stop_conflict)
            if should_stop and pano_stop_conflict:
                should_stop = False
                step_metadata['lightweight_panoramic_stop_veto_blocked'] = True
                step_metadata['stop_gate_blocked_action'] = -1
            else:
                step_metadata['lightweight_panoramic_stop_veto_blocked'] = False
            if should_stop and stop_veto_runtime_far_direct_block:
                should_stop = False
                step_metadata['stop_gate_veto_runtime_far_blocked'] = True
                step_metadata['stop_gate_blocked_action'] = -1
            else:
                step_metadata['stop_gate_veto_runtime_far_blocked'] = False
            if (
                should_stop
                and stop_veto_ready_streak_gate_passed
                and stop_veto_action_far
                and action_far_or_approach
            ):
                should_stop = False
                step_metadata['stop_gate_veto_action_far_blocked'] = True
                step_metadata['stop_gate_blocked_action'] = -1
            else:
                step_metadata['stop_gate_veto_action_far_blocked'] = False
            if (
                should_stop
                and stop_veto_ready_streak_gate_passed
                and stop_veto_action_approach_conflict
                and action_approach_conflict
            ):
                should_stop = False
                step_metadata['stop_gate_veto_action_approach_conflict_blocked'] = True
                step_metadata['stop_gate_blocked_action'] = -1
            else:
                step_metadata['stop_gate_veto_action_approach_conflict_blocked'] = False
            if (
                should_stop
                and stop_veto_ready_streak_gate_passed
                and stop_veto_action_stop_mode_conflict
                and action_stop_mode_conflict
            ):
                should_stop = False
                step_metadata['stop_gate_veto_action_stop_mode_conflict_blocked'] = True
                step_metadata['stop_gate_blocked_action'] = -1
            else:
                step_metadata['stop_gate_veto_action_stop_mode_conflict_blocked'] = False
            step_metadata['stop_gate_passed'] = bool(should_stop)
            if should_stop:
                step_metadata['action_number'] = -1
                agent_action = PolarAction.stop
            elif step_metadata.get('action_number') == -1:
                fallback_action = int(os.environ.get('VLMNAV_STOP_GATE_FALLBACK_ACTION', '1'))
                max_action = len(a_final)
                if fallback_action < 0 or fallback_action > max_action:
                    fallback_action = 0
                step_metadata['stop_gate_blocked_action'] = -1
                step_metadata['action_number'] = fallback_action
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))
            else:
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'candidate_action_scores': list(self.last_candidate_action_scores),
            'images': images
        }
        return agent_action, metadata

    def _construct_terminal_certificate_prompt(self, goal) -> str:
        """Return the fixed public input surface for the visual arrival model.

        This stays separate from ``_construct_prompt(..., 'stopping')`` so
        train-only rollout collection can serialize the exact future
        certificate prompt without changing the frozen policy's legacy Stop
        behavior on incidental action ``-1`` proposals.
        """
        protocol = self._terminal_certificate_protocol_id()
        if protocol == 'objectnav_state_conditioned_causal_stop_bottleneck_v3':
            return (
                "Protocol: objectnav_state_conditioned_causal_stop_bottleneck_v3\n\n"
                f"Navigate to the nearest {goal}. Use only this current egocentric RGB image.\n"
                "A separate action policy proposes movement. This module produces spatial conclusions only; deterministic runtime logic decides whether to Stop.\n\n"
                "First route the current visual state:\n"
                "- SEARCH: the target is absent, ambiguous, or not reliably confirmed; activate target_visibility only.\n"
                "- TARGET_ENGAGED: the target may be visually confirmed; activate target_visibility and viewpoint_proximity. This same state covers both approach-needed and stop-ready cases, so the state itself never determines Stop.\n\n"
                "Return exactly one compact JSON object with these top-level fields and no prose:\n"
                '{"navigation_state":"SEARCH_or_TARGET_ENGAGED","ability":["state_allowed_abilities"],"spatial_reasoning":{"ability_name":{"visual_observation":"specific visible spatial evidence","conclusion":"allowed categorical spatial relation"}}}\n\n'
                "Categorical conclusions:\n"
                "- target_visibility: CONFIRMED or NOT_CONFIRMED.\n"
                "- viewpoint_proximity: ADDITIONAL_APPROACH_REQUIRED or NEAR_CURRENT_VIEWPOINT.\n\n"
                "Hard constraints:\n"
                "- SEARCH uses exactly [target_visibility] with conclusion=NOT_CONFIRMED.\n"
                "- TARGET_ENGAGED uses exactly [target_visibility, viewpoint_proximity] with target_visibility conclusion=CONFIRMED; viewpoint_proximity independently distinguishes ADDITIONAL_APPROACH_REQUIRED from NEAR_CURRENT_VIEWPOINT.\n"
                "- Every visual_observation must state concrete visible spatial evidence from this image rather than repeat the categorical conclusion.\n"
                "- Do not output Stop, Continue, done, should_stop, an action, a score, or a decision field.\n"
                "- Do not use pose, map, simulator distance, semantic masks, target coordinates, oracle geometry, or private metadata."
            )
        if protocol == 'objectnav_state_conditioned_evidence_stop_v2':
            return (
                "Protocol: objectnav_state_conditioned_evidence_stop_v2\n\n"
                f"Navigate to the nearest {goal}. Use only this current egocentric RGB image.\n"
                "A separate action policy proposes movement; this module decides only whether to Stop.\n\n"
                "First select the visual navigation state, which activates only the evidence needed now:\n"
                "- SEARCH: activate target_visibility only; the target is absent, ambiguous, or not reliably confirmed.\n"
                "- APPROACH: activate target_visibility and viewpoint_proximity; the target is confirmed but visible space, depth ordering, scale, cropping, or an intervening structure shows that approach is still needed.\n"
                "- ARRIVE_VERIFY: activate target_visibility and viewpoint_proximity; the target is confirmed and the current viewpoint is already visually close enough to finish.\n\n"
                "Return exactly one compact JSON object with these top-level fields and no prose:\n"
                '{"navigation_state":"SEARCH_or_APPROACH_or_ARRIVE_VERIFY",'
                '"ability":["state_allowed_abilities"],'
                '"spatial_reasoning":{"ability_name":"one short image-grounded judgment"},'
                '"decision_bridge":"one concise sentence linking the activated evidence to Continue or Stop",'
                '"stop":false_or_true}\n\n'
                "Hard constraints:\n"
                "- SEARCH uses exactly [target_visibility] and stop=false.\n"
                "- APPROACH uses exactly [target_visibility, viewpoint_proximity] and stop=false.\n"
                "- ARRIVE_VERIFY uses exactly [target_visibility, viewpoint_proximity] and stop=true.\n"
                "- spatial_reasoning keys equal ability exactly and every judgment must cite visible evidence from this image.\n"
                "- Do not use pose, map, simulator distance, semantic masks, target coordinates, oracle geometry, or private metadata.\n"
                "- No candidate actions, scores, long chain of thought, markdown, or extra fields."
            )
        if protocol == 'objectnav_state_conditioned_spatial_stop_v1':
            return (
                "Protocol: objectnav_state_conditioned_spatial_stop_v1\n\n"
                f"Navigate to the nearest {goal}. Use only this current egocentric RGB image; "
                "do not use pose, map, distance, semantic masks, simulator state, or room priors alone.\n\n"
                "First select the visual navigation state. The state determines which spatial "
                "evidence is allowed to drive Stop:\n"
                "- SEARCH: judge target identity only; the target is not visually confirmed.\n"
                "- APPROACH: the target identity is confirmed, then judge that translation or "
                "alignment is still needed.\n"
                "- ARRIVE_VERIFY: the target identity is confirmed, then verify a visually close "
                "final viewpoint from scale, occlusion, intervening space, and depth ordering.\n\n"
                "Return exactly one compact JSON object matching one of these schemas and no prose:\n"
                '{"navigation_state":"SEARCH","identity_evidence":"NOT_CONFIRMED","stop":0}\n'
                '{"navigation_state":"APPROACH","identity_evidence":"CONFIRMED",'
                '"proximity_evidence":"APPROACH_NEEDED","stop":0}\n'
                '{"navigation_state":"ARRIVE_VERIFY","identity_evidence":"CONFIRMED",'
                '"proximity_evidence":"STOP_READY","stop":1}\n'
                "Stop is permitted only by the complete ARRIVE_VERIFY evidence chain."
            )
        if protocol == 'objectnav_visual_arrival_certificate_v2':
            return (
                "Protocol: objectnav_visual_arrival_certificate_v2\n\n"
                f"The agent is navigating to the nearest {goal}. Use only this current "
                "egocentric RGB image. Do not use room priors alone, pose, a map, "
                "distance, or any hidden signal.\n\n"
                "Issue an arrival certificate only when the correct target is visibly "
                "confirmed and the current camera pose itself appears to be a valid close "
                "final viewpoint. Require direct visual evidence of identity and arrival: "
                "target appearance, scale/cropping, visible free space, depth ordering, "
                "and no meaningful remaining translation or turn. Do not certify if target "
                "is absent, uncertain, partly guessed from room context, visible but still "
                "far, occluded, behind a threshold, or if another move/alignment is still needed.\n\n"
                "Return exactly one compact JSON object and no prose:\n"
                '{"arrival_certified":0_or_1}\n'
                "Set arrival_certified=1 only for a visually grounded arrival certificate. "
                "Default to 0 whenever uncertain."
            )
        if protocol != 'objectnav_terminal_visual_certificate_v1':
            raise ValueError(
                'Unsupported terminal certificate protocol: ' + repr(protocol)
            )
        return (
            "Protocol: objectnav_terminal_visual_certificate_v1\n\n"
            f"The agent is navigating to the nearest {goal}. Use only this current "
            "egocentric RGB image. Do not use room priors alone, pose, a map, "
            "or any hidden distance signal.\n\n"
            "First classify the current navigation state using visual spatial evidence:\n"
            "- search: the target is absent or not visually confirmed;\n"
            "- approach: the correct target is visible but additional translation or alignment is needed;\n"
            "- terminal: the correct target is visibly confirmed and this camera pose already appears close enough to finish.\n\n"
            "A terminal certificate requires both target identity/visibility and visual "
            "proximity evidence: target scale, occlusion, free traversable space, depth "
            "ordering, and intervening furniture or thresholds. Visible alone is never enough.\n\n"
            "Return exactly one compact JSON object and no prose:\n"
            '{"terminal_state":"search_or_approach_or_terminal",'
            '"target_visible":0_or_1,"near_enough":0_or_1,"done":0_or_1}\n'
            "Set done=1 if and only if terminal_state=terminal, target_visible=1, and near_enough=1."
        )

    def _construct_prompt(
        self,
        goal: str,
        prompt_type:str,
        num_actions: int=0,
        candidate_features=None,
        memory: str = "",
        spatial_state: str = "",
        protocol_id_override: str = "",
        temporal_reference_available: bool = False,
    ):
        structured_spatial_prompt = os.environ.get('VLMNAV_STRUCTURED_SPATIAL_PROMPT', '').strip().lower() in (
            '1',
            'true',
            'yes',
            'y',
            'on',
        )
        protocol_id = (protocol_id_override or os.environ.get('VLMNAV_PROTOCOL_ID', '')).strip()
        if not protocol_id:
            model_name = str(getattr(self.actionVLM, "name", "")).lower()
            if "goalbalanced_stop_v13" in model_name:
                protocol_id = 'objectnav_cot_v2_abilitymode'
        protocol = None
        if protocol_id:
            try:
                protocols = _load_objectnav_cot_protocol()
                protocol = protocols.get(protocol_id)
            except Exception as exc:
                logging.warning(
                    "Could not load VLMNAV_PROTOCOL_ID=%s protocol; falling back to existing prompt: %s",
                    protocol_id,
                    exc,
                )

        def _append_original_prompt_length_limit(prompt: str, prompt_type: str) -> str:
            enabled = os.environ.get(
                "VLMNAV_ORIGINAL_PROMPT_LENGTH_LIMIT",
                "",
            ).strip().lower() in ("1", "true", "yes", "y", "on")
            if not enabled:
                return prompt
            if prompt_type == "stopping":
                raw_words = os.environ.get("VLMNAV_STOP_PROMPT_MAX_WORDS", "50")
            else:
                raw_words = os.environ.get("VLMNAV_ACTION_PROMPT_MAX_WORDS", "80")
            try:
                max_words = int(raw_words)
            except ValueError:
                max_words = 0
            if max_words <= 0:
                return prompt
            return prompt + f" Keep the response concise: no more than {max_words} words total."

        def _append_same_pose_stop_crosscheck(prompt: str) -> str:
            if (
                self._action_first_stop_confirmation_enabled()
                and self._action_first_append_crosscheck_prompt()
            ):
                return prompt + blind_stop_crosscheck_prompt_suffix(goal)
            if not (
                (
                    self._same_pose_stop_crosscheck_enabled()
                    or self._blind_same_pose_stop_crosscheck_enabled()
                )
                and self._current_stop_signal()
            ):
                return prompt
            if self._blind_same_pose_stop_crosscheck_enabled():
                return prompt + blind_stop_crosscheck_prompt_suffix(goal)
            return (
                prompt
                + "\n\nSame-position stop cross-check:\n"
                + f"- An independent stopping verifier flagged that the {goal} may already be close enough, but this signal can be wrong.\n"
                + "- Independently inspect the current image. Choose action -1 to STOP NOW only if the actual goal is clearly visible and the current camera pose already appears close enough that no translation is needed.\n"
                + "- If the goal is only visible at approach distance, uncertain, partly blocked, or still has traversable space before it, choose the best numbered short red-arrow action instead.\n"
                + "- Return the normal action object, using {'action': -1} only for an independently confirmed immediate stop."
            )

        def _append_explicit_target_visibility_prompt(prompt: str) -> str:
            enabled = os.environ.get(
                "VLMNAV_EXPLICIT_TARGET_VISIBILITY_PROMPT",
                "",
            ).strip().lower() in ("1", "true", "yes", "y", "on")
            if not enabled:
                return prompt
            return (
                prompt
                + "\n\nBefore the final action dict, write one compact target_visibility spatial-reasoning JSON using the same field names as the SFT schema:\n"
                + "{\"ability\":[\"target_visibility\"],\"spatial_reasoning\":{\"target_visibility\":{\"claim\":\"...\",\"visual_evidence\":\"...\",\"navigation_relevance\":\"...\",\"evidence_limit\":\"...\"}}}\n"
                + "In this JSON, use only the current image and the goal. visual_evidence must cite concrete visible objects, regions, occlusions, openings, free space, or viewpoint relations. "
                + "Phrase navigation_relevance as one concrete information-value sentence tied to target_visibility and this image. "
                + "Keep each field concise, and then return the original action dict {'action': <action_key>}."
            )

        def _append_spatial_cot_v2_action_prompt(prompt: str) -> str:
            enabled = os.environ.get(
                "VLMNAV_SPATIAL_COT_V2_ACTION_PROMPT",
                "",
            ).strip().lower() in ("1", "true", "yes", "y", "on")
            if not enabled:
                return prompt
            return (
                prompt
                + "\n\nBefore the final action dict, write one decision-free spatial reasoning JSON for ObjectNav. "
                "The purpose is to reason about the current image and goal so the next navigation action is better grounded.\n"
                "Use one or more abilities from exactly this set: "
                "target_visibility, spatial_relation, local_affordance, self_orientation, semantic_layout_topology, metric_distance.\n"
                "Return this JSON shape:\n"
                "{\n"
                '  "ability": ["one_or_more_abilities"],\n'
                '  "spatial_reasoning": {\n'
                '    "<ability_name>": {\n'
                '      "claim": "one concrete spatial claim grounded in the current image",\n'
                '      "visual_evidence": "specific visible evidence such as objects, openings, free space, occlusion, layout, viewpoint, or distance cues",\n'
                '      "navigation_relevance": "why this spatial fact matters for moving toward the goal",\n'
                '      "evidence_limit": "what remains uncertain from this single view"\n'
                "    }\n"
                "  }\n"
                "}\n"
                "Do not include decision, reason, candidate_comparison, stop, done, action, action_id, or candidate ranking inside the spatial reasoning JSON. "
                "After the JSON, output exactly one final executable action dict using the original VLMNav format: {'action': <action_key>}."
            )

        def _finalize_multifloor_action_prompt(prompt: str) -> str:
            if hasattr(self, "multifloor") and self.multifloor is not None:
                if self.multifloor.enabled or multifloor_enabled():
                    self.multifloor.enabled = multifloor_enabled()
                    if self.multifloor.enabled:
                        prompt = self.multifloor.strip_no_stairs_clause(prompt)
                        suffix = self.multifloor.prompt_suffix()
                        if suffix:
                            prompt = prompt + suffix
            return self._append_gym_move_prior(prompt, goal)

        if prompt_type == 'stopping':
            if self._terminal_certificate_prompt_enabled():
                return self._construct_terminal_certificate_prompt(goal)
            force_original_stop_prompt = os.environ.get(
                "VLMNAV_FORCE_ORIGINAL_STOP_PROMPT",
                "",
            ).strip().lower() in ("1", "true", "yes", "y", "on")
            if force_original_stop_prompt:
                stopping_prompt = (
                    f"The agent has has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                    f"Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note a chair is NOT sofa which is NOT a bed. "
                    f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. Format your answer in the json {{'done': <1 or 0>}}"
                )
                return _append_original_prompt_length_limit(stopping_prompt, "stopping")

            stop_context = getattr(self, "_stopping_prompt_context", None)
            singleview_stopcot_prompt = os.environ.get(
                "VLMNAV_SINGLEVIEW_STOPCOT_PROMPT",
                "",
            ).strip().lower() in ("1", "true", "yes", "y", "on")
            singleview_stopboundary_done_prompt = os.environ.get(
                "VLMNAV_SINGLEVIEW_STOPBOUNDARY_DONE_PROMPT",
                "",
            ).strip().lower() in ("1", "true", "yes", "y", "on")
            if singleview_stopboundary_done_prompt:
                stopping_prompt = (
                    "Protocol: objectnav_direct_stop_boundary_v1\n\n"
                    f"You are the ObjectNav stopping verifier. The agent is navigating to the nearest {goal}.\n"
                    "Use only the current single egocentric image to judge whether the current camera position is visually consistent with being sufficiently close to a valid viewpoint for the goal object.\n"
                    "Do not estimate metric distance from the camera to the object. The stopping condition concerns proximity to a valid goal viewpoint, not proximity to the object's surface or center.\n"
                    "Use coarse visual evidence such as target visibility, target scale or cropping, remaining traversable space, depth ordering, and intervening structures. No single cue, including apparent object size, is sufficient by itself.\n\n"
                    "Stop decision policy:\n"
                    f"- Default to done=0. Use done=1 only when the {goal} is visually confirmed and appears close enough to stop from the current camera position.\n"
                    f"- done=0 when the {goal} is absent, uncertain, a different object, reflected/picture-like, inferred only from room context, or visible but still not close enough.\n"
                    "- Treat done=1 as requiring both reliable target confirmation and a viewpoint that already appears close enough to finish. If the target is merely visible but there is still clear traversable space, a doorway threshold, a counter, large furniture, glass, or another intervening structure between the camera and a reasonable viewing position, prefer done=0.\n"
                    "- Large, centered, or salient in the image is not enough. The key question is whether this exact position already looks close enough for a reliable stop.\n"
                    "- A stop target does not need perfect centering, a frontal view, or a nicer alignment step. However, continue when the image clearly indicates that substantial navigation toward a valid viewing position remains.\n"
                    "- If one more movement or alignment step still seems necessary, output done=0.\n\n"
                    "Return exactly one compact JSON object and nothing else:\n"
                    '{"done":0_or_1}'
                )
                return stopping_prompt
            if singleview_stopcot_prompt:
                stopping_prompt = (
                    "Protocol: objectnav_cot_v1_stopping_cot_visibility_distance_v1\n\n"
                    f"You are the ObjectNav stopping verifier. The agent is navigating to the nearest {goal}.\n"
                    "Use only the current single egocentric image to judge whether the current camera position is visually consistent with being sufficiently close to a valid viewpoint for the goal object.\n"
                    "Do not estimate metric distance from the camera to the object. The stopping condition concerns proximity to a valid goal viewpoint, not proximity to the object's surface or center.\n"
                    "Use coarse visual evidence such as target visibility, target scale or cropping, remaining traversable space, depth ordering, and intervening structures. No single cue, including apparent object size, is sufficient by itself.\n"
                    "Focus the reasoning only on target visibility and target distance. Do not plan navigation actions.\n\n"
                    "Stop decision policy:\n"
                    f"- Default to stop=false. Use stop=true only when the {goal} is visually confirmed and appears close enough to stop from the current camera position.\n"
                    f"- stop=false when the {goal} is absent, uncertain, a different object, reflected/picture-like, inferred only from room context, or visible but still not close enough.\n"
                    "- Treat stop=true as requiring both reliable target confirmation and a viewpoint that already appears close enough to finish. If the target is merely visible but there is still clear traversable space, a doorway threshold, a counter, large furniture, glass, or another intervening structure between the camera and a reasonable viewing position, prefer stop=false.\n"
                    "- Large, centered, or salient in the image is not enough. The key question is whether this exact position already looks close enough for a reliable stop.\n"
                    "- A stop target does not need perfect centering, a frontal view, or a nicer alignment step. However, continue when the image clearly indicates that substantial navigation toward a valid viewing position remains.\n"
                    "- If one more movement or alignment step still seems necessary, output stop=false.\n\n"
                    "Return exactly one compact JSON object:\n"
                    "{\n"
                    '  "spatial_reasoning": {\n'
                    f'    "target_visibility": "one concise free-form judgment about whether the {goal} is visually confirmed in the current image",\n'
                    f'    "distance_measurement": "one concise free-form judgment about whether the confirmed {goal} appears close enough to stop or still not close enough"\n'
                    "  },\n"
                    '  "reason": "one short sentence linking the visual evidence to the stop decision",\n'
                    '  "stop": true_or_false\n'
                    "}"
                )
                return stopping_prompt
            multiview_prompt_enabled = os.environ.get(
                "VLMNAV_STOP_MULTIVIEW_PROMPT",
                "1",
            ).strip().lower() in ("1", "true", "yes", "y", "on")
            if (
                multiview_prompt_enabled
                and isinstance(stop_context, dict)
                and stop_context.get("format") == "standard_multiview_contact_sheet"
            ):
                display_angles = stop_context.get("display_angles_deg") or []
                angle_text = ", ".join(f"+{int(angle) % 360:03d}" for angle in display_angles)
                rows = int(stop_context.get("layout_rows") or 1)
                cols = int(stop_context.get("layout_cols") or len(display_angles) or 1)
                tile_gap = int(stop_context.get("tile_gap") or 0)
                multiview_text = (
                    f"The image is a clean {rows}-by-{cols} multi-view contact sheet from the same current location"
                )
                if tile_gap > 0:
                    multiview_text += "; black dividers separate neighboring views"
                if angle_text:
                    multiview_text += (
                        f", and left to right the yaw offsets are {angle_text} degrees"
                    )
                multiview_text += ". "
                stopping_prompt = (
                    f"The agent has has been tasked with navigating to a {goal.upper()}. "
                    "The agent has sent you an image taken from its current location. "
                    f"{multiview_text}"
                    f"Your job is to determine whether the agent is VERY CLOSE to a {goal}. "
                    "The panorama is only for checking whether the target is reachable from the current position, not for stopping whenever the target appears in any view. "
                    "Return 1 only if the target appears immediately nearby from the agent's current position; if it is across the room, small, behind furniture, or just a background sighting in any view, return 0. "
                    "Note a chair is NOT sofa which is NOT a bed. "
                    f"First, tell me what you see in the image, and tell me if there is a {goal}. "
                    f"Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, "
                    f"Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. "
                    "Format your answer in the json {'done': <1 or 0>}"
                )
                return stopping_prompt
            cot_stopping_prompt = os.environ.get(
                'VLMNAV_COT_STOPPING_PROMPT',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            if protocol is not None and cot_stopping_prompt:
                return protocol[1](goal=goal)
            strict_stopping_prompt = os.environ.get(
                'VLMNAV_STRICT_STOPPING_PROMPT',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            if strict_stopping_prompt:
                stopping_prompt = (
                    f"The agent has been tasked with navigating to a {goal.upper()}. "
                    "The image is from the agent's current location. "
                    f"Return 1 only if an actual {goal} is clearly confirmed and close enough that no further movement is needed. "
                    f"Return 0 if there is no {goal}, if it is only suspected, if it is in the background or across the room, "
                    "if it is behind a door/window/counter/furniture, or if the agent still needs to move closer. "
                    "Do not stop just because the target is visible. "
                    f"Note a chair is NOT sofa which is NOT a bed. Format your answer exactly as JSON {{'done': <1 or 0>}}."
                )
                return stopping_prompt
            stopping_prompt = (f"The agent has has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
            f'Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note a chair is NOT sofa which is NOT a bed. '
            f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. Format your answer in the json {{'done': <1 or 0>}}")
            return _append_original_prompt_length_limit(stopping_prompt, "stopping")
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return _finalize_multifloor_action_prompt(baseline_prompt)
        if prompt_type == 'pivot':
            pivot_prompt = f"NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            return pivot_prompt
        if prompt_type == 'action':
            wmnav_lite_route_prior = os.environ.get(
                'VLMNAV_WMNAV_LITE_ROUTE_PRIOR',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            wmnav_lite_candidate0_prompt = os.environ.get(
                'VLMNAV_WMNAV_LITE_CANDIDATE0_PROMPT',
                os.environ.get('VLMNAV_WMNAV_LITE_CANDIDATE0_VETO', ''),
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            wmnav_lite_visible_approach_prior = os.environ.get(
                'VLMNAV_WMNAV_LITE_VISIBLE_APPROACH_PRIOR',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            wmnav_lite_short_path_prior = os.environ.get(
                'VLMNAV_WMNAV_LITE_SHORT_PATH_PRIOR',
                '',
            ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
            if protocol is not None:
                candidate_lines = [
                    "- candidate_-1: executable action -1; stop only when the goal is localized in the immediate foreground and within the success distance.",
                    "- candidate_0: executable action 0; turn around or reorient when no forward red-arrow candidate is reliable."
                ]
                valid_actions = [-1, 0]
                if candidate_features:
                    candidate_lines.extend(candidate_features)
                else:
                    for idx in range(1, max(int(num_actions), 0) + 1):
                        candidate_lines.append(
                            f"- candidate_{idx}: executable action {idx}; visible red-arrow label {idx}."
                        )
                for idx in range(1, max(int(num_actions), 0) + 1):
                    valid_actions.append(idx)
                prompt = protocol[0](
                    goal=goal,
                    candidate_lines=candidate_lines,
                    valid_actions=valid_actions,
                    memory=memory or "",
                    context_name="Visible action space",
                    spatial_state=spatial_state,
                    temporal_reference_available=bool(temporal_reference_available),
                )
                if wmnav_lite_route_prior:
                    if candidate_features:
                        prompt += (
                            "\n\nWMNav-lite single-call route prior:\n"
                            "- This is not a separate planner. Use only the current image and candidate metadata.\n"
                            "- Use distance_m, heading_deg, frontier_like, reliable_forward, route_role, route_role_hint, route_prior_score, and explored_overlap_score only as local route hints.\n"
                            "- If the target is hidden or uncertain, prefer a reliable forward candidate with route_role=frontier_transition_option or high route_prior_score that moves through open space, a hallway, doorway, frontier, or new room.\n"
                            "- Prefer low explored_overlap_score when searching because it indicates less checked nearby map space.\n"
                            "- Do not chase route_prior_score if the image clearly shows that the arrow is blocked, goes into irrelevant clutter, or conflicts with a visible target.\n"
                        )
                    else:
                        prompt += (
                            "\n\nWMNav-lite single-call route prior:\n"
                            "- This is not a separate planner. Use only the current image, valid red-arrow candidates, free-space layout, and target visibility.\n"
                            "- If the target is hidden or uncertain, prefer a forward candidate that enters promising topology: open doorway, hallway, room transition, visible opening, or unexplored-looking frontier.\n"
                            "- Prefer meaningful transitions into connected areas over circling within already checked or irrelevant space.\n"
                            "- Do not choose an arrow that is visibly blocked, points into clutter, or conflicts with a clearly visible target approach.\n"
                        )
                if wmnav_lite_visible_approach_prior:
                    if candidate_features:
                        prompt += (
                            "\n\nWMNav-lite visible-target approach prior:\n"
                            "- If the target is clearly visible but not in the immediate stop region, choose the reliable_forward candidate that most directly approaches it through free space.\n"
                            "- For visible target approach, direct_route_option and small |heading_deg| usually beat frontier exploration.\n"
                            "- Do not detour into a new room/frontier when the current image already gives a safe direct approach to the visible target.\n"
                        )
                    else:
                        prompt += (
                            "\n\nWMNav-lite visible-target approach prior:\n"
                            "- Use this prior only when the target is clearly visible in the current image but still not close enough to stop.\n"
                            "- Choose the red-arrow candidate whose endpoint visually moves most directly toward the target through open free space.\n"
                            "- Do not turn away, enter a new room, or choose candidate_0 when a visible red arrow can safely approach the target.\n"
                            "- If the target is hidden, uncertain, or only suspected, ignore this prior and follow the normal search policy.\n"
                        )
                if wmnav_lite_candidate0_prompt:
                    prompt += (
                        "\n\nWMNav-lite candidate_0 rule:\n"
                        "- candidate_0 is recovery only. If any reliable_forward candidate exists and is not visibly blocked, do not choose candidate_0.\n"
                        "- Use candidate_0 only for blocked views, no reliable red-arrow movement, strong heading confusion, or route recovery after bad progress.\n"
                    )
                if wmnav_lite_short_path_prior:
                    prompt += (
                        "\n\nWMNav-lite short-path prior:\n"
                        "- Prefer the shortest likely successful route, not broad exploration.\n"
                        "- If the target is visible, approach it directly through free space or stop only when it is immediate foreground and within success distance.\n"
                        "- If the target is hidden, choose the open forward candidate with the clearest progress into useful connected space; avoid side detours, local circling, and checking minor clutter unless all forward options are poor.\n"
                        "- Do not enter a new room or frontier when a current-room candidate makes clearer goal-directed progress.\n"
                    )
                return _finalize_multifloor_action_prompt(
                    _append_same_pose_stop_crosscheck(prompt)
                )
            if structured_spatial_prompt:
                candidate_lines = [
                    "- candidate_-1: executable action -1; stop only when the goal is localized in the immediate foreground and within the success distance.",
                    "- candidate_0: executable action 0; turn around or reorient when no forward red-arrow candidate is reliable."
                ]
                if candidate_features:
                    candidate_lines.extend(candidate_features)
                else:
                    for idx in range(1, max(int(num_actions), 0) + 1):
                        candidate_lines.append(
                            f"- candidate_{idx}: executable action {idx}; visible red-arrow label {idx}."
                        )
                action_prompt = (
                    "You are doing navigation-relevant spatial reasoning for ObjectNav from an egocentric observation image.\n\n"
                    f"Goal category: {goal}\n\n"
                    "Visible action space:\n"
                    + "\n".join(candidate_lines)
                    + "\n\nUse the red-arrow numbers in the image. In the structured JSON, write the chosen action as a concrete candidate string from the visible action space, such as candidate_1 or candidate_-1. "
                    "After the JSON, output the executable VLMnav action dict with the same raw number, for example {'action': 1} or {'action': -1}.\n\n"
                    "Six spatial reasoning abilities to use:\n"
                    "- target_visibility: decide whether the goal is visible and whether visibility should dominate the next move\n"
                    "- metric_property_estimation: judge near/far distance, stop boundary, and expected progress\n"
                    "- route_planning: choose the locally executable action that best advances the navigation plan\n"
                    "- scene_topological_reasoning: reason about rooms, openings, frontiers, and transitions between regions\n"
                    "- cross_view_reasoning: compare current observation with recent spatial memory if available to avoid revisits and recover progress\n"
                    "- egocentric_allocentric_transformation: map red-arrow egocentric directions to persistent route or branch choices\n\n"
                    "Return a structured JSON object with exactly these top-level keys: ability, evidence, state, spatial_reasoning, reason, decision.\n"
                    "- ability: a list of one or more abilities from the list above.\n"
                    "- evidence: short grounded strings about target visibility, distance/range, candidate arrows, openings/frontiers, and uncertainty.\n"
                    "- state: near_goal_stop, topology_correction, target_guided_approach, frontier_transition, or frontier_exploration.\n"
                    "- spatial_reasoning: an object with short fields target_relation, scene_relation, spatial_inference, navigation_use.\n"
                    "- decision: an object with action as a concrete valid candidate string, mode as approach/explore/replan/stop, and should_stop as false for normal movement. Choose candidate_-1 and should_stop true only when the goal is immediate foreground/within reach; if the goal is visible but still needs approach, choose a movement candidate and should_stop false.\n\n"
                    "Do not use free-form prose outside the JSON except the final executable action dict. "
                    "The final line must be an executable action dict using a concrete raw number; use raw number -1 only when decision.action is candidate_-1."
                )
                if wmnav_lite_route_prior:
                    if candidate_features:
                        action_prompt += (
                            "\n\nWMNav-lite single-call route prior:\n"
                            "- This is not a separate planner. Use only the current image and candidate metadata.\n"
                            "- Use distance_m, heading_deg, frontier_like, reliable_forward, route_role, route_role_hint, route_prior_score, and explored_overlap_score only as local route hints.\n"
                            "- If the target is hidden or uncertain, prefer a reliable forward candidate with route_role=frontier_transition_option or high route_prior_score that moves through open space, a hallway, doorway, frontier, or new room.\n"
                            "- Prefer low explored_overlap_score when searching because it indicates less checked nearby map space.\n"
                            "- Do not chase route_prior_score if the image clearly shows that the arrow is blocked, goes into irrelevant clutter, or conflicts with a visible target.\n"
                        )
                    else:
                        action_prompt += (
                            "\n\nWMNav-lite single-call route prior:\n"
                            "- This is not a separate planner. Use only the current image, valid red-arrow candidates, free-space layout, and target visibility.\n"
                            "- If the target is hidden or uncertain, prefer a forward candidate that enters promising topology: open doorway, hallway, room transition, visible opening, or unexplored-looking frontier.\n"
                            "- Prefer meaningful transitions into connected areas over circling within already checked or irrelevant space.\n"
                            "- Do not choose an arrow that is visibly blocked, points into clutter, or conflicts with a clearly visible target approach.\n"
                        )
                if wmnav_lite_visible_approach_prior:
                    if candidate_features:
                        action_prompt += (
                            "\n\nWMNav-lite visible-target approach prior:\n"
                            "- If the target is clearly visible but not in the immediate stop region, choose the reliable_forward candidate that most directly approaches it through free space.\n"
                            "- For visible target approach, direct_route_option and small |heading_deg| usually beat frontier exploration.\n"
                            "- Do not detour into a new room/frontier when the current image already gives a safe direct approach to the visible target.\n"
                        )
                    else:
                        action_prompt += (
                            "\n\nWMNav-lite visible-target approach prior:\n"
                            "- Use this prior only when the target is clearly visible in the current image but still not close enough to stop.\n"
                            "- Choose the red-arrow candidate whose endpoint visually moves most directly toward the target through open free space.\n"
                            "- Do not turn away, enter a new room, or choose candidate_0 when a visible red arrow can safely approach the target.\n"
                            "- If the target is hidden, uncertain, or only suspected, ignore this prior and follow the normal search policy.\n"
                        )
                if wmnav_lite_candidate0_prompt:
                    action_prompt += (
                        "\n\nWMNav-lite candidate_0 rule:\n"
                        "- candidate_0 is recovery only. If any reliable_forward candidate exists and is not visibly blocked, do not choose candidate_0.\n"
                        "- Use candidate_0 only for blocked views, no reliable red-arrow movement, strong heading confusion, or route recovery after bad progress.\n"
                    )
                if wmnav_lite_short_path_prior:
                    action_prompt += (
                        "\n\nWMNav-lite short-path prior:\n"
                        "- Prefer the shortest likely successful route, not broad exploration.\n"
                        "- If the target is visible, approach it directly through free space or stop only when it is immediate foreground and within success distance.\n"
                        "- If the target is hidden, choose the open forward candidate with the clearest progress into useful connected space; avoid side detours, local circling, and checking minor clutter unless all forward options are poor.\n"
                        "- Do not enter a new room or frontier when a current-room candidate makes clearer goal-directed progress.\n"
                    )
                return _finalize_multifloor_action_prompt(
                    _append_same_pose_stop_crosscheck(action_prompt)
                )
            action_prompt = (
            f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
            f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
            f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
            f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
            )
            action_prompt = _append_explicit_target_visibility_prompt(action_prompt)
            action_prompt = _append_spatial_cot_v2_action_prompt(action_prompt)
            action_prompt = _append_same_pose_stop_crosscheck(action_prompt)
            return _finalize_multifloor_action_prompt(
                _append_original_prompt_length_limit(action_prompt, "action")
            )

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')
