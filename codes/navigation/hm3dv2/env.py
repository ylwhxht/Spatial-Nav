import gzip
import ast
import json
import logging
import math
import os
import re
import random
import random
import sys
import time
import requests
import traceback
import habitat_sim

import pandas as pd
import numpy as np
from habitat_sim.utils.common import quat_to_angle_axis

from PIL import Image
from simWrapper import PolarAction, SimWrapper
from agent import *
from utils import *
from persistent_episode_pool import PersistentEpisodePool
from navigation_runtime import (
    candidate_angle_metadata,
    candidate_model_displacement,
    compass_habitat_step_cost,
    strict_habitat_step_horizon_decision,
)
from terminal_certificate_storage import (
    image_name_allowed_by_storage_contract,
    prepare_terminal_certificate_deployment_jpeg,
    resolve_terminal_certificate_storage_contract,
)
from objectnav_sparse_runtime import load_sparse_objectnav_runtime_from_environment


def _load_json_maybe_gzip(path: str):
    """Load a dataset shard; supports `.json.gz` (official) or plain `.json`."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as fp:
            return json.load(fp)
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def _json_safe_for_post(obj):
    """Recursively convert numpy / pandas scalars so requests json= works."""
    if isinstance(obj, dict):
        return {str(k): _json_safe_for_post(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_for_post(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _count_run_complete_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    n = 0
    try:
        with open(path, "r", errors="ignore") as fp:
            for line in fp:
                if "RUN COMPLETE" in line:
                    n += 1
    except OSError:
        return 0
    return n


def _completed_episode_indices(worker_dir: str, start_ndx: int, end_ndx: int):
    completed = set()
    if not os.path.isdir(worker_dir):
        return completed
    try:
        entries = list(os.scandir(worker_dir))
    except OSError:
        return completed
    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            episode_ndx = int(entry.name.split("_", 1)[0])
        except (TypeError, ValueError):
            continue
        if episode_ndx < start_ndx or episode_ndx >= end_ndx:
            continue
        summary_path = os.path.join(entry.path, "episode_summary.json")
        pkl_path = os.path.join(entry.path, "df_results.pkl")
        if os.path.isfile(summary_path) or os.path.isfile(pkl_path):
            completed.add(episode_ndx)
    return completed


def _contiguous_completed_prefix(worker_dir: str, start_ndx: int, end_ndx: int) -> int:
    completed = _completed_episode_indices(worker_dir, start_ndx, end_ndx)
    episode_ndx = start_ndx
    while episode_ndx in completed and episode_ndx < end_ndx:
        episode_ndx += 1
    return episode_ndx - start_ndx


def _normalize_objectnav_dataset(dataset_name):
    aliases = {
        'hm3d_v0.1': 'hm3d_v1',
        'hm3d_v1': 'hm3d_v1',
        'hm3d_v0.2': 'hm3d_v2',
        'hm3d_v2': 'hm3d_v2',
    }
    value = str(dataset_name or 'hm3d_v2').strip().lower()
    if value not in aliases:
        raise ValueError(
            f"Unsupported ObjectNav dataset '{dataset_name}'. "
            "Expected one of hm3d_v1 / hm3d_v2 (or hm3d_v0.1 / hm3d_v0.2 aliases)."
        )
    return aliases[value]


def _first_existing_path(candidates, description):
    for path in candidates:
        if os.path.exists(path):
            return path
    joined = '\n'.join(f'  - {path}' for path in candidates)
    raise FileNotFoundError(f'Unable to resolve {description}. Tried:\n{joined}')


def _vlmnav_data_root():
    override = os.environ.get('VLMNAV_DATASET_ROOT') or os.environ.get('DATASET_ROOT')
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def _vlmnav_scene_root_override(dataset: str):
    if dataset == 'hm3d_v1':
        override = os.environ.get('VLMNAV_HM3D_V1_SCENE_ROOT')
    elif dataset == 'hm3d_v2':
        override = os.environ.get('VLMNAV_HM3D_V2_SCENE_ROOT')
    else:
        override = None
    if override:
        path = os.path.abspath(os.path.expanduser(override))
        if not os.path.exists(path):
            raise FileNotFoundError(f'Configured HM3D scene root does not exist: {path}')
        return path
    return None


def _vlmnav_objectnav_root_override(dataset: str):
    if dataset == 'hm3d_v1':
        override = os.environ.get('VLMNAV_OBJECTNAV_V1_ROOT')
    elif dataset == 'hm3d_v2':
        override = os.environ.get('VLMNAV_OBJECTNAV_V2_ROOT')
    else:
        override = None
    if override:
        path = os.path.abspath(os.path.expanduser(override))
        if not os.path.exists(path):
            raise FileNotFoundError(f'Configured ObjectNav dataset root does not exist: {path}')
        return path
    return None


def _resolve_objectnav_dataset_paths(dataset_name):
    dataset = _normalize_objectnav_dataset(dataset_name)
    data_root = _vlmnav_data_root()
    scene_root_override = _vlmnav_scene_root_override(dataset)
    objectnav_root_override = _vlmnav_objectnav_root_override(dataset)

    if dataset == 'hm3d_v1':
        if scene_root_override:
            scene_root = scene_root_override
        else:
            scene_root = _first_existing_path(
                [
                    os.path.join(data_root, 'hm3d_v1'),
                    os.path.join(data_root, 'scene_datasets', 'hm3d_v1'),
                ],
                'HM3D v1 scene root',
            )
        scene_config = _first_existing_path(
            [
                os.path.join(scene_root, 'hm3d_annotated_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'hm3d_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'hm3d_annotated_val_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'hm3d_val_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'hm3d_v0.1_annotated_val.scene_dataset_config.json'),
                os.path.join(scene_root, 'hm3d_v0.1_val.scene_dataset_config.json'),
                os.path.join(scene_root, 'val', 'hm3d_annotated_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'val', 'hm3d_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'val', 'hm3d_annotated_val_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'val', 'hm3d_val_basis.scene_dataset_config.json'),
            ],
            'HM3D v1 scene dataset config',
        )
        if objectnav_root_override:
            objectnav_root = objectnav_root_override
        else:
            objectnav_root = _first_existing_path(
                [
                    os.path.join(data_root, 'objectnav_hm3d_v1'),
                    os.path.join(data_root, 'datasets', 'objectnav_hm3d_v1'),
                ],
                'ObjectNav HM3D v1 dataset root',
            )
    else:
        if scene_root_override:
            scene_root = scene_root_override
        else:
            scene_root = _first_existing_path(
                [
                    os.path.join(data_root, 'scene_datasets', 'hm3d'),
                    os.path.join(data_root, 'hm3d_v2'),
                ],
                'HM3D v2 scene root',
            )
        scene_config = _first_existing_path(
            [
                os.path.join(scene_root, 'hm3d_annotated_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'hm3d_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'hm3d_annotated_val_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'hm3d_val_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'val', 'hm3d_annotated_val_basis.scene_dataset_config.json'),
                os.path.join(scene_root, 'val', 'hm3d_val_basis.scene_dataset_config.json'),
            ],
            'HM3D v2 scene dataset config',
        )
        if objectnav_root_override:
            objectnav_root = objectnav_root_override
        else:
            objectnav_root = _first_existing_path(
                [
                    os.path.join(data_root, 'datasets', 'objectnav_hm3d_v2'),
                    os.path.join(data_root, 'objectnav_hm3d_v2_unzip'),
                    os.path.join(data_root, 'objectnav_hm3d_v2'),
                ],
                'ObjectNav HM3D v2 dataset root',
            )

    return {
        'dataset': dataset,
        'data_root': data_root,
        'scene_root': scene_root,
        'scene_config': scene_config,
        'objectnav_root': objectnav_root,
    }


class Env:
    """
    Base class for creating an environment for embodied navigation tasks.
    This class defines the setup, logging, running, and evaluation of episodes.
    """

    task = 'Not defined'

    def __init__(self, cfg: dict):
        """
        Initializes the environment with the provided configuration.

        Args:
            cfg (dict): Configuration dictionary containing environment, simulation, and agent settings.
        """
        self.cfg = cfg['env_cfg']
        self.sim_cfg = cfg['sim_cfg']
        self.logs_root = os.environ.get("VLMNAV_LOGS_ROOT", "logs").strip() or "logs"
        if self.cfg['name'] == 'default':
            self.cfg['name'] = f'default_{random.randint(0, 1000)}'
        self._initialize_logging(cfg)
        self._initialize_agent(cfg)
        self.outer_run_name = self.task + '_' + self.cfg['name']
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper: SimWrapper = None
        self.num_episodes = 0
        self._initialize_experiment()

    def _initialize_agent(self, cfg: dict):
        """Initializes the agent for the environment."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        agent_cls = globals()[cfg['agent_cls']]
        self.agent: Agent = agent_cls(cfg['agent_cfg']) 

    def _initialize_logging(self, cfg: dict):
        """
        Initializes logging for the environment.

        Args:
            cfg (dict): Configuration dictionary containing logging settings.
        """
        self.log_file = os.path.join(
            self.logs_root,
            f'{cfg["task"]}_{self.cfg["name"]}',
            f'{self.cfg["instance"]}_of_{self.cfg["instances"]}.txt',
        )
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        root.addHandler(fh)
        if self.cfg['parallel'] and os.environ.get('VLMNAV_PARALLEL_LOG_STDERR', '1') not in ('0', 'false', 'no'):
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            root.addHandler(sh)
        elif not self.cfg['parallel']:
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            root.addHandler(sh)

    def _initialize_experiment(self):
        """
        Abstract method for setting up the environment and initializing all required variables.
        Should be implemented in derived classes.
        """
        raise NotImplementedError

    def run_experiment(self):
        """
        Runs the experiment by iterating over episodes.
        """
        instance_size = math.ceil(self.num_episodes / self.cfg['instances'])
        start_ndx = self.cfg['instance'] * instance_size
        loop_end = min(start_ndx + self.cfg['num_episodes'], self.num_episodes)
        loop_start = start_ndx
        completed_episode_indices = set()
        shared_pool = None

        # Skip episodes already finished in a previous launch. Prefer the on-disk
        # completion artifacts written in _post_episode(); fall back to legacy log
        # scanning only when those artifacts are not available yet.
        if os.environ.get("VLMNAV_AUTO_RESUME", "").lower() in ("1", "true", "yes"):
            err_path = os.path.join(
                self.logs_root,
                self.outer_run_name,
                f'{self.cfg["instance"]}_of_{self.cfg["instances"]}_error.txt',
            )
            worker_dir = os.path.join(
                self.logs_root,
                self.outer_run_name,
                f'{self.cfg["instance"]}_of_{self.cfg["instances"]}',
            )
            completed_episode_indices = _completed_episode_indices(worker_dir, start_ndx, loop_end)
            n_done = _contiguous_completed_prefix(worker_dir, start_ndx, loop_end)
            resume_source = None
            if n_done > 0:
                resume_source = f"{worker_dir} completion artifacts"
            elif os.path.isfile(err_path):
                n_done = _count_run_complete_lines(err_path)
                resume_source = f"{err_path} 'RUN COMPLETE' lines"
            shard_cap = self.cfg["num_episodes"]
            n_done = min(n_done, shard_cap)
            cand = start_ndx + n_done
            if cand < loop_end:
                loop_start = cand
                if resume_source is not None:
                    logging.info(
                        f"VLMNAV_AUTO_RESUME: from {resume_source} counted {n_done} completed episode(s) "
                        f"in this shard; resuming at episode_ndx={loop_start} "
                        f"(range [{start_ndx}, {loop_end}))"
                    )
            elif n_done > 0:
                if resume_source is not None:
                    logging.info(
                        f"VLMNAV_AUTO_RESUME: {resume_source} shows {n_done} completed episode(s) "
                        f"for this shard [{start_ndx}, {loop_end}); worker has nothing left to run."
                    )

        index_offset = int(os.environ.get("VLMNAV_EPISODE_INDEX_OFFSET", "0") or 0)
        episode_index_list_path = os.environ.get("VLMNAV_EPISODE_INDEX_LIST_PATH", "").strip()
        absolute_index = os.environ.get("VLMNAV_EPISODE_INDEX_ABSOLUTE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        if episode_index_list_path:
            with open(episode_index_list_path, "r", encoding="utf-8") as fp:
                requested_episode_indices = json.load(fp)
            if not isinstance(requested_episode_indices, list):
                raise ValueError(
                    f"VLMNAV_EPISODE_INDEX_LIST_PATH must contain a JSON list: {episode_index_list_path}"
                )
            episode_indices = []
            seen_episode_indices = set()
            for raw_episode_ndx in requested_episode_indices:
                if isinstance(raw_episode_ndx, bool) or not isinstance(raw_episode_ndx, int):
                    raise ValueError(
                        f"invalid episode index {raw_episode_ndx!r} in {episode_index_list_path}"
                    )
                if raw_episode_ndx < 0 or raw_episode_ndx >= self.num_episodes:
                    raise ValueError(
                        f"episode index {raw_episode_ndx} outside [0, {self.num_episodes}) "
                        f"in {episode_index_list_path}"
                    )
                if raw_episode_ndx not in seen_episode_indices:
                    seen_episode_indices.add(raw_episode_ndx)
                    episode_indices.append(raw_episode_ndx)

            shared_pool_enabled = os.environ.get(
                "VLMNAV_PERSISTENT_SHARED_POOL", ""
            ).strip().lower() in ("1", "true", "yes", "on")
            if shared_pool_enabled:
                log_root = os.path.join(self.logs_root, self.outer_run_name)
                pool_dir = os.environ.get("VLMNAV_SHARED_POOL_DIR", "").strip()
                if not pool_dir:
                    pool_dir = os.path.join(log_root, ".persistent_episode_pool")
                shared_pool = PersistentEpisodePool(pool_dir, log_root)
                seeded = shared_pool.seed_from_summaries()
                episode_indices = [
                    episode_ndx
                    for episode_ndx in episode_indices
                    if not shared_pool.is_done(episode_ndx)
                ]
                logging.info(
                    f"VLMNAV_PERSISTENT_SHARED_POOL: pool={pool_dir} "
                    f"candidates={len(episode_indices)} seeded={seeded}"
                )
            else:
                worker_dir = os.path.join(
                    self.logs_root,
                    self.outer_run_name,
                    f'{self.cfg["instance"]}_of_{self.cfg["instances"]}',
                )
                completed_episode_indices = _completed_episode_indices(worker_dir, 0, self.num_episodes)
                episode_indices = [
                    episode_ndx
                    for episode_ndx in episode_indices
                    if episode_ndx not in completed_episode_indices
                ]
            logging.info(
                f"VLMNAV_EPISODE_INDEX_LIST_PATH: loaded {len(requested_episode_indices)} requested "
                f"episode(s), {len(episode_indices)} remaining from {episode_index_list_path}"
            )
        elif absolute_index and self.num_episodes > 0:
            episode_indices = [index_offset % self.num_episodes]
            logging.info(
                f"VLMNAV_EPISODE_INDEX_ABSOLUTE: running episode index {episode_indices[0]} "
                f"from offset {index_offset} mod {self.num_episodes}"
            )
        else:
            episode_indices = list(range(loop_start, loop_end))
        shuffle_seed_raw = os.environ.get("VLMNAV_EPISODE_SHUFFLE_SEED", "")
        if (
            not episode_index_list_path
            and not absolute_index
            and index_offset
            and not completed_episode_indices
            and self.num_episodes > 0
        ):
            episode_indices = [(episode_ndx + index_offset) % self.num_episodes for episode_ndx in episode_indices]
            logging.info(
                f"VLMNAV_EPISODE_INDEX_OFFSET: shifted shard episode indices by {index_offset} "
                f"mod {self.num_episodes}"
            )
        if episode_index_list_path:
            shuffle_seed_raw = ""
        if shuffle_seed_raw and not completed_episode_indices:
            try:
                shuffle_seed = int(shuffle_seed_raw)
                random.Random(shuffle_seed).shuffle(episode_indices)
                logging.info(f"VLMNAV_EPISODE_SHUFFLE_SEED: shuffled shard episode order with seed={shuffle_seed}")
            except ValueError:
                logging.warning(f"VLMNAV_EPISODE_SHUFFLE_SEED ignored invalid value: {shuffle_seed_raw}")

        def execute_episode(episode_ndx, shared):
            self.wandb_log_data = {
                'episode_ndx': episode_ndx,
                'instance': self.inner_run_name,
                'total_episodes': (
                    self.num_episodes
                    if shared
                    else self.cfg['instances'] * self.cfg['num_episodes']
                ),
                'task': self.task,
                'task_data': {},
                'spl': 0,
                'goal_reached': False
            }

            try:
                self._run_episode(episode_ndx)
                return True
            except Exception as e:
                log_exception(e)
                self.simWrapper.reset()
                return False

        if shared_pool is None:
            for episode_ndx in episode_indices:
                if episode_ndx in completed_episode_indices:
                    continue
                execute_episode(episode_ndx, False)
        else:
            idle_sleep_s = float(
                os.environ.get("VLMNAV_SHARED_POOL_IDLE_SLEEP_S", "1") or 1
            )
            while True:
                pending = [
                    episode_ndx
                    for episode_ndx in episode_indices
                    if not shared_pool.is_done(episode_ndx)
                ]
                if not pending:
                    break
                claimed_any = False
                for episode_ndx in pending:
                    with shared_pool.try_claim(episode_ndx) as claimed:
                        if not claimed:
                            continue
                        claimed_any = True
                        if execute_episode(episode_ndx, True):
                            summary_path = os.path.join(
                                self._episode_dir(), "episode_summary.json"
                            )
                            if not shared_pool.mark_done(episode_ndx, summary_path):
                                logging.error(
                                    f"VLMNAV_PERSISTENT_SHARED_POOL: valid summary missing "
                                    f"after episode {episode_ndx}: {summary_path}"
                                )
                if not claimed_any:
                    time.sleep(max(0.1, idle_sleep_s))

        if episode_index_list_path:
            exit_stagger_s = float(os.environ.get("VLMNAV_EPISODE_LIST_EXIT_STAGGER_S", "3") or 0)
            if exit_stagger_s > 0:
                exit_delay_s = max(0, int(self.cfg["instance"])) * exit_stagger_s
                if exit_delay_s > 0:
                    logging.info(
                        f"VLMNAV_EPISODE_INDEX_LIST_PATH: delaying process teardown by "
                        f"{exit_delay_s:.1f}s for instance={self.cfg['instance']}"
                    )
                    time.sleep(exit_delay_s)


    def _run_episode(self, episode_ndx: int):
        """
        Runs a single episode.

        Args:
            episode_ndx (int): The index of the episode to run.
        """
        obs = self._initialize_episode(episode_ndx)

        logging.info(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(self.cfg['max_steps']):
            try:
                habitat_step_budget = self._habitat_step_budget()
                if habitat_step_budget and self.habitat_steps >= habitat_step_budget:
                    logging.info(
                        "Episode terminated at estimated Habitat-step budget: %s/%s",
                        self.habitat_steps,
                        habitat_step_budget,
                    )
                    break
                agent_action = self._step_env(obs)
                if agent_action is None:
                    break
                obs = self.simWrapper.step(agent_action)

            except Exception as e:
                log_exception(e)
                if getattr(self, "_habitat_step_hard_horizon_invariant_violation", False):
                    logging.error(
                        "Aborting episode after a strict Habitat-step invariant violation"
                    )
                    break

            finally:
                self.step += 1
        self._post_episode()

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode. This method should be implemented in derived classes.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        self.step = 0
        self.habitat_steps = 0
        self.last_agent_action = None
        self._habitat_step_hard_horizon_invariant_violation = False
        if self._strict_habitat_step_budget_enabled() and self._habitat_step_budget() <= 0:
            raise RuntimeError(
                "VLMNAV_REQUIRE_HABITAT_STEP_BUDGET=1 requires a positive "
                "VLMNAV_HABITAT_STEP_BUDGET before an episode starts"
            )
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0
        self.prev_agent_position = None
        self._last_motion_distance_sequence_id = None
        self._visited_position_counts = {}
        self._previous_room_id = None
        self._previous_chosen_action_theta_deg = None
        self._lightweight_panoramic_last_navigation_mode = None
        self._lightweight_panoramic_mode_streak = 0
        self._lightweight_panoramic_last_scan_step = None
        self._lightweight_panoramic_scan_count = 0
        self._lightweight_panoramic_global_state = "free"
        self._lightweight_panoramic_view_direction_deg = 0.0
        self._lightweight_panoramic_lock_until_step = None
        self._lightweight_panoramic_lock_context = None
        self._lightweight_panoramic_distance_history = []
        self._lightweight_panoramic_best_distance = None
        self._lightweight_panoramic_last_improvement_step = None
        self._lightweight_panoramic_current_distance = None
        self._lightweight_panoramic_last_public_progress_step = 0
        self._action0_pano_replace_count = 0

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment. This method should be implemented in derived classes.

        Args:
            obs (dict): The current observation. Contains agent state and sensor observations.

        Returns:
            PolarAction: The next action to be taken by the agent.
        """
        logging.info(f'Step {self.step}')
        self._accumulate_motion_distance(obs)
        self._accumulate_habitat_steps(obs)

        return None

    def _habitat_step_budget(self) -> int:
        try:
            budget = int(os.environ.get("VLMNAV_HABITAT_STEP_BUDGET", "0"))
        except (TypeError, ValueError):
            budget = 0
        return max(0, budget)

    def _strict_habitat_step_budget_enabled(self) -> bool:
        """Whether this run must enforce the macro-action horizon preflight.

        This is intentionally opt-in.  Historical launchers keep their old
        post-hoc accounting semantics unless they explicitly request the strict
        protocol through ``VLMNAV_REQUIRE_HABITAT_STEP_BUDGET=1``.
        """
        return os.environ.get(
            "VLMNAV_REQUIRE_HABITAT_STEP_BUDGET",
            "0",
        ).strip().lower() in ("1", "true", "yes", "y", "on")

    def _calculate_habitat_steps(self, action: PolarAction) -> int:
        return compass_habitat_step_cost(
            action.r,
            action.theta,
            is_stop=getattr(action, "type", None) == "stop",
        )

    def _strict_habitat_horizon_for_action(self, action: PolarAction) -> dict:
        """Return a pre-execution hard-horizon decision for ``action``.

        Stop is terminal and is never sent to ``SimWrapper.step`` in this
        executor, so it must not consume or be rejected by the movement
        horizon.  Every executable macro action is priced before the simulator
        sees it.
        """
        strict = self._strict_habitat_step_budget_enabled()
        budget = self._habitat_step_budget()
        terminal_stop = action is PolarAction.stop
        action_cost = 0 if terminal_stop or action is None else self._calculate_habitat_steps(action)
        decision = strict_habitat_step_horizon_decision(
            consumed_steps=int(getattr(self, "habitat_steps", 0) or 0),
            action_cost=action_cost,
            budget=budget,
            enabled=strict,
        )
        decision["terminal_stop"] = bool(terminal_stop)
        decision["counts_toward_horizon"] = bool(not terminal_stop and action is not None)
        return decision

    def _accumulate_habitat_steps(self, obs: dict) -> int:
        """Charge the previously selected action on its next observation."""
        budget = self._habitat_step_budget()
        delta = 0
        previous_action = getattr(self, "last_agent_action", None)
        if budget and previous_action is not None:
            delta = self._calculate_habitat_steps(previous_action)
            if (
                self._strict_habitat_step_budget_enabled()
                and self.habitat_steps + delta > budget
            ):
                # The ObjectNav execution path asks the pure horizon helper
                # before calling SimWrapper.step.  Reaching this branch means
                # a new execution path bypassed that gate; fail closed instead
                # of emitting a falsely valid strict-horizon result.
                self._habitat_step_hard_horizon_invariant_violation = True
                self.last_agent_action = None
                raise RuntimeError(
                    "strict Habitat-step invariant violated: "
                    f"{self.habitat_steps}+{delta}>{budget}"
                )
            self.habitat_steps += delta
            self.last_agent_action = None
            logging.info(
                "Estimated Habitat steps: %s/%s (+%s)",
                self.habitat_steps,
                budget,
                delta,
            )
        obs["habitat_steps"] = int(self.habitat_steps)
        obs["habitat_step_budget"] = int(budget)
        obs["habitat_step_hard_horizon_enabled"] = bool(
            self._strict_habitat_step_budget_enabled()
        )
        return int(delta)

    def _accumulate_motion_distance(self, obs: dict) -> float:
        """Accumulate physical substep travel once per simulator action."""
        agent_state = obs['agent_state']
        feedback = obs.get('motion_feedback') if isinstance(obs, dict) else None
        distance = None
        if isinstance(feedback, dict):
            try:
                sequence_id = int(feedback.get('sequence_id'))
            except (TypeError, ValueError):
                sequence_id = None
            if sequence_id is not None and sequence_id != self._last_motion_distance_sequence_id:
                self._last_motion_distance_sequence_id = sequence_id
                try:
                    distance = float(feedback.get('traveled_distance'))
                except (TypeError, ValueError):
                    distance = None
        if distance is None and self.prev_agent_position is not None:
            distance = float(np.linalg.norm(agent_state.position - self.prev_agent_position))
        if distance is not None and np.isfinite(distance):
            self.agent_distance_traveled += max(0.0, float(distance))
        self.prev_agent_position = np.asarray(agent_state.position, dtype=float).copy()
        return max(0.0, float(distance or 0.0))

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        episode_dir = self._episode_dir()
        os.makedirs(episode_dir, exist_ok=True)
        self._finalize_unfinished_episode_status()
        self.df.to_pickle(f'{episode_dir}/df_results.pkl')
        with open(f'{episode_dir}/episode_summary.json', 'w', encoding='utf-8') as file:
            json.dump(self._build_episode_summary(), file, ensure_ascii=False, indent=2)
        self.simWrapper.reset()
        self.agent.reset()

        # Ensure that all subgoals for this episode are represented in wandb_log_data,
        # even if some of them did not reach a clean 'done' state due to errors/early termination.
        try:
            # GOAT uses a list of subgoals; ObjectNav uses a dict — only backfill for list episodes.
            if hasattr(self, "current_episode") and isinstance(self.current_episode, list):
                total_goals = len(self.current_episode)
                task_data = self.wandb_log_data.setdefault("task_data", {})
                logged_goals = task_data.get("goal_data", [])

                # For any goal that never reached metrics['done'], add a failure entry
                if total_goals > len(logged_goals):
                    for idx in range(len(logged_goals), total_goals):
                        goal = self.current_episode[idx]
                        logged_goals.append(
                            {
                                "goal_mode": goal.get("mode", "unknown"),
                                "goal_reached": False,
                                "spl": 0.0,
                            }
                        )
                    task_data["goal_data"] = logged_goals
        except Exception as e:
            logging.error(f"Error while finalizing goal_data for episode {getattr(self, 'curr_run_name', 'unknown')}: {e}")

        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = float(self.agent.get_spend())
                nrows = len(self.df)
                if nrows > 0 and 'success' in self.df.columns:
                    self.wandb_log_data['default_rate'] = float(
                        len(self.df[self.df['success'] == 0]) / nrows
                    )
                else:
                    self.wandb_log_data['default_rate'] = 0.0
                agg_host = os.environ.get('AGGREGATOR_HOST', '127.0.0.1')
                payload = _json_safe_for_post(self.wandb_log_data)
                url = f"http://{agg_host}:{self.cfg['port']}/log"
                response = requests.post(url, json=payload, timeout=60)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics {response.status_code}: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info('\n===================RUN COMPLETE===================\n')
        if self.cfg['log_freq'] == 1 and os.environ.get('VLMNAV_DISABLE_IMAGE_LOGGING', '0') != '1':
            create_gif(self._episode_dir())

    def _finalize_unfinished_episode_status(self):
        """
        Persist a terminal status when recoverable step errors leave the last
        logged row in the default running state.
        """
        if self.df.empty or 'finish_status' not in self.df.columns:
            return

        last_idx = self.df.index[-1]
        finish_status = self.df.at[last_idx, 'finish_status']
        if pd.notna(finish_status) and str(finish_status).strip().lower() not in ('', 'running'):
            return

        try:
            max_steps = int(self.cfg.get('max_steps', 0))
        except Exception:
            max_steps = 0
        current_step = int(getattr(self, 'step', 0) or 0)
        habitat_step_budget = self._habitat_step_budget()
        if getattr(self, "_habitat_step_hard_horizon_invariant_violation", False):
            terminal_status = 'strict_habitat_horizon_invariant_violation'
        elif habitat_step_budget and self.habitat_steps >= habitat_step_budget:
            terminal_status = 'habitat_max_steps'
        else:
            terminal_status = 'max_steps' if max_steps and current_step >= max_steps else 'aborted'

        self.df.at[last_idx, 'done'] = True
        self.df.at[last_idx, 'finish_status'] = terminal_status
        self.df.at[last_idx, 'goal_reached'] = False
        self.df.at[last_idx, 'spl'] = 0.0
        self.df.at[last_idx, 'exception_finalized'] = True
        self.df.at[last_idx, 'exception_finalized_step'] = current_step

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        """
        Appends the step metadata to the dataframe, and saves the images and general metadata to disk.

        Args:
            images (dict): Images generated during the step.
            step_metadata (dict): Metadata for the current step.
            logging_data (dict): General logging data.
        """
        storage_contract = resolve_terminal_certificate_storage_contract(os.environ)
        prepared_terminal_certificate_jpeg = None
        if storage_contract is not None:
            terminal_image = images.get('terminal_certificate_input_rgb')
            if terminal_image is None:
                raise RuntimeError(
                    "deployment JPEG storage contract requires one "
                    "terminal_certificate_input_rgb image on every logged step"
                )
            prepared_terminal_certificate_jpeg, storage_metadata = (
                prepare_terminal_certificate_deployment_jpeg(terminal_image)
            )
            existing_source_hash = str(
                step_metadata.get('terminal_certificate_input_rgb_pixel_sha256') or ''
            ).strip()
            if (
                existing_source_hash
                and existing_source_hash
                != storage_metadata['terminal_certificate_input_rgb_source_pixel_sha256']
            ):
                raise RuntimeError(
                    "captured terminal-certificate RGB changed before deployment storage"
                )
            step_metadata = dict(step_metadata)
            step_metadata.update(storage_metadata)

        row = self._sanitize_for_json(dict(step_metadata))
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self._append_step_trace(row, logging_data)

        save_every_step_image = os.environ.get("VLMNAV_SAVE_EVERY_STEP_IMAGE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        save_stop_signal_image = (
            os.environ.get("VLMNAV_SAVE_STOP_SIGNAL_IMAGES", "").lower()
            in ("1", "true", "yes")
            and bool(step_metadata.get("stop_signal"))
        )
        image_logging_disabled = os.environ.get(
            "VLMNAV_DISABLE_IMAGE_LOGGING", ""
        ).lower() in ("1", "true", "yes")
        if (
            not image_logging_disabled
            and (
                save_every_step_image
                or save_stop_signal_image
                or self.step % self.cfg['log_freq'] == 0
                or step_metadata['success'] == 0
            )
        ):
            path = f'{self._episode_dir()}/step{self.step}'
            if not step_metadata['success']:
                path += '_ERROR'
            os.makedirs(path, exist_ok=True)
            for name, im in images.items():
                if not image_name_allowed_by_storage_contract(name, storage_contract):
                    continue
                if name == 'terminal_certificate_input_rgb' and storage_contract is not None:
                    output_path = os.path.join(path, str(storage_contract['filename']))
                    with open(output_path, 'wb') as file:
                        file.write(prepared_terminal_certificate_jpeg)
                    continue
                im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                im.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w', encoding='utf-8') as file:
                file.write('STEP_METADATA\n')
                for k, v in row.items():
                    file.write(f'{k}\n{self._stringify_for_text(v)}\n\n')
                file.write('LOGGING_DATA\n')
                for k, v in logging_data.items():
                    file.write(f'{k}\n{self._stringify_for_text(v)}\n\n')

    def _calculate_metrics(
        self,
        agent_state: habitat_sim.AgentState,
        agent_action: PolarAction,
        geodesic_path: int,
        max_steps: int,
        strict_horizon_decision=None,
    ):
        """
        Calculates the navigation metrics at a given step.

        Args:
            agent_state: The state of the agent.
            agent_action: The action taken by the agent.
            geodesic_path: The shortest path to the goal.
            max_steps (int): Maximum steps allowed for the episode.

        Returns:
            dict: A dictionary containing calculated metrics.
        """
        metrics = {}
        self.path_calculator.requested_start = agent_state.position
        metrics['distance_to_goal'] = self.simWrapper.get_path(self.path_calculator)
        metrics['spl'] = 0
        metrics['goal_reached'] = False
        metrics['done'] = False
        metrics['finish_status'] = 'running'
        terminal_action_was_stop = agent_action is PolarAction.stop
        require_explicit_stop = os.environ.get(
            'VLMNAV_REQUIRE_EXPLICIT_STOP_FOR_SUCCESS',
            '',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        metrics['terminal_action_was_stop'] = bool(terminal_action_was_stop)
        metrics['success_requires_explicit_stop'] = bool(require_explicit_stop)
        habitat_step_budget = self._habitat_step_budget()
        habitat_budget_reached = bool(
            habitat_step_budget and self.habitat_steps >= habitat_step_budget
        )
        strict_horizon_enabled = self._strict_habitat_step_budget_enabled()
        strict_horizon_blocked = bool(
            isinstance(strict_horizon_decision, dict)
            and strict_horizon_decision.get("strict")
            and not strict_horizon_decision.get("allow_execution")
        )
        policy_step_budget_reached = bool(self.step + 1 == max_steps)
        force_stop_at_budget = os.environ.get(
            'VLMNAV_FORCE_STOP_AT_EPISODE_BUDGET',
            '1',
        ).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        # At a fixed evaluation horizon the executor issues a terminal Stop at
        # the final pose.  This decision is unconditional and therefore uses
        # no goal distance, oracle visibility, or other simulator-private
        # signal.  Keep it separate from the model's own Stop for diagnostics.
        budget_forced_stop = bool(
            force_stop_at_budget
            and not terminal_action_was_stop
            and (policy_step_budget_reached or habitat_budget_reached)
        )
        effective_terminal_action_was_stop = bool(
            terminal_action_was_stop or budget_forced_stop
        )
        metrics['terminal_stop_was_forced_by_budget'] = budget_forced_stop
        metrics['effective_terminal_action_was_stop'] = effective_terminal_action_was_stop
        metrics['force_stop_at_episode_budget'] = bool(force_stop_at_budget)
        metrics['habitat_steps'] = int(self.habitat_steps)
        metrics['habitat_step_budget'] = int(habitat_step_budget)
        metrics['habitat_step_budget_reached'] = habitat_budget_reached
        metrics['habitat_step_hard_horizon_enabled'] = bool(strict_horizon_enabled)
        metrics['habitat_step_hard_horizon_invariant_violation'] = bool(
            getattr(self, "_habitat_step_hard_horizon_invariant_violation", False)
        )
        metrics['habitat_step_hard_horizon_blocked'] = bool(strict_horizon_blocked)
        if isinstance(strict_horizon_decision, dict):
            metrics['habitat_step_next_action_cost'] = int(
                strict_horizon_decision.get('action_cost', 0) or 0
            )
            metrics['habitat_step_remaining_before_action'] = int(
                strict_horizon_decision.get('remaining_steps', 0) or 0
            )
            metrics['habitat_step_action_execution_allowed'] = bool(
                strict_horizon_decision.get('allow_execution')
            )
            metrics['habitat_step_horizon_block_reason'] = str(
                strict_horizon_decision.get('blocked_reason') or ''
            )
        else:
            metrics['habitat_step_next_action_cost'] = None
            metrics['habitat_step_remaining_before_action'] = max(
                0,
                int(habitat_step_budget) - int(self.habitat_steps),
            )
            metrics['habitat_step_action_execution_allowed'] = True
            metrics['habitat_step_horizon_block_reason'] = ''

        if (
            agent_action is PolarAction.stop
            or self.step + 1 == max_steps
            or habitat_budget_reached
            or strict_horizon_blocked
        ):
            metrics['done'] = True

            success_distance_met = metrics['distance_to_goal'] < self.cfg['success_threshold']
            explicit_stop_met = effective_terminal_action_was_stop or not require_explicit_stop
            if success_distance_met and explicit_stop_met:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': metrics['goal_reached']
                })
            else:
                if agent_action is PolarAction.stop:
                    metrics['finish_status'] = 'fp'
                elif habitat_budget_reached or strict_horizon_blocked:
                    metrics['finish_status'] = 'habitat_max_steps'
                else:
                    metrics['finish_status'] = 'max_steps'

        return metrics

    def _episode_dir(self):
        return os.path.join(
            self.logs_root,
            self.outer_run_name,
            self.inner_run_name,
            self.curr_run_name,
        )

    def _sanitize_for_json(self, value):
        if isinstance(value, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_for_json(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _stringify_for_text(self, value):
        value = self._sanitize_for_json(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        return str(value)

    def _append_step_trace(self, row: dict, logging_data: dict):
        episode_dir = self._episode_dir()
        os.makedirs(episode_dir, exist_ok=True)
        trace_path = os.path.join(episode_dir, 'step_traces.jsonl')
        payload = {
            'step': int(self.step),
            'step_metadata': self._sanitize_for_json(row),
            'logging_data': self._sanitize_for_json(logging_data),
        }
        with open(trace_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(payload, ensure_ascii=False) + '\n')
        self._append_model_io_log(row, logging_data)

    def _append_model_io_log(self, row: dict, logging_data: dict):
        if os.environ.get('VLMNAV_MODEL_IO_LOG', '1').strip().lower() in ('0', 'false', 'no', 'off'):
            return
        episode_dir = self._episode_dir()
        payload = {
            'event': 'model_io',
            'run_name': self.curr_run_name,
            'step': int(self.step),
            'object': row.get('object'),
            'finish_status': row.get('finish_status'),
            'parsed_action': row.get('action_number'),
            'action_success': row.get('success'),
            'action_decision_action': row.get('action_decision_action'),
            'action_decision_mode': row.get('action_decision_mode'),
            'action_navigation_mode': row.get('action_navigation_mode'),
            'action_should_stop': row.get('action_should_stop'),
            'called_stopping': row.get('called_stopping'),
            'stop_signal': row.get('stop_signal'),
            'stop_signal_ready': row.get('stop_signal_ready'),
            'action_wants_stop': row.get('action_wants_stop'),
            'stop_gate_require_action_stop': row.get('stop_gate_require_action_stop'),
            'stop_gate_passed': row.get('stop_gate_passed'),
            'stop_gate_blocked_action': row.get('stop_gate_blocked_action'),
            'action_guided_choice_enabled': row.get('action_guided_choice_enabled'),
            'action_guided_choice_mode': row.get('action_guided_choice_mode'),
            'action_guided_choice_action': row.get('action_guided_choice_action'),
            'action_guided_choice_action_number': row.get('action_guided_choice_action_number'),
            'action_guided_choice_valid': row.get('action_guided_choice_valid'),
            'action_guided_choice_applied': row.get('action_guided_choice_applied'),
            'action_guided_choice_original_action_number': row.get('action_guided_choice_original_action_number'),
            'action_guided_choice_response': row.get('action_guided_choice_response'),
            'action_label_scorer_enabled': row.get('action_label_scorer_enabled'),
            'action_label_scorer_mode': row.get('action_label_scorer_mode'),
            'action_label_scorer_status': row.get('action_label_scorer_status'),
            'action_label_scorer_action': row.get('action_label_scorer_action'),
            'action_label_scorer_action_number': row.get('action_label_scorer_action_number'),
            'action_label_scorer_valid': row.get('action_label_scorer_valid'),
            'action_label_scorer_applied': row.get('action_label_scorer_applied'),
            'action_label_scorer_original_action_number': row.get('action_label_scorer_original_action_number'),
            'action_label_scorer_preserve_stop': row.get('action_label_scorer_preserve_stop'),
            'action_label_scorer_preserved_stop': row.get('action_label_scorer_preserved_stop'),
            'action_label_scorer_hybrid_action': row.get('action_label_scorer_hybrid_action'),
            'action_label_scorer_hybrid_action_number': row.get('action_label_scorer_hybrid_action_number'),
            'action_label_scorer_execute_skip_reason': row.get('action_label_scorer_execute_skip_reason'),
            'action_label_scorer_execute_min_margin': row.get('action_label_scorer_execute_min_margin'),
            'action_label_scorer_margin_passed': row.get('action_label_scorer_margin_passed'),
            'action_label_scorer_score': row.get('action_label_scorer_score'),
            'action_label_scorer_score_margin': row.get('action_label_scorer_score_margin'),
            'action_label_scorer_scores': row.get('action_label_scorer_scores'),
            'action_label_scorer_missing_labels': row.get('action_label_scorer_missing_labels'),
            'action_label_scorer_generated': row.get('action_label_scorer_generated'),
            'action_label_scorer_top_logprobs': row.get('action_label_scorer_top_logprobs'),
            'action_label_scorer_error': row.get('action_label_scorer_error'),
            'model_response': logging_data.get('RESPONSE'),
            'guided_choice_original_response': logging_data.get('GUIDED_CHOICE_ORIGINAL_RESPONSE'),
            'guided_choice_response': logging_data.get('GUIDED_CHOICE_RESPONSE'),
            'label_scorer_original_response': logging_data.get('LABEL_SCORER_ORIGINAL_RESPONSE'),
            'label_scorer_response': logging_data.get('LABEL_SCORER_RESPONSE'),
            'stopping_response': logging_data.get('STOPPING RESPONSE'),
            'stopping_prompt': logging_data.get('STOPPING PROMPT'),
            'terminal_certificate_response': logging_data.get('TERMINAL CERTIFICATE RESPONSE'),
            'terminal_certificate_prompt': logging_data.get('TERMINAL CERTIFICATE PROMPT'),
            'public_inference_only': bool(row.get('public_inference_only', False)),
            'private_observation_fields_removed': row.get('private_observation_fields_removed'),
            'privileged_runtime_gates_disabled': bool(
                row.get('privileged_runtime_gates_disabled', False)
            ),
        }
        for key in (
            'action0_pano_enabled',
            'action0_pano_triggered',
            'action0_pano_success',
            'action0_pano_selected_angle',
            'action0_pano_selected_score',
            'action0_pano_score_margin',
            'action0_pano_min_score',
            'action0_pano_min_score_margin',
            'action0_pano_scores',
            'action0_pano_fallback_reason',
            'action0_pano_count',
            'action0_pano_actual_action',
            'action0_pano_gate_decision',
            'action0_pano_gate_reason',
            'action0_pano_max_per_episode',
            'action0_pano_min_step',
            'action0_pano_block_near_goal_dist',
            'action0_pano_current_distance',
            'action0_pano_mode_gate',
            'action0_pano_mode_hit',
            'lightweight_panoramic_enabled',
            'lightweight_panoramic_success',
            'lightweight_panoramic_gate_enabled',
            'lightweight_panoramic_gate_decision',
            'lightweight_panoramic_gate_reason',
            'lightweight_panoramic_last_navigation_mode',
            'lightweight_panoramic_current_navigation_mode',
            'lightweight_panoramic_mode_streak',
            'lightweight_panoramic_last_scan_step',
            'lightweight_panoramic_steps_since_scan',
            'lightweight_panoramic_scan_count',
            'lightweight_panoramic_max_scans_per_episode',
            'lightweight_panoramic_progress_gate',
            'lightweight_panoramic_progress_require_mode',
            'lightweight_panoramic_progress_mode_hit',
            'lightweight_panoramic_progress_ready',
            'lightweight_panoramic_progress_current_distance',
            'lightweight_panoramic_progress_best_distance',
            'lightweight_panoramic_progress_steps_since_improvement',
            'lightweight_panoramic_progress_stagnation_steps',
            'lightweight_panoramic_block_near_goal_dist',
            'lightweight_panoramic_near_goal_blocked',
            'lightweight_panoramic_near_goal_block_distance',
            'lightweight_panoramic_min_select_score',
            'lightweight_panoramic_min_score_margin',
            'lightweight_panoramic_selected_score',
            'lightweight_panoramic_score_margin',
            'lightweight_panoramic_reject_reason',
            'lightweight_panoramic_override_action_enabled',
            'lightweight_panoramic_override_applied',
            'lightweight_panoramic_override_from_action',
            'lightweight_panoramic_override_to_action',
            'lightweight_panoramic_override_angle_diff',
            'lightweight_panoramic_override_reason',
            'stop_ready_streak',
            'stop_ready_streak_min',
            'stop_ready_streak_override',
            'stop_signal_ready_for_gate',
            'stop_ready_streak_require_action_near',
            'stop_ready_streak_require_action_stop',
            'stop_gate_stable_action_near',
            'stop_gate_stable_override',
            'stop_gate_veto_runtime_far_enabled',
            'stop_gate_veto_runtime_far_thresh',
            'stop_gate_veto_runtime_far_direct_block',
            'stop_gate_veto_runtime_far_blocked',
            'stop_gate_veto_action_far_enabled',
            'stop_gate_veto_action_far_blocked',
            'stop_gate_veto_action_approach_conflict_enabled',
            'stop_gate_veto_action_approach_conflict_blocked',
            'stop_gate_veto_action_stop_mode_conflict_enabled',
            'stop_gate_veto_action_stop_mode_conflict_blocked',
            'stop_gate_veto_target_rel_dist_thresh',
            'stop_gate_veto_target_runtime_far',
            'stop_gate_veto_max_ready_streak',
            'stop_gate_veto_ready_streak_gate_passed',
            'stop_gate_action_far_or_approach',
            'stop_gate_action_approach_conflict',
            'stop_gate_action_stop_mode_conflict',
            'stop_signal_target_rel_dist_override',
            'stop_signal_runtime_near_override',
            'lightweight_panoramic_stop_veto_enabled',
            'lightweight_panoramic_stop_conflict',
            'lightweight_panoramic_stop_veto_blocked',
            'runtime_loop_guard_enabled',
            'runtime_loop_guard_applied',
            'runtime_loop_guard_ready',
            'runtime_loop_guard_current_action_number',
            'runtime_loop_guard_original_action_number',
            'runtime_loop_guard_fallback_action_number',
            'runtime_loop_guard_previous_streak',
            'runtime_loop_guard_candidate_streak',
            'runtime_loop_guard_min_streak',
            'runtime_loop_guard_revisit_count',
            'runtime_loop_guard_min_revisit',
            'runtime_loop_guard_step',
            'runtime_loop_guard_min_step',
            'runtime_loop_guard_public_history_ready',
            'runtime_loop_guard_execution_failed',
            'runtime_loop_guard_stop_active',
            'action_first_stop_confirmation_enabled',
            'action_first_stop_confirmation_require_final_action',
            'action_first_stop_confirmation_crosscheck_prompt',
            'action_first_stop_confirmation_verifier_model',
            'action_first_stop_confirmation_input_images',
            'action_first_stop_confirmation_prompt_sha256',
            'action_first_stop_confirmation_rgb_log_key',
            'action_first_stop_confirmation_rgb_pixel_sha256',
            'action_first_stop_confirmation_rgb_shape',
            'action_first_stop_confirmation_rgb_dtype',
            'action_first_stop_confirmation_rgb_capture_error',
            'action_first_stop_confirmation_called',
            'action_first_stop_confirmation_outcome',
            'action_first_stop_confirmation_trigger',
            'action_first_stop_proposal_action_number',
            'action_first_stop_proposal_legacy_wants_stop',
            'action_first_stop_proposal_suppressed_inconsistent',
            'terminal_certificate_every_step_enabled',
            'terminal_certificate_prompt_audit_enabled',
            'terminal_certificate_called',
            'terminal_certificate_stop',
            'terminal_certificate_outcome',
            'terminal_certificate_error',
            'terminal_certificate_policy_action_number',
            'terminal_certificate_overrode_move',
            'terminal_certificate_terminal_state',
            'terminal_certificate_target_visible',
            'terminal_certificate_near_enough',
            'terminal_certificate_done',
            'terminal_certificate_input_rgb_log_key',
            'terminal_certificate_input_rgb_pixel_sha256',
            'terminal_certificate_input_rgb_shape',
            'terminal_certificate_input_rgb_dtype',
            'terminal_certificate_input_rgb_capture_error',
            'terminal_certificate_input_rgb_storage_contract',
            'terminal_certificate_input_rgb_filename',
            'terminal_certificate_input_rgb_storage_format',
            'terminal_certificate_input_rgb_storage_mime_type',
            'terminal_certificate_input_rgb_storage_max_dimension',
            'terminal_certificate_input_rgb_storage_resize_resample',
            'terminal_certificate_input_rgb_storage_jpeg_quality',
            'terminal_certificate_input_rgb_pixel_semantics',
            'terminal_certificate_input_rgb_source_pixel_sha256',
            'terminal_certificate_input_rgb_source_shape',
            'terminal_certificate_input_rgb_source_dtype',
            'terminal_certificate_input_rgb_preencode_pixel_sha256',
            'terminal_certificate_input_rgb_preencode_shape',
            'terminal_certificate_input_rgb_encoded_file_sha256',
            'terminal_certificate_input_rgb_encoded_bytes',
            'terminal_certificate_only_image_logging',
            'terminal_certificate_prompt_sha256',
        ):
            if key in row:
                payload[key] = row.get(key)
        if 'action_contract_repaired' in row:
            payload['action_contract_repaired'] = row.get('action_contract_repaired')
        if 'action_contract_repair_reason' in row:
            payload['action_contract_repair_reason'] = row.get('action_contract_repair_reason')
        payload = self._sanitize_for_json(payload)
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        model_io_path = os.path.join(episode_dir, 'model_io.jsonl')
        with open(model_io_path, 'a', encoding='utf-8') as file:
            file.write(text + '\n')
        if os.environ.get('VLMNAV_MODEL_IO_STDOUT', '1').strip().lower() not in ('0', 'false', 'no', 'off'):
            logging.info('[VLMNAV_MODEL_IO] %s', text)

    def _position_key(self, position, decimals: int = 1):
        arr = np.asarray(position, dtype=float)
        if arr.shape[0] < 3:
            return None
        return tuple(np.round(arr[[0, 2]], decimals=decimals).tolist())

    def _get_agent_yaw_deg(self, rotation):
        try:
            theta, axis = quat_to_angle_axis(rotation)
            if axis[1] < 0:
                theta = 2 * np.pi - theta
            return float(np.degrees(theta))
        except Exception:
            return None

    def _get_goal_positions_for_logging(self):
        current_episode = getattr(self, 'current_episode', None)
        if isinstance(current_episode, dict):
            positions = current_episode.get('object_positions') or current_episode.get('view_positions') or []
            return [np.asarray(pos, dtype=float) for pos in positions]
        if isinstance(current_episode, list):
            curr_goal_ndx = getattr(self, 'curr_goal_ndx', 0)
            if 0 <= curr_goal_ndx < len(current_episode):
                goal = current_episode[curr_goal_ndx] or {}
                positions = goal.get('view_points') or []
                if not positions and goal.get('image_position') is not None:
                    positions = [goal['image_position']]
                return [np.asarray(pos, dtype=float) for pos in positions]
        return []

    def _infer_room_info(self, position):
        room_info = {
            'room_id': None,
            'room_label': None,
            'room_source': None,
        }
        try:
            sim = getattr(self.simWrapper, 'sim', None)
            semantic_scene = getattr(sim, 'semantic_scene', None)
            regions = getattr(semantic_scene, 'regions', None)
            if not regions:
                return room_info

            pos = np.asarray(position, dtype=float)
            best_match = None
            best_volume = None
            for idx, region in enumerate(regions):
                aabb = getattr(region, 'aabb', None)
                center = np.asarray(getattr(aabb, 'center', []), dtype=float)
                sizes = np.asarray(getattr(aabb, 'sizes', []), dtype=float)
                if center.shape[0] != 3 or sizes.shape[0] != 3:
                    continue
                half = sizes / 2.0 + 1e-4
                if np.all(np.abs(pos - center) <= half):
                    volume = float(np.prod(np.maximum(sizes, 1e-6)))
                    if best_volume is None or volume < best_volume:
                        category = getattr(region, 'category', None)
                        label = None
                        if category is not None:
                            name_attr = getattr(category, 'name', None)
                            try:
                                label = name_attr() if callable(name_attr) else name_attr
                            except Exception:
                                label = str(category)
                        best_match = {
                            'room_id': str(getattr(region, 'id', idx)),
                            'room_label': label,
                            'room_source': 'semantic_region_aabb',
                        }
                        best_volume = volume
            if best_match is not None:
                return best_match
        except Exception:
            pass
        return room_info

    def _compute_target_geometry(self, agent_state: habitat_sim.AgentState):
        fields = {
            'target_visible_heuristic': False,
            'visible_target_count_heuristic': 0,
            'target_relative_angle_deg': None,
            'target_relative_distance': None,
            'target_relative_position': None,
            'nearest_target_position': None,
            'target_room_id': None,
            'target_room_label': None,
            'same_room_as_target': None,
        }
        goal_positions = self._get_goal_positions_for_logging()
        if not goal_positions:
            return fields

        sensor_state = agent_state.sensor_states['color_sensor']
        resolution = self.agent.resolution
        focal_length = self.agent.focal_length
        fov = float(self.agent.fov)

        nearest_dist = None
        nearest_pos = None
        nearest_angle = None
        visible_count = 0
        for pos in goal_positions:
            local = global_to_local(agent_state.position, agent_state.rotation, pos)
            planar_dist = float(np.linalg.norm([local[0], local[2]]))
            angle_deg = float(np.degrees(np.arctan2(local[0], -local[2])))
            px = agent_frame_to_image_coords(
                local,
                agent_state,
                sensor_state,
                resolution=resolution,
                focal_length=focal_length,
            )
            in_fov = local[2] < 0 and abs(angle_deg) <= (fov / 2.0)
            if px is not None and in_fov:
                x, y = px
                if 0 <= x < resolution[1] and 0 <= y < resolution[0]:
                    visible_count += 1
            if nearest_dist is None or planar_dist < nearest_dist:
                nearest_dist = planar_dist
                nearest_pos = pos
                nearest_angle = angle_deg

        fields['target_visible_heuristic'] = visible_count > 0
        fields['visible_target_count_heuristic'] = int(visible_count)
        if nearest_pos is not None:
            local = global_to_local(agent_state.position, agent_state.rotation, nearest_pos)
            fields['target_relative_angle_deg'] = nearest_angle
            fields['target_relative_distance'] = float(nearest_dist)
            fields['target_relative_position'] = np.asarray(local, dtype=float).tolist()
            fields['nearest_target_position'] = np.asarray(nearest_pos, dtype=float).tolist()
            target_room = self._infer_room_info(nearest_pos)
            fields['target_room_id'] = target_room.get('room_id')
            fields['target_room_label'] = target_room.get('room_label')
        agent_room_id = getattr(self, '_current_room_id_for_step', None)
        if agent_room_id is not None and fields['target_room_id'] is not None:
            fields['same_room_as_target'] = bool(agent_room_id == fields['target_room_id'])
        return fields

    def _normalize_candidate_actions(self, agent_metadata):
        if not isinstance(agent_metadata, dict):
            return []
        actions = agent_metadata.get('a_final')
        if actions is None:
            return []
        if isinstance(actions, dict):
            actions = list(actions.keys())
        return list(actions)

    def _compute_candidate_action_fields(self, agent_state: habitat_sim.AgentState, step_metadata: dict, agent_metadata: dict):
        candidate_actions = self._normalize_candidate_actions(agent_metadata)
        candidate_scores = []
        if isinstance(agent_metadata, dict):
            candidate_scores = list(agent_metadata.get('candidate_action_scores') or [])
        fields = {
            'candidate_action_count': int(len(candidate_actions)),
            'frontier_count': int(len(candidate_actions)),
            'candidate_actions': [],
            'candidate_action_scores': self._sanitize_for_json(candidate_scores),
            'visible_doorway_count_heuristic': int(sum(1 for item in candidate_scores if item.get('frontier_like'))),
            'chosen_action_r': None,
            'chosen_action_theta_deg': None,
            'chosen_action_theta_deg_model_frame': None,
            'chosen_action_yaw_command_deg': None,
            'chosen_action_world': None,
            'candidate_geometry_frame_version': 3,
            'privileged_candidate_labels_logged_post_inference': False,
        }
        log_privileged_labels = os.environ.get(
            'VLMNAV_LOG_PRIVILEGED_CANDIDATE_LABELS',
            '0',
        ).strip().lower() not in ('0', 'false', 'no', 'off')
        original_requested_start = None
        if log_privileged_labels:
            try:
                original_requested_start = np.asarray(
                    self.path_calculator.requested_start,
                    dtype=np.float32,
                ).copy()
            except Exception:
                original_requested_start = None
        for idx, action in enumerate(candidate_actions, start=1):
            try:
                r, theta = action
                model_theta = float(theta)
                angle_fields = candidate_angle_metadata(model_theta)
                local_goal = np.asarray(
                    candidate_model_displacement(float(r), model_theta),
                    dtype=float,
                )
                world_goal = local_to_global(agent_state.position, agent_state.rotation, local_goal)
                candidate_fields = {
                    'index': idx,
                    'r': float(r),
                    **angle_fields,
                    'landing_position': np.asarray(world_goal, dtype=float).tolist(),
                    'geometry_frame_version': 3,
                    'landing_position_source': 'model_bearing_from_agent_pose',
                }
                if log_privileged_labels:
                    try:
                        self.path_calculator.requested_start = np.asarray(world_goal, dtype=np.float32)
                        geodesic = float(self.simWrapper.get_path(self.path_calculator))
                        if np.isfinite(geodesic):
                            candidate_fields['oracle_geodesic_distance_to_goal'] = geodesic
                        else:
                            candidate_fields['oracle_geodesic_distance_to_goal'] = None
                        candidate_fields['oracle_label_private'] = True
                        fields['privileged_candidate_labels_logged_post_inference'] = True
                    except Exception:
                        candidate_fields['oracle_geodesic_distance_to_goal'] = None
                        candidate_fields['oracle_label_private'] = True
                fields['candidate_actions'].append(candidate_fields)
            except Exception:
                continue

        if original_requested_start is not None:
            try:
                self.path_calculator.requested_start = original_requested_start
            except Exception:
                pass

        try:
            action_number = int(step_metadata.get('action_number'))
        except Exception:
            action_number = None
        if action_number is not None and 1 <= action_number <= len(fields['candidate_actions']):
            chosen = fields['candidate_actions'][action_number - 1]
            fields['chosen_action_r'] = chosen['r']
            fields['chosen_action_theta_deg'] = chosen['theta_deg_model_frame']
            fields['chosen_action_theta_deg_model_frame'] = chosen['theta_deg_model_frame']
            fields['chosen_action_yaw_command_deg'] = chosen['yaw_command_deg']
            fields['chosen_action_world'] = chosen['landing_position']
        return fields

    def _compute_revisit_fields(self, agent_state: habitat_sim.AgentState):
        fields = {
            'position_key': None,
            'visit_count_current_cell': None,
            'revisit_count': None,
            'loop_risk': False,
        }
        key = self._position_key(agent_state.position)
        if key is None:
            return fields
        count = self._visited_position_counts.get(key, 0) + 1
        self._visited_position_counts[key] = count
        fields['position_key'] = list(key)
        fields['visit_count_current_cell'] = int(count)
        fields['revisit_count'] = int(max(count - 1, 0))
        fields['loop_risk'] = bool(count >= 4)
        return fields

    def _compute_stop_fields(self, step_metadata: dict, logging_data: dict):
        fields = {
            'stop_signal': bool(step_metadata.get(
                'stop_signal',
                step_metadata.get('called_stopping', False),
            )),
            'stop_source': None,
            'stop_response_present': False,
            'stop_response_text': None,
        }
        stop_response = logging_data.get('STOPPING RESPONSE')
        if stop_response is not None and str(stop_response).strip():
            fields['stop_response_present'] = True
            fields['stop_response_text'] = str(stop_response)
            fields['stop_source'] = 'vlm_stopping'
        return fields

    def _compute_motion_feedback_fields(self, obs: dict):
        feedback = obs.get('motion_feedback') if isinstance(obs, dict) else None
        if not isinstance(feedback, dict):
            return {}
        return {
            'execution_feedback_public': True,
            'execution_sequence_id': feedback.get('sequence_id'),
            'execution_requested_distance': feedback.get('requested_distance'),
            'execution_requested_yaw_rad': feedback.get('requested_yaw_rad'),
            'execution_net_displacement': feedback.get('net_displacement'),
            'execution_traveled_distance': feedback.get('traveled_distance'),
            'execution_ratio': feedback.get('execution_ratio'),
            'execution_collision_substeps': feedback.get('collision_substeps'),
            'execution_slide_rejected_substeps': feedback.get('slide_rejected_substeps'),
            'execution_underexecuted': bool(feedback.get('underexecuted', False)),
        }

    def _build_spatial_debug_fields(self, obs: dict, step_metadata: dict, logging_data: dict, agent_metadata: dict):
        agent_state = obs['agent_state']
        room_info = self._infer_room_info(agent_state.position)
        current_room_id = room_info.get('room_id')
        fields = {
            'agent_yaw_deg': self._get_agent_yaw_deg(agent_state.rotation),
            'agent_forward_xz': np.asarray(
                habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)[[0, 2]],
                dtype=float,
            ).tolist(),
            'agent_room_id': current_room_id,
            'current_room_id': current_room_id,
            'current_room_label': room_info.get('room_label'),
            'room_source': room_info.get('room_source'),
            'room_transition_this_step': bool(
                self._previous_room_id is not None and current_room_id is not None and self._previous_room_id != current_room_id
            ),
            'crossed_doorway_this_step': bool(
                self._previous_room_id is not None and current_room_id is not None and self._previous_room_id != current_room_id
            ),
        }
        self._current_room_id_for_step = current_room_id
        fields.update(self._compute_target_geometry(agent_state))
        fields.update(self._compute_candidate_action_fields(agent_state, step_metadata, agent_metadata))
        fields.update(self._compute_revisit_fields(agent_state))
        fields.update(self._compute_stop_fields(step_metadata, logging_data))
        fields.update(self._compute_motion_feedback_fields(obs))
        prev_theta = getattr(self, '_previous_chosen_action_theta_deg', None)
        curr_theta = fields.get('chosen_action_theta_deg')
        angle_delta = None
        if prev_theta is not None and curr_theta is not None:
            angle_delta = abs(((curr_theta - prev_theta + 180.0) % 360.0) - 180.0)
        fields['replan_angle_delta_deg'] = angle_delta
        fields['replan_flag'] = bool(angle_delta is not None and angle_delta > 45.0)
        if curr_theta is not None:
            self._previous_chosen_action_theta_deg = curr_theta
        self._previous_room_id = current_room_id
        return self._sanitize_for_json(fields)

    def _build_episode_summary(self):
        current_episode = getattr(self, 'current_episode', {}) or {}
        summary = {
            'run_name': self.curr_run_name,
            'scene_id': self.curr_run_name.split('_', 1)[1] if '_' in self.curr_run_name else self.curr_run_name,
            'steps_taken': int(len(self.df)),
            'habitat_steps': int(getattr(self, 'habitat_steps', 0) or 0),
            'habitat_step_budget': int(self._habitat_step_budget()),
            'habitat_step_hard_horizon_enabled': bool(
                self._strict_habitat_step_budget_enabled()
            ),
            'habitat_step_hard_horizon_invariant_violation': bool(
                getattr(self, '_habitat_step_hard_horizon_invariant_violation', False)
            ),
            'agent_distance_traveled': float(self.agent_distance_traveled),
        }
        if isinstance(current_episode, dict):
            summary.update({
                'object': current_episode.get('object'),
                'shortest_path': current_episode.get('shortest_path'),
                'target_instance_count': len(current_episode.get('object_positions', []) or []),
                'viewpoint_count': len(current_episode.get('view_positions', []) or []),
            })
        if self.df.empty:
            summary.update({
                'finish_status': 'empty',
                'goal_reached': False,
                'spl': 0.0,
                'final_distance_to_goal': None,
                'min_distance_to_goal': None,
                'room_transition_count': 0,
                'max_revisit_count': 0,
            })
            return self._sanitize_for_json(summary)

        last = self.df.iloc[-1]
        summary.update({
            'finish_status': str(last.get('finish_status', 'unknown')),
            'goal_reached': bool(last.get('goal_reached', False)),
            'terminal_action_was_stop': bool(last.get('terminal_action_was_stop', False)),
            'terminal_stop_was_forced_by_budget': bool(last.get('terminal_stop_was_forced_by_budget', False)),
            'effective_terminal_action_was_stop': bool(last.get('effective_terminal_action_was_stop', last.get('terminal_action_was_stop', False))),
            'force_stop_at_episode_budget': bool(last.get('force_stop_at_episode_budget', False)),
            'success_requires_explicit_stop': bool(last.get('success_requires_explicit_stop', False)),
            'habitat_step_hard_horizon_blocked': bool(
                last.get('habitat_step_hard_horizon_blocked', False)
            ),
            'habitat_step_next_action_cost': (
                int(last.get('habitat_step_next_action_cost'))
                if pd.notna(last.get('habitat_step_next_action_cost'))
                else None
            ),
            'habitat_step_remaining_before_action': (
                int(last.get('habitat_step_remaining_before_action'))
                if pd.notna(last.get('habitat_step_remaining_before_action'))
                else None
            ),
            'habitat_step_action_execution_allowed': bool(
                last.get('habitat_step_action_execution_allowed', True)
            ),
            'habitat_step_horizon_block_reason': str(
                last.get('habitat_step_horizon_block_reason', '') or ''
            ),
            'spl': float(last.get('spl', 0.0)),
            'final_distance_to_goal': float(last.get('distance_to_goal')) if pd.notna(last.get('distance_to_goal')) else None,
            'min_distance_to_goal': float(pd.to_numeric(self.df['distance_to_goal'], errors='coerce').min()) if 'distance_to_goal' in self.df.columns else None,
            'room_transition_count': int(self.df['room_transition_this_step'].fillna(False).astype(bool).sum()) if 'room_transition_this_step' in self.df.columns else 0,
            'target_visible_heuristic_count': int(self.df['target_visible_heuristic'].fillna(False).astype(bool).sum()) if 'target_visible_heuristic' in self.df.columns else 0,
            'max_revisit_count': int(pd.to_numeric(self.df['revisit_count'], errors='coerce').fillna(0).max()) if 'revisit_count' in self.df.columns else 0,
            'called_stopping_count': int(self.df['called_stopping'].fillna(False).astype(bool).sum()) if 'called_stopping' in self.df.columns else 0,
            'replan_count': int(self.df['replan_flag'].fillna(False).astype(bool).sum()) if 'replan_flag' in self.df.columns else 0,
            'max_visible_doorway_count_heuristic': int(pd.to_numeric(self.df['visible_doorway_count_heuristic'], errors='coerce').fillna(0).max()) if 'visible_doorway_count_heuristic' in self.df.columns else 0,
        })
        return self._sanitize_for_json(summary)

class GOATEnv(Env):
    """
    Environment for the GOAT task, extending the base Env class.
    This class defines the setup, initialization, and running of GOAT episodes.
    """

    task = 'GOAT'

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.split = 'val' if 'val' in self.cfg['split'] else 'train'
        self.sim_cfg['scene_config'] = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

        self.all_episodes = []
        self.goals = {}
        for f in sorted(os.listdir(f'data/datasets/goat_bench/hm3d/v1/{self.cfg["split"]}/content')):
            path = f'data/datasets/goat_bench/hm3d/v1/{self.cfg["split"]}/content/{f}'
            js = _load_json_maybe_gzip(path)
            hsh = f.split('.')[0]
            self.goals[hsh] = js['goals']
            self.all_episodes += js['episodes']
        self.num_episodes = len(self.all_episodes)

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the GOAT task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)

        episode = self.all_episodes[episode_ndx]
        f, glb = episode['scene_id'].split('/')[-2:]
        hsh = f[6:]
        goals = self.goals[hsh]
        self.sim_cfg['scene_id'] = f[2:5]
        self.sim_cfg['scene_path'] = f'data/scene_datasets/hm3d/{self.split}/{f}/{glb}'
        self.simWrapper = SimWrapper(self.sim_cfg)
        self.current_episode = []

        for goal in episode['tasks']:
            name = goal[0]
            mode = goal[1]
            subgoal = {'name': name, 'mode': mode, 'id': goal[2], 'view_points': []}
            for obj in goals[f'{f[6:]}.basis.glb_{name}']:
                if mode == 'object':
                    subgoal['view_points'] += [a['agent_state']['position'] for a in obj['view_points']]
                else:
                    if obj['object_id'] == goal[2]:
                        subgoal['view_points'] = [a['agent_state']['position'] for a in obj['view_points']]
                        if mode == 'description':
                            subgoal['lang_desc'] = obj['lang_desc']
                        if mode == 'image':
                            image_ndx = goal[3]
                            subgoal['image_position'] = obj['image_goals'][image_ndx]['position']
                            subgoal['image_rotation'] = obj['image_goals'][image_ndx]['rotation']

            self.current_episode.append(subgoal)

        logging.info(f'\nRUNNING EPISODE {episode_ndx}, SCENE: {self.simWrapper.scene_id}')
        for i, subgoal in enumerate(self.current_episode):
            logging.info(f'Goal {i}: {subgoal["name"]}, {subgoal["mode"]}')

        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_goal_ndx = 0
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"
        self.last_goal_reset = -1
        self.path_calculator.requested_ends = np.array(self.current_episode[self.curr_goal_ndx]['view_points'], dtype=np.float32)
        self.path_calculator.requested_start = self.init_pos
        self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)

        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment for the GOAT task.

        Args:
            obs (dict): The current observation.

        Returns:
            list: The next action to be taken by the agent.
        """
        super()._step_env(obs)

        goal = self.current_episode[self.curr_goal_ndx]
        obs['goal'] = goal
        if goal['mode'] == 'image':
            position = goal['image_position']
            rotation = goal['image_rotation']
            goal_im = self.simWrapper.get_goal_image(position, rotation)
            put_text_on_image(goal_im, f"GOAL IMAGE: {goal['name']}", bg_color=(255, 255, 255), location='top_center')
            obs['goal']['goal_image'] = goal_im

        agent_state = obs['agent_state']
        agent_action, metadata = self.agent.step(obs)
        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']

        metrics = self._calculate_metrics(
            agent_state,
            agent_action,
            self.curr_shortest_path,
            self.last_goal_reset + 1 + self.cfg['max_steps_per_subgoal'],
        )
        step_metadata.update(metrics)
        step_metadata.update(self._build_spatial_debug_fields(obs, step_metadata, logging_data, metadata))

        if metrics['done']:
            # Log explicit subtask failure/success for easier debugging
            if not metrics['goal_reached']:
                logging.info(
                    f"SUBTASK FAILED | episode={self.curr_run_name} "
                    f"goal_idx={self.curr_goal_ndx} "
                    f"name={goal.get('name', 'unknown')} "
                    f"mode={goal.get('mode', 'unknown')} "
                    f"finish_status={metrics.get('finish_status', 'unknown')} "
                    f"distance_to_goal={metrics.get('distance_to_goal', -1):.3f}"
                )
            else:
                logging.info(
                    f"SUBTASK SUCCEEDED | episode={self.curr_run_name} "
                    f"goal_idx={self.curr_goal_ndx} "
                    f"name={goal.get('name', 'unknown')} "
                    f"mode={goal.get('mode', 'unknown')} "
                    f"spl={metrics.get('spl', 0.0):.3f}"
                )

            self.wandb_log_data['task_data'].setdefault('goal_data', []).append(
                {
                    'goal_mode': goal['mode'],
                    'goal_reached': metrics['goal_reached'],
                    'spl': metrics['spl'],
                }
            )

            if 'spl' in self.wandb_log_data:
                del self.wandb_log_data['spl']
            if 'goal_reached' in self.wandb_log_data:
                del self.wandb_log_data['goal_reached']

            if self.curr_goal_ndx + 1 == len(self.current_episode):
                agent_action = None
            else:
                self.agent.reset_goal()
                self.agent_distance_traveled = 0
                agent_action = PolarAction.null
                self.curr_goal_ndx += 1
                self.last_goal_reset = self.step
                goal = self.current_episode[self.curr_goal_ndx]
                self.path_calculator.requested_ends = np.array(goal['view_points'], dtype=np.float32)
                self.path_calculator.requested_start = obs['agent_state'].position
                self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)

                logging.info(f'New goal {goal["mode"]}: {goal["name"]}, GEODESIC: {self.curr_shortest_path}')

        self._log(images, step_metadata, logging_data)

        return agent_action

class ObjectNavEnv(Env):
    """
    Environment for the ObjectNav task, extending the base Env class.
    This class defines the setup, initialization, and running of ObjectNav episodes.
    """

    task = 'ObjectNav'

    def _resolve_split_scene_config(self):
        split = str(self.cfg.get('split', 'val'))
        scene_root = self.dataset_info['scene_root']
        candidates = [
            os.path.join(scene_root, split, f'hm3d_annotated_{split}_basis.scene_dataset_config.json'),
            os.path.join(scene_root, split, f'hm3d_{split}_basis.scene_dataset_config.json'),
            os.path.join(scene_root, f'hm3d_annotated_{split}_basis.scene_dataset_config.json'),
            os.path.join(scene_root, f'hm3d_{split}_basis.scene_dataset_config.json'),
            self.dataset_info['scene_config'],
        ]
        return _first_existing_path(
            candidates,
            f"ObjectNav {self.dataset_info['dataset']} scene dataset config for split={split}",
        )

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.all_episodes = []
        self._objectnav_sparse_runtime = None
        self._sparse_goal_objects_by_global_index = {}
        self.dataset_info = _resolve_objectnav_dataset_paths(self.cfg.get('dataset'))
        self.sim_cfg['scene_config'] = self._resolve_split_scene_config()
        self.goals = {}
        content_dir = os.path.join(self.dataset_info['objectnav_root'], self.cfg['split'], 'content')
        logging.info(
            "ObjectNav dataset=%s scene_root=%s episodes_dir=%s",
            self.dataset_info['dataset'],
            self.dataset_info['scene_root'],
            content_dir,
        )

        sparse_runtime = load_sparse_objectnav_runtime_from_environment(
            cfg_split=self.cfg['split'],
            dataset_name=self.dataset_info['dataset'],
            content_dir=content_dir,
        )
        if sparse_runtime is not None:
            self._objectnav_sparse_runtime = sparse_runtime
            self.all_episodes = sparse_runtime.episodes_by_global_index
            self._sparse_goal_objects_by_global_index = (
                sparse_runtime.goal_objects_by_global_index
            )
            # Keep global HM3Dv2 indices and range checks identical to the
            # canonical train dataset; only resident payloads are sparse.
            self.num_episodes = sparse_runtime.canonical_episode_count
            logging.info(
                "ObjectNav sparse train runtime selected=%d canonical_episodes=%d "
                "manifest_sha256=%s bundle_sha256=%s content_sha256=%s",
                len(sparse_runtime.selected_indices),
                sparse_runtime.canonical_episode_count,
                sparse_runtime.source_manifest_sha256,
                sparse_runtime.bundle_sha256,
                sparse_runtime.source_content_sha256,
            )
            return

        for f in sorted(os.listdir(content_dir)):
            path = os.path.join(content_dir, f)
            js = _load_json_maybe_gzip(path)
            hsh = f.split('.')[0]
            self.goals[hsh] = js['goals_by_category']
            self.all_episodes += js['episodes']

        self.num_episodes = len(self.all_episodes)

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the ObjectNav task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)
        self._validate_strict_habitat_step_runtime_contract()
        episode, f, all_objects = self._objectnav_episode_payload(episode_ndx)
        self.sim_cfg['scene_id'] = f[1][2:5]
        scene_candidates = [
            os.path.join(self.dataset_info['scene_root'], f[1], f[2]),
            os.path.join(self.dataset_info['scene_root'], self.cfg['split'], f[1], f[2]),
        ]
        if f[2].endswith('.basis.glb'):
            scene_file_glb = f[2].replace('.basis.glb', '.glb')
            scene_candidates.extend([
                os.path.join(self.dataset_info['scene_root'], f[1], scene_file_glb),
                os.path.join(self.dataset_info['scene_root'], self.cfg['split'], f[1], scene_file_glb),
            ])
        self.sim_cfg['scene_path'] = _first_existing_path(scene_candidates, f"scene file for episode {episode_ndx}")
        self.simWrapper = SimWrapper(self.sim_cfg)

        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)
        logging.info(f'RUNNING EPISODE {episode_ndx} with {episode["object_category"]} and {len(all_objects)} instances. GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]}')
        if episode['object_category'] == 'tv_monitor':
            episode['object_category'] = 'tv screen'
        self.current_episode = {
            'object': episode['object_category'],
            'shortest_path': episode['info']['geodesic_distance'],
            'object_positions': [a['position'] for a in all_objects],
            'view_positions': view_positions
        }
        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _objectnav_episode_payload(self, episode_ndx: int):
        """Resolve one canonical global episode and its exact goal objects.

        Kept separate from simulator construction so the sparse/legacy data
        wiring can be verified byte-for-byte on CPU.
        """
        episode = self.all_episodes[episode_ndx]
        f = episode['scene_id'].split('/')[1:]
        if self._objectnav_sparse_runtime is None:
            goals = self.goals[f[1][6:]]
            all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        else:
            try:
                all_objects = self._sparse_goal_objects_by_global_index[episode_ndx]
            except KeyError as exc:
                raise RuntimeError(
                    f"episode {episode_ndx} is not present in the validated sparse runtime"
                ) from exc
        return episode, f, all_objects

    def _lightweight_panoramic_planner_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_PLANNER")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_planner", False))

    def _validate_strict_habitat_step_runtime_contract(self) -> None:
        """Reject unmetered simulator scans in a strict-horizon rollout.

        The selected Compass macro action is priced by
        :meth:`_strict_habitat_horizon_for_action`.  Multiview stop capture,
        panoramic planning, and the action-0 recovery scan invoke
        ``SimWrapper.step`` independently, so treating them as free would
        make a claimed 500-step horizon false.  The terminal-certificate
        wrappers deliberately use the single-view/no-scan configuration.
        """
        if not self._strict_habitat_step_budget_enabled():
            return
        violations = []
        if self._stop_multiview_num_views() > 1:
            violations.append(
                "VLMNAV_STOP_MULTIVIEW_NUM_VIEWS must be 1"
            )
        if self._lightweight_panoramic_planner_enabled():
            violations.append(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_PLANNER must be disabled"
            )
        if self._action0_pano_replace_enabled():
            violations.append(
                "VLMNAV_ACTION0_PANO_REPLACE must be disabled"
            )
        if violations:
            raise RuntimeError(
                "strict Habitat-step protocol forbids unmetered simulator "
                "scans: " + "; ".join(violations)
            )

    def _lightweight_panoramic_mode_gate_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_MODE_GATE")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_mode_gate", True))

    def _lightweight_panoramic_initial_scan_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_INITIAL_SCAN")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_initial_scan", True))

    def _lightweight_panoramic_reactive_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_REACTIVE")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_reactive", False))

    def _lightweight_panoramic_context_only_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_CONTEXT_ONLY")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_context_only", False))

    def _lightweight_panoramic_score_only_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_SCORE_ONLY")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_score_only", False))

    def _lightweight_panoramic_override_action_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_OVERRIDE_ACTION")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_override_action", False))

    def _lightweight_panoramic_override_max_angle_diff(self) -> float:
        try:
            value = float(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_OVERRIDE_MAX_ANGLE_DIFF",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_override_max_angle_diff", 55)),
            ))
        except (TypeError, ValueError):
            value = 55.0
        return max(0.0, value)

    def _lightweight_panoramic_override_min_candidate_dist(self) -> float:
        try:
            value = float(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_OVERRIDE_MIN_CANDIDATE_DIST",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_override_min_candidate_dist", 0.75)),
            ))
        except (TypeError, ValueError):
            value = 0.75
        return max(0.0, value)

    def _lightweight_panoramic_override_theta_field(self) -> str:
        value = os.environ.get(
            "VLMNAV_LIGHTWEIGHT_PANORAMIC_OVERRIDE_THETA_FIELD",
            str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_override_theta_field", "model")),
        )
        value = str(value).strip().lower()
        if value not in {"model", "execute"}:
            value = "model"
        return value

    def _lightweight_panoramic_override_allowed_actions(self) -> set[str]:
        value = os.environ.get(
            "VLMNAV_LIGHTWEIGHT_PANORAMIC_OVERRIDE_ALLOWED_ACTIONS",
            str(
                getattr(self.agent, "cfg", {}).get(
                    "lightweight_panoramic_override_allowed_actions",
                    "candidate0",
                )
            ),
        )
        tokens = {
            token.strip().lower()
            for token in str(value).replace(";", ",").split(",")
            if token.strip()
        }
        if "any" in tokens:
            return {"candidate0", "forward"}
        return tokens or {"candidate0"}

    def _lightweight_panoramic_reactive_action_filter(self) -> set[str]:
        value = os.environ.get(
            "VLMNAV_LIGHTWEIGHT_PANORAMIC_REACTIVE_ACTION_FILTER",
            str(
                getattr(self.agent, "cfg", {}).get(
                    "lightweight_panoramic_reactive_action_filter",
                    "candidate0",
                )
            ),
        )
        tokens = {
            token.strip().lower()
            for token in str(value).replace(";", ",").split(",")
            if token.strip()
        }
        if "any" in tokens:
            return {"candidate0", "forward"}
        return tokens or {"candidate0"}

    def _lightweight_panoramic_override_min_switch_margin_deg(self) -> float:
        try:
            value = float(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_OVERRIDE_MIN_SWITCH_MARGIN_DEG",
                str(
                    getattr(self.agent, "cfg", {}).get(
                        "lightweight_panoramic_override_min_switch_margin_deg",
                        20,
                    )
                ),
            ))
        except (TypeError, ValueError):
            value = 20.0
        return max(0.0, value)

    def _lightweight_panoramic_min_interval(self) -> int:
        try:
            value = int(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_MIN_INTERVAL",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_min_interval", 3)),
            ))
        except (TypeError, ValueError):
            value = 3
        return max(0, value)

    def _lightweight_panoramic_min_step(self) -> int:
        try:
            value = int(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_MIN_STEP",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_min_step", 0)),
            ))
        except (TypeError, ValueError):
            value = 0
        return max(0, value)

    def _lightweight_panoramic_lock_steps(self) -> int:
        try:
            value = int(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_LOCK_STEPS",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_lock_steps", 8)),
            ))
        except (TypeError, ValueError):
            value = 8
        return max(0, value)

    def _lightweight_panoramic_max_scans_per_episode(self) -> int:
        try:
            value = int(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_MAX_SCANS_PER_EPISODE",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_max_scans_per_episode", 0)),
            ))
        except (TypeError, ValueError):
            value = 0
        return max(0, value)

    def _lightweight_panoramic_min_select_score(self) -> float:
        try:
            value = float(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_MIN_SELECT_SCORE",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_min_select_score", 0)),
            ))
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, value)

    def _lightweight_panoramic_min_score_margin(self) -> float:
        try:
            value = float(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_MIN_SCORE_MARGIN",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_min_score_margin", 0)),
            ))
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, value)

    def _lightweight_panoramic_min_mode_streak(self) -> int:
        try:
            value = int(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_MIN_MODE_STREAK",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_min_mode_streak", 2)),
            ))
        except (TypeError, ValueError):
            value = 2
        return max(1, value)

    def _lightweight_panoramic_min_action_repeat_streak(self) -> int:
        try:
            value = int(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_MIN_ACTION_REPEAT_STREAK",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_min_action_repeat_streak", 0)),
            ))
        except (TypeError, ValueError):
            value = 0
        return max(0, value)

    def _lightweight_panoramic_action_repeat_streak(self):
        history = list(getattr(self.agent, "prompt_action_history", []) or [])
        if not history:
            return None, 0
        last_action = history[-1]
        streak = 0
        for action in reversed(history):
            if action != last_action:
                break
            streak += 1
        try:
            last_action = int(last_action)
        except (TypeError, ValueError):
            last_action = None
        return last_action, int(streak)

    def _lightweight_panoramic_progress_gate_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_PROGRESS_GATE")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_progress_gate", False))

    def _lightweight_panoramic_progress_require_mode(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_PROGRESS_REQUIRE_MODE")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_progress_require_mode", False))

    def _lightweight_panoramic_stagnation_steps(self) -> int:
        try:
            value = int(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_STAGNATION_STEPS",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_stagnation_steps", 12)),
            ))
        except (TypeError, ValueError):
            value = 12
        return max(1, value)

    def _lightweight_panoramic_progress_eps(self) -> float:
        try:
            value = float(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_PROGRESS_EPS",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_progress_eps", 0.3)),
            ))
        except (TypeError, ValueError):
            value = 0.3
        return max(0.0, value)

    def _lightweight_panoramic_min_dist_to_goal(self) -> float:
        try:
            value = float(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_MIN_DIST_TO_GOAL",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_min_dist_to_goal", 2.0)),
            ))
        except (TypeError, ValueError):
            value = 2.0
        return max(0.0, value)

    def _lightweight_panoramic_block_near_goal_dist(self) -> float:
        try:
            value = float(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_BLOCK_NEAR_GOAL_DIST",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_block_near_goal_dist", 0.0)),
            ))
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, value)

    def _lightweight_panoramic_near_goal_block_metadata(self) -> dict:
        block_dist = self._lightweight_panoramic_block_near_goal_dist()
        return {
            "lightweight_panoramic_block_near_goal_dist": float(block_dist),
            "lightweight_panoramic_near_goal_blocked": False,
            "lightweight_panoramic_near_goal_block_distance": None,
            "lightweight_panoramic_near_goal_block_source": "disabled_public_inference",
        }

    def _lightweight_panoramic_progress_metadata(self) -> dict:
        last_improvement_step = getattr(self, "_lightweight_panoramic_last_public_progress_step", 0)
        stagnation_steps = self._lightweight_panoramic_stagnation_steps()
        steps_since_improvement = max(0, int(self.step) - int(last_improvement_step or 0))
        stagnant = bool(
            int(steps_since_improvement) >= int(stagnation_steps)
        )
        ready = bool(stagnant)
        metadata = {
            "lightweight_panoramic_progress_gate": bool(self._lightweight_panoramic_progress_gate_enabled()),
            "lightweight_panoramic_progress_current_distance": None,
            "lightweight_panoramic_progress_best_distance": None,
            "lightweight_panoramic_progress_last_improvement_step": last_improvement_step,
            "lightweight_panoramic_progress_steps_since_improvement": steps_since_improvement,
            "lightweight_panoramic_progress_stagnation_steps": int(stagnation_steps),
            "lightweight_panoramic_progress_eps": float(self._lightweight_panoramic_progress_eps()),
            "lightweight_panoramic_progress_min_dist_to_goal": None,
            "lightweight_panoramic_progress_distance_ready": False,
            "lightweight_panoramic_progress_far_enough": True,
            "lightweight_panoramic_progress_stagnant": bool(stagnant),
            "lightweight_panoramic_progress_ready": bool(ready),
            "lightweight_panoramic_progress_source": "public_execution_feedback",
        }
        return metadata

    def _update_lightweight_panoramic_progress_state(self, step_metadata: dict):
        try:
            displacement = float(
                step_metadata.get("previous_execution_net_displacement")
                or step_metadata.get("execution_net_displacement")
                or 0.0
            )
        except (TypeError, ValueError):
            return
        if not np.isfinite(displacement):
            return
        eps = self._lightweight_panoramic_progress_eps()
        revisit = int(step_metadata.get("prompt_revisit_count") or 0)
        if displacement > max(0.05, float(eps)) and revisit <= 1:
            self._lightweight_panoramic_last_public_progress_step = int(self.step)

    def _lightweight_panoramic_lock_active(self) -> bool:
        lock_until = getattr(self, "_lightweight_panoramic_lock_until_step", None)
        return lock_until is not None and int(self.step) < int(lock_until)

    def _lightweight_panoramic_context(self) -> dict:
        context = dict(getattr(self, "_lightweight_panoramic_lock_context", None) or {})
        lock_until = getattr(self, "_lightweight_panoramic_lock_until_step", None)
        context.update({
            "global_state": getattr(self, "_lightweight_panoramic_global_state", "free"),
            "view_direction_deg": float(getattr(self, "_lightweight_panoramic_view_direction_deg", 0.0)),
            "lock_active": bool(self._lightweight_panoramic_lock_active()),
            "lock_until_step": lock_until,
            "lock_steps_remaining": (
                None if lock_until is None else max(0, int(lock_until) - int(self.step))
            ),
        })
        return context

    def _normalize_navigation_mode(self, mode) -> str:
        if mode is None:
            return ""
        normalized = str(mode).strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in {"enter_new_area", "enter_new_region", "frontier_transition"}:
            return "explore_new_area"
        return normalized

    def _lightweight_panoramic_trigger_modes(self) -> set[str]:
        raw = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_TRIGGER_MODES")
        if raw is None:
            raw = getattr(
                self.agent,
                "cfg",
                {},
            ).get("lightweight_panoramic_trigger_modes", "explore_new_area,recover_route")
        values = raw if isinstance(raw, (list, tuple, set)) else str(raw).split(",")
        modes = set()
        for value in values:
            mode = self._normalize_navigation_mode(value)
            if mode:
                modes.add(mode)
        return modes

    def _lightweight_panoramic_reactive_trigger_modes(self) -> set[str]:
        raw = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_REACTIVE_TRIGGER_MODES")
        if raw is None:
            raw = "recover_route,topology_correction,replan"
        values = raw if isinstance(raw, (list, tuple, set)) else str(raw).split(",")
        modes = set()
        for value in values:
            mode = self._normalize_navigation_mode(value)
            if mode:
                modes.add(mode)
        return modes

    def _should_run_reactive_lightweight_panoramic_planner(self, probe_metadata: dict) -> tuple[bool, dict]:
        last_scan_step = getattr(self, "_lightweight_panoramic_last_scan_step", None)
        steps_since_scan = None if last_scan_step is None else int(self.step - last_scan_step)
        action_number = probe_metadata.get("action_number")
        try:
            action_number_int = int(action_number)
        except (TypeError, ValueError):
            action_number_int = None
        nav_mode = self._normalize_navigation_mode(probe_metadata.get("action_navigation_mode"))
        decision_mode = self._normalize_navigation_mode(probe_metadata.get("action_decision_mode"))
        trigger_modes = self._lightweight_panoramic_reactive_trigger_modes()
        mode_hit = bool((nav_mode and nav_mode in trigger_modes) or (decision_mode and decision_mode in trigger_modes))
        metadata = {
            "lightweight_panoramic_reactive": True,
            "lightweight_panoramic_progress_gate": bool(
                self._lightweight_panoramic_progress_gate_enabled()
            ),
            "lightweight_panoramic_min_select_score": float(
                self._lightweight_panoramic_min_select_score()
            ),
            "lightweight_panoramic_min_score_margin": float(
                self._lightweight_panoramic_min_score_margin()
            ),
            "lightweight_panoramic_probe_action_number": action_number_int,
            "lightweight_panoramic_probe_navigation_mode": nav_mode or None,
            "lightweight_panoramic_probe_decision_mode": decision_mode or None,
            "lightweight_panoramic_last_scan_step": last_scan_step,
            "lightweight_panoramic_steps_since_scan": steps_since_scan,
            "lightweight_panoramic_global_state": getattr(
                self, "_lightweight_panoramic_global_state", "free"
            ),
            "lightweight_panoramic_view_direction_deg": float(
                getattr(self, "_lightweight_panoramic_view_direction_deg", 0.0)
            ),
            "lightweight_panoramic_lock_active": bool(self._lightweight_panoramic_lock_active()),
            "lightweight_panoramic_lock_until_step": getattr(
                self, "_lightweight_panoramic_lock_until_step", None
            ),
        }
        scan_count = int(getattr(self, "_lightweight_panoramic_scan_count", 0) or 0)
        max_scans = self._lightweight_panoramic_max_scans_per_episode()
        metadata["lightweight_panoramic_scan_count"] = int(scan_count)
        metadata["lightweight_panoramic_max_scans_per_episode"] = int(max_scans)
        if max_scans > 0 and scan_count >= max_scans:
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": f"reactive_max_scans:{scan_count}/{max_scans}",
            })
            return False, metadata
        near_goal_metadata = self._lightweight_panoramic_near_goal_block_metadata()
        metadata.update(near_goal_metadata)
        min_step = self._lightweight_panoramic_min_step()
        metadata["lightweight_panoramic_min_step"] = int(min_step)
        if int(self.step) < int(min_step):
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": f"reactive_min_step:{min_step}",
            })
            return False, metadata
        min_repeat = self._lightweight_panoramic_min_action_repeat_streak()
        repeat_action, repeat_streak = self._lightweight_panoramic_action_repeat_streak()
        metadata["lightweight_panoramic_min_action_repeat_streak"] = int(min_repeat)
        metadata["lightweight_panoramic_repeated_action"] = repeat_action
        metadata["lightweight_panoramic_repeated_action_streak"] = int(repeat_streak)
        action_filter = self._lightweight_panoramic_reactive_action_filter()
        metadata["lightweight_panoramic_reactive_action_filter"] = ",".join(sorted(action_filter))
        if min_repeat > 0 and repeat_streak < min_repeat:
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": f"reactive_repeat_streak:{repeat_streak}/{min_repeat}",
            })
            return False, metadata
        if bool(near_goal_metadata.get("lightweight_panoramic_near_goal_blocked", False)):
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": (
                    "reactive_near_goal_block:"
                    f"{near_goal_metadata.get('lightweight_panoramic_near_goal_block_distance')}/"
                    f"{near_goal_metadata.get('lightweight_panoramic_block_near_goal_dist')}"
                ),
            })
            return False, metadata
        if self._lightweight_panoramic_lock_active():
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": "reactive_direction_locked",
            })
            return False, metadata
        min_interval = self._lightweight_panoramic_min_interval()
        if last_scan_step is not None and self.step - last_scan_step < min_interval:
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": f"reactive_min_interval:{min_interval}",
            })
            return False, metadata
        stop_active = bool(
            probe_metadata.get("called_stopping")
            or probe_metadata.get("stop_signal")
            or probe_metadata.get("stop_signal_ready")
            or probe_metadata.get("action_should_stop")
            or action_number_int == -1
        )
        if stop_active:
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": "reactive_stop_or_stop_signal",
            })
            return False, metadata
        if self._lightweight_panoramic_progress_gate_enabled():
            progress_metadata = self._lightweight_panoramic_progress_metadata()
            metadata.update(progress_metadata)
            if not bool(progress_metadata.get("lightweight_panoramic_progress_ready", False)):
                if not bool(progress_metadata.get("lightweight_panoramic_progress_distance_ready", False)):
                    reason = "reactive_progress_no_distance"
                elif not bool(progress_metadata.get("lightweight_panoramic_progress_far_enough", False)):
                    reason = (
                        "reactive_progress_near_goal:"
                        f"{progress_metadata.get('lightweight_panoramic_progress_current_distance')}/"
                        f"{progress_metadata.get('lightweight_panoramic_progress_min_dist_to_goal')}"
                    )
                else:
                    reason = (
                        "reactive_progress_not_stagnant:"
                        f"{progress_metadata.get('lightweight_panoramic_progress_steps_since_improvement')}/"
                        f"{progress_metadata.get('lightweight_panoramic_progress_stagnation_steps')}"
                    )
                metadata.update({
                    "lightweight_panoramic_gate_decision": False,
                    "lightweight_panoramic_gate_reason": reason,
                })
                return False, metadata
        if action_number_int == 0 and "candidate0" in action_filter and mode_hit:
            metadata.update({
                "lightweight_panoramic_gate_decision": True,
                "lightweight_panoramic_gate_reason": f"reactive_action0:{nav_mode or decision_mode}",
                "lightweight_panoramic_progress_mode_hit": True,
            })
            return True, metadata
        if (
            action_number_int is not None
            and action_number_int > 0
            and "forward" in action_filter
            and mode_hit
        ):
            metadata.update({
                "lightweight_panoramic_gate_decision": True,
                "lightweight_panoramic_gate_reason": (
                    f"reactive_forward_action:{action_number_int}:{nav_mode or decision_mode}"
                ),
                "lightweight_panoramic_progress_mode_hit": True,
            })
            return True, metadata
        reason = "reactive_action_not_recovery"
        if action_number_int == 0 and "candidate0" not in action_filter:
            reason = "reactive_action0_disabled"
        elif action_number_int is not None and action_number_int > 0 and "forward" not in action_filter:
            reason = f"reactive_forward_disabled:{action_number_int}"
        elif action_number_int != 0:
            reason = f"reactive_action_not_turnaround:{action_number_int}"
        elif not mode_hit:
            reason = f"reactive_mode_not_trigger:{nav_mode or decision_mode or 'none'}"
        metadata.update({
            "lightweight_panoramic_gate_decision": False,
            "lightweight_panoramic_gate_reason": reason,
        })
        return False, metadata

    def _should_run_lightweight_panoramic_planner(self) -> tuple[bool, dict]:
        gate_enabled = self._lightweight_panoramic_mode_gate_enabled()
        progress_gate_enabled = self._lightweight_panoramic_progress_gate_enabled()
        last_mode = self._normalize_navigation_mode(
            getattr(self, "_lightweight_panoramic_last_navigation_mode", None)
        )
        last_scan_step = getattr(self, "_lightweight_panoramic_last_scan_step", None)
        steps_since_scan = None if last_scan_step is None else int(self.step - last_scan_step)
        metadata = {
            "lightweight_panoramic_gate_enabled": bool(gate_enabled),
            "lightweight_panoramic_progress_gate": bool(progress_gate_enabled),
            "lightweight_panoramic_initial_scan": bool(
                self._lightweight_panoramic_initial_scan_enabled()
            ),
            "lightweight_panoramic_min_interval": int(
                self._lightweight_panoramic_min_interval()
            ),
            "lightweight_panoramic_lock_steps": int(
                self._lightweight_panoramic_lock_steps()
            ),
            "lightweight_panoramic_min_select_score": float(
                self._lightweight_panoramic_min_select_score()
            ),
            "lightweight_panoramic_min_score_margin": float(
                self._lightweight_panoramic_min_score_margin()
            ),
            "lightweight_panoramic_min_mode_streak": int(
                self._lightweight_panoramic_min_mode_streak()
            ),
            "lightweight_panoramic_trigger_modes": ",".join(
                sorted(self._lightweight_panoramic_trigger_modes())
            ),
            "lightweight_panoramic_last_navigation_mode": last_mode or None,
            "lightweight_panoramic_last_scan_step": last_scan_step,
            "lightweight_panoramic_steps_since_scan": steps_since_scan,
            "lightweight_panoramic_global_state": getattr(
                self, "_lightweight_panoramic_global_state", "free"
            ),
            "lightweight_panoramic_view_direction_deg": float(
                getattr(self, "_lightweight_panoramic_view_direction_deg", 0.0)
            ),
            "lightweight_panoramic_lock_active": bool(self._lightweight_panoramic_lock_active()),
            "lightweight_panoramic_lock_until_step": getattr(
                self, "_lightweight_panoramic_lock_until_step", None
            ),
        }
        scan_count = int(getattr(self, "_lightweight_panoramic_scan_count", 0) or 0)
        max_scans = self._lightweight_panoramic_max_scans_per_episode()
        metadata["lightweight_panoramic_scan_count"] = int(scan_count)
        metadata["lightweight_panoramic_max_scans_per_episode"] = int(max_scans)
        if max_scans > 0 and scan_count >= max_scans:
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": f"max_scans:{scan_count}/{max_scans}",
            })
            return False, metadata
        near_goal_metadata = self._lightweight_panoramic_near_goal_block_metadata()
        metadata.update(near_goal_metadata)
        min_step = self._lightweight_panoramic_min_step()
        metadata["lightweight_panoramic_min_step"] = int(min_step)
        if int(self.step) < int(min_step):
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": f"min_step:{min_step}",
            })
            return False, metadata
        min_repeat = self._lightweight_panoramic_min_action_repeat_streak()
        repeat_action, repeat_streak = self._lightweight_panoramic_action_repeat_streak()
        metadata["lightweight_panoramic_min_action_repeat_streak"] = int(min_repeat)
        metadata["lightweight_panoramic_repeated_action"] = repeat_action
        metadata["lightweight_panoramic_repeated_action_streak"] = int(repeat_streak)
        if min_repeat > 0 and repeat_streak < min_repeat:
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": f"repeat_streak:{repeat_streak}/{min_repeat}",
            })
            return False, metadata
        if bool(near_goal_metadata.get("lightweight_panoramic_near_goal_blocked", False)):
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": (
                    "near_goal_block:"
                    f"{near_goal_metadata.get('lightweight_panoramic_near_goal_block_distance')}/"
                    f"{near_goal_metadata.get('lightweight_panoramic_block_near_goal_dist')}"
                ),
            })
            return False, metadata

        if not gate_enabled:
            metadata.update({
                "lightweight_panoramic_gate_decision": True,
                "lightweight_panoramic_gate_reason": "mode_gate_disabled",
            })
            return True, metadata

        if self._lightweight_panoramic_lock_active():
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": "direction_locked",
            })
            return False, metadata

        min_interval = self._lightweight_panoramic_min_interval()
        interval_ready = last_scan_step is None or self.step - last_scan_step >= min_interval
        if self.step == 0 and self._lightweight_panoramic_initial_scan_enabled():
            metadata.update({
                "lightweight_panoramic_gate_decision": True,
                "lightweight_panoramic_gate_reason": "initial_scan",
            })
            return True, metadata

        if not interval_ready:
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": f"min_interval:{min_interval}",
            })
            return False, metadata

        if progress_gate_enabled:
            progress_metadata = self._lightweight_panoramic_progress_metadata()
            metadata.update(progress_metadata)
            trigger_modes = self._lightweight_panoramic_trigger_modes()
            progress_require_mode = self._lightweight_panoramic_progress_require_mode()
            progress_mode_hit = bool(last_mode and last_mode in trigger_modes)
            metadata["lightweight_panoramic_progress_require_mode"] = bool(progress_require_mode)
            metadata["lightweight_panoramic_progress_mode_hit"] = bool(progress_mode_hit)
            if progress_require_mode and not progress_mode_hit:
                metadata.update({
                    "lightweight_panoramic_gate_decision": False,
                    "lightweight_panoramic_gate_reason": f"progress_mode_not_trigger:{last_mode or 'none'}",
                })
                return False, metadata
            if bool(progress_metadata.get("lightweight_panoramic_progress_ready", False)):
                metadata.update({
                    "lightweight_panoramic_gate_decision": True,
                    "lightweight_panoramic_gate_reason": (
                        "progress_stagnation:"
                        f"{progress_metadata.get('lightweight_panoramic_progress_steps_since_improvement')}/"
                        f"{progress_metadata.get('lightweight_panoramic_progress_stagnation_steps')}"
                    ),
                })
                return True, metadata
            if not bool(progress_metadata.get("lightweight_panoramic_progress_distance_ready", False)):
                reason = "progress_no_distance"
            elif not bool(progress_metadata.get("lightweight_panoramic_progress_far_enough", False)):
                reason = (
                    "progress_near_goal:"
                    f"{progress_metadata.get('lightweight_panoramic_progress_current_distance')}/"
                    f"{progress_metadata.get('lightweight_panoramic_progress_min_dist_to_goal')}"
                )
            else:
                reason = (
                    "progress_not_stagnant:"
                    f"{progress_metadata.get('lightweight_panoramic_progress_steps_since_improvement')}/"
                    f"{progress_metadata.get('lightweight_panoramic_progress_stagnation_steps')}"
                )
            metadata.update({
                "lightweight_panoramic_gate_decision": False,
                "lightweight_panoramic_gate_reason": reason,
            })
            return False, metadata

        trigger_modes = self._lightweight_panoramic_trigger_modes()
        mode_streak = int(getattr(self, "_lightweight_panoramic_mode_streak", 0))
        min_mode_streak = self._lightweight_panoramic_min_mode_streak()
        recovery_modes = {"recover_route", "topology_correction", "replan"}
        mode_ready = mode_streak >= min_mode_streak or last_mode in recovery_modes
        metadata["lightweight_panoramic_mode_streak_required"] = int(min_mode_streak)
        if last_mode in trigger_modes and mode_ready:
            metadata.update({
                "lightweight_panoramic_gate_decision": True,
                "lightweight_panoramic_gate_reason": f"navigation_mode:{last_mode}:streak:{mode_streak}",
            })
            return True, metadata

        if last_mode in trigger_modes and not mode_ready:
            reason = f"navigation_mode_streak_too_short:{last_mode}:{mode_streak}/{min_mode_streak}"
        else:
            reason = "no_previous_navigation_mode" if not last_mode else f"navigation_mode_not_trigger:{last_mode}"
        metadata.update({
            "lightweight_panoramic_gate_decision": False,
            "lightweight_panoramic_gate_reason": reason,
        })
        return False, metadata

    def _update_lightweight_panoramic_nav_state(self, step_metadata: dict):
        self._update_lightweight_panoramic_progress_state(step_metadata)
        mode = self._normalize_navigation_mode(step_metadata.get("action_navigation_mode"))
        if not mode:
            mode = self._normalize_navigation_mode(step_metadata.get("action_decision_mode"))
        if not mode:
            return
        prev_mode = self._normalize_navigation_mode(
            getattr(self, "_lightweight_panoramic_last_navigation_mode", None)
        )
        if mode == prev_mode:
            self._lightweight_panoramic_mode_streak = int(
                getattr(self, "_lightweight_panoramic_mode_streak", 0)
            ) + 1
        else:
            self._lightweight_panoramic_mode_streak = 1
        self._lightweight_panoramic_last_navigation_mode = mode
        target_approach_modes = {
            "near_goal_stop",
            "stop_at_goal",
            "target_guided_approach",
            "approach",
            "approach_visible_target",
            "approach_target",
            "visible_target_approach",
        }
        if mode in target_approach_modes:
            self._lightweight_panoramic_global_state = "target_approach"
            self._lightweight_panoramic_lock_until_step = None
            self._lightweight_panoramic_lock_context = None
        elif mode in ("recover_route", "topology_correction", "replan"):
            self._lightweight_panoramic_global_state = "recovery"
            self._lightweight_panoramic_lock_until_step = None
        elif mode in ("explore_new_area", "frontier_exploration", "explore"):
            self._lightweight_panoramic_global_state = "search_transition"
        else:
            self._lightweight_panoramic_global_state = "free"
        try:
            theta = step_metadata.get("chosen_action_theta_deg")
            if theta is not None:
                self._lightweight_panoramic_view_direction_deg = (
                    float(getattr(self, "_lightweight_panoramic_view_direction_deg", 0.0)) + float(theta)
                ) % 360.0
        except (TypeError, ValueError):
            pass

    def _lightweight_panoramic_num_views(self) -> int:
        try:
            value = int(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_NUM_VIEWS",
                str(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_num_views", 6)),
            ))
        except (TypeError, ValueError):
            value = 6
        return max(4, value)

    def _stop_multiview_num_views(self) -> int:
        try:
            value = int(os.environ.get("VLMNAV_STOP_MULTIVIEW_NUM_VIEWS", "1"))
        except (TypeError, ValueError):
            value = 1
        return max(1, value)

    def _stop_multiview_enabled(self) -> bool:
        return self._stop_multiview_num_views() > 1

    def _stop_multiview_defer_until_action_stop_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_DEFER_STOP_UNTIL_ACTION_STOP")
        if env_value is None:
            return False
        return env_value.strip().lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _stop_multiview_env_bool(name: str, default: bool) -> bool:
        env_value = os.environ.get(name)
        if env_value is None:
            return default
        return env_value.strip().lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _stop_multiview_parse_rgb(text: str) -> tuple[int, int, int]:
        try:
            parts = [part.strip() for part in str(text).split(",")]
            if len(parts) != 3:
                raise ValueError
            return tuple(max(0, min(255, int(part))) for part in parts)  # type: ignore[return-value]
        except Exception:
            return (0, 0, 0)

    def _stop_multiview_sensor_fov(self) -> float:
        try:
            cfg_fov = getattr(self.agent, "cfg", {}).get("sensor_cfg", {}).get("fov", 131.0)
        except Exception:
            cfg_fov = 131.0
        try:
            value = float(os.environ.get("VLMNAV_STOP_MULTIVIEW_SENSOR_FOV", str(cfg_fov)))
        except (TypeError, ValueError):
            value = float(cfg_fov or 131.0)
        return max(1.0, min(179.0, value))

    def _stop_multiview_center_crop_fov(self) -> float:
        try:
            value = float(os.environ.get("VLMNAV_STOP_MULTIVIEW_CENTER_CROP_FOV", "90"))
        except (TypeError, ValueError):
            value = 90.0
        return max(0.0, min(179.0, value))

    def _stop_multiview_view_size(self) -> tuple[int, int]:
        try:
            width = int(os.environ.get("VLMNAV_STOP_MULTIVIEW_VIEW_WIDTH", "512"))
        except (TypeError, ValueError):
            width = 512
        try:
            height = int(os.environ.get("VLMNAV_STOP_MULTIVIEW_VIEW_HEIGHT", "288"))
        except (TypeError, ValueError):
            height = 288
        return max(16, width), max(16, height)

    def _stop_multiview_layout_cols(self, num_views: int) -> int:
        try:
            value = int(os.environ.get("VLMNAV_STOP_MULTIVIEW_LAYOUT_COLS", str(num_views)))
        except (TypeError, ValueError):
            value = int(num_views)
        return max(1, value)

    def _stop_multiview_tile_gap(self) -> int:
        try:
            value = int(os.environ.get("VLMNAV_STOP_MULTIVIEW_TILE_GAP", "6"))
        except (TypeError, ValueError):
            value = 6
        return max(0, value)

    def _stop_multiview_crop_view(self, image: np.ndarray, sensor_fov: float, crop_fov: float) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(image)[:, :, :3])
        if crop_fov <= 0.0 or crop_fov >= sensor_fov:
            return arr
        height, width = arr.shape[:2]
        ratio = math.tan(math.radians(crop_fov) / 2.0) / math.tan(math.radians(sensor_fov) / 2.0)
        crop_w = max(1, min(width, int(round(width * ratio))))
        crop_h = max(1, min(height, int(round(crop_w * height / width))))
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        return arr[top:top + crop_h, left:left + crop_w]

    def _build_standard_stop_multiview_image(
        self,
        images: list[np.ndarray],
        angles: list[int],
    ) -> tuple[np.ndarray, list[int], dict]:
        """Build the clean stop-module contact sheet used by all multiview stop calls."""
        sensor_fov = self._stop_multiview_sensor_fov()
        crop_fov = self._stop_multiview_center_crop_fov()
        view_width, view_height = self._stop_multiview_view_size()
        reverse_order = self._stop_multiview_env_bool("VLMNAV_STOP_MULTIVIEW_REVERSE_ORDER", True)
        tile_gap = self._stop_multiview_tile_gap()
        tile_gap_color = self._stop_multiview_parse_rgb(
            os.environ.get("VLMNAV_STOP_MULTIVIEW_TILE_GAP_COLOR", "0,0,0")
        )
        layout_cols = self._stop_multiview_layout_cols(len(images))

        paired = list(zip(images, angles))
        if reverse_order:
            paired = list(reversed(paired))

        tiles = []
        display_angles = []
        for image, angle in paired:
            cropped = self._stop_multiview_crop_view(image, sensor_fov, crop_fov)
            tile = Image.fromarray(cropped[:, :, :3], mode="RGB").resize(
                (view_width, view_height),
                Image.BILINEAR,
            )
            tiles.append(np.asarray(tile, dtype=np.uint8))
            display_angles.append(int(angle))

        rows = int(math.ceil(len(tiles) / layout_cols))
        canvas_h = rows * view_height + max(0, rows - 1) * tile_gap
        canvas_w = layout_cols * view_width + max(0, layout_cols - 1) * tile_gap
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :] = np.asarray(tile_gap_color, dtype=np.uint8)
        for idx, tile in enumerate(tiles):
            row = idx // layout_cols
            col = idx % layout_cols
            y0 = row * (view_height + tile_gap)
            x0 = col * (view_width + tile_gap)
            canvas[y0:y0 + view_height, x0:x0 + view_width] = tile

        spec = (
            f"clean_contact_sheet_crop{crop_fov:g}_"
            f"{rows}x{layout_cols}_{'reverse' if reverse_order else 'forward'}_gap{tile_gap}"
        )
        metadata = {
            "stop_multiview_standard_input": True,
            "stop_multiview_input_format": spec,
            "stop_multiview_sensor_fov": float(sensor_fov),
            "stop_multiview_center_crop_fov": float(crop_fov),
            "stop_multiview_view_width": int(view_width),
            "stop_multiview_view_height": int(view_height),
            "stop_multiview_layout_cols": int(layout_cols),
            "stop_multiview_layout_rows": int(rows),
            "stop_multiview_reverse_order": bool(reverse_order),
            "stop_multiview_tile_gap": int(tile_gap),
            "stop_multiview_tile_gap_color": ",".join(str(v) for v in tile_gap_color),
            "stop_multiview_display_angles_deg": display_angles,
        }
        return canvas, display_angles, metadata

    def _run_stop_multiview_capture(self, obs: dict) -> tuple[dict, dict, dict]:
        num_views = self._stop_multiview_num_views()
        metadata = {
            "stop_multiview_enabled": bool(num_views > 1),
            "stop_multiview_num_views": int(num_views),
            "stop_multiview_success": False,
        }
        logging_images = {}
        if num_views <= 1:
            return obs, metadata, logging_images

        score_step = 2 * np.pi / num_views
        scan_action = PolarAction(0, -score_step)
        restore_action = PolarAction(0, score_step)
        capture_images = []
        capture_angles = []
        working_obs = obs

        try:
            capture_images.append(working_obs["color_sensor"].copy()[:, :, :3])
            capture_angles.append(0)
            for view_idx in range(1, num_views):
                working_obs = self.simWrapper.step(scan_action)
                capture_images.append(working_obs["color_sensor"].copy()[:, :, :3])
                capture_angles.append(int(round(view_idx * 360 / num_views)) % 360)

            restored_obs = working_obs
            for _ in range(1, num_views):
                restored_obs = self.simWrapper.step(restore_action)

            stop_contact_sheet, display_angles, contact_sheet_metadata = self._build_standard_stop_multiview_image(
                capture_images,
                capture_angles,
            )
            restored_obs["stopping_images_override"] = [stop_contact_sheet]
            restored_obs["stop_multiview_prompt_context"] = {
                "format": "standard_multiview_contact_sheet",
                "num_views": int(len(capture_images)),
                "angles_deg": list(capture_angles),
                "display_angles_deg": list(display_angles),
                "input_format": contact_sheet_metadata["stop_multiview_input_format"],
                "layout_cols": contact_sheet_metadata["stop_multiview_layout_cols"],
                "layout_rows": contact_sheet_metadata["stop_multiview_layout_rows"],
                "reverse_order": contact_sheet_metadata["stop_multiview_reverse_order"],
                "tile_gap": contact_sheet_metadata["stop_multiview_tile_gap"],
                "center_crop_fov": contact_sheet_metadata["stop_multiview_center_crop_fov"],
                "sensor_fov": contact_sheet_metadata["stop_multiview_sensor_fov"],
            }
            metadata.update({
                "stop_multiview_success": True,
                "stop_multiview_raw_input_images": int(len(capture_images)),
                "stop_multiview_input_images": 1,
                "stop_multiview_angles_deg": capture_angles,
                "stop_multiview_restore_steps": int(max(0, num_views - 1)),
                "stop_multiview_step_degrees": float(360.0 / num_views),
            })
            metadata.update(contact_sheet_metadata)

            log_image = os.environ.get(
                "VLMNAV_STOP_MULTIVIEW_LOG_IMAGE",
                "0",
            ).strip().lower() in ("1", "true", "yes", "y", "on")
            if log_image:
                logging_images["stop_multiview_pano"] = stop_contact_sheet.copy()
            return restored_obs, metadata, logging_images
        except Exception as exc:
            logging.exception("Stop multiview capture failed; falling back to the current view.")
            obs["stopping_images_override"] = [obs["color_sensor"]]
            metadata.update({
                "stop_multiview_success": False,
                "stop_multiview_error": str(exc),
                "stop_multiview_input_images": 1,
                "stop_multiview_standard_input": False,
            })
            return obs, metadata, logging_images

    def _run_deferred_stop_verification(
        self,
        obs: dict,
        agent_action,
        metadata: dict,
    ) -> tuple[dict, object, dict, dict]:
        """Run the six-view stop verifier only after the action policy asks to stop."""
        step_metadata = metadata.get("step_metadata", {})
        logging_data = metadata.get("logging_data", {})
        stop_multiview_metadata = {
            "stop_multiview_enabled": self._stop_multiview_enabled(),
            "stop_multiview_num_views": int(self._stop_multiview_num_views()),
            "stop_multiview_deferred_until_action_stop": True,
            "stop_multiview_success": False,
            "stop_multiview_skipped": True,
            "stop_multiview_skip_reason": "action_did_not_request_stop",
            "stop_multiview_input_images": 0,
            "stop_multiview_raw_input_images": 0,
        }
        stop_multiview_images = {}
        action_wants_stop = bool(step_metadata.get("action_wants_stop", False))
        disable_stopping_module = os.environ.get(
            "VLMNAV_DISABLE_STOPPING_MODULE",
            "",
        ).strip().lower() in ("1", "true", "yes", "y", "on")

        step_metadata["stop_gate_require_action_stop"] = True
        step_metadata["stop_deferred_until_action_stop"] = True
        step_metadata["stop_verification_run"] = False
        step_metadata["stop_verification_skipped"] = not action_wants_stop

        if not action_wants_stop or disable_stopping_module:
            if disable_stopping_module:
                stop_multiview_metadata["stop_multiview_skip_reason"] = "stopping_module_disabled"
                step_metadata["stop_verification_skipped"] = True
            return obs, agent_action, stop_multiview_metadata, stop_multiview_images

        obs, stop_multiview_metadata, stop_multiview_images = self._run_stop_multiview_capture(obs)
        obs["goal"] = self.current_episode["object"]
        stopping_images = obs.get("stopping_images_override") if isinstance(obs, dict) else None
        if not isinstance(stopping_images, list) or not stopping_images:
            stopping_images = [obs["color_sensor"]]

        stop_prompt_context = obs.get("stop_multiview_prompt_context") if isinstance(obs, dict) else None
        self.agent._stopping_prompt_context = stop_prompt_context if isinstance(stop_prompt_context, dict) else None
        try:
            called_stop, stopping_response = self.agent._stopping_module(
                stopping_images,
                self.current_episode["object"],
            )
        except Exception as exc:
            logging.error("Deferred stop verifier failed: %s", exc)
            called_stop, stopping_response = False, ""
            step_metadata["stop_verification_error"] = str(exc)
        finally:
            self.agent._stopping_prompt_context = None

        step_ndx = max(0, int(getattr(self.agent, "step_ndx", 1)) - 1)
        if called_stop:
            logging.info("Deferred stop verifier called stop")
            self.agent.stopping_calls.append(step_ndx)

        model_name = str(getattr(self.agent.actionVLM, "name", "")).lower()
        is_sensenova_qwen25 = ("sensenova" in model_name) and ("qwen2.5" in model_name)
        current_stop_signal = (
            len(self.agent.stopping_calls) >= 1 and self.agent.stopping_calls[-1] == step_ndx
        )
        stop_ready_streak = 0
        expected_stop_step = step_ndx
        for stop_step in reversed(self.agent.stopping_calls):
            if stop_step == expected_stop_step:
                stop_ready_streak += 1
                expected_stop_step -= 1
                continue
            if stop_step < expected_stop_step:
                break
        try:
            stop_ready_streak_min = int(os.environ.get("VLMNAV_STOP_READY_STREAK_MIN", "0"))
        except ValueError:
            stop_ready_streak_min = 0
        stop_ready_streak_min = max(0, stop_ready_streak_min)
        legacy_stop_signal_ready = (
            current_stop_signal
            if is_sensenova_qwen25
            else (len(self.agent.stopping_calls) >= 2 and self.agent.stopping_calls[-2] == step_ndx - 1)
        )
        require_legacy_ready = os.environ.get(
            "VLMNAV_DEFERRED_STOP_REQUIRE_LEGACY_READY",
            "",
        ).strip().lower() in ("1", "true", "yes", "y", "on")
        stop_signal_ready = bool(legacy_stop_signal_ready if require_legacy_ready else current_stop_signal)
        stop_signal_ready_for_gate = bool(
            stop_signal_ready
            and (
                stop_ready_streak_min <= 1
                or stop_ready_streak >= stop_ready_streak_min
            )
        )

        parsed_stop = {}
        stop_confidence = None
        try:
            parsed_stop = self.agent._eval_response(stopping_response) if stopping_response else {}
            if isinstance(parsed_stop, dict):
                for key in ("confidence", "score", "prob", "probability"):
                    if key in parsed_stop:
                        stop_confidence = float(parsed_stop[key])
                        break
                if stop_confidence is None and "stop" in parsed_stop:
                    value = parsed_stop.get("stop")
                    if isinstance(value, bool):
                        stop_confidence = 1.0 if value else 0.0
                    elif isinstance(value, (int, float)):
                        stop_confidence = 1.0 if int(value) == 1 else 0.0
                    elif isinstance(value, str):
                        stop_confidence = 1.0 if value.strip().lower() in {"true", "1", "yes", "stop"} else 0.0
                if stop_confidence is None and "done" in parsed_stop:
                    stop_confidence = 1.0 if int(parsed_stop["done"]) == 1 else 0.0
        except Exception:
            parsed_stop = {}
            stop_confidence = None

        for key in (
            "ability",
            "spatial_reasoning",
            "stop",
            "target_visibility",
            "distance_measurement",
            "reason",
            "target_visible",
            "target_identity_clear",
            "distance_judgment",
            "blocking_or_gap",
            "stop_reason",
            "done",
        ):
            if isinstance(parsed_stop, dict) and key in parsed_stop:
                step_metadata[f"stop_cot_{key}"] = parsed_stop.get(key)

        step_metadata["called_stopping"] = bool(called_stop)
        step_metadata["stop_confidence"] = stop_confidence
        step_metadata["stop_decision_source"] = "deferred_vlm_stopping"
        step_metadata["stop_verification_run"] = True
        step_metadata["stop_verification_skipped"] = False
        step_metadata["stop_signal"] = bool(current_stop_signal)
        step_metadata["stop_signal_ready"] = bool(stop_signal_ready)
        step_metadata["stop_ready_streak"] = int(stop_ready_streak)
        step_metadata["stop_ready_streak_min"] = int(stop_ready_streak_min)
        step_metadata["stop_signal_ready_for_gate"] = bool(stop_signal_ready_for_gate)
        step_metadata["stop_gate_require_action_stop"] = True
        step_metadata["stop_gate_passed"] = bool(stop_signal_ready_for_gate and action_wants_stop)
        step_metadata["stop_multiview_input_images"] = int(len(stopping_images))
        logging_data["STOPPING RESPONSE"] = stopping_response

        if step_metadata["stop_gate_passed"]:
            step_metadata["action_number"] = -1
            agent_action = PolarAction.stop
            try:
                if getattr(self.agent, "prompt_action_history", None):
                    self.agent.prompt_action_history[-1] = -1
            except Exception:
                pass
        elif step_metadata.get("action_number") == -1:
            fallback_action = int(os.environ.get("VLMNAV_STOP_GATE_FALLBACK_ACTION", "1"))
            max_action = len(metadata.get("a_final") or [])
            if fallback_action < 0 or fallback_action > max_action:
                fallback_action = 0
            step_metadata["stop_gate_blocked_action"] = -1
            step_metadata["action_number"] = fallback_action
            agent_action = self.agent._action_number_to_polar(
                step_metadata["action_number"],
                list(metadata.get("a_final") or []),
            )

        return obs, agent_action, stop_multiview_metadata, stop_multiview_images

    def _concat_lightweight_panoramic(self, images: list[np.ndarray], angles: list[int]) -> np.ndarray:
        height, width = images[0].shape[:2]
        cols = 3
        rows = math.ceil(len(images) / cols)
        pad = 10
        canvas = np.zeros((rows * height + (rows + 1) * pad, cols * width + (cols + 1) * pad, 3), dtype=np.uint8)
        for i, image in enumerate(images):
            row = i // cols
            col = i % cols
            panel = np.ascontiguousarray(image[:, :, :3].copy())
            cv2.putText(
                panel,
                f"Angle {angles[i]}",
                (max(12, width // 25), max(36, height // 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.7, height / 360.0),
                (255, 0, 0),
                max(2, int(height / 180)),
                cv2.LINE_AA,
            )
            y0 = pad + row * (height + pad)
            x0 = pad + col * (width + pad)
            canvas[y0:y0 + height, x0:x0 + width] = panel
        return canvas

    def _parse_lightweight_panoramic_scores(self, response: str, score_angles: list[int]) -> tuple[dict[int, float], dict[int, str]]:
        if not response:
            return {}, {}
        text = str(response)
        start = text.find("{")
        end = text.rfind("}")
        parsed = None
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
            try:
                parsed = json.loads(candidate)
            except Exception:
                try:
                    parsed = ast.literal_eval(candidate)
                except Exception:
                    parsed = None
        scores: dict[int, float] = {}
        reasons: dict[int, str] = {}
        if isinstance(parsed, dict):
            for angle in score_angles:
                for key in (str(angle), angle):
                    if key not in parsed:
                        continue
                    value = parsed[key]
                    if isinstance(value, dict):
                        reason_value = value.get("Explanation", value.get("explanation", value.get("Reason", value.get("reason"))))
                        if reason_value is not None:
                            reasons[int(angle)] = str(reason_value).strip()
                        value = value.get("Score", value.get("score"))
                    try:
                        scores[int(angle)] = float(value)
                    except (TypeError, ValueError):
                        pass
                    break
        if scores:
            return scores, reasons

        for angle in score_angles:
            match = re.search(
                rf"['\"]?{angle}['\"]?\s*[:=]\s*(?:\{{[^}}]*?['\"]?[Ss]core['\"]?\s*[:=]\s*)?([0-9]+(?:\.[0-9]+)?)",
                text,
            )
            if match:
                scores[int(angle)] = float(match.group(1))
        return scores, reasons

    def _lightweight_panoramic_depth_score(self, depth_image) -> float:
        try:
            depth = np.asarray(depth_image, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            h, w = depth.shape[:2]
            roi = depth[int(h * 0.45):int(h * 0.9), int(w * 0.25):int(w * 0.75)]
            valid = roi[np.isfinite(roi) & (roi > 0.05)]
            if valid.size == 0:
                return 0.0
            far_fraction = float(np.mean(valid > 1.2))
            median_depth = float(np.median(valid))
            return max(0.0, min(10.0, 6.0 * far_fraction + 2.0 * min(median_depth, 2.5) / 2.5))
        except Exception:
            return 0.0

    def _lightweight_panoramic_require_target_mention_for_high_score(self) -> bool:
        env_value = os.environ.get("VLMNAV_LIGHTWEIGHT_PANORAMIC_REQUIRE_TARGET_MENTION")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("lightweight_panoramic_require_target_mention", False))

    def _lightweight_panoramic_target_mention_min_score(self) -> float:
        try:
            value = float(os.environ.get(
                "VLMNAV_LIGHTWEIGHT_PANORAMIC_TARGET_MENTION_MIN_SCORE",
                str(getattr(self.agent, "cfg", {}).get(
                    "lightweight_panoramic_target_mention_min_score",
                    9.0,
                )),
            ))
        except (TypeError, ValueError):
            value = 9.0
        return max(0.0, value)

    def _lightweight_panoramic_reason_mentions_goal(self, reason: str, goal: str) -> bool:
        reason_text = str(reason or "").lower()
        goal_text = str(goal or "").lower()
        terms = [term for term in re.findall(r"[a-z0-9]+", goal_text) if len(term) >= 2]
        if "tv" in terms or "screen" in terms:
            terms.extend(["television", "monitor"])
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", reason_text):
                return True
        return False

    def _lightweight_panoramic_prompt(self, goal: str, score_angles: list[int]) -> str:
        angle_schema = ", ".join(
            f"'{angle}': {{'Score': <0-10>, 'Explanation': <short reason>}}"
            for angle in score_angles
        )
        return (
            f"The agent is navigating to the nearest {goal.upper()}. "
            "You are given a lightweight panoramic image made from the current location. "
            "Each panel is labeled by its relative rotation angle. "
            "Score only the labeled candidate directions by whether that direction is worth facing before local action selection. "
            "Use target visibility, open traversable space, doorways, hallways, room transitions, and likely target-room priors. "
            "Use this scale: 0-2 means blocked wall, dead end, mirror/reflection, or no usable floor; "
            "3-5 means locally traversable but weak or already checked; "
            "6-8 means a useful recovery heading such as open floor, doorway, hallway, room transition, or likely target-room prior even when the target is not visible; "
            "9-10 means the target itself is visible and the direction is both visually promising and locally usable. "
            "Do not give 10 merely because the target is visible: reserve 10 for a clearly visible target with an obvious close, traversable approach. "
            "Ignore target-like objects seen only in mirrors, glass, screens, or other reflections; reflected objects are not reachable navigation targets. "
            "A direction mostly facing a mirror, wall, sink, counter, cabinet, or other non-traversable surface should receive a low score unless it also contains a clear doorway or usable path. "
            "Prefer a real doorway, hallway, open floor path, or room transition over a reflected or ambiguous target-like shape. "
            "If the target is visible but across the room, partly blocked, behind furniture, or still needs approach, score the direction as promising but not a stop-ready direction. "
            f"For score 9 or 10, the explanation must explicitly name {goal.upper()} and why that target direction is usable. "
            "Do not give 9 or 10 to a different object, surface, rug, decoration, or generic room feature. "
            "Do not give 0 merely because the target is absent; an open doorway, hallway, or traversable room transition should receive 6-8. "
            "Give 0 only if the direction is blocked/dead-end and the target is absent. "
            "Do not plan a full path and do not use memory; this is only a one-step view-selection prior. "
            f"Return JSON only in this schema: {{{angle_schema}}}."
        )

    def _action0_pano_replace_enabled(self) -> bool:
        env_value = os.environ.get("VLMNAV_ACTION0_PANO_REPLACE")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(getattr(self.agent, "cfg", {}).get("action0_pano_replace", False))

    def _action0_pano_env_bool(self, name: str, default: bool) -> bool:
        env_value = os.environ.get(name)
        if env_value is None:
            return bool(default)
        return env_value.strip().lower() in ("1", "true", "yes", "y", "on")

    def _action0_pano_env_int(self, name: str, default: int, minimum: int = 0) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            value = int(default)
        return max(int(minimum), int(value))

    def _action0_pano_env_float(self, name: str, default: float, minimum: float = 0.0) -> float:
        try:
            value = float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            value = float(default)
        return max(float(minimum), float(value))

    def _action0_pano_min_step(self) -> int:
        return self._action0_pano_env_int("VLMNAV_ACTION0_PANO_MIN_STEP", 3)

    def _action0_pano_max_per_episode(self) -> int:
        return self._action0_pano_env_int("VLMNAV_ACTION0_PANO_MAX_PER_EPISODE", 2)

    def _action0_pano_min_score(self) -> float:
        return self._action0_pano_env_float("VLMNAV_ACTION0_PANO_MIN_SCORE", 8.0)

    def _action0_pano_min_score_margin(self) -> float:
        return self._action0_pano_env_float("VLMNAV_ACTION0_PANO_MIN_SCORE_MARGIN", 1.5)

    def _action0_pano_block_near_goal_dist(self) -> float:
        return self._action0_pano_env_float("VLMNAV_ACTION0_PANO_BLOCK_NEAR_GOAL_DIST", 1.0)

    def _action0_pano_mode_gate_enabled(self) -> bool:
        return self._action0_pano_env_bool("VLMNAV_ACTION0_PANO_MODE_GATE", True)

    def _action0_pano_trigger_modes(self) -> set[str]:
        raw = os.environ.get(
            "VLMNAV_ACTION0_PANO_TRIGGER_MODES",
            "recover_route,topology_correction,replan",
        )
        modes = set()
        for value in str(raw).replace(";", ",").split(","):
            mode = self._normalize_navigation_mode(value)
            if mode:
                modes.add(mode)
        return modes

    def _action0_pano_current_distance_to_goal(self, obs: dict):
        # Goal geodesics are privileged labels and cannot gate inference behavior.
        return None

    def _should_run_action0_pano_replace(self, step_metadata: dict, obs: dict) -> tuple[bool, dict]:
        nav_mode = self._normalize_navigation_mode(step_metadata.get("action_navigation_mode"))
        decision_mode = self._normalize_navigation_mode(step_metadata.get("action_decision_mode"))
        trigger_modes = self._action0_pano_trigger_modes()
        mode_hit = bool(
            (nav_mode and nav_mode in trigger_modes)
            or (decision_mode and decision_mode in trigger_modes)
        )
        count = int(getattr(self, "_action0_pano_replace_count", 0) or 0)
        max_per_episode = self._action0_pano_max_per_episode()
        min_step = self._action0_pano_min_step()
        block_dist = self._action0_pano_block_near_goal_dist()
        current_distance = self._action0_pano_current_distance_to_goal(obs)
        metadata = {
            "action0_pano_gate_decision": False,
            "action0_pano_gate_reason": "",
            "action0_pano_min_step": int(min_step),
            "action0_pano_max_per_episode": int(max_per_episode),
            "action0_pano_min_score": float(self._action0_pano_min_score()),
            "action0_pano_min_score_margin": float(self._action0_pano_min_score_margin()),
            "action0_pano_block_near_goal_dist": float(block_dist),
            "action0_pano_current_distance": current_distance,
            "action0_pano_mode_gate": bool(self._action0_pano_mode_gate_enabled()),
            "action0_pano_trigger_modes": ",".join(sorted(trigger_modes)),
            "action0_pano_navigation_mode": nav_mode or None,
            "action0_pano_decision_mode": decision_mode or None,
            "action0_pano_mode_hit": bool(mode_hit),
        }
        if int(self.step) < int(min_step):
            metadata["action0_pano_gate_reason"] = f"min_step:{int(self.step)}/{min_step}"
            return False, metadata
        if max_per_episode > 0 and count >= max_per_episode:
            metadata["action0_pano_gate_reason"] = f"max_per_episode:{count}/{max_per_episode}"
            return False, metadata
        if block_dist > 0 and current_distance is not None and float(current_distance) <= float(block_dist):
            metadata["action0_pano_gate_reason"] = f"near_goal:{float(current_distance):.3f}<={block_dist:.3f}"
            return False, metadata
        stop_active = bool(
            step_metadata.get("called_stopping")
            or step_metadata.get("stop_signal")
            or step_metadata.get("stop_signal_ready")
            or step_metadata.get("action_should_stop")
        )
        if stop_active:
            metadata["action0_pano_gate_reason"] = "stop_or_stop_signal"
            return False, metadata
        if self._action0_pano_mode_gate_enabled() and not mode_hit:
            metadata["action0_pano_gate_reason"] = f"mode_not_trigger:{nav_mode or decision_mode or 'none'}"
            return False, metadata
        metadata.update({
            "action0_pano_gate_decision": True,
            "action0_pano_gate_reason": "candidate0_recovery_gate_pass",
        })
        return True, metadata

    def _action0_pano_prompt(self, goal: str, score_angles: list[int]) -> str:
        angle_schema = ", ".join(
            f"'{angle}': {{'Score': <0-10>, 'Explanation': <short reason>}}"
            for angle in score_angles
        )
        return (
            f"The agent is navigating to the nearest {goal.upper()}. "
            "The normal ObjectNav policy already chose candidate_0, meaning it wants to reorient rather than move forward. "
            "You are given a six-panel panoramic image captured from the current location. "
            "Each panel is labeled by its relative rotation angle. "
            "Score each direction only as the next observation/reorientation heading. "
            "Prefer a heading with a visible target, open traversable space, a doorway, hallway, room transition, or a plausible route toward the target. "
            "Do not choose a mirror/reflection, wall, cabinet, counter, sink, or blocked surface unless it also contains a real usable passage. "
            "Do not plan a full route and do not decide stop or forward movement. "
            "Return JSON only in this schema: "
            f"{{{angle_schema}}}."
        )

    def _run_action0_pano_replace(self, obs: dict, goal: str):
        num_views = 6
        score_step = 2 * np.pi / num_views
        half_step = score_step / 2
        first_scan_action = PolarAction(0, -half_step)
        scan_action = PolarAction(0, -score_step)
        reverse_scan_action = PolarAction(0, score_step)
        score_angles = [int(round((i + 0.5) * 360 / num_views)) % 360 for i in range(num_views)]

        pano_images = []
        obs = self.simWrapper.step(first_scan_action)
        pano_images.append(obs["color_sensor"].copy()[:, :, :3])
        for _ in range(1, num_views):
            obs = self.simWrapper.step(scan_action)
            pano_images.append(obs["color_sensor"].copy()[:, :, :3])

        panoramic_image = self._concat_lightweight_panoramic(pano_images, score_angles)
        prompt = self._action0_pano_prompt(goal, score_angles)
        response = ""
        scores = {}
        reasons = {}
        selected_angle = None
        selected_score = None
        selected_score_margin = None
        fallback_reason = ""
        success = False

        try:
            response = self.agent.actionVLM.call([panoramic_image], prompt)
            scores, reasons = self._parse_lightweight_panoramic_scores(response, score_angles)
            valid_scores = {
                int(angle): float(scores[angle])
                for angle in score_angles
                if angle in scores
            }
            if not valid_scores:
                fallback_reason = "no_valid_scores"
            else:
                ordered = sorted(valid_scores.items(), key=lambda item: item[1], reverse=True)
                selected_angle, selected_score = ordered[0]
                if len(ordered) >= 2:
                    selected_score_margin = float(ordered[0][1]) - float(ordered[1][1])
                else:
                    selected_score_margin = float(selected_score)
                min_score = self._action0_pano_min_score()
                min_margin = self._action0_pano_min_score_margin()
                if float(selected_score) < float(min_score):
                    fallback_reason = f"low_select_score:{float(selected_score):.2f}<{float(min_score):.2f}"
                elif float(selected_score_margin) < float(min_margin):
                    fallback_reason = (
                        "low_score_margin:"
                        f"{float(selected_score_margin):.2f}<"
                        f"{float(min_margin):.2f}"
                    )
                else:
                    success = True
        except Exception as exc:
            logging.error("Action0 panoramic replacement failed: %s", exc)
            fallback_reason = f"exception:{type(exc).__name__}"

        if success:
            selected_index = score_angles.index(int(selected_angle))
            clockwise_steps = selected_index + 1
            counter_steps = num_views - clockwise_steps
            if clockwise_steps <= counter_steps:
                for _ in range(clockwise_steps):
                    obs = self.simWrapper.step(scan_action)
            else:
                for _ in range(counter_steps):
                    obs = self.simWrapper.step(reverse_scan_action)
            self._action0_pano_replace_count = int(getattr(self, "_action0_pano_replace_count", 0)) + 1
        else:
            obs = self.simWrapper.step(first_scan_action)
            if not fallback_reason:
                fallback_reason = "unknown"

        metadata = {
            "action0_pano_enabled": True,
            "action0_pano_triggered": True,
            "action0_pano_success": bool(success),
            "action0_pano_selected_angle": int(selected_angle) if selected_angle is not None else None,
            "action0_pano_selected_score": float(selected_score) if selected_score is not None else None,
            "action0_pano_score_margin": (
                float(selected_score_margin) if selected_score_margin is not None else None
            ),
            "action0_pano_min_score": float(self._action0_pano_min_score()),
            "action0_pano_min_score_margin": float(self._action0_pano_min_score_margin()),
            "action0_pano_scores": scores,
            "action0_pano_reasons": reasons,
            "action0_pano_fallback_reason": "" if success else fallback_reason,
            "action0_pano_count": int(getattr(self, "_action0_pano_replace_count", 0)),
            "action0_pano_num_views": int(num_views),
        }
        logging_data = {
            "ACTION0_PANO_PROMPT": prompt,
            "ACTION0_PANO_RESPONSE": response,
        }
        images = {
            "action0_pano": panoramic_image,
        }
        return obs, metadata, logging_data, images

    def _run_lightweight_panoramic_planner(self, obs: dict, goal: str):
        num_views = self._lightweight_panoramic_num_views()
        score_step = 2 * np.pi / num_views
        half_step = score_step / 2
        first_scan_action = PolarAction(0, -half_step)
        scan_action = PolarAction(0, -score_step)
        reverse_scan_action = PolarAction(0, score_step)

        pano_images = []
        pano_depths = []
        obs = self.simWrapper.step(first_scan_action)
        pano_images.append(obs["color_sensor"].copy()[:, :, :3])
        pano_depths.append(obs.get("depth_sensor"))
        for _ in range(1, num_views):
            obs = self.simWrapper.step(scan_action)
            pano_images.append(obs["color_sensor"].copy()[:, :, :3])
            pano_depths.append(obs.get("depth_sensor"))

        score_angles = [int(round((i + 0.5) * 360 / num_views)) % 360 for i in range(num_views)]
        panoramic_image = self._concat_lightweight_panoramic(pano_images, score_angles)
        prompt = self._lightweight_panoramic_prompt(goal, score_angles)
        response = ""
        scores = {}
        raw_scores = {}
        depth_scores = {}
        reasons = {}
        selected_index = -1
        selected_angle = 0
        selected_score = None
        selected_score_margin = None
        reject_reason = ""
        success = False
        context_only = self._lightweight_panoramic_context_only_enabled()
        score_only = self._lightweight_panoramic_score_only_enabled()
        depth_blend = os.environ.get(
            "VLMNAV_LIGHTWEIGHT_PANORAMIC_DEPTH_BLEND",
            "0",
        ).strip().lower() in ("1", "true", "yes", "y", "on")
        try:
            response = self.agent.actionVLM.call([panoramic_image], prompt)
            raw_scores, reasons = self._parse_lightweight_panoramic_scores(response, score_angles)
            scores = dict(raw_scores)
            if depth_blend:
                for angle, depth in zip(score_angles, pano_depths):
                    depth_scores[int(angle)] = self._lightweight_panoramic_depth_score(depth)
                for angle in score_angles:
                    if angle in raw_scores and angle in depth_scores:
                        scores[angle] = 0.8 * float(raw_scores[angle]) + 0.2 * float(depth_scores[angle])
            valid_scores = {angle: scores[angle] for angle in score_angles if angle in scores}
            if valid_scores:
                selected_angle = max(valid_scores, key=valid_scores.get)
                selected_index = score_angles.index(selected_angle)
                selected_score = scores.get(int(selected_angle))
                min_select_score = self._lightweight_panoramic_min_select_score()
                ordered_scores = sorted(float(score) for score in valid_scores.values())
                if len(ordered_scores) >= 2:
                    selected_score_margin = float(selected_score) - float(ordered_scores[-2])
                if float(selected_score) >= float(min_select_score):
                    selected_reason = reasons.get(int(selected_angle), "")
                    min_score_margin = self._lightweight_panoramic_min_score_margin()
                    target_mention_min_score = self._lightweight_panoramic_target_mention_min_score()
                    if (
                        min_score_margin > 0
                        and selected_score_margin is not None
                        and float(selected_score_margin) < float(min_score_margin)
                    ):
                        reject_reason = (
                            "low_score_margin:"
                            f"{float(selected_score_margin):.2f}<"
                            f"{float(min_score_margin):.2f}"
                        )
                    elif (
                        self._lightweight_panoramic_require_target_mention_for_high_score()
                        and float(selected_score) >= float(target_mention_min_score)
                        and not self._lightweight_panoramic_reason_mentions_goal(selected_reason, goal)
                    ):
                        reject_reason = (
                            "selected_reason_missing_target:"
                            f"{float(selected_score):.2f}>={float(target_mention_min_score):.2f}"
                        )
                    else:
                        success = True
                else:
                    reject_reason = (
                        f"low_select_score:{float(selected_score):.2f}<"
                        f"{float(min_select_score):.2f}"
                    )
        except Exception as exc:
            logging.error("Lightweight panoramic planner failed: %s", exc)

        if success and not context_only and not score_only:
            # The paper-style scan samples half-step offset directions:
            # 30, 90, 150, 210, 270, 330 for six views. After the scan, the
            # agent is at the last sampled direction and turns to the selected
            # direction by the shortest sampled-angle rotation.
            clockwise_steps = selected_index + 1
            counter_steps = num_views - clockwise_steps
            if clockwise_steps <= counter_steps:
                for _ in range(clockwise_steps):
                    obs = self.simWrapper.step(scan_action)
            else:
                for _ in range(counter_steps):
                    obs = self.simWrapper.step(reverse_scan_action)
        else:
            # Return from the last half-step-offset direction to the original heading.
            # In context-only or score-only mode, the panoramic scan is only a
            # weak prompt/action prior and must not change the local candidate
            # coordinate frame used by the final ObjectNav action.
            obs = self.simWrapper.step(first_scan_action)

        metadata = {
            "lightweight_panoramic_enabled": True,
            "lightweight_panoramic_success": bool(success),
            "lightweight_panoramic_selected_angle": int(selected_angle),
            "lightweight_panoramic_selected_index": int(selected_index),
            "lightweight_panoramic_scores": scores,
            "lightweight_panoramic_raw_scores": raw_scores,
            "lightweight_panoramic_depth_scores": depth_scores,
            "lightweight_panoramic_reasons": reasons,
            "lightweight_panoramic_selected_reason": reasons.get(int(selected_angle), ""),
            "lightweight_panoramic_selected_score": selected_score,
            "lightweight_panoramic_min_select_score": float(self._lightweight_panoramic_min_select_score()),
            "lightweight_panoramic_min_score_margin": float(self._lightweight_panoramic_min_score_margin()),
            "lightweight_panoramic_score_margin": selected_score_margin,
            "lightweight_panoramic_target_mention_min_score": float(
                self._lightweight_panoramic_target_mention_min_score()
            ),
            "lightweight_panoramic_reject_reason": reject_reason,
            "lightweight_panoramic_num_views": int(num_views),
            "lightweight_panoramic_context_only": bool(context_only),
            "lightweight_panoramic_score_only": bool(score_only),
        }
        if success:
            selected_depth_score = depth_scores.get(int(selected_angle))
            if context_only:
                self._lightweight_panoramic_lock_until_step = None
                self._lightweight_panoramic_global_state = "context_hint"
            elif score_only:
                self._lightweight_panoramic_lock_until_step = None
                self._lightweight_panoramic_global_state = "score_hint"
            else:
                self._lightweight_panoramic_view_direction_deg = (
                    float(getattr(self, "_lightweight_panoramic_view_direction_deg", 0.0))
                    + float(selected_angle)
                ) % 360.0
                lock_steps = self._lightweight_panoramic_lock_steps()
                self._lightweight_panoramic_lock_until_step = int(self.step) + int(lock_steps)
                self._lightweight_panoramic_global_state = "direction_locked"
            self._lightweight_panoramic_lock_context = {
                "selected_angle": int(selected_angle),
                "selected_score": selected_score,
                "selected_reason": reasons.get(int(selected_angle), ""),
                "selected_depth_score": selected_depth_score,
                "scores": scores,
                "depth_scores": depth_scores,
                "gate_reason": "",
                "context_only": bool(context_only),
            }
        logging_data = {
            "LIGHTWEIGHT_PANORAMIC_PROMPT": prompt,
            "LIGHTWEIGHT_PANORAMIC_RESPONSE": response,
        }
        images = {
            "panoramic": panoramic_image,
            "color_origin": pano_images[0],
        }
        return obs, metadata, logging_data, images

    def _signed_angle_delta_deg(self, angle_a: float, angle_b: float) -> float:
        return (float(angle_a) - float(angle_b) + 180.0) % 360.0 - 180.0

    def _normalize_signed_angle_deg(self, angle: float) -> float:
        return self._signed_angle_delta_deg(float(angle), 0.0)

    def _select_lightweight_panoramic_override_action(
        self,
        probe_metadata: dict,
        planner_metadata: dict,
    ) -> tuple:
        override_metadata = {
            "lightweight_panoramic_override_action_enabled": bool(
                self._lightweight_panoramic_override_action_enabled()
            ),
            "lightweight_panoramic_override_applied": False,
            "lightweight_panoramic_override_reason": "",
            "lightweight_panoramic_override_theta_field": self._lightweight_panoramic_override_theta_field(),
            "lightweight_panoramic_override_max_angle_diff": float(
                self._lightweight_panoramic_override_max_angle_diff()
            ),
            "lightweight_panoramic_override_min_candidate_dist": float(
                self._lightweight_panoramic_override_min_candidate_dist()
            ),
            "lightweight_panoramic_override_allowed_actions": ",".join(
                sorted(self._lightweight_panoramic_override_allowed_actions())
            ),
            "lightweight_panoramic_override_min_switch_margin_deg": float(
                self._lightweight_panoramic_override_min_switch_margin_deg()
            ),
        }
        if not bool(planner_metadata.get("lightweight_panoramic_success")):
            override_metadata["lightweight_panoramic_override_reason"] = "scan_not_successful"
            return None, override_metadata

        step_metadata = dict(probe_metadata.get("step_metadata") or {})
        try:
            original_action = int(step_metadata.get("action_number"))
        except (TypeError, ValueError):
            original_action = None
        override_metadata["lightweight_panoramic_override_from_action"] = original_action
        allowed_actions = self._lightweight_panoramic_override_allowed_actions()
        if original_action == 0:
            if "candidate0" not in allowed_actions:
                override_metadata["lightweight_panoramic_override_reason"] = (
                    "candidate0_override_disabled"
                )
                return None, override_metadata
        elif original_action is not None and original_action > 0:
            if "forward" not in allowed_actions:
                override_metadata["lightweight_panoramic_override_reason"] = (
                    f"forward_override_disabled:{original_action}"
                )
                return None, override_metadata
        else:
            override_metadata["lightweight_panoramic_override_reason"] = (
                f"original_action_not_overridable:{original_action}"
            )
            return None, override_metadata

        selected_angle_raw = planner_metadata.get("lightweight_panoramic_selected_angle")
        try:
            selected_angle = self._normalize_signed_angle_deg(float(selected_angle_raw))
        except (TypeError, ValueError):
            override_metadata["lightweight_panoramic_override_reason"] = "missing_selected_angle"
            return None, override_metadata
        override_metadata["lightweight_panoramic_override_selected_angle_deg"] = float(selected_angle)

        candidates = list(step_metadata.get("candidate_actions") or [])
        if not candidates:
            for idx, action in enumerate(probe_metadata.get("a_final") or [], start=1):
                try:
                    r, theta = action
                    candidates.append({
                        "index": idx,
                        "r": float(r),
                        **candidate_angle_metadata(float(theta)),
                        "geometry_frame_version": 3,
                    })
                except Exception:
                    continue
        if not candidates:
            override_metadata["lightweight_panoramic_override_reason"] = "no_candidate_actions"
            return None, override_metadata

        theta_field = self._lightweight_panoramic_override_theta_field()
        theta_key = f"theta_deg_{theta_field}_frame"
        min_dist = self._lightweight_panoramic_override_min_candidate_dist()
        max_angle_diff = self._lightweight_panoramic_override_max_angle_diff()
        best = None
        for candidate in candidates:
            try:
                index = int(candidate.get("index"))
                dist = float(candidate.get("r"))
                theta = float(candidate.get(theta_key))
            except (TypeError, ValueError):
                continue
            if index <= 0 or dist < min_dist:
                continue
            diff = abs(self._signed_angle_delta_deg(theta, selected_angle))
            if best is None or diff < best["angle_diff"]:
                best = {
                    "index": index,
                    "r": dist,
                    "theta": theta,
                    "angle_diff": diff,
                    "candidate": candidate,
                }

        if best is None:
            override_metadata["lightweight_panoramic_override_reason"] = "no_forward_candidate"
            return None, override_metadata
        override_metadata.update({
            "lightweight_panoramic_override_candidate": int(best["index"]),
            "lightweight_panoramic_override_candidate_r": float(best["r"]),
            "lightweight_panoramic_override_candidate_theta_deg": float(best["theta"]),
            "lightweight_panoramic_override_angle_diff": float(best["angle_diff"]),
        })
        if float(best["angle_diff"]) > float(max_angle_diff):
            override_metadata["lightweight_panoramic_override_reason"] = (
                f"angle_diff_too_large:{float(best['angle_diff']):.1f}>{float(max_angle_diff):.1f}"
            )
            return None, override_metadata

        if original_action is not None and original_action > 0:
            original_candidate = None
            for candidate in candidates:
                try:
                    if int(candidate.get("index")) == int(original_action):
                        original_candidate = candidate
                        break
                except (TypeError, ValueError):
                    continue
            if original_candidate is None:
                override_metadata["lightweight_panoramic_override_reason"] = (
                    "missing_original_forward_candidate"
                )
                return None, override_metadata
            if int(best["index"]) == int(original_action):
                override_metadata["lightweight_panoramic_override_reason"] = (
                    "selected_same_as_original_forward"
                )
                return None, override_metadata
            try:
                original_theta = float(original_candidate.get(theta_key))
                original_diff = abs(self._signed_angle_delta_deg(original_theta, selected_angle))
            except (TypeError, ValueError):
                override_metadata["lightweight_panoramic_override_reason"] = (
                    "missing_original_forward_theta"
                )
                return None, override_metadata
            min_switch_margin = self._lightweight_panoramic_override_min_switch_margin_deg()
            override_metadata.update({
                "lightweight_panoramic_override_original_theta_deg": float(original_theta),
                "lightweight_panoramic_override_original_angle_diff": float(original_diff),
            })
            if float(best["angle_diff"]) + float(min_switch_margin) > float(original_diff):
                override_metadata["lightweight_panoramic_override_reason"] = (
                    "insufficient_forward_switch_gain:"
                    f"{float(original_diff - best['angle_diff']):.1f}<"
                    f"{float(min_switch_margin):.1f}"
                )
                return None, override_metadata

        override_metadata.update({
            "lightweight_panoramic_override_applied": True,
            "lightweight_panoramic_override_to_action": int(best["index"]),
            "lightweight_panoramic_override_reason": (
                "candidate_0_replaced_by_panoramic_direction"
                if original_action == 0
                else "forward_candidate_replaced_by_panoramic_direction"
            ),
        })
        return int(best["index"]), override_metadata

    def _apply_lightweight_panoramic_action_override(
        self,
        probe_metadata: dict,
        override_action_number: int,
        override_metadata: dict,
    ):
        step_metadata = probe_metadata.get("step_metadata") or {}
        step_metadata.update(override_metadata)
        step_metadata["lightweight_panoramic_reactive_original_action_number"] = (
            override_metadata.get("lightweight_panoramic_override_from_action")
        )
        step_metadata["action_number"] = int(override_action_number)
        step_metadata["action_decision_action"] = f"candidate_{int(override_action_number)}"
        try:
            candidates = list(step_metadata.get("candidate_actions") or [])
            if 1 <= int(override_action_number) <= len(candidates):
                chosen = candidates[int(override_action_number) - 1]
                step_metadata["chosen_action_r"] = chosen.get("r")
                step_metadata["chosen_action_theta_deg"] = chosen.get("theta_deg_model_frame")
                step_metadata["chosen_action_theta_deg_model_frame"] = chosen.get(
                    "theta_deg_model_frame"
                )
                step_metadata["chosen_action_yaw_command_deg"] = chosen.get(
                    "yaw_command_deg"
                )
                step_metadata["chosen_action_world"] = chosen.get("landing_position")
        except Exception:
            pass
        try:
            return self.agent._action_number_to_polar(
                int(override_action_number),
                probe_metadata.get("a_final") or [],
            )
        except Exception:
            return None

    def _finalize_agent_choose_action(self, obs: dict, agent_action, metadata: dict):
        """Finish the side effects normally performed by VLMNavAgent.step."""
        step_metadata = metadata['step_metadata']
        step_metadata.update(self.agent.cfg)
        try:
            if step_metadata['action_number'] == 0:
                self.agent.turned = self.agent.step_ndx
        except Exception:
            pass

        agent_state = obs['agent_state']
        chosen_action_image = obs['color_sensor'].copy()
        try:
            self.agent._project_onto_image(
                metadata['a_final'],
                chosen_action_image,
                agent_state,
                agent_state.sensor_states['color_sensor'],
                chosen_action=step_metadata.get('action_number'),
            )
            metadata['images']['color_sensor_chosen'] = chosen_action_image
        except Exception as exc:
            logging.error("Failed to render chosen action image in reactive pano path: %s", exc)
            metadata['images']['color_sensor_chosen'] = chosen_action_image

        try:
            self.agent.prompt_action_history.append(int(step_metadata['action_number']))
            max_history = int(os.environ.get("VLMNAV_WMNAV_LITE_ACTION_HISTORY_MAX", "32"))
            max_history = max(1, max_history)
            if len(self.agent.prompt_action_history) > max_history:
                self.agent.prompt_action_history = self.agent.prompt_action_history[-max_history:]
        except Exception:
            pass

        self.agent.step_ndx += 1
        return agent_action, metadata

    def _run_reactive_lightweight_panoramic_agent_step(
        self,
        obs: dict,
        planner_metadata: dict,
        planner_logging_data: dict,
        planner_images: dict,
    ):
        agent_state = obs.get('agent_state')
        if self.agent.step_ndx == 0 and agent_state is not None:
            self.agent.init_pos = agent_state.position

        probe_action, probe_metadata = self.agent._choose_action(obs)
        probe_step_metadata = dict(probe_metadata.get('step_metadata') or {})
        run_planner, gate_metadata = self._should_run_reactive_lightweight_panoramic_planner(
            probe_step_metadata
        )
        planner_metadata.update(gate_metadata)
        if not run_planner:
            return self._finalize_agent_choose_action(obs, probe_action, probe_metadata), obs, planner_metadata, planner_logging_data, planner_images

        next_obs, scan_metadata, scan_logging_data, scan_images = self._run_lightweight_panoramic_planner(
            obs,
            self.current_episode['object'],
        )
        planner_metadata.update(scan_metadata)
        planner_metadata.update(gate_metadata)
        planner_logging_data.update(scan_logging_data)
        planner_images.update(scan_images)
        self._lightweight_panoramic_last_scan_step = int(self.step)
        self._lightweight_panoramic_scan_count = int(
            getattr(self, "_lightweight_panoramic_scan_count", 0)
        ) + 1
        planner_metadata["lightweight_panoramic_last_scan_step"] = int(self.step)
        planner_metadata["lightweight_panoramic_steps_since_scan"] = 0
        planner_metadata["lightweight_panoramic_scan_count"] = int(
            self._lightweight_panoramic_scan_count
        )
        planner_metadata["lightweight_panoramic_reactive_reprompt"] = bool(
            planner_metadata.get("lightweight_panoramic_success")
        )
        next_obs['goal'] = self.current_episode['object']

        if self._lightweight_panoramic_override_action_enabled():
            override_action_number, override_metadata = self._select_lightweight_panoramic_override_action(
                probe_metadata,
                planner_metadata,
            )
            planner_metadata.update(override_metadata)
            probe_step_metadata.update(override_metadata)
            if override_action_number is not None:
                override_action = self._apply_lightweight_panoramic_action_override(
                    probe_metadata,
                    override_action_number,
                    override_metadata,
                )
                if override_action is not None:
                    probe_action = override_action
            planner_metadata["lightweight_panoramic_reactive_reprompt"] = False
            return self._finalize_agent_choose_action(
                next_obs,
                probe_action,
                probe_metadata,
            ), next_obs, planner_metadata, planner_logging_data, planner_images

        if self._lightweight_panoramic_score_only_enabled():
            planner_metadata["lightweight_panoramic_reactive_reprompt"] = False
            planner_metadata["lightweight_panoramic_score_only_kept_probe_action"] = True
            return self._finalize_agent_choose_action(next_obs, probe_action, probe_metadata), next_obs, planner_metadata, planner_logging_data, planner_images

        if not bool(planner_metadata.get("lightweight_panoramic_success")):
            return self._finalize_agent_choose_action(next_obs, probe_action, probe_metadata), next_obs, planner_metadata, planner_logging_data, planner_images

        context = self._lightweight_panoramic_context()
        context_active = bool(context.get("lock_active") or planner_metadata.get("lightweight_panoramic_success"))
        if context_active:
            next_obs['lightweight_panoramic_context'] = context
        planner_metadata.update({
            "lightweight_panoramic_context_active": context_active,
            "lightweight_panoramic_global_state": context.get("global_state"),
            "lightweight_panoramic_view_direction_deg": context.get("view_direction_deg"),
            "lightweight_panoramic_lock_active": context.get("lock_active"),
            "lightweight_panoramic_lock_until_step": context.get("lock_until_step"),
            "lightweight_panoramic_lock_steps_remaining": context.get("lock_steps_remaining"),
        })
        planner_metadata["lightweight_panoramic_reactive_original_action_number"] = probe_step_metadata.get(
            "action_number"
        )
        final_action, final_metadata = self.agent._choose_action(next_obs)
        return self._finalize_agent_choose_action(next_obs, final_action, final_metadata), next_obs, planner_metadata, planner_logging_data, planner_images

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment for the ObjectNav task.

        Args:
            obs (dict): The current observation.

        Returns:
            list: The next action to be taken by the agent.
        """
        super()._step_env(obs)
        obs['goal'] = self.current_episode['object']
        planner_metadata = {"lightweight_panoramic_enabled": False}
        planner_logging_data = {}
        planner_images = {}
        stop_multiview_metadata = {"stop_multiview_enabled": False}
        stop_multiview_images = {}
        terminal_certificate_every_step = os.environ.get(
            "VLMNAV_TERMINAL_CERTIFICATE_EVERY_STEP",
            "0",
        ).strip().lower() in ("1", "true", "yes", "y", "on")
        planner_enabled = self._lightweight_panoramic_planner_enabled()
        reactive_planner = bool(planner_enabled and self._lightweight_panoramic_reactive_enabled())
        defer_stop_multiview = bool(
            self._stop_multiview_enabled()
            and self._stop_multiview_defer_until_action_stop_enabled()
            and not terminal_certificate_every_step
        )
        if planner_enabled and not reactive_planner:
            run_planner, gate_metadata = self._should_run_lightweight_panoramic_planner()
            planner_metadata.update(gate_metadata)
            if run_planner:
                obs, planner_metadata, planner_logging_data, planner_images = self._run_lightweight_panoramic_planner(
                    obs,
                    self.current_episode['object'],
                )
                planner_metadata.update(gate_metadata)
                self._lightweight_panoramic_last_scan_step = int(self.step)
                self._lightweight_panoramic_scan_count = int(
                    getattr(self, "_lightweight_panoramic_scan_count", 0)
                ) + 1
                planner_metadata["lightweight_panoramic_last_scan_step"] = int(self.step)
                planner_metadata["lightweight_panoramic_steps_since_scan"] = 0
                planner_metadata["lightweight_panoramic_scan_count"] = int(
                    self._lightweight_panoramic_scan_count
                )
                obs['goal'] = self.current_episode['object']
            context = self._lightweight_panoramic_context()
            scan_success = bool(planner_metadata.get("lightweight_panoramic_success"))
            context_active = bool(context.get("lock_active") or scan_success)
            if context_active:
                obs['lightweight_panoramic_context'] = context
            planner_metadata.update({
                "lightweight_panoramic_context_active": context_active,
                "lightweight_panoramic_global_state": context.get("global_state"),
                "lightweight_panoramic_view_direction_deg": context.get("view_direction_deg"),
                "lightweight_panoramic_lock_active": context.get("lock_active"),
                "lightweight_panoramic_lock_until_step": context.get("lock_until_step"),
                "lightweight_panoramic_lock_steps_remaining": context.get("lock_steps_remaining"),
            })
        if terminal_certificate_every_step and self._stop_multiview_enabled():
            # The terminal certificate is trained and evaluated on exactly one
            # raw egocentric RGB.  A contact sheet would violate that contract
            # and silently turn the visual surface into a different model.
            stop_multiview_metadata = {
                "stop_multiview_enabled": True,
                "stop_multiview_suppressed_for_terminal_certificate": True,
                "stop_multiview_num_views": int(self._stop_multiview_num_views()),
                "stop_multiview_success": False,
                "stop_multiview_skipped": True,
                "stop_multiview_skip_reason": "terminal_certificate_requires_single_raw_rgb",
                "stop_multiview_input_images": 0,
                "stop_multiview_raw_input_images": 0,
            }
        elif self._stop_multiview_enabled() and not defer_stop_multiview:
            obs, stop_multiview_metadata, stop_multiview_images = self._run_stop_multiview_capture(obs)
            obs['goal'] = self.current_episode['object']
        elif defer_stop_multiview:
            obs["defer_stop_verification_until_action_stop"] = True
            stop_multiview_metadata = {
                "stop_multiview_enabled": True,
                "stop_multiview_num_views": int(self._stop_multiview_num_views()),
                "stop_multiview_deferred_until_action_stop": True,
                "stop_multiview_success": False,
                "stop_multiview_skipped": True,
                "stop_multiview_skip_reason": "pending_action_decision",
                "stop_multiview_input_images": 0,
                "stop_multiview_raw_input_images": 0,
            }
        agent_state = obs['agent_state']
        if reactive_planner:
            (
                (agent_action, metadata),
                obs,
                planner_metadata,
                planner_logging_data,
                planner_images,
            ) = self._run_reactive_lightweight_panoramic_agent_step(
                obs,
                planner_metadata,
                planner_logging_data,
                planner_images,
            )
            agent_state = obs['agent_state']
        else:
            agent_action, metadata = self.agent.step(obs)
        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']
        if defer_stop_multiview:
            obs, agent_action, stop_multiview_metadata, stop_multiview_images = self._run_deferred_stop_verification(
                obs,
                agent_action,
                metadata,
            )
            agent_state = obs['agent_state']
        if self._action0_pano_replace_enabled():
            step_metadata.setdefault("action0_pano_enabled", True)
            step_metadata.setdefault("action0_pano_triggered", False)
            step_metadata.setdefault("action0_pano_success", False)
            step_metadata.setdefault("action0_pano_count", int(getattr(self, "_action0_pano_replace_count", 0)))
            try:
                action_number_int = int(step_metadata.get("action_number"))
            except (TypeError, ValueError):
                action_number_int = None
            if action_number_int == 0:
                run_action0_pano, action0_gate_metadata = self._should_run_action0_pano_replace(
                    step_metadata,
                    obs,
                )
                step_metadata.update(action0_gate_metadata)
                step_metadata["action0_pano_original_action_number"] = 0
                if run_action0_pano:
                    (
                        action0_obs,
                        action0_metadata,
                        action0_logging_data,
                        action0_images,
                    ) = self._run_action0_pano_replace(
                        obs,
                        self.current_episode['object'],
                    )
                    step_metadata.update(action0_metadata)
                    logging_data.update(action0_logging_data)
                    images.update(action0_images)
                    if bool(action0_metadata.get("action0_pano_success")):
                        obs = action0_obs
                        agent_state = obs['agent_state']
                        agent_action = PolarAction.null
                        step_metadata["action0_pano_actual_action"] = "selected_rotation"
                    else:
                        obs = action0_obs
                        agent_state = obs['agent_state']
                        step_metadata["action0_pano_actual_action"] = "fallback_original_action0"
                else:
                    step_metadata["action0_pano_actual_action"] = "gate_keep_original_action0"
                    step_metadata["action0_pano_fallback_reason"] = action0_gate_metadata.get(
                        "action0_pano_gate_reason",
                        "gate_blocked",
                    )
        step_metadata.update(planner_metadata)
        step_metadata.update(stop_multiview_metadata)
        logging_data.update(planner_logging_data)
        images.update(planner_images)
        images.update(stop_multiview_images)

        # Price the selected macro action before it reaches SimWrapper.step.
        # In strict mode this is a hard guard: an action that would cross the
        # configured Compass/Habitat horizon is logged as an unexecuted
        # horizon terminal and is never left in last_agent_action for a later
        # post-hoc charge.
        strict_horizon_decision = self._strict_habitat_horizon_for_action(agent_action)
        if not strict_horizon_decision['allow_execution']:
            self.last_agent_action = None
        else:
            # CompassNav charges this selected PolarAction when the next
            # simulator observation enters _step_env. Terminal selections are
            # never executed.
            self.last_agent_action = agent_action
        metrics = self._calculate_metrics(
            agent_state,
            agent_action,
            self.current_episode['shortest_path'],
            self.cfg['max_steps'],
            strict_horizon_decision=strict_horizon_decision,
        )
        step_metadata.update(metrics)
        step_metadata.update(self._build_spatial_debug_fields(obs, step_metadata, logging_data, metadata))
        self._update_lightweight_panoramic_nav_state(step_metadata)
        step_metadata["lightweight_panoramic_current_navigation_mode"] = self._normalize_navigation_mode(
            step_metadata.get("action_navigation_mode")
        ) or None
        step_metadata["lightweight_panoramic_mode_streak"] = int(
            getattr(self, "_lightweight_panoramic_mode_streak", 0)
        )

        self._log(images, step_metadata, logging_data)

        if metrics['done']:
            agent_action = None

        return agent_action
