import logging
import habitat_sim
import numpy as np
import magnum as mn

from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
from utils import local_to_global


class PolarAction:
    
    default = None #Default action if the VLM response is not parsed
    stop = None #Stop action
    null = None #Null action to just get current observation

    def __init__(self, r, theta, type=None):
        self.theta = theta
        self.r = r
        self.type = type

PolarAction.stop = PolarAction(0, 0, 'stop')
PolarAction.null = PolarAction(0, 0, 'null')
    

class SimWrapper:
    """
    A wrapper for Habitat-Sim that initializes agents, sensors, and manages movement and sensor observations.
    """
    def __init__(self, cfg):
        """
        Initialize the simulator with the specified configurations.

        :param cfg: Dictionary with configurations for the simulator, agents, and sensors.
        """
        self.scene_id = cfg['scene_id']
        self.use_goal_image_agent = cfg['use_goal_image_agent']
        self.allow_slide = cfg['allow_slide']
        self.limit_slide_deviation = bool(cfg.get('limit_slide_deviation', True))
        self.max_slide_lateral_ratio = float(cfg.get('max_slide_lateral_ratio', 0.75))
        self.max_slide_lateral_distance = float(cfg.get('max_slide_lateral_distance', 0.12))
        self._motion_sequence_id = 0
        self.last_motion_feedback = self._empty_motion_feedback()
        self.sensor_pitch = cfg['sensor_cfg']['pitch']
        self.fov = cfg['sensor_cfg']['fov']
        self.sensor_height = cfg['sensor_cfg']['height']
        self.resolution = (
            1080 // cfg['sensor_cfg']['res_factor'],
            1920 // cfg['sensor_cfg']['res_factor']
        )

        # Simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = cfg['scene_path']
        backend_cfg.scene_dataset_config_file = cfg['scene_config']
        backend_cfg.enable_physics = True
        if 'gpu_device_id' in cfg:
            backend_cfg.gpu_device_id = int(cfg['gpu_device_id'])

        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.radius = cfg['agent_radius']
        agent_cfg.height = cfg['agent_height']
        agent_cfg.sensor_specifications = self._create_sensor_specs()

        agents = [agent_cfg]
        if self.use_goal_image_agent:
            goal_cfg = self._create_goal_agent_config(fov=cfg['goal_image_agent_fov'])
            agents.append(goal_cfg)

        self.sim_cfg = habitat_sim.Configuration(backend_cfg, agents)

        try:
            self.sim = habitat_sim.Simulator(self.sim_cfg)
            logging.info('Simulator initialized!')
        except Exception as e:
            logging.error(f"Error initializing simulator: {e}")
            raise RuntimeError(f"Could not initialize simulator: {e}")

    def _create_sensor_specs(self):
        """
        Create sensor specifications for the agent.

        :return: List of sensor specifications.
        """
        sensor_specs = []
        sensor_specs.append(self._create_rgb_sensor_spec())
        sensor_specs.append(self._create_depth_sensor_spec())
        return sensor_specs

    def _create_rgb_sensor_spec(self):
        """
        Create an RGB camera sensor specification.

        :return: RGB sensor specification.
        """
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = f"color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = self.resolution
        rgb_sensor_spec.hfov = self.fov
        rgb_sensor_spec.position = mn.Vector3([0, self.sensor_height, 0])
        rgb_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0, 0])
        return rgb_sensor_spec

    def _create_depth_sensor_spec(self):
        """
        Create a depth camera sensor specification.

        :return: Depth sensor specification.
        """
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = f"depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = self.resolution
        depth_sensor_spec.hfov = self.fov
        depth_sensor_spec.position = mn.Vector3([0, self.sensor_height, 0])
        depth_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0, 0])
        return depth_sensor_spec

    def _create_goal_agent_config(self, fov):
        """
        Create a configuration for the goal image capture agent.

        :param fov: Field of view for the goal agent.
        :return: Agent configuration for the goal agent.
        """
        goal_cfg = habitat_sim.agent.AgentConfiguration()
        goal_sensor_spec = habitat_sim.CameraSensorSpec()
        goal_sensor_spec.uuid = "goal_sensor"
        goal_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        goal_sensor_spec.resolution = self.resolution
        goal_sensor_spec.hfov = fov
        goal_sensor_spec.orientation = mn.Vector3([0, 0, 0])
        goal_sensor_spec.position = mn.Vector3([0, 0, 0])
        goal_cfg.sensor_specifications = [goal_sensor_spec]
        return goal_cfg

    def step(self, action: PolarAction):
        """
        Move the agent based on the specified action and magnitude.

        :param action: The action to perform.
        """
        agent = self.sim.get_agent(0)
        curr_state = agent.get_state()
        start_position = np.asarray(curr_state.position, dtype=float).copy()
        requested_distance = 0.0
        requested_theta = 0.0
        traveled_distance = 0.0
        collision_substeps = 0
        slide_rejected_substeps = 0
        expected_position = start_position.copy()

        if action is PolarAction.null:
            new_agent_state = curr_state
        else:
            requested_distance = max(0.0, float(action.r))
            requested_theta = float(action.theta)
            new_agent_state = habitat_sim.AgentState()
            new_agent_state.position = np.copy(curr_state.position)
            new_agent_state.rotation = curr_state.rotation
            self._rotate_yaw(new_agent_state, new_agent_state.rotation, action.theta)
            expected_position = local_to_global(
                new_agent_state.position,
                new_agent_state.rotation,
                np.array([0.0, 0.0, -requested_distance]),
            )
            movement = self._move_forward(
                new_agent_state,
                new_agent_state.position,
                new_agent_state.rotation,
                requested_distance,
            )
            traveled_distance = float(movement['traveled_distance'])
            collision_substeps = int(movement['collision_substeps'])
            slide_rejected_substeps = int(movement['slide_rejected_substeps'])
            agent.set_state(new_agent_state)

        observation = self.sim.get_sensor_observations(0)
        final_state = agent.get_state()
        end_position = np.asarray(final_state.position, dtype=float).copy()
        actual_distance = float(np.linalg.norm(end_position - start_position))
        execution_ratio = (
            actual_distance / requested_distance if requested_distance > 1e-6 else 1.0
        )
        self._motion_sequence_id += 1
        self.last_motion_feedback = {
            'sequence_id': int(self._motion_sequence_id),
            'action_type': getattr(action, 'type', None),
            'requested_distance': float(requested_distance),
            'requested_yaw_rad': float(requested_theta),
            'start_position': start_position.tolist(),
            'expected_end_position': np.asarray(expected_position, dtype=float).tolist(),
            'end_position': end_position.tolist(),
            'net_displacement': float(actual_distance),
            'traveled_distance': float(traveled_distance),
            'execution_ratio': float(execution_ratio),
            'collision_substeps': int(collision_substeps),
            'slide_rejected_substeps': int(slide_rejected_substeps),
            'underexecuted': bool(requested_distance > 0.05 and execution_ratio < 0.5),
        }
        observation['agent_state'] = final_state
        observation['motion_feedback'] = dict(self.last_motion_feedback)
        return observation

    def _move_forward(self, agent_state, curr_position, curr_rotation, magnitude):
        """
        Move the agent_state forward by a specified magnitude.

        :param agent_state: The state of the agent to be updated.
        :param curr_position: Current position of the agent.
        :param curr_rotation: Current rotation of the agent.
        :param magnitude: Distance to move forward.
        """
        local_point = np.array([0, 0, -magnitude])
        global_point = local_to_global(curr_position, curr_rotation, local_point)
        delta = (global_point - curr_position) / 10
        new_position = np.copy(curr_position)
        traveled_distance = 0.0
        collision_substeps = 0
        slide_rejected_substeps = 0
        requested_step = float(np.linalg.norm(delta))
        direction = delta / requested_step if requested_step > 1e-9 else np.zeros(3, dtype=float)

        for _ in range(10):
            if self.allow_slide:
                candidate = self.sim.pathfinder.try_step(new_position, new_position + delta)
            else:
                candidate = self.sim.pathfinder.try_step_no_sliding(new_position, new_position + delta)

            actual_delta = np.asarray(candidate, dtype=float) - np.asarray(new_position, dtype=float)
            progress = float(np.dot(actual_delta, direction)) if requested_step > 1e-9 else 0.0
            lateral = float(np.linalg.norm(actual_delta - progress * direction))
            excessive_slide = bool(
                self.allow_slide
                and self.limit_slide_deviation
                and lateral > self.max_slide_lateral_distance
                and lateral > max(progress, 0.0) * self.max_slide_lateral_ratio
            )
            if excessive_slide:
                slide_rejected_substeps += 1
                candidate = self.sim.pathfinder.try_step_no_sliding(
                    new_position,
                    new_position + delta,
                )
                actual_delta = (
                    np.asarray(candidate, dtype=float)
                    - np.asarray(new_position, dtype=float)
                )
                progress = (
                    float(np.dot(actual_delta, direction))
                    if requested_step > 1e-9
                    else 0.0
                )
                if requested_step > 1e-6 and progress < requested_step * 0.5:
                    collision_substeps += 1
                    break

            moved = float(np.linalg.norm(actual_delta))
            if requested_step > 1e-6 and progress < requested_step * 0.5:
                collision_substeps += 1
            traveled_distance += moved
            new_position = candidate

        agent_state.position = new_position
        return {
            'traveled_distance': float(traveled_distance),
            'collision_substeps': int(collision_substeps),
            'slide_rejected_substeps': int(slide_rejected_substeps),
        }

    def _empty_motion_feedback(self):
        return {
            'sequence_id': int(getattr(self, '_motion_sequence_id', 0)),
            'action_type': None,
            'requested_distance': 0.0,
            'requested_yaw_rad': 0.0,
            'start_position': None,
            'expected_end_position': None,
            'end_position': None,
            'net_displacement': 0.0,
            'traveled_distance': 0.0,
            'execution_ratio': 1.0,
            'collision_substeps': 0,
            'slide_rejected_substeps': 0,
            'underexecuted': False,
        }

    def _rotate_yaw(self, agent_state, curr_rotation, magnitude):
        """
        Rotate the agent_state by a specified angle.

        :param agent_state: The state of the agent to be updated.
        :param curr_rotation: Current rotation of the agent.
        :param magnitude: Angle in radians to rotate counterclockwise.
        """
        theta, axis = quat_to_angle_axis(curr_rotation)
        if axis[1] < 0:  # Ensure consistent rotation direction
            theta = 2 * np.pi - theta
        new_theta = theta + magnitude
        agent_state.rotation = quat_from_angle_axis(new_theta, np.array([0, 1, 0]))

    def get_goal_image(self, goal_position, goal_rotation):
        """
        Capture an image from the goal agent's perspective.

        :param goal_position: Position of the goal agent.
        :param goal_rotation: Rotation of the goal agent.
        :return: The captured image from the goal sensor.
        """
        assert self.use_goal_image_agent, "Goal image agent is not enabled."

        goal_agent = self.sim.get_agent(1)
        new_agent_state = habitat_sim.AgentState()
        new_agent_state.position = goal_position
        new_agent_state.rotation = goal_rotation
        goal_agent.set_state(new_agent_state)

        observations = self.sim.get_sensor_observations(1)
        return observations['goal_sensor']

    def set_state(self, pos, quat):
        """
        Set the agent's state to the specified position and orientation.

        :param pos: The position to set.
        :param quat: The quaternion orientation to set.
        """
        agent = self.sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = pos
        agent_state.rotation = quat
        agent.set_state(agent_state)

    def get_path(self, path):
        """
        Find a path to the specified target position.

        :param path: Target position for pathfinding.
        :return: The path found by the pathfinder.
        """
        if self.sim.pathfinder.find_path(path):
            return path.geodesic_distance
        else:
            logging.info('NO PATH FOUND')
            return 1000

    def reset(self):
        """
        Close the simulator to clean up memory.
        """
        try:
            self.sim.close()
        except:
            pass
