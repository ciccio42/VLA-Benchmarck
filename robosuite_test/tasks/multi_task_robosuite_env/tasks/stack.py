from robosuite.environments.manipulation.stack import Stack as DefaultStack
import sys
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from multi_task_robosuite_env.arena import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
import robosuite.utils.transform_utils as T
from multi_task_robosuite_env.sampler import BoundarySampler
from multi_task_robosuite_env.tasks import bluewood, greenwood, redwood, grayplaster, lemon, darkwood
from collections import OrderedDict
from robosuite.utils.transform_utils import convert_quat, quat2mat


NAMES = {'r': 'red_block', 'g': 'green_block', 'b': 'blue_block'}
OFFSET = 0.0


class Stack(DefaultStack):

    def __init__(self,
                 robots,
                 env_configuration="default",
                 controller_configs=None,
                 mount_types="default",
                 gripper_types="default",
                 robot_offset=None,
                 initialization_noise="default",
                 table_full_size=(0.8, 0.8, 0.05),
                 table_friction=(1., 5e-3, 1e-4),
                 table_offset=(0, 0, 0.82),
                 use_camera_obs=True,
                 use_object_obs=True,
                 reward_scale=1.0,
                 reward_shaping=False,
                 placement_initializer=None,
                 has_renderer=False,
                 has_offscreen_renderer=True,
                 render_camera="frontview",
                 render_collision_mesh=False,
                 render_visual_mesh=True,
                 render_gpu_device_id=-1,
                 control_freq=20,
                 horizon=1000,
                 ignore_done=False,
                 hard_reset=True,
                 camera_names="agentview",
                 camera_heights=256,
                 camera_widths=256,
                 camera_depths=False,
                 camera_poses=None,
                 camera_attribs=None,
                 camera_gripper=None,
                 task_id=0,
                 object_type=None,
                 y_ranges=[[0.16, 0.19], [0.05, 0.09],
                           [-0.08, -0.03], [-0.19, -0.15]],
                 env_conf=None,
                 size=False,
                 shape=False,
                 color=False,
                 ):

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.robot_offset = robot_offset
        self.object_set = env_conf['object_set']
        print(f"Object set {self.object_set}")
        self.env_name = "block_stack"
        self.y_ranges = env_conf['y_ranges']
        self.x_ranges = env_conf['x_ranges']

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # camera poses and attributes
        self.camera_names = camera_names
        self.camera_poses = camera_poses
        self.camera_attribs = camera_attribs
        self.camera_gripper = camera_gripper
        self.camera_height = camera_heights
        self.camera_width = camera_widths

        self.task_id = task_id
        self.random_size = size
        self.random_shape = shape
        self.random_color = color

        self._obj_dim = {'cubeA': [0.07, 0.07, 0.07],
                         'cubeB': [0.07, 0.07, 0.07],
                         'cubeC': [0.07, 0.07, 0.07]}

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=env_conf['ignore_done'],
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        if self.robot_offset is None:
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0])
        else:
            xpos = self.robot_offset
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # add desired cameras
        if self.camera_poses is not None:
            for camera_name in self.camera_names:
                if camera_name != "robot0_eye_in_hand":
                    mujoco_arena.set_camera(camera_name=camera_name,
                                            pos=self.camera_poses[camera_name][0],
                                            quat=self.camera_poses[camera_name][1],
                                            camera_attribs=self.camera_attribs)

        # modify robot0_eye_in_hand
        if self.robots[0].robot_model.default_gripper == "Robotiq85Gripper":
            self.robots[0].robot_model.set_camera(camera_name="eye_in_hand",
                                                  pos=self.camera_gripper["Robotiq85Gripper"]["pose"][0],
                                                  quat=self.camera_gripper["Robotiq85Gripper"]["pose"][1],
                                                  root=self.camera_gripper["Robotiq85Gripper"]["root"],
                                                  camera_attribs=self.camera_attribs)

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        material_dict = {'g': greenwood, 'r': redwood, 'b': bluewood}
        color_dict = {'g': [0, 1, 0, 1], 'r': [1, 0, 0, 1], 'b': [0, 0, 1, 1]}
        task_name = ['rgb', 'rbg', 'bgr', 'brg', 'grb', 'gbr']
        task = task_name[self.task_id]
        print(f"Task-name {task} - First block on second block")
        size_noise = 0
        if self.random_size:
            while np.abs(size_noise) < 0.0015:
                size_noise = np.random.uniform(-0.004, 0.004)
        if self.random_color:
            material_dict = {'g': grayplaster, 'r': lemon, 'b': darkwood}
            color_dict = {'g': [0, 0, 0, 1], 'r': [
                0, 0, 0, 1], 'b': [0, 0, 0, 1]}

        if not self.random_shape:
            self.cubeA = BoxObject(
                name="cubeA",
                size_min=[0.024 + size_noise, 0.024 +
                          size_noise, 0.024 + size_noise],
                size_max=[0.024 + size_noise, 0.024 +
                          size_noise, 0.024 + size_noise],
                rgba=color_dict[task[0]],
                material=material_dict[task[0]],
            )
            self.cubeB = BoxObject(
                name="cubeB",
                size_min=[0.028 + size_noise, 0.028 +
                          size_noise, 0.028 + size_noise],
                size_max=[0.028 + size_noise, 0.028 +
                          size_noise, 0.028 + size_noise],
                rgba=color_dict[task[1]],
                material=material_dict[task[1]],
            )
            self.cubeC = BoxObject(
                name="cubeC",
                size_min=[0.024 + size_noise, 0.024 +
                          size_noise, 0.024 + size_noise],
                size_max=[0.024 + size_noise, 0.024 +
                          size_noise, 0.024 + size_noise],
                rgba=color_dict[task[2]],
                material=material_dict[task[2]],
            )
        else:
            self.cubeA = CylinderObject(
                name="cubeA",
                size_min=[0.021 + size_noise, 0.021 + size_noise],
                size_max=[0.021 + size_noise, 0.021 + size_noise],
                rgba=color_dict[task[0]],
                material=material_dict[task[0]],
                friction=2,
            )
            self.cubeB = CylinderObject(
                name="cubeB",
                size_min=[0.024 + size_noise, 0.024 + size_noise],
                size_max=[0.024 + size_noise, 0.024 + size_noise],
                rgba=color_dict[task[1]],
                material=material_dict[task[1]],
                friction=2,
            )
            self.cubeC = CylinderObject(
                name="cubeC",
                size_min=[0.021 + size_noise, 0.021 + size_noise],
                size_max=[0.021 + size_noise, 0.021 + size_noise],
                rgba=color_dict[task[2]],
                material=material_dict[task[2]],
                friction=2,
            )

        self.cubes = [self.cubeA, self.cubeC, self.cubeB]
        self.cube_names = {
            'cubeA': NAMES[task[0]], 'cubeB': NAMES[task[1]], 'cubeC':  NAMES[task[2]]}

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes,
        )

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(
            name="ObjectSampler")

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        self.placement_initializer.append_sampler(
            BoundarySampler(
                name="ObjectSampler",
                mujoco_objects=self.cubes[:2],
                x_range=self.x_ranges[0],
                y_range=self.y_ranges[0],
                rotation=[0, np.pi / 16],
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.000,
                addtional_dist=0.05
            )
        )

        self.placement_initializer.append_sampler(
            BoundarySampler(
                name="TargetSampler",
                mujoco_objects=self.cubeB,
                x_range=self.x_ranges[1],
                y_range=self.y_ranges[1],
                rotation=[0, np.pi / 16],
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.000,
                addtional_dist=0.001
            )
        )

    def _get_reference(self):
        super()._get_reference()
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()

        # position and rotation of the second cube
        cubeC_pos = np.array(self.sim.data.body_xpos[self.cubeC_body_id])
        cubeC_quat = convert_quat(
            np.array(self.sim.data.body_xquat[self.cubeC_body_id]), to="xyzw"
        )
        di["cubeC_pos"] = cubeC_pos
        di["cubeC_quat"] = cubeC_quat

        di['obj_bb'] = self._create_bb(di)

        return di

    def initialize_time(self, control_freq):
        self.sim.model.vis.quality.offsamples = 8
        super().initialize_time(control_freq)

    def _create_bb(self, di):
        """
            Create bb around each object in the scene for each camera of interest
        """

        def plot_bb(img, obj_bb):
            import cv2
            # draw bb
            for obj_name in obj_bb.keys():
                center = obj_bb[obj_name]['center']
                upper_left_corner = obj_bb[obj_name]['upper_left_corner']
                bottom_right_corner = obj_bb[obj_name]['bottom_right_corner']
                img = cv2.circle(
                    img, center, radius=1, color=(0, 0, 255), thickness=-1)
                img = cv2.rectangle(
                    img, upper_left_corner,
                    bottom_right_corner, (255, 0, 0), 1)
            cv2.imwrite("test_bb.png", img)
            # cv2.imshow("Test", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        import logging
        logging.basicConfig(
            format='%(levelname)s:%(message)s', level=logging.INFO)
        logger = logging.getLogger("BB-Creator")

        obj_bb = OrderedDict()
        # 1. For each camera of interest get the pose
        for camera_name in self.camera_names:
            if "in_hand" not in camera_name:
                obj_bb[camera_name] = OrderedDict()
                r_camera_world = quat2mat(convert_quat(
                    np.array(self.camera_poses[camera_name][1]), to="xyzw")).T
                p_camera_world = - \
                    r_camera_world @  np.reshape(np.expand_dims(
                        np.array(self.camera_poses[camera_name][0]), axis=0), (3, 1))
                # Transformation matrix
                T_camera_world = np.concatenate(
                    (r_camera_world, p_camera_world), axis=1)
                T_camera_world = np.concatenate(
                    (T_camera_world, np.array([[0, 0, 0, 1]])), axis=0)

                # 2. For each object compute bb
                for i, obj in enumerate(self.cubes):
                    obj_name = obj.name
                    obj_pos = np.array(
                        self.sim.data.body_xpos[self.sim.model.body_name2id(obj.root_body)])

                    obj_quat = T.convert_quat(
                        self.sim.data.body_xquat[self.sim.model.body_name2id(
                            obj.root_body)], to="xyzw"
                    )

                    obj_bb[camera_name][obj_name] = dict()

                    # 2. Create transformation matrix
                    T_camera_world = np.concatenate(
                        (r_camera_world, p_camera_world), axis=1)
                    T_camera_world = np.concatenate(
                        (T_camera_world, np.array([[0, 0, 0, 1]])), axis=0)
                    # logger.debug(T_camera_world)
                    p_world_object = np.expand_dims(
                        np.insert(obj_pos, 3, 1), 0).T
                    p_camera_object = T_camera_world @ p_world_object
                    logger.debug(
                        f"\nP_world_object:\n{p_world_object} - \nP_camera_object:\n {p_camera_object}")

                    # 3. Cnversion into pixel coordinates of object center
                    f = 0.5 * self.camera_height / \
                        np.tan(int(self.camera_attribs['fovy']) * np.pi / 360)

                    p_x_center = int(
                        (p_camera_object[0][0] / - p_camera_object[2][0]) * f + self.camera_width / 2)

                    p_y_center = int(
                        (- p_camera_object[1][0] / - p_camera_object[2][0]) * f + self.camera_height / 2)
                    logger.debug(
                        f"\nImage coordinate: px {p_x_center}, py {p_y_center}")

                    p_x_corner_list = []
                    p_y_corner_list = []
                    # 3.1 create a box around the object
                    for i in range(8):
                        if i == 0:  # upper-left front corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[self._obj_dim[obj_name][2]/2],
                                        [-self._obj_dim[obj_name][0]/2-OFFSET],
                                        [self._obj_dim[obj_name][1]/2+OFFSET],
                                        [0]])
                        elif i == 1:  # upper-right front corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[self._obj_dim[obj_name][2]/2],
                                        [self._obj_dim[obj_name][0]/2+OFFSET],
                                        [self._obj_dim[obj_name][1]/2+OFFSET],
                                        [0]])
                        elif i == 2:  # bottom-left front corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[self._obj_dim[obj_name][2]/2],
                                        [-self._obj_dim[obj_name][0]/2-OFFSET],
                                        [-self._obj_dim[obj_name][1]/2-OFFSET],
                                        [0]])
                        elif i == 3:  # bottom-right front corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[self._obj_dim[obj_name][2]/2],
                                        [self._obj_dim[obj_name][0]/2+OFFSET],
                                        [-self._obj_dim[obj_name][1]/2-OFFSET],
                                        [0]])
                        elif i == 4:  # upper-left back corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[-self._obj_dim[obj_name][2]/2],
                                        [-self._obj_dim[obj_name][0]/2-OFFSET],
                                        [self._obj_dim[obj_name][1]/2+OFFSET],
                                        [0]])
                        elif i == 5:  # upper-right back corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[-self._obj_dim[obj_name][2]/2],
                                        [self._obj_dim[obj_name][0]/2+OFFSET],
                                        [self._obj_dim[obj_name][1]/2+OFFSET],
                                        [0]])
                        elif i == 6:  # bottom-left back corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[-self._obj_dim[obj_name][2]/2],
                                        [-self._obj_dim[obj_name][0]/2-OFFSET],
                                        [-self._obj_dim[obj_name][1]/2-OFFSET],
                                        [0]])
                        elif i == 7:  # bottom-right back corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[-self._obj_dim[obj_name][2]/2],
                                        [self._obj_dim[obj_name][0]/2+OFFSET],
                                        [-self._obj_dim[obj_name][1]/2-OFFSET],
                                        [0]])

                        p_camera_object_corner = T_camera_world @ p_world_object_corner
                        logger.debug(
                            f"\nP_world_object_upper_left:\n{p_world_object_corner} -   \nP_camera_object_upper_left:\n {p_camera_object_corner}")

                        # 3.1 Upper-left corner and bottom right corner in pixel coordinate
                        p_x_corner = int(
                            (p_camera_object_corner[0][0] / - p_camera_object_corner[2][0]) * f + self.camera_width / 2)

                        p_y_corner = int(
                            (- p_camera_object_corner[1][0] / - p_camera_object_corner[2][0]) * f + self.camera_height / 2)
                        logger.debug(
                            f"\nImage coordinate upper_left corner: px {p_x_corner}, py {p_y_corner}")

                        p_x_corner_list.append(p_x_corner)
                        p_y_corner_list.append(p_y_corner)

                    x_min = min(p_x_corner_list)
                    y_min = min(p_y_corner_list)
                    x_max = max(p_x_corner_list)
                    y_max = max(p_y_corner_list)
                    # save bb
                    obj_bb[camera_name][obj_name]['center'] = [
                        p_x_center, p_y_center]
                    obj_bb[camera_name][obj_name]['upper_left_corner'] = [
                        x_max, y_max]
                    obj_bb[camera_name][obj_name]['bottom_right_corner'] = [
                        x_min, y_min]

                    # plot_bb(img=np.array(di[f'{camera_name}_image'][::-1, :, ::-1]),
                    #         obj_bb=obj_bb[camera_name])
        return obj_bb


class SawyerBlockStacking(Stack):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, **kwargs):
        obj = np.random.randint(6) if force_object is None else force_object
        super().__init__(task_id=obj, robots=['Sawyer'], **kwargs)


class PandaBlockStacking(Stack):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, task_id=None, **kwargs):
        obj = np.random.randint(6) if task_id is None else task_id
        super().__init__(task_id=obj, robots=['Panda'], **kwargs)


class UR5eBlockStacking(Stack):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, task_id=None, **kwargs):
        obj = np.random.randint(6) if task_id is None else task_id
        super().__init__(task_id=obj, robots=['UR5e'], **kwargs)


if __name__ == '__main__':
    from robosuite.environments.manipulation.pick_place import PickPlace
    import robosuite
    from robosuite.controllers import load_controller_config
    import debugpy
    import os
    import sys
    import yaml
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    import cv2

    controller = load_controller_config(default_controller="IK_POSE")

    # load env conf
    with open('/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/config/BlockStacking.yaml', 'r') as file:
        env_conf = yaml.safe_load(file)

    env = UR5eBlockStacking(task_id=1,
                            has_renderer=False,
                            controller_configs=controller,
                            has_offscreen_renderer=True,
                            reward_shaping=False,
                            use_camera_obs=True,
                            camera_heights=env_conf['camera_heights'],
                            camera_widths=env_conf['camera_widths'],
                            render_camera='frontview',
                            camera_depths=False,
                            camera_names=env_conf['camera_names'],
                            camera_poses=env_conf['camera_poses'],
                            camera_attribs=env_conf['camera_attribs'],
                            camera_gripper=env_conf['camera_gripper'],
                            env_conf=env_conf)

    for i in range(1000):
        if i % 200 == 0:
            env.reset()
            print(env.task_id)
        low, high = env.action_spec
        action = np.random.uniform(low=low, high=high)
        obs, _, _, _ = env.step(action)
        cv2.imwrite("debug.png", obs['camera_front_image'][::-1, :, ::-1])
        # env.render()
