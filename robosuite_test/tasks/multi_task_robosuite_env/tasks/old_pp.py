from collections import OrderedDict
from robosuite.environments.manipulation.pick_place import PickPlace as DefaultPickPlace

import numpy as np

from robosuite.utils.transform_utils import convert_quat, quat2mat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.objects import (
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject,
)
from multi_task_robosuite_env.arena import TableArena, BinsArena
from multi_task_robosuite_env.objects.custom_xml_objects import *
from multi_task_robosuite_env.sampler import BoundarySampler
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from multi_task_robosuite_env.objects.meta_xml_objects import Bin
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements
import robosuite.utils.transform_utils as T

OFFSET = 0.0


class PickPlace(SingleArmEnv):
    def __init__(self,
                 robots,
                 randomize_goal=False,
                 single_object_mode=0,
                 default_bin=3,
                 no_clear=False,
                 force_success=False,
                 use_novel_objects=False,
                 num_objects=4,
                 env_configuration="default",
                 controller_configs=None,
                 mount_types="default",
                 gripper_types="default",
                 robot_offset=None,
                 initialization_noise="default",
                 table_full_size=(0.39, 0.49, 0.82),
                 table_friction=(1., 5e-3, 1e-4),
                 bin1_pos=(0.1, -0.25, 0.8),
                 bin2_pos=(0.1, 0.28, 0.8),
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
                 **kwargs):

        self._randomize_goal = randomize_goal
        self._no_clear = no_clear
        self._default_bin = default_bin
        self._force_success = force_success
        self._was_closed = False
        self._use_novel_objects = use_novel_objects
        self._num_objects = num_objects
        if randomize_goal:
            assert single_object_mode == 2, "only works with single_object_mode==2!"

        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        self.obj_names = ["Milk", "Bread", "Cereal", "Can"]
        if object_type is not None:
            assert (
                object_type in self.object_to_id.keys()
            ), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.object_id = self.object_to_id[
                object_type
            ]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.robot_offset = robot_offset

        # settings for bin position
        self.bin1_pos = np.array(bin1_pos)
        self.bin2_pos = np.array(bin2_pos)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # camera poses and attributes
        self.camera_names = camera_names
        self.camera_poses = camera_poses
        self.camera_attribs = camera_attribs
        self.camera_gripper = camera_gripper
        self.camera_height = camera_heights
        self.camera_width = camera_widths

        self._obj_dim = {'milk': [0.05, 0.12, 0.080],
                         'bread': [0.045, 0.055, 0.045],
                         'cereal': [0.045, 0.055, 0.045],
                         'can': [0.045, 0.065, 0.045]}

        super().__init__(robots=robots,
                         mount_types=mount_types,
                         gripper_types=gripper_types,
                         initialization_noise=None,
                         camera_names=camera_names,
                         camera_heights=camera_heights,
                         camera_widths=camera_widths,
                         camera_depths=camera_depths,
                         **kwargs)

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(
            name="ObjectSampler")

        # can sample anywhere in bin
        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.04
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-bin_x_half, bin_x_half],
                y_range=[-bin_y_half, bin_y_half],
                rotation=[0, 0],
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=0.,
            )
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        if self.robot_offset is None:
            xpos = self.robots[0].robot_model.base_xpos_offset["bins"](
                self.table_full_size[0])
        else:
            xpos = self.robot_offset
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = BinsArena(
            bin1_pos=self.bin1_pos,
            table_full_size=self.table_full_size,
            table_friction=self.table_friction
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

        # store some arena attributes
        self.bin_size = mujoco_arena.table_full_size

        self.objects = []
        self.visual_objects = []
        for vis_obj_cls, obj_name in zip(
                (MilkVisualObject, BreadVisualObject,
                 CerealVisualObject, CanVisualObject),
                self.obj_names,
        ):
            vis_name = "Visual" + obj_name
            vis_obj = vis_obj_cls(name=vis_name)
            self.visual_objects.append(vis_obj)

        randomized_object_list = [[MilkObject, MilkObject2, MilkObject3], [BreadObject, BreadObject2, BreadObject3],
                                  [CerealObject, CerealObject2, CerealObject3], [CanObject, CanObject2, CanObject3]]

        if self._use_novel_objects:
            idx = np.random.randint(0, 3, 4)
            object_seq = (
                randomized_object_list[0][idx[0]], randomized_object_list[1][idx[1]
                                                                             ], randomized_object_list[2][idx[2]],
                randomized_object_list[3][idx[3]])
        else:
            object_seq = (MilkObject3, BreadObject3, CerealObject, CokeCan2)

        object_seq = object_seq[:self._num_objects]

        for obj_cls, obj_name in zip(
                object_seq,
                self.obj_names,
        ):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.visual_objects + self.objects,
        )
        compiler = self.model.root.find('compiler')
        compiler.set('inertiafromgeom', 'auto')
        if compiler.attrib['inertiagrouprange'] == "0 0":
            compiler.attrib.pop('inertiagrouprange')

        # Generate placement initializer
        self._get_placement_initializer()

    def clear_objects(self, obj):
        if self._no_clear:
            return
        super().clear_objects(obj)

    def _get_reference(self):
        super()._get_reference()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        for obj in (self.visual_objects + self.objects):
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(
                obj.root_body)
            self.obj_geom_id[obj.name] = [
                self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.objects))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.objects), 3))
        for i, obj in enumerate(self.objects):
            bin_id = i
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.
            bin_x_low += self.bin_size[0] / 4.
            bin_y_low += self.bin_size[1] / 4.
            self.target_bin_placements[i, :] = [
                bin_x_low, bin_y_low, self.bin2_pos[2]]

        if self.single_object_mode == 2:
            self.target_bin_placements = self.target_bin_placements[self._bin_mappings]

    def _reset_internal(self):

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the visual object body locations
                if "visual" in obj.name.lower():
                    self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                else:
                    # Set the collision object joints
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate(
                        [np.array(obj_pos), np.array(obj_quat)]))

        # Set the bins to the desired position
        self.sim.model.body_pos[self.sim.model.body_name2id(
            "bin1")] = self.bin1_pos
        self.sim.model.body_pos[self.sim.model.body_name2id(
            "bin2")] = self.bin2_pos

        self._was_closed = False
        if self.single_object_mode == 2:
            # randomly target bins if in single_object_mode==2
            self._bin_mappings = np.arange(self._num_objects)
            if self._randomize_goal:
                np.random.shuffle(self._bin_mappings)
            else:
                self._bin_mappings[:] = self._default_bin

        super()._reset_internal()

    def reward(self, action=None):
        if self.single_object_mode == 2:
            return float(self._check_success())
        else:
            # compute sparse rewards
            self._check_success()
            reward = np.sum(self.objects_in_bins)

            # add in shaped rewards
            if self.reward_shaping:
                staged_rewards = self.staged_rewards()
                reward += max(staged_rewards)
            if self.reward_scale is not None:
                reward *= self.reward_scale
                if self.single_object_mode == 0:
                    reward /= 4.0
            return reward

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin2_pos[0]
        bin_y_low = self.bin2_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0] / 2
        bin_y_high = bin_y_low + self.bin_size[1] / 2

        res = True
        if (
            bin_x_low < obj_pos[0] < bin_x_high
            and bin_y_low < obj_pos[1] < bin_y_high
            and self.bin2_pos[2] < obj_pos[2] < self.bin2_pos[2] + 0.1
        ):
            res = False
        return res

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        if self.single_object_mode == 2:
            obj_str = self.objects[self.object_id].name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            return not self.not_in_bin(obj_pos, self._bin_mappings[self.object_id])
        else:
            # remember objects that are in the correct bins
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            for i, obj in enumerate(self.objects):
                obj_str = obj.name
                obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
                dist = np.linalg.norm(gripper_site_pos - obj_pos)
                r_reach = 1 - np.tanh(10.0 * dist)
                self.objects_in_bins[i] = int(
                    (not self.not_in_bin(obj_pos, i)) and r_reach < 0.6)

            # returns True if a single object is in the correct bin
            if self.single_object_mode in {1, 2}:
                return np.sum(self.objects_in_bins) > 0

            # returns True if all objects are in correct bins
            return np.sum(self.objects_in_bins) == len(self.objects)

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

        if self.single_object_mode == 2:
            di['target-box-id'] = self._bin_mappings[self.object_id]
            di['target-object'] = self.object_id

        # add observation for all objects
        pr = self.robots[0].robot_model.naming_prefix

        # remember the keys to collect into object info
        object_state_keys = []

        # for conversion to relative gripper frame
        gripper_pose = T.pose2mat((di[pr + "eef_pos"], di[pr + "eef_quat"]))
        world_pose_in_gripper = T.pose_inv(gripper_pose)

        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            obj_pos = np.array(
                self.sim.data.body_xpos[self.obj_body_id[obj_str]])
            obj_quat = T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
            )
            di["{}_pos".format(obj_str)] = obj_pos
            di["{}_quat".format(obj_str)] = obj_quat

            # get relative pose of object in gripper frame
            object_pose = T.pose2mat((obj_pos, obj_quat))
            rel_pose = T.pose_in_A_to_pose_in_B(
                object_pose, world_pose_in_gripper)
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            di["{}_to_{}eef_pos".format(obj_str, pr)] = rel_pos
            di["{}_to_{}eef_quat".format(obj_str, pr)] = rel_quat

            object_state_keys.append("{}_pos".format(obj_str))
            object_state_keys.append("{}_quat".format(obj_str))
            object_state_keys.append("{}_to_{}eef_pos".format(obj_str, pr))
            object_state_keys.append("{}_to_{}eef_quat".format(obj_str, pr))

        di["object-state"] = np.concatenate([di[k] for k in object_state_keys])

        di['obj_bb'] = self._create_bb(di)

        return di

    def _create_bb(self, di):
        """
            Create bb around each object in the scene for each camera of interest
        """
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
                for i, obj in enumerate(self.objects):
                    obj_name = obj.name
                    obj_bb[camera_name][obj_name] = dict()

                    # convert obj pos in camera coordinate
                    if "nut" in obj_name:
                        obj_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(
                            f'{obj_name}_handle_site')]
                        obj_quat = T.mat2quat(
                            np.reshape(self.sim.data.site_xmat[self.sim.model.site_name2id(
                                f'{obj_name}_handle_site')], (3, 3)))
                    else:
                        obj_pos = np.array(
                            self.sim.data.body_xpos[self.obj_body_id[obj_name]])

                    if obj_name == 'bin':
                        obj_pos[2] = obj_pos[2] + 0.08
                    obj_quat = T.convert_quat(
                        self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
                    )

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

        return obj_bb

    def initialize_time(self, control_freq):
        self.sim.model.vis.quality.offsamples = 8
        super().initialize_time(control_freq)


class UR5ePickPlaceDistractor(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, randomize_goal=True, task_id=None, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        if task_id is None:
            obj_id = np.random.randint(0, 4)
            obj = items[obj_id]
        else:
            obj = items[int(task_id/len(items))]
        super().__init__(robots=['UR5e'], single_object_mode=2,
                         object_type=obj, no_clear=True, randomize_goal=randomize_goal, **kwargs)


class SawyerPickPlaceDistractor(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, randomize_goal=True, task_id=None, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        if task_id is None:
            obj_id = np.random.randint(0, 8)
            obj = items[obj_id]
        else:
            obj = items[int(task_id/len(items))]
        super().__init__(robots=['Sawyer'], single_object_mode=2,
                         object_type=obj, no_clear=True, randomize_goal=randomize_goal, **kwargs)


class PandaPickPlaceDistractor(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, randomize_goal=True, task_id=None, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        if task_id is None:
            obj_id = np.random.randint(0, 8)
            obj = items[obj_id]
        else:
            obj = items[int(task_id/len(items))]
        super().__init__(robots=['Panda'], single_object_mode=2, object_type=obj,
                         no_clear=True, randomize_goal=randomize_goal, **kwargs)


if __name__ == '__main__':
    from robosuite.controllers import load_controller_config
    from robosuite.environments.manipulation.pick_place import PickPlace
    import robosuite
    from robosuite.controllers import load_controller_config
    import debugpy
    import yaml
    import cv2

    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    controller = load_controller_config(default_controller="IK_POSE")
    # load env conf
    with open('/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/config/OldPickPlaceDistractor.yaml', 'r') as file:
        env_conf = yaml.safe_load(file)

    env = UR5ePickPlaceDistractor(has_renderer=False,
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
                                  robot_offset=env_conf['robot_offset'],
                                  mount_types=env_conf['mount_types'],
                                  env_conf=env_conf)
    obs = env.reset()
    cv2.imwrite(
        "debug.png", obs['camera_lateral_right_image'][::-1, :, ::-1])
    for i in range(10000):
        if i % 200 == 0:
            env.reset()
        low, high = env.action_spec
        action = np.random.uniform(low=low, high=high)
        obs, _, _, _ = env.step(action)
        cv2.imwrite(
            "debug.png", obs['camera_lateral_right_image'][::-1, :, ::-1])
        # env.render()
