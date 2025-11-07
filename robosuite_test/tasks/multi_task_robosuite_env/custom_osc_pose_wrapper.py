from robosuite.wrappers.wrapper import Wrapper
import numpy as np
from pyquaternion import Quaternion
import robosuite.utils.transform_utils as T
import copy
import cv2

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
osc_pose_logger = logging.getLogger(name="OSCPOSELogger")


class CustomOSCPoseWrapper(Wrapper):
    def __init__(self, env, ranges):
        super().__init__(env)
        self.action_repeat = 5
        self.ranges = ranges

    def _get_rel_action(self, action, base_pos, base_quat):
        if action.shape[0] == 7:
            cmd_quat = T.axisangle2quat(action[3:6])
            quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
            aa = T.quat2axisangle(quat)
            return np.concatenate((action[:3] - base_pos, aa, action[6:]))
        else:
            cmd_quat = Quaternion(angle=action[3] * np.pi, axis=action[4:7])
            cmd_quat = np.array(
                [cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
            quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
            aa = T.quat2axisangle(quat)
            return np.concatenate((action[:3] - base_pos, aa, action[7:]))

    def _project_point(self, point, sim, camera='agentview', frame_width=320, frame_height=320):
        model_matrix = np.zeros((3, 4))
        model_matrix[:3, :3] = sim.data.get_camera_xmat(camera).T

        fovy = sim.model.cam_fovy[sim.model.camera_name2id(camera)]
        f = 0.5 * frame_height / np.tan(fovy * np.pi / 360)
        camera_matrix = np.array(
            ((f, 0, frame_width / 2), (0, f, frame_height / 2), (0, 0, 1)))

        MVP_matrix = camera_matrix.dot(model_matrix)
        cam_coord = np.ones((4, 1))
        cam_coord[:3, 0] = point - sim.data.get_camera_xpos(camera)

        clip = MVP_matrix.dot(cam_coord)
        row, col = clip[:2].reshape(-1) / clip[2]
        row, col = row, frame_height - col
        return int(max(col, 0)), int(max(row, 0))

    def _get_real_depth(self, depth_img):
        # Make sure that depth values are normalized
        assert np.all(depth_img >= 0.0) and np.all(depth_img <= 1.0)
        extent = self.env.sim.model.stat.extent
        far = self.env.sim.model.vis.map.zfar * extent
        near = self.env.sim.model.vis.map.znear * extent
        return near / (1.0 - depth_img * (1.0 - near / far))

    def post_proc_obs(self, obs, env):
        new_obs = {}
        from PIL import Image
        robot_name = env.robots[0].robot_model.naming_prefix
        for k in obs.keys():
            if k.startswith(robot_name):
                name = k[len(robot_name):]
                if isinstance(obs[k], np.ndarray):
                    new_obs[name] = obs[k].copy()
                else:
                    new_obs[name] = obs[k]
            else:
                if isinstance(obs[k], np.ndarray):
                    new_obs[k] = obs[k].copy()
                else:
                    new_obs[k] = obs[k]

        frame_height, frame_width = self.env.camera_heights[0], self.env.camera_widths[0]
        if self.env.use_camera_obs:
            for camera_name in self.env.camera_names:
                # save image observation
                new_width = int(obs[f"{camera_name}_image"].shape[1] * 1)
                new_height = int(obs[f"{camera_name}_image"].shape[0] * 1)
                new_dim = (new_width, new_height)
                # cv2.imwrite("pre_flip_debug_img.png",
                #             obs[f"{camera_name}_image"])
                new_obs[f"{camera_name}_image"] = cv2.resize(
                    obs[f"{camera_name}_image"].copy()[::-1,], new_dim)
                # cv2.imwrite("post_flip_debug_img.png",
                #             new_obs[f"{camera_name}_image"])
                if self.env.camera_depths:
                    new_obs[f"{camera_name}_depth_norm"] = np.array(
                        obs[f"{camera_name}_depth"].copy()[::-1]*255, dtype=np.uint8)
                    new_obs[f"{camera_name}_depth"] = self._get_real_depth(
                        obs[f"{camera_name}_depth"]).copy()[::-1]

        aa = T.quat2axisangle(obs[robot_name+'eef_quat'])
        flip_points = np.array(self._project_point(obs[robot_name+'eef_pos'], env.sim,
                                                   camera="camera_front", frame_width=frame_width, frame_height=frame_height))
        flip_points[0] = frame_height - flip_points[0]
        flip_points[1] = frame_width - flip_points[1]
        new_obs['extent'] = self.env.sim.model.stat.extent
        new_obs['zfar'] = self.env.sim.model.vis.map.zfar
        new_obs['znear'] = self.env.sim.model.vis.map.znear
        new_obs['eef_point'] = flip_points
        new_obs['ee_aa'] = np.concatenate(
            (obs[robot_name+'eef_pos'], aa)).astype(np.float32)
        return new_obs

    def convert_rotation_gripper_to_world(self, action, base_pos, base_quat):
        """
            Take the current delta defined with respect to the gripper frame and convert it with respect to the world frame

            Args:
                action (): .
                base_pos (): .  
                base_quat (): . 
        """
        if action.shape[0] == 7:
            # Create rotation matrix R^{world}_{new_ee}
            R_w_new_ee = T.quat2mat(T.axisangle2quat(action[3:6]))
            # Create rotation matrix R^{ee}_{world}
            R_ee_world = T.matrix_inverse(T.quat2mat(base_quat))
            # Compute the delta with rispect to the base frame
            delta_world = R_w_new_ee @ R_ee_world
            # osc_pose_logger.debug(f"Delta world {T.mat2euler(delta_world)}")
            euler = -T.mat2euler(delta_world)
            aa = T.quat2axisangle(T.mat2quat(T.euler2mat(euler)))
            return np.concatenate((action[:3] - base_pos, aa, action[6:]))
        else:
            # retrieve command quaternion
            cmd_quat = Quaternion(angle=action[3], axis=action[4:7])
            cmd_quat = np.array(
                [cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
            # Create rotation matrix R^{world}_{new_ee}
            R_w_new_ee = T.quat2mat(cmd_quat)
            # Create rotation matrix R^{ee}_{world}
            R_ee_world = T.matrix_inverse(T.quat2mat(base_quat))
            # Compute the delta with rispect to the base frame
            delta_world = R_w_new_ee @ R_ee_world
            # osc_pose_logger.debug(f"Delta world {T.mat2euler(delta_world)}")
            euler = -T.mat2euler(delta_world)
            aa = T.quat2axisangle(T.mat2quat(T.euler2mat(euler)))
            return np.concatenate((action[:3] - base_pos, aa, action[7:]))

    def step(self, action):
        reward = -100.0
        osc_pose_logger.debug("-------------------------------------------")
        #for _ in range(self.action_repeat):
        dist = np.inf
        cnt = 10
        prev_dist = 0.0
        #
        #while dist > 0.004:
        #for _ in range(self.action_repeat):
        while dist > 0.004 and cnt > 0:
            # take the current position and gripper orientation with respect to world
            osc_pose_logger.debug(f"Target position {action[:3]}")
            base_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
            base_quat = T.mat2quat(np.reshape(
                self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id], (3, 3)))
            global_action = self.convert_rotation_gripper_to_world(
                action, base_pos, base_quat)
            osc_pose_logger.debug(f"Global delta position {global_action[:3]}")
            obs, reward_t, done, info = self.env.step(global_action)
            reward = max(reward, reward_t)
            
            dist = np.linalg.norm(
                self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id] - action[:3])
            
            if round(prev_dist,4) == round(dist,4):
                # if the distance does not change, the robot is not moving
                # print(f"Robot is not moving, {dist}")
                cnt -= 1
            else:
                prev_dist = dist
            
        osc_pose_logger.debug(
            "----------------------------------------------\n\n")
        return self.post_proc_obs(obs, self.env), reward, done, info

    def reset(self):
        obs = super().reset()
        return self.post_proc_obs(obs, self.env)

    def _get_observation(self):
        return self.post_proc_obs(self.env._get_observation(), self.env)
