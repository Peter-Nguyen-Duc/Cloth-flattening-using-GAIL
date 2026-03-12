


from typing import Tuple, Union

import mujoco as mj
import numpy as np
import torch


from utils.mj import get_joint_q, get_joint_dq, get_joint_ddq, get_joint_torque, get_body_pose, get_body_vel
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

import matplotlib
matplotlib.use('Agg')


import pickle

from typing import Tuple, Union



from spatialmath.base import r2q, qqmul, q2r, eul2r, qconj


from robots.ur_robot import URRobot
from collections import deque


from utils.rtb import make_tf



from shapely import Polygon



DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class URSim_SKRL_env(MujocoEnv):

    # Not sure what this guy is
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        # "render_fps": 100,
    }


    # set default episode_len for truncate episodes
    def __init__(self, args, name, render_mode="human"):
        # For expert training set




        self.args = args
        self.actuated_joints = 6
        self.render_mode = render_mode
        observation_space_size = 55


        # # change shape of observation to your observation space size
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_space_size,), 
            dtype=np.float64
        )
        
        super().__init__(
            model_path=os.path.abspath(args.scene_path),
            frame_skip=14,
            observation_space=self.observation_space,
            render_mode=render_mode
        )

        # UR5e robot interface
        self.ur5e = URRobot(args, self.data, self.model, robot_type=URRobot.Type.UR5e)


        # # constrained action space
        self.action_space = Box(
            -np.pi*2, 
            np.pi*2, 
            shape=(self.actuated_joints,), 
            dtype=np.float64
        )


        self.step_number = 0
        self.all_steps = 0
        self.episode_len = args.episode_timeout
        
        



        # Log error value between target and current position

        # Log position of the block, and joint value to see if its been flicked

        self.block_joint_value = []




        self.episode_count = 0
        self.name = name

        self.previous_action = [-1]*self.actuated_joints 


        self.robot_joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


        # Variants on the cloth flattening task
        self.spacing0025scene =  "scenes/c1_cloth_spacing_randomization/cloth_spacing0_025.xml"
        self.spacing003scene =   "scenes/c1_cloth_spacing_randomization/cloth_spacing0_030.xml"
        self.spacing0035scene =  "scenes/c1_cloth_spacing_randomization/cloth_spacing0_035.xml"
        self.spacing004scene =   "scenes/c1_cloth_spacing_randomization/cloth_spacing0_040.xml"
        self.spacing0045scene =  "scenes/c1_cloth_spacing_randomization/cloth_spacing0_045.xml"
        self.spacing005scene =   "scenes/c1_cloth_spacing_randomization/cloth_spacing0_050.xml"
        self.spacing0055scene =  "scenes/c1_cloth_spacing_randomization/cloth_spacing0_055.xml"
        self.spacing006scene =   "scenes/c1_cloth_spacing_randomization/cloth_spacing0_060.xml"
        self.spacing0065scene =  "scenes/c1_cloth_spacing_randomization/cloth_spacing0_065.xml"
        self.spacing007scene =   "scenes/c1_cloth_spacing_randomization/cloth_spacing0_070.xml"



        # Load start state:
        if self.args.scene_path == self.spacing0025scene:
            qpos_path = "config_files/C1_start_state/spacing_0_025/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_025/qvel_start_state.pkl"
        elif self.args.scene_path == self.spacing003scene:
            qpos_path = "config_files/C1_start_state/spacing_0_030/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_030/qvel_start_state.pkl"
        elif self.args.scene_path == self.spacing0035scene:
            qpos_path = "config_files/C1_start_state/spacing_0_035/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_035/qvel_start_state.pkl"
        elif self.args.scene_path == self.spacing004scene:
            qpos_path = "config_files/C1_start_state/spacing_0_040/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_040/qvel_start_state.pkl"
        elif self.args.scene_path == self.spacing0045scene:
            qpos_path = "config_files/C1_start_state/spacing_0_045/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_045/qvel_start_state.pkl"
        elif self.args.scene_path == self.spacing005scene: # basic environment
            qpos_path = "config_files/C1_start_state/spacing_0_050/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_050/qvel_start_state.pkl"
        elif self.args.scene_path == self.spacing0055scene:
            qpos_path = "config_files/C1_start_state/spacing_0_055/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_055/qvel_start_state.pkl"
        elif self.args.scene_path == self.spacing006scene:
            qpos_path = "config_files/C1_start_state/spacing_0_060/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_060/qvel_start_state.pkl"
        elif self.args.scene_path == self.spacing0065scene:
            qpos_path = "config_files/C1_start_state/spacing_0_065/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_065/qvel_start_state.pkl"
        elif self.args.scene_path == self.spacing007scene:
            print("WARNING environment: ", self.args.scene_path)
            print("Does not have a supported start state yet!")

            qpos_path = "config_files/C1_start_state/spacing_0_065/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_065/qvel_start_state.pkl"

        else:
            print("utilizing the standard start state")
            qpos_path = "config_files/C1_start_state/spacing_0_050/qpos_start_state.pkl"
            qvel_path = "config_files/C1_start_state/spacing_0_050/qvel_start_state.pkl"



        qpos_dir = os.path.abspath(qpos_path)
        qvel_dir = os.path.abspath(qvel_path)

        qpos_file = open(qpos_dir, "rb")
        qvel_file = open(qvel_dir, "rb")

        self.Q_list = pickle.load(qpos_file)
        self.Q_vel_list = pickle.load(qvel_file)


        self.data_points = 0


        path_expert_data = os.path.abspath("GAIL/expert_demo/gello_demonstrations/C1_expert_data_gello_500/expert_memory.pkl")

        expert_data_file = open(path_expert_data, "rb")
        self.expert_memory = pickle.load(expert_data_file)
        

        # DEBUG; This is for getting the observation space size.. 
        # But mujoco must initialize before its possible to run it
        self.reset()
        # print("Observation test: ", self._get_obs())
        assert len(self._get_obs()) == observation_space_size, f"Observation space size is not correct. Expected {observation_space_size}, got {len(self._get_obs())}"



    def change_scene_path(self, env_path):
        super().close()

        super().__init__(
            model_path=os.path.abspath(env_path),
            frame_skip=14,
            observation_space=self.observation_space,
            render_mode=self.render_mode
        )



    def define_cloth_joints(self):
        return ['c1_free_joint_1', 'c1_free_joint_2', 'c1_free_joint_3', 'c1_free_joint_4', 'c1_free_joint_5', 'c1_free_joint_6', 'c1_free_joint_7', 'c1_free_joint_8', 'c1_free_joint_9', 'unnamed_composite_0J0_1_0', 'unnamed_composite_0J1_1_0', 'unnamed_composite_0J2_1_0', 'unnamed_composite_0J0_1_1', 'unnamed_composite_0J1_1_1', 'unnamed_composite_0J2_1_1', 'unnamed_composite_0J0_1_2', 'unnamed_composite_0J1_1_2', 'unnamed_composite_0J2_1_2', 'unnamed_composite_0J0_1_3', 'unnamed_composite_0J1_1_3', 'unnamed_composite_0J2_1_3', 'unnamed_composite_0J0_1_4', 'unnamed_composite_0J1_1_4', 'unnamed_composite_0J2_1_4', 'unnamed_composite_0J0_1_5', 'unnamed_composite_0J1_1_5', 'unnamed_composite_0J2_1_5', 'unnamed_composite_0J0_1_6', 'unnamed_composite_0J1_1_6', 'unnamed_composite_0J2_1_6', 'unnamed_composite_0J0_1_7', 'unnamed_composite_0J1_1_7', 'unnamed_composite_0J2_1_7', 'unnamed_composite_0J0_1_8', 'unnamed_composite_0J1_1_8', 'unnamed_composite_0J2_1_8', 'unnamed_composite_0J0_2_0', 'unnamed_composite_0J1_2_0', 'unnamed_composite_0J2_2_0', 'unnamed_composite_0J0_2_1', 'unnamed_composite_0J1_2_1', 'unnamed_composite_0J2_2_1', 'unnamed_composite_0J0_2_2', 'unnamed_composite_0J1_2_2', 'unnamed_composite_0J2_2_2', 'unnamed_composite_0J0_2_3', 'unnamed_composite_0J1_2_3', 'unnamed_composite_0J2_2_3', 'unnamed_composite_0J0_2_4', 'unnamed_composite_0J1_2_4', 'unnamed_composite_0J2_2_4', 'unnamed_composite_0J0_2_5', 'unnamed_composite_0J1_2_5', 'unnamed_composite_0J2_2_5', 'unnamed_composite_0J0_2_6', 'unnamed_composite_0J1_2_6', 'unnamed_composite_0J2_2_6', 'unnamed_composite_0J0_2_7', 'unnamed_composite_0J1_2_7', 'unnamed_composite_0J2_2_7', 'unnamed_composite_0J0_2_8', 'unnamed_composite_0J1_2_8', 'unnamed_composite_0J2_2_8', 'unnamed_composite_0J0_3_0', 'unnamed_composite_0J1_3_0', 'unnamed_composite_0J2_3_0', 'unnamed_composite_0J0_3_1', 'unnamed_composite_0J1_3_1', 'unnamed_composite_0J2_3_1', 'unnamed_composite_0J0_3_2', 'unnamed_composite_0J1_3_2', 'unnamed_composite_0J2_3_2', 'unnamed_composite_0J0_3_3', 'unnamed_composite_0J1_3_3', 'unnamed_composite_0J2_3_3', 'unnamed_composite_0J0_3_4', 'unnamed_composite_0J1_3_4', 'unnamed_composite_0J2_3_4', 'unnamed_composite_0J0_3_5', 'unnamed_composite_0J1_3_5', 'unnamed_composite_0J2_3_5', 'unnamed_composite_0J0_3_6', 'unnamed_composite_0J1_3_6', 'unnamed_composite_0J2_3_6', 'unnamed_composite_0J0_3_7', 'unnamed_composite_0J1_3_7', 'unnamed_composite_0J2_3_7', 'unnamed_composite_0J0_3_8', 'unnamed_composite_0J1_3_8', 'unnamed_composite_0J2_3_8', 'unnamed_composite_0J0_4_0', 'unnamed_composite_0J1_4_0', 'unnamed_composite_0J2_4_0', 'unnamed_composite_0J0_4_1', 'unnamed_composite_0J1_4_1', 'unnamed_composite_0J2_4_1', 'unnamed_composite_0J0_4_2', 'unnamed_composite_0J1_4_2', 'unnamed_composite_0J2_4_2', 'unnamed_composite_0J0_4_3', 'unnamed_composite_0J1_4_3', 'unnamed_composite_0J2_4_3', 'unnamed_composite_0J0_4_4', 'unnamed_composite_0J1_4_4', 'unnamed_composite_0J2_4_4', 'unnamed_composite_0J0_4_5', 'unnamed_composite_0J1_4_5', 'unnamed_composite_0J2_4_5', 'unnamed_composite_0J0_4_6', 'unnamed_composite_0J1_4_6', 'unnamed_composite_0J2_4_6', 'unnamed_composite_0J0_4_7', 'unnamed_composite_0J1_4_7', 'unnamed_composite_0J2_4_7', 'unnamed_composite_0J0_4_8', 'unnamed_composite_0J1_4_8', 'unnamed_composite_0J2_4_8', 'unnamed_composite_0J0_5_0', 'unnamed_composite_0J1_5_0', 'unnamed_composite_0J2_5_0', 'unnamed_composite_0J0_5_1', 'unnamed_composite_0J1_5_1', 'unnamed_composite_0J2_5_1', 'unnamed_composite_0J0_5_2', 'unnamed_composite_0J1_5_2', 'unnamed_composite_0J2_5_2', 'unnamed_composite_0J0_5_3', 'unnamed_composite_0J1_5_3', 'unnamed_composite_0J2_5_3', 'unnamed_composite_0J0_5_4', 'unnamed_composite_0J1_5_4', 'unnamed_composite_0J2_5_4', 'unnamed_composite_0J0_5_5', 'unnamed_composite_0J1_5_5', 'unnamed_composite_0J2_5_5', 'unnamed_composite_0J0_5_6', 'unnamed_composite_0J1_5_6', 'unnamed_composite_0J2_5_6', 'unnamed_composite_0J0_5_7', 'unnamed_composite_0J1_5_7', 'unnamed_composite_0J2_5_7', 'unnamed_composite_0J0_5_8', 'unnamed_composite_0J1_5_8', 'unnamed_composite_0J2_5_8', 'unnamed_composite_0J0_6_0', 'unnamed_composite_0J1_6_0', 'unnamed_composite_0J2_6_0', 'unnamed_composite_0J0_6_1', 'unnamed_composite_0J1_6_1', 'unnamed_composite_0J2_6_1', 'unnamed_composite_0J0_6_2', 'unnamed_composite_0J1_6_2', 'unnamed_composite_0J2_6_2', 'unnamed_composite_0J0_6_3', 'unnamed_composite_0J1_6_3', 'unnamed_composite_0J2_6_3', 'unnamed_composite_0J0_6_4', 'unnamed_composite_0J1_6_4', 'unnamed_composite_0J2_6_4', 'unnamed_composite_0J0_6_5', 'unnamed_composite_0J1_6_5', 'unnamed_composite_0J2_6_5', 'unnamed_composite_0J0_6_6', 'unnamed_composite_0J1_6_6', 'unnamed_composite_0J2_6_6', 'unnamed_composite_0J0_6_7', 'unnamed_composite_0J1_6_7', 'unnamed_composite_0J2_6_7', 'unnamed_composite_0J0_6_8', 'unnamed_composite_0J1_6_8', 'unnamed_composite_0J2_6_8', 'unnamed_composite_0J0_7_0', 'unnamed_composite_0J1_7_0', 'unnamed_composite_0J2_7_0', 'unnamed_composite_0J0_7_1', 'unnamed_composite_0J1_7_1', 'unnamed_composite_0J2_7_1', 'unnamed_composite_0J0_7_2', 'unnamed_composite_0J1_7_2', 'unnamed_composite_0J2_7_2', 'unnamed_composite_0J0_7_3', 'unnamed_composite_0J1_7_3', 'unnamed_composite_0J2_7_3', 'unnamed_composite_0J0_7_4', 'unnamed_composite_0J1_7_4', 'unnamed_composite_0J2_7_4', 'unnamed_composite_0J0_7_5', 'unnamed_composite_0J1_7_5', 'unnamed_composite_0J2_7_5', 'unnamed_composite_0J0_7_6', 'unnamed_composite_0J1_7_6', 'unnamed_composite_0J2_7_6', 'unnamed_composite_0J0_7_7', 'unnamed_composite_0J1_7_7', 'unnamed_composite_0J2_7_7', 'unnamed_composite_0J0_7_8', 'unnamed_composite_0J1_7_8', 'unnamed_composite_0J2_7_8', 'unnamed_composite_0J0_8_0', 'unnamed_composite_0J1_8_0', 'unnamed_composite_0J2_8_0', 'unnamed_composite_0J0_8_1', 'unnamed_composite_0J1_8_1', 'unnamed_composite_0J2_8_1', 'unnamed_composite_0J0_8_2', 'unnamed_composite_0J1_8_2', 'unnamed_composite_0J2_8_2', 'unnamed_composite_0J0_8_3', 'unnamed_composite_0J1_8_3', 'unnamed_composite_0J2_8_3', 'unnamed_composite_0J0_8_4', 'unnamed_composite_0J1_8_4', 'unnamed_composite_0J2_8_4', 'unnamed_composite_0J0_8_5', 'unnamed_composite_0J1_8_5', 'unnamed_composite_0J2_8_5', 'unnamed_composite_0J0_8_6', 'unnamed_composite_0J1_8_6', 'unnamed_composite_0J2_8_6', 'unnamed_composite_0J0_8_7', 'unnamed_composite_0J1_8_7', 'unnamed_composite_0J2_8_7', 'unnamed_composite_0J0_8_8', 'unnamed_composite_0J1_8_8', 'unnamed_composite_0J2_8_8']
            




            
    def _r(self) -> float:
        '''
        Summary: This is the reward function for the RL mujoco simulation task.
                    The reward is based on the proximity to 0

        ARGS:
            args: The arguments that is passed from the main script
            robot: The robot model that also contains all mujoco data and model information

        RETURNS:
            punishment: Quantitative value that represents the performance of the robot, based on intended task specified in this function.
        '''

        # edge_indices = [[0,0], [0,8], [4,8], [8,8], [8,0], [4,0]]
        edge_indices = [[2,8], [8,8], [8,0], [2,0]]
        
        # add an orientation error on the cloth, to make sure the robot is holding the cloth the right way

        positions = []

        z_position_error = 0
        desired_z_position = 0.1768

        orientation_error = 0


        # Expert demo: -0.35
        desired_x_distance = 0.35
        

        
        for val in edge_indices:
            cloth_body_name = "unnamed_composite_0B" + str(val[0]) + "_" + str(val[1])

            cloth_pose = get_body_pose(self.data, self.model, cloth_body_name)

            positions.append([cloth_pose.t[0],
                            cloth_pose.t[1]])
            
            z_position_error += abs(cloth_pose.t[2] - desired_z_position)


        if len(edge_indices) == 4:
            orientation_error += positions[0][0] - positions[1][0]
            orientation_error += positions[3][0] - positions[2][0]

        elif len(edge_indices) == 6:
            orientation_error += positions[0][0] - positions[4][0]
            orientation_error += positions[1][0] - positions[3][0]

            
        # Put it into a sigmoid to emphasize the sign error, and diminish the small errors

        orientation_error = 1 / (1 + np.exp(-orientation_error*10)) # scale it to have a steeper curve



        # Only a problem when the z position is as desired (steady state)
        
        z_position_error *= max(0, 1 - z_position_error) # scale it to have a steeper curve


        cloth_Polygon = Polygon(positions)
        area = cloth_Polygon.area # [meters]

        if len(edge_indices) == 4:
            area *= 8.447722159
        if len(edge_indices) == 6:
            area *= 6.666666667
        # area is at mst 6,666666667

        # Scale based on the area
        # 0.025 spacing: 0,04 m^2 				→ scale: 4.00
        # 0.030 spacing: 0.0576 m^2 			→ scale: 2,777777778
        # 0.035 spacing: 0,0784 m^2 			→ scale: 2,040816327
        # 0.040 spacing:  0.1024 m^2 			→ scale: 1.5625
        # 0.045 spacing: 0,1296 m^2 			→ scale: 1,234567901
        # 0.050 spacing (CONTROL): 0.16 m^2 	→ scale: 1
        # 0.055 spacing: 0,1936 m^2 			→ scale: 0,8264
        # 0.060 spacing: 0.2304 m^2 			→ scale: 0,6944
        # 0.065 spacing: 0,2704 m^2 			→ scale: 0,591715976
        # 0.070 spacing: 0.3136 m^2 			→ scale: 0.510204082

        # normalize the reward if the cloth size is changed.
        if self.args.scene_path == self.spacing0025scene:
            area *= 4.00
        elif self.args.scene_path == self.spacing003scene:
            area *= 2.777777778
        elif self.args.scene_path == self.spacing0035scene:
            area *= 2.040816327
        elif self.args.scene_path == self.spacing004scene:
            area *= 1.5625
        elif self.args.scene_path == self.spacing0045scene:
            area *= 1.234567901
        elif self.args.scene_path == self.spacing005scene:
            area *= 1
        elif self.args.scene_path == self.spacing0055scene:
            area *= 0.8264
        elif self.args.scene_path == self.spacing006scene:
            area *= 0.694444444
        elif self.args.scene_path == self.spacing0065scene:
            area *= 0.591715976
        elif self.args.scene_path == self.spacing007scene:
            area *= 0.510204082



        # normalize the range [0 - 0.136], to [0 - 1] (full area is nearly impossible to achieve)

        # area_normalized = min( area * 10, 1.3)
        # area_normalized *= max(0, 1 - z_position_error)


        
        #normalize error between [0 and 1] from [0, 3]

        z_position_error_average  = z_position_error/3




        performance_metric = area - z_position_error_average - orientation_error

        # penalize if robot end effector is below x treshold length, to avoid the robot missing the table
        ee_pose = self.ur5e.get_ee_pose()

        end_of_table = 0.1

        robot_x_error = abs(end_of_table - ee_pose.t[0])





        reward = performance_metric

        # print("reward: ", reward, " Area norm: ", area_normalized, " Z pos error: ", z_position_error_average, " Sig x dist: ", sig_x_distance)


        return reward





    def _d(self) -> bool:

        

        # for i in range(len(joints)):
        #     if abs(joints[i]) > self.out_of_bounds: # If the joint is outside the range
        #         return True

        #     if i >= self.actuated_joints - 1:
        #         break



        # If timeout
        if self.step_number >= self.episode_len:
            return True




        return False


    def _get_obs(self) -> np.ndarray:
        joint = self.get_robot_q()
        joint_vel = self.get_robot_dq()


        # Order as [joint1 pos, joint1 vel, joint2 pos, joint2 vel, ...]
        observation = np.array([])


        for i in range(self.actuated_joints):
            observation = np.append(observation, joint[i])
        
        for i in range(self.actuated_joints):
            observation = np.append(observation, joint_vel[i])


        # Add cartesian position of the end effector
        ee_pose = self.ur5e.get_ee_pose()

        cartesian_ee_pos = ee_pose.t
        cartesian_ee_rot = r2q(ee_pose.R)

        observation = np.append(observation, cartesian_ee_pos)
        observation = np.append(observation, cartesian_ee_rot)

        # Cartesian space coordinates for each point on the cloth.



        #get the vertices of the cloth to get the area. put both the vertices and the area into the observation state.


        edge_indices = [[0,0], [0,8], [4,8], [8,8], [8,0], [4,0]]
        


        
        for val in edge_indices:
            cloth_body_name = "unnamed_composite_0B" + str(val[0]) + "_" + str(val[1])

            cloth_pose = get_body_pose(self.data, self.model, cloth_body_name)

    
            observation = np.append(observation, cloth_pose.t[0])
            observation = np.append(observation, cloth_pose.t[1])
            observation = np.append(observation, cloth_pose.t[2])




        for val in edge_indices:
            cloth_body_name = "unnamed_composite_0B" + str(val[0]) + "_" + str(val[1])

            cloth_vel = get_body_vel(self.data, self.model, cloth_body_name)

                            
            observation = np.append(observation, cloth_vel.t[0])
            observation = np.append(observation, cloth_vel.t[1])
            observation = np.append(observation, cloth_vel.t[2])



        # Add simple reward function to the observation state

        # reward = self._r()
        # observation = np.append(observation, reward)


        return observation



    

    # Get position of foldable block NOTE: task related function, should go somewhere else when cleaning up code.
    def get_box_position(self, block_pos, block_rot):
        """
        Summary: Get the position of the block in the robot frame
        ARGS:
            block_pos: The position of the block
            block_rot: The rotation of the block (quaternion)
        """
        

        # Get robot base position
        self.ur5e_base_cartesian = [0.22331901, 0.37537452, 0.08791326]
        self.ur5e_rot_quat = [ -0.19858483999999996, -0.00311175, 0.0012299899999999998, 0.98007799]
        self.ur5e_base_SE3 = make_tf(self.ur5e_base_cartesian, self.ur5e_rot_quat)





        block_pos = np.append(block_pos, 1)

        # position above the block to pick it up in box frame
        box_shift_translation_box_space = [-0.14,   0.2,   -0.075,  1.]

        # Convert from box space to world frame
        box_rotation = make_tf([0, 0, 0], block_rot)
        box_shift_translation = np.array(box_rotation) @ box_shift_translation_box_space

        # add the translation to the block position for correct pick up position
        block_pos[0] = block_pos[0] + box_shift_translation[0]
        block_pos[1] = block_pos[1] + box_shift_translation[1]
        block_pos[2] = block_pos[2] + abs(box_shift_translation[2]) # Never pick the box from beneath

        # Convert to robot frame
        point_translated = np.array(self.ur5e_base_SE3.inv()) @ block_pos
        point_translated = point_translated[0:3] / point_translated[3]

        # gained by designing rotation in world frame
        desired_rotation = [ 0.0000,   -0.7071,    0.7071,    0.0000]

        return point_translated, desired_rotation
    

    def joint_change_for_cartesian_pose(self, desired_position, desired_rotation):
        """
        Summary: 
            This function is used for the expert data generation where we get the 
            joint change value for the desired cartesian pose

        ARGS:
            desired_pos: The desired cartesian pose [x,y,z]
            desired_rot: The desired rotation in quaternion euler angles
        """
        
        # Position 2: charge flick downwards
        jac=np.zeros((6,self.model.nv))
        id=self.model.body("wrist_3_link").id


        mj.mj_jacBody(self.model, self.data, jac[:3], jac[3:], id) #Get geometric jacobian
        # Only the first 6 indexes are the ur5e robot joints

        jac_ur5e = jac[:, :6]

        pose = self.ur5e.get_ee_pose()

        quat_rot = r2q(pose.R)

        flick_rot = desired_rotation # in robot frame

        quat_flick = qqmul(quat_rot, r2q(eul2r(flick_rot)))


        def angular_velocities(q1, q2, dt): # https://mariogc.com/post/angular-velocity-quaternions/#the-angular-velocities
            return (2 / dt) * np.array([
                q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
                q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
                q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

        angular_velocity_flick = angular_velocities(quat_rot, quat_flick, 2.0)

        # Translation flick does not work


        
        world_to_robot = q2r([-0.19858483999999996, -0.00311175, 0.0012299899999999998, 0.98007799])

        ee_pose_world_frame = world_to_robot @ pose.R

        angular_velocity_flick = ee_pose_world_frame @ angular_velocity_flick
        desired_position = ee_pose_world_frame @ desired_position


        desired_cartesian_change = np.concatenate((desired_position, angular_velocity_flick))
        # Calculate the joint change
        joint_change = np.linalg.inv(jac_ur5e) @ (desired_cartesian_change)

        return joint_change



    def joint_space_control(self, input_val):
        
        # Remade into a velocity controller


        Kp = 30


        # Calculate the desired velocity based on a p controller



        q_tilde = np.array(input_val) - np.array(self.get_robot_q())


        u = Kp*(q_tilde)



        return u
        


    def step(
        self,
        a: Union[np.ndarray, list, torch.Tensor],
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Placeholder function.

        Returns:
            Tuple: A tuple.
        """


        # a = self.expert_memory[self.step_number+200*self.episode_count][1]

        # saturate action values 
        for i, val in enumerate(a):
            a[i] = min(max(a[i], -5), 5)




        self.do_simulation(a, self.frame_skip)

        self.step_number += 1


        reward = self._r()
        done = self._d()
        obs = self._get_obs()   



        # if self.step_number == 180:
        #     self.data_points += 1
        #     print(f"R ({self.data_points}): ", reward)



        robot_joints = self.get_robot_q()
        # robot_vel = self.get_robot_dq()
        # robot_acc = self.get_robot_ddq()





        infos =  {"Robot joint 0":  torch.tensor(robot_joints[0]),
                  }

        if self.render_mode == "human":
            self.render()
        
        truncated = self.step_number > self.episode_len




        return obs, reward, done, truncated, infos


    def reset(self): # overloading reset_model to fit more libraries 
        return self.reset_model(), {}
    

    def reset_model(
        self
    ):     
        
        self.all_steps += self.step_number



        self.episode_count += 1
        self.step_number = 0


        self.set_state(self.Q_list, self.Q_vel_list)
        self.memory_single_episode = deque()


        self.record_mode = False

        return  self._get_obs()



    def get_robot_q(self):

        # Return the 6 first values of the qpos
        pos = []

        for i, joint_name in enumerate(self.robot_joint_names):
            pos.append(get_joint_q(self.data, self.model, joint_name)[0])

        return pos
    


    def get_robot_dq(self):

        # Return the 6 first values of the qpos
        vel = []

        for i, joint_name in enumerate(self.robot_joint_names):
            vel.append(get_joint_dq(self.data, self.model, joint_name)[0])
        return vel
    

    def get_robot_ddq(self):

        # Return the 6 first values of the qpos
        acc = []

        for i, joint_name in enumerate(self.robot_joint_names):
            acc.append(get_joint_ddq(self.data, self.model, joint_name)[0])
        return acc



    def get_robot_torque(self):

        # Return the 6 first values of the qpos
        Torque = []

        for i, joint_name in enumerate(self.robot_joint_names):
            Torque.append(get_joint_torque(self.data, self.model, joint_name)[0])
        return Torque




    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]