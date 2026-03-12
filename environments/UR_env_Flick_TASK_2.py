



import argparse
import time
from threading import Lock
from typing import Tuple, Union

import mujoco as mj
import numpy as np
import torch


from utils.mj import get_mj_data, get_mj_model, attach, get_joint_names, get_joint_q, get_joint_dq, get_joint_ddq, get_joint_torque, body_name2id, set_joint_q
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
from typing import Union, Tuple, List
from skrl.memories.torch import Memory

import pickle

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from torch.distributions import Normal

from spatialmath.base import r2q, qqmul, q2r, eul2r, qconj
import spatialmath as sm

from robots.ur_robot import URRobot
from collections import deque


from utils.rtb import make_tf
from utils.math import angular_distance

import copy


from learning.AIRL.src.airl_model import AIRL
from learning.AIRL.utils.utils import log_prob_density

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
        observation_space_size = 23


        # # change shape of observation to your observation space size
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_space_size,), 
            dtype=np.float64
        )
        
        super().__init__(
            model_path=os.path.abspath(args.scene_path),
            frame_skip=7,
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
        

        self.plot_data_after_count = 3
        self.episode_count = 0
        self.name = name

        self.saved_expert_demonstrations = 0

        self.memory = deque()

        self.memory_single_episode = deque()



        self.robot_joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


        # Load start state:
        qpos_dir = os.path.abspath("learning/config_files/F2_start_state/backup/qpos_start_state.pkl")
        qvel_dir = os.path.abspath("learning/config_files/F2_start_state/backup/qvel_start_state.pkl")

        qpos_file = open(qpos_dir, "rb")
        qvel_file = open(qvel_dir, "rb")

        self.Q_list = pickle.load(qpos_file)
        self.Q_vel_list = pickle.load(qvel_file)


        # Load expert data [AIRL]
        # path_expert_data = os.path.abspath("learning/AIRL/expert_demo/F2_algorithm_demonstrator/expert_memory.pkl")
        
        # expert_data_file = open(path_expert_data, "rb")
        # self.expert_memory = pickle.load(expert_data_file)


        self.home_pos_robot = np.array([2.8, -1.5708, np.pi/2, -1.5708, -1.5708, -np.pi/2])    

            
        # DEBUG; This is for getting the observation space size.. 
        # But mujoco must initialize before its possible to run it
        self.reset()
        # print("Observation test: ", self._get_obs())
        assert len(self._get_obs()) == observation_space_size, f"Observation space size is not correct. Expected {observation_space_size}, got {len(self._get_obs())}"



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



        # Get the joint value closest in the list
        performance_score = 0

        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_q(self.data, self.model, "f2_block_1_joint")
        performance_score += abs(block_joint)

        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_q(self.data, self.model, "f2_block_2_joint")
        performance_score += abs( block_joint)

    


        reward = 1 - performance_score/(4.5)

        # print(f"joint {i} - Total: Punishment: ", punishment)
        # input()
        return reward[0]





    def _d(self) -> bool:

    
    
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
            

            
        # # Add cartesian position of the end effector
        ee_pose = self.ur5e.get_ee_pose()

        cartesian_ee_pos = ee_pose.t
        cartesian_ee_rot = r2q(ee_pose.R)

        observation = np.append(observation, cartesian_ee_pos)
        observation = np.append(observation, cartesian_ee_rot)




        # Add block joint value as observation - Joint value 1
        block_joint = get_joint_q(self.data, self.model, "f2_block_1_joint")
        observation = np.append(observation, block_joint)


        # Add block joint value as observation - Joint value 2
        block_joint = get_joint_q(self.data, self.model, "f2_block_2_joint")
        observation = np.append(observation, block_joint)


        # Add block joint value as observation - Joint value 1
        block_vel = get_joint_dq(self.data, self.model, "f2_block_1_joint")
        observation = np.append(observation, block_vel)


        # Add block joint value as observation - Joint value 2
        block_vel = get_joint_dq(self.data, self.model, "f2_block_2_joint")
        observation = np.append(observation, block_vel)



        
        return observation



    def step(
        self,
        a: Union[np.ndarray, list, torch.Tensor],
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Placeholder function.

        Returns:
            Tuple: A tuple.
        """
                
        # Expert data generation


        if self.args.expert_demonstration:
            if self.step_number > self.time_marks[0] and self.step_number < self.time_marks[1]: 
                # flick downwards position
                joint_change = self.joint_change_for_cartesian_pose(self.flick_downwards[0], self.flick_downwards[1])
                self.a_expert = joint_change
                
            elif self.step_number > self.time_marks[1] and self.step_number < self.time_marks[2]:
                # flick upwards position
                joint_change = self.joint_change_for_cartesian_pose(self.flick_upwards[0], self.flick_upwards[1])
                self.a_expert = joint_change

            elif self.step_number > self.time_marks[2] and self.step_number < self.time_marks[3]:
                # flick final rest

                joint_change = self.joint_change_for_cartesian_pose(self.flick_final_rest[0], self.flick_final_rest[1])
                self.a_expert = joint_change 

            elif self.step_number > self.time_marks[3] or self.step_number < self.time_marks[0]:
                self.a_expert = [0] * len(self.home_pos_randomized)


            a = self.a_expert



        # To check the expert demonstrator aciton values
        # test = self.expert_memory[self.step_number + 350*2]

        # saturate action values 
        for i, val in enumerate(a):
            a[i] = min(max(a[i], -10), 10)


        pre_state = self._get_obs()
        self.do_simulation(a, self.frame_skip)



        self.step_number += 1





        reward = self._r()
        done = self._d()
        obs = self._get_obs()   


        if self.args.expert_demonstration:
            # Store the expert data in the memory
            self.memory_single_episode.append([pre_state, a, reward, done, 0, obs]) # the log_pis should be 0, as it is calculated online in IRL


        infos =  { }

        if self.render_mode == "human":
            self.render()
        
        truncated = self.step_number > self.episode_len





        return obs, reward, done, truncated, infos


    def joint_change_for_cartesian_pose(self, desired_velocity, desired_rotation_velocity):
        """
        Summary: 
            This function is used for the expert data generation where we get the 
            joint change value for the desired cartesian pose

        ARGS:
            desired_pos: The desired cartesian pose [x,y,z]
            desired_rot: The desired rotation in quaternion euler angles
        """
        


        pose = self.ur5e.get_ee_pose()

        quat_rot = r2q(pose.R)

        flick_rot = desired_rotation_velocity # in robot frame

        quat_flick = qqmul(quat_rot, r2q(eul2r(flick_rot)))


        def angular_velocities(q1, q2, speed): # https://mariogc.com/post/angular-velocity-quaternions/#the-angular-velocities
            return (speed) * np.array([
                q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
                q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
                q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

        # make pythagorean of desired velocity and use it as inverse speed
        speed = np.linalg.norm(desired_rotation_velocity)


        angular_velocity_flick = angular_velocities(quat_rot, quat_flick, speed)


        
        world_to_robot = q2r([-0.19858483999999996, -0.00311175, 0.0012299899999999998, 0.98007799])

        ee_pose_world_frame = world_to_robot @ pose.R

        angular_velocity_flick = ee_pose_world_frame @ angular_velocity_flick
        desired_position = ee_pose_world_frame @ desired_velocity

        desired_cartesian_change = np.concatenate(( desired_position, angular_velocity_flick))
        # Calculate the joint change

        jac=np.zeros((6,self.model.nv))
        id=self.model.body("wrist_3_link").id

        mj.mj_jacBody(self.model, self.data, jac[:3], jac[3:], id) #Get geometric jacobian

        jac_ur5e = jac[:, :6]


        joint_change = np.linalg.inv(jac_ur5e) @ (desired_cartesian_change)

        return joint_change
    



    def reset(self): # overloading reset_model to fit more libraries 
        return self.reset_model(), {}
    

    def save_expert_data(self):
        '''
        Summary: This function is used to save the expert data to a file.
        ARGS:
            None
        RETURNS:
            None

        '''

        # Ensure the directory exists before saving the .npy file
        directory = f"./state_action_data/{self.name}"
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist


        # Save the memory to a pickle file
        with open(f"./state_action_data/{self.name}/expert_memory.pkl", "wb") as file:
            pickle.dump(self.memory, file)

        # Calculate the number of datasets (entries in self.state_actions)
        num_memory = len(self.memory)

        # Write metadata to a .txt file, overwriting the old file
        with open(f"./state_action_data/{self.name}/metadata_expert_memory.txt", "w") as file:
            file.write(f"Number of datasets: {num_memory}\n")

        print("Files saved!")



    def reset_model(
        self
    ):     
        self.all_steps += self.step_number
        self.episode_count += 1
        self.step_number = 0


        # If make expert demonstration, then load all memories into expert data:

        if self.args.expert_demonstration and len(self.memory_single_episode) > 0:
            
            reward = self.memory_single_episode[-1][2]  # Get the last reward of the episode


            # If reward is within bound of optimal reward, then save the memory to a file as expert demonstrator
            if reward > 0.995:
                self.saved_expert_demonstrations += 1
                print(f"Saving expert demonstration: {self.saved_expert_demonstrations} - with reward: {reward:.8f}")


                for val in self.memory_single_episode:
                    self.memory.append(val)



        

        self.memory_single_episode = deque()




        # Generate expert trajectory for following

        self.home_pos_randomized = self.home_pos_robot
        # Generate random variations of the expert flick
        # -------------- Flick downwards position ----------------
        noise_range_pos = [0.200, 0.200, 0.200]
        noise_range_rot = [0.20, 0.20, 0.20]

        desired_pos = [-0.300, 0.00, 0.000]
        desired_rot = [-1.0, 0.00, 0.0]

        # Add noise to the desired position
        desired_pos = [desired_pos[i] + np.random.uniform(-noise_range_pos[i], noise_range_pos[i]) for i in range(3)]
        desired_rot = [desired_rot[i] + np.random.uniform(-noise_range_rot[i], noise_range_rot[i]) for i in range(3)]

        self.flick_downwards = [desired_pos, desired_rot]



        # --------------- Flick upwards position ----------------
        # pos noise
        noise_range_pos = [0.200, 0.200, 0.200]
        noise_range_rot = [0.20, 0.20, 0.20]


        desired_pos = [0.100, 0.5, 0.000]
        desired_rot = [10.0, 0.0, 0.0]


        # Add noise to the desired position
        desired_pos = [desired_pos[i] + np.random.uniform(-noise_range_pos[i], noise_range_pos[i]) for i in range(3)]
        desired_rot = [desired_rot[i] + np.random.uniform(-noise_range_rot[i], noise_range_rot[i]) for i in range(3)]

        self.flick_upwards = [desired_pos, desired_rot]



        # --------------- Flick rest position ----------------
        # pos noise
        noise_range_pos = [0.200, 0.200, 0.200]
        noise_range_rot = [0.20, 0.20, 0.20]


        desired_pos = [-0.100, 0.00, 0]
        desired_rot = [-1.0, 0.0, 0.0]


        # Add noise to the desired position
        desired_pos = [desired_pos[i] + np.random.uniform(-noise_range_pos[i], noise_range_pos[i]) for i in range(3)]
        desired_rot = [desired_rot[i] + np.random.uniform(-noise_range_rot[i], noise_range_rot[i]) for i in range(3)]

        self.flick_final_rest = [desired_pos, desired_rot]



        # ----------- Randomize time ------------
        self.time_marks = [10, 60, 140, 200]

        time_marks_randomization_range = [15, 15, 15, 50]

        self.time_marks  = [self.time_marks[i] + np.random.uniform(-time_marks_randomization_range[i], time_marks_randomization_range[i]) for i in range(4)]




        self.set_state(self.Q_list, self.Q_vel_list)

        attach(
            self.data,
            self.model,
            "attach",
            "f2_block_free_joint",
            self.ur5e.T_world_base @ self.ur5e.get_ee_pose()
        )


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