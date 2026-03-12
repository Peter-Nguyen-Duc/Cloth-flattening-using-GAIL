import argparse
from config_files.generate_config_file import generate_config_file
import numpy as np
from GAIL.src.IRL import train_IRL 

#Flick task environment
from environments.UR_env_Flick_TASK_C1 import URSim_SKRL_env


def make_args():
    parser = argparse.ArgumentParser(description='PyTorch GAIL')
    parser.add_argument('--render', action="store_true", default=False, 
                        help='if you dont want to render, set this to False')
    parser.add_argument('--gamma', type=float, default=0.70, 
                        help='discounted factor (default: 0.99)')
    parser.add_argument('--lamda', type=float, default=0.65, 
                        help='GAE hyper-parameter (default: 0.98)')
    parser.add_argument('--hidden_size', type=int, default=512, 
                        help='hidden unit size of actor, critic and discrim networks (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=8e-5, 
                        help='learning rate of models (default: 3e-4)')
    parser.add_argument('--l2_rate', type=float, default=1e-3, 
                        help='l2 regularizer coefficient (default: 1e-3)')
    parser.add_argument('--entropy_gain_PPO', type=float, default=1e-3, 
                        help='gain for entropy of PPO (default: 1e-3)')
    parser.add_argument('--critic_gain_PPO', type=float, default=0.5, 
                        help='critic gain for PPO (default: 1e-3)')
    parser.add_argument('--clip_param_actor', type=float, default=0.2, 
                        help='clipping parameter for PPO (default: 0.2)')
    parser.add_argument('--clip_param_critic', type=float, default= 10, 
                        help='clipping parameter for PPO (default: 0.2)')
    parser.add_argument('--discrim_update_num', type=int, default=10, 
                        help='update number of discriminator (default: 2)')
    parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                        help='update number of actor-critic (default: 10)')
    parser.add_argument('--total_sample_size', type=int, default=1, 
                        help='Mini batch size')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size to update (default: 64)')
    parser.add_argument('--suspend_accu_exp', type=float, default=0.80,
                        help='accuracy for suspending discriminator about expert data (default: 0.8)')
    parser.add_argument('--suspend_accu_gen', type=float, default=0.80,
                        help='accuracy for suspending discriminator about generated data (default: 0.8)')
    parser.add_argument('--max_iter_num', type=int, default=1000000,
                        help='maximal number of main iterations (default: 4000)')
    parser.add_argument('--seed', type=int, default=np.random.randint(1e6),
                        help='random seed (default: 500)')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='tensorboardx logs directory')
    parser.add_argument('--max_grad_norm', type=float, default=10,
                        help='gradient clipping of PPO')
    parser.add_argument('--hidden_layer_nodes_r', type=tuple, default=(256,256),
                        help='layer design for AIRL network')
    parser.add_argument('--hidden_layer_nodes_v', type=tuple, default=(256,256),
                        help='layer design for AIRL network')
    parser.add_argument('--irl_learning_rate', type=float, default=3e-4,
                        help='AIRL learning rate')
    parser.add_argument('--airl_epochs', type=int, default=5,
                        help='Epoch count of the training of AIRL')
    parser.add_argument('--AIRL_L2_weight_decay', type=float, default=1e-1,
                        help='Weight decay for the reward network of AIRL')
    parser.add_argument('--discriminator_delay_value', type=int, default=20,
                        help='Weight decay for the reward network of AIRL')
    parser.add_argument('--model_path', type=str, default="",
                        help='AIRL learning rate')
    parser.add_argument('--model_candidate', type=str, default="",
                        help='Epoch count of the training of AIRL')
    parser.add_argument('--disc_model_path', type=str, default="",
                        help='Weight decay for the reward network of AIRL')
    parser.add_argument('--disc_model_candidate', type=str, default="",
                        help='Weight decay for the reward network of AIRL')
    parser.add_argument('--discrim_type', type=str, default="",
                        help='Set to either AIRL or GAIL to train with the respective models')
    parser.add_argument('--expert_address', type=str, default="",
                        help='Weight decay for the reward network of AIRL')
    parser.add_argument('--train_RL', type=bool, default=True,
                        help='enable or disable train with RL for debugging')
    parser.add_argument('--train_IRL', type=bool, default=True,
                        help='enable or disable train with IRL for debugging')
    parser.add_argument('--render_mode', type=str, default="rgb_array",
                        help='enable or disable train with IRL for debugging')
    parser.add_argument('--task_device', type=str, default="cuda:0",
                        help='device to run the task on, e.g. "cuda:0" or "cpu"')
    parser.add_argument('--expert_data_size', type=int, default=-1,
                        help='Size of the expert data to be used for training, -1 for all data')
    parser.add_argument('--use_IRL_reward', type=bool, default=True,
                        help='This boolean decides whether to use the reward from AIRL or the environment reward for the PPO training')
    parser.add_argument('--record_reward_data', type=bool, default=False,
                        help='This boolean decides whether to use the reward from AIRL or the environment reward for the PPO training')


    args = parser.parse_args()

    return args


def main():
    # hmm, https://github.com/HumanCompatibleAI/imitation/tree/master
    args = make_args()
    env_args = generate_config_file()
    env_args.scene_path =  "scenes/c1_cloth_spacing_randomization/cloth_spacing0_050.xml"


    
    # env_args.domain_randomization_list = [  "scenes/c1_cloth_spacing_randomization/cloth_spacing0_030.xml",
    #                                         "scenes/c1_cloth_spacing_randomization/cloth_spacing0_040.xml",
    #                                         "scenes/c1_cloth_spacing_randomization/cloth_spacing0_050.xml",
    #                                         "scenes/c1_cloth_spacing_randomization/cloth_spacing0_060.xml"]


    env_args.episode_timeout = 200


    # ------- PPO settings -------
    args.gamma = 0.99
    args.lamda = 0.85
    args.actor_critic_update_num = 30
    args.total_sample_size = 100 # set to 100 when training
    args.batch_size = int(350 * args.total_sample_size / 10) # Ensure its int (both nummerically and in datatype)
    args.hidden_size = 256
    args.learning_rate = 2e-4
    args.max_grad_norm = 10

    args.entropy_gain_PPO = 0
    args.critic_gain_PPO = 1
    args.l2_rate = 1e-6


    # ------- IRL settings -------
    args.hidden_layer_nodes_r=(256,256)
    args.hidden_layer_nodes_v=(256,256)
    args.irl_learning_rate = 1e-4

    args.discrim_update_num = 5
    args.airl_epocs = args.discrim_update_num
    args.AIRL_L2_weight_decay = 1e-6
    args.discriminator_delay_value = 40

    args.expert_data_size = 350*100 # Use all expert data by default, can be set to a specific number

    args.discrim_type = "GAIL" # Set to either "AIRL" or "GAIL" to train with the respective models

    # ------- General settings -------
    args.max_iter_num = 80 # set high when training (i set it to 1000000)


    # Load RL model
    args.model_path = "GAIL/Saved_models/cloth_flattening_Baseline_25-09-11_02-08-54"
    args.model_candidate = "latest"

    # Load IRL model
    args.disc_model_path = "GAIL/Saved_models/cloth_flattening_Baseline_25-09-11_02-08-54"
    args.disc_model_candidate = "latest"


    args.expert_address = "GAIL/expert_demo/gello_demonstrations/C1_expert_data_gello_500/expert_memory.pkl"

    args.train_RL = True
    args.train_IRL = False

    args.use_IRL_reward = True


    args.env_render_mode = "human"  # "human" or "rgb_array" 



    # ------- ENV args settings -------
    env_args.agent_disabled = False
    env_args.expert_demonstration = False



    # Set true if recording data
    env_args.save_state_actions = False

    # Set true if recording reward data
    args.record_reward_data = False



    args.task_device = "cpu"  # Set to "cuda:0" for GPU or "cpu" for CPU, does not work with GPU at the moment
    train_IRL(args, env_args, URSim_SKRL_env, Train_at_start=False)

    print("Training completed!")
    print("current time: ", np.datetime64('now', 's'))

if __name__=="__main__":

    main()