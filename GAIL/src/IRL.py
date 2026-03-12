import os
import pickle

import numpy as np
from collections import deque

import torch

import torch.optim as optim
from tensorboardX import SummaryWriter 

from GAIL.utils.utils import *
from GAIL.src.model import Actor, Critic, Discriminator
from GAIL.src.airl_model import AIRL
from GAIL.src.train_model import train_actor_critic, train_discrim

# ----------- Import peter stuff -----------
# from learning.SKRL_UR_env_Q_learning import URSim_SKRL_env


#from learning.environments.SKRL_UR_env_PPO import URSim_SKRL_env


from config_files.generate_config_file import generate_config_file

import time

# Calculating the log_probability (log_pis), of the actions taken using the GAIL repo implementation
from GAIL.utils.utils import log_prob_density
from GAIL.src.memory_process import process_mem, process_expert_data

torch.set_default_dtype(torch.float32)

import copy



def mahalanobis_distance(action, mean, std):
    """
    This function calculates the mahalanobis distance between the action point, and the gaussian variables.
    """

    mahalanobis_distance_val = torch.sqrt((action - mean).pow(2) / std.pow(2))

    return mahalanobis_distance_val

def euclidean_distance(action, action2):
    """
    This function calculates the mahalanobis distance between the action point, and the gaussian variables.
    """

    euclidean_val = torch.sqrt((action - action2).pow(2))

    return euclidean_val


def get_IRL_reward(memory, AIRL_trainer, normalization_val = None):


    with torch.no_grad():

        states, actions, rewards, masks, log_pis, next_states = process_mem(memory)

        observation_space = len(states[0])
        next_state = torch.tensor(np.array([entry[5] for entry in memory])).type(dtype=torch.float32).view(-1, observation_space)    # Convert masks to array


        



        AIRL_trainer.disc.eval()
        irl_reward = AIRL_trainer.get_reward(states, actions, masks, next_state)

        debug = 0
        for i, val in enumerate(memory):
            if normalization_val != None:
                val[2] = irl_reward[i] / normalization_val # Normalize reward based on expert reward
            else:
                val[2] = irl_reward[i]
            
            debug += irl_reward[i]

            memory[i] = val


    return memory

def get_GAIL_reward(memory, discrim, normalization_val=None):
    with torch.no_grad():

        states, actions, rewards, masks, log_pis, next_states = process_mem(memory)

        observation_space = len(states[0])
        next_state = torch.tensor(np.array([entry[5] for entry in memory])).type(dtype=torch.float32).view(-1, observation_space)    # Convert masks to array


    
        irl_reward = discrim.get_reward(states, actions)

        debug = 0
        for i, val in enumerate(memory):
            if normalization_val != None:
                val[2] = irl_reward[i] / normalization_val # Normalize reward based on expert reward
            else:
                val[2] = irl_reward[i]
            
            debug += irl_reward[i]

            memory[i] = val


    return memory



def check_nan_in_model(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            return True
        else:
            return False

def save_parameters(args, env_args, directory):
    """
    SUMMARY: 
        This file is for saving a specific params file that is given in the main code

    ARGS:
        params: The param file that will be saved
        directory: The directory at which it will be saved in.
    """


    # Save parameters to file
    params_file = os.path.join(directory, "params.txt")
    args_dict = vars(args)  # Convert namespace to dictionary
    env_args_dict = vars(env_args)
    with open(params_file, "w") as f:
        
        f.write(" --------------------- Arguments ---------------------\n")

        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n\n --------------------- Environment arguments ---------------------\n\n")

        for key, value in env_args_dict.items():
            f.write(f"{key}: {value}\n")




    print(f"Parameters saved to {params_file}")


def run_rollout_loop(actor, episodes_per_mini_batch, memory, env, args, env_args, debug_expert_memory=None, URSim_SKRL_env=None, env_name = None):

    episodes_per_mini_batch = args.total_sample_size
    episodes_count = 0


    while episodes_count < episodes_per_mini_batch: 

        if len(env_args.domain_randomization_list) > 0:
            env_path = np.random.choice(env_args.domain_randomization_list)
            env_args.scene_path = env_path
            
            env = URSim_SKRL_env(args = env_args, name=env_name, render_mode=args.env_render_mode)


        state, _ = env.reset()
        steps = 0

        for _ in range(10000): 

            steps += 1


            with torch.no_grad():
                

                state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)

                mu, std = actor(state_tensor)
            
                action = get_action(mu, std)[0]

                # Deploy with expert action


                action_tensor = torch.tensor(action, dtype=torch.float32).view(1, -1)

                log_pis = log_prob_density(action_tensor, mu, std).item()


            next_state, reward, done, _, _ = env.step(action)

            if done:    
                # print("end reward: ", reward)
                mask = 1
            else:
                mask = 0

            # if steps == 1:
            #     print("states: ", state)
            #     print("actions: ", action)
            #     input()


            memory.append([state_tensor.detach(), action_tensor.detach(), reward, mask, log_pis, next_state])
            


            state = next_state



            if done:
                break
        



        episodes_count += 1




    return memory




def get_rewards(memory):
    '''
    Summary: This function retrieves the average reward of each episode, along with the average reward gained at the end of each episode
    
    ARGS:
        memory [tuple]: This is the iteration object that is passed in

    RETURN:
        reward_average [float]: The average reward of the entire memory
        rewards_average_end [float]: The average reward at the end of each episode
    '''


    reward_epoch_end = 0
    reward_sum = 0
    mem_epoch_count = 0

    for transitions in memory:
        if transitions[3] == True:
            reward_epoch_end += transitions[2] # Add all rewards
            mem_epoch_count += 1

        reward_sum += transitions[2]


    rewards_average_end = reward_epoch_end / mem_epoch_count
    reward_average = reward_sum / len(memory)


    return reward_average, rewards_average_end


def train_IRL(args, env_args, URSim_SKRL_env, Train_at_start=True, save_path=""):
    # Run script in background:
    # $ nohup python -u -m learning.airl_UR.main_airl > IRL_learning_test_1.log 3>&1 &

    Training_model = "PPO"

    if args.discrim_type == "AIRL":
        Train_With_AIRL = True # Set to false to enable GAIL
    elif args.discrim_type == "GAIL":
        Train_With_AIRL = False
    else:
        print(f"ERROR: discrim type {args.discrim_type} is not supported for this training environment")


    if args.model_path == "":
        use_pretrained_RL  = False
    else:
        use_pretrained_RL = True


    if args.disc_model_path == "":
        use_pretrained_IRL  = False
    else:
        use_pretrained_IRL = True



    environment_name = time.strftime("%y-%m-%d_%H-%M-%S") + "URSim_SKRL_env_" + Training_model

    env = URSim_SKRL_env(args = env_args, name=environment_name, render_mode=args.env_render_mode)


    #env.seed(args.seed)
    args.seed = np.random.randint(1e6)
    torch.manual_seed(args.seed)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    # running_state = ZFilter((num_inputs,), clip=5) # Maybe this has some features in the future that is worth exploring


    # Initiating Actor and critic


    actor = copy.deepcopy(Actor(num_inputs, num_actions, args).to(args.task_device))
    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate, weight_decay=args.l2_rate)

    critic = copy.deepcopy(Critic(num_inputs, args).to(args.task_device))
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate, weight_decay=args.l2_rate) 



    # Initializing the AIRL discriminator
    if args.discrim_type == "AIRL":
        AIRL_trainer = AIRL(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device(args.task_device), #  "cuda" if args.cuda else "cpu"
            seed=args.seed,
            units_disc_r=args.hidden_layer_nodes_r,
            units_disc_v=args.hidden_layer_nodes_v,
            lr_disc=args.irl_learning_rate,
            epoch_disc=args.airl_epocs,
            weight_decay_L2 = args.AIRL_L2_weight_decay,
            gamma=args.gamma,
            state_only=False,
            value_shaping=True
            #rollout_length=args.rollout_length
        )
    elif args.discrim_type == "GAIL":
        # Initiating GAIL discriminator
        discrim = Discriminator(num_inputs + num_actions, args)
        discrim_optim = optim.Adam(discrim.parameters(), lr=args.irl_learning_rate)




    if use_pretrained_RL:
        actor_state_dict = torch.load(args.model_path + "/" + args.model_candidate + "_actor.pkl")
        actor.load_state_dict(actor_state_dict)

        actor_optim_state_dict = torch.load(args.model_path + "/" + args.model_candidate + "_actor_optim.pkl")
        actor_optim.load_state_dict(actor_optim_state_dict)


        critic_state_dict = torch.load(args.model_path + "/" + args.model_candidate + "_critic.pkl")
        critic.load_state_dict(critic_state_dict)

        critic_optim_state_dict = torch.load(args.model_path + "/" + args.model_candidate + "_critic_optim.pkl")
        critic_optim.load_state_dict(critic_optim_state_dict)
    else:
        args.model_path = None



    if use_pretrained_IRL:
        # AIRL load pretrained
        if args.discrim_type == "AIRL":
            discrim_state_dict = torch.load(args.disc_model_path + "/" + args.disc_model_candidate + "_AIRL_discrim.pkl")
            AIRL_trainer.disc.load_state_dict(discrim_state_dict)

            discrim_state_dict = torch.load(args.disc_model_path + "/" + args.disc_model_candidate + "_AIRL_discrim_optim.pkl")
            AIRL_trainer.optim_disc.load_state_dict(discrim_state_dict)

            
        elif args.discrim_type == "GAIL":
            discrim_state_dict_GAIL = torch.load(args.disc_model_path + "/" + args.disc_model_candidate + "_GAIL_discrim.pkl")
            discrim.load_state_dict(discrim_state_dict_GAIL)

            discrim_state_dict_GAIL = torch.load(args.disc_model_path + "/" + args.disc_model_candidate + "_GAIL_discrim_optim.pkl")
            discrim_optim.load_state_dict(discrim_state_dict_GAIL)



    for g in actor_optim.param_groups:
        g['lr'] = args.learning_rate
        

    for g in critic_optim.param_groups:
        g['lr'] = args.learning_rate


    if args.discrim_type == "AIRL":
                
        for g in AIRL_trainer.optim_disc.param_groups:
            g['weight_decay'] = args.AIRL_L2_weight_decay
    
    elif args.discrim_type == "GAIL":

        for g in discrim_optim.param_groups:
            g['weight_decay'] = args.AIRL_L2_weight_decay
    


    # Load expert data
    path_expert_data = os.path.abspath(args.expert_address)


    expert_data_file = open(path_expert_data, "rb")
    
    expert_memory = pickle.load(expert_data_file)
    
    # Define the size you want for the subset
    n = args.expert_data_size  # Example size, you can change this to any number you need


    # Initialize lists to hold the first 'n' entries
    states = []
    actions = []
    dones = []
    log_pis = []
    next_states = []

    # Manually iterate and collect the first 'n' entries
    for i, entry in enumerate(expert_memory):
        if i >= n:
            break
        states.append(entry[0])
        actions.append(entry[1])
        dones.append(entry[3])
        log_pis.append(entry[4])
        next_states.append(entry[5])

    # Convert the lists to tensors
    states_exp = torch.tensor(np.array(states)).to(torch.float)
    actions_exp = torch.tensor(np.array(actions)).to(torch.float)
    dones_exp = torch.tensor(dones).to(torch.int8)
    log_pis_exp = torch.tensor(log_pis)
    next_states_exp = torch.tensor(np.array(next_states)).to(torch.float)

    # Create the expert_data list with the first 'n' entries
    expert_data = [
        states_exp,
        actions_exp,
        [],  # Rewards exp that is not used
        dones_exp,
        [],  # New log pis has not been calculated yet
        next_states_exp,
    ]

    if save_path == "":
        path = os.path.abspath("learning/GAIL/Saved_models/")
    else:
        path = save_path + "/Saved_models/"


    name = "GAIL_CLOTH_TASK_" + time.strftime("%y-%m-%d_%H-%M-%S") 


    # Make directories for training
    if args.train_RL == True or args.train_IRL == True or args.record_reward_data :
        
        if save_path == "":
            log_address = "learning/GAIL/logs/" + name
        else:
            log_address = save_path + "/logs/" + name

        writer = SummaryWriter(log_address)


        directory = path+ "/" + name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory) 


        save_parameters(args=args, env_args=env_args, directory=directory)




    episodes = 0
    train_discrim_flag = True
    discriminator_delay = args.discriminator_delay_value
    training_start_time = time.time()
    best_model_score = -np.inf


    episodes_per_mini_batch = args.total_sample_size

    # For restarting training without immediately training the discriminator
    if not(Train_at_start):
        train_discrim_flag = False
        discriminator_delay = 0 


    for iter in range(args.max_iter_num):
        memory = deque()


        actor.eval(), critic.eval()



        memory = run_rollout_loop(actor, episodes_per_mini_batch, memory, env,args,env_args, debug_expert_memory=expert_memory, URSim_SKRL_env=URSim_SKRL_env, env_name=environment_name)

        reward_average, rewards_average_end = get_rewards(memory=memory)



        # Save the best model so far before training the models
        if rewards_average_end > best_model_score: # first begin save models after 15 minutes
            best_model_score = rewards_average_end

            if args.train_RL:
                best_actor = copy.deepcopy(actor.state_dict())
                best_critic = copy.deepcopy(critic.state_dict())
                best_actor_optim = copy.deepcopy(actor_optim.state_dict())
                best_critic_optim = copy.deepcopy(critic_optim.state_dict())

                torch.save(best_actor, directory+'best_actor.pkl')
                torch.save(best_critic, directory+'best_critic.pkl')

                torch.save(best_actor_optim, directory+'best_actor_optim.pkl')
                torch.save(best_critic_optim, directory+'best_critic_optim.pkl')
            


            if args.train_IRL:
                if args.discrim_type == "AIRL":
                    best_airl_dis = copy.deepcopy(AIRL_trainer.disc.state_dict())
                    best_airl_dis_optim = copy.deepcopy(AIRL_trainer.optim_disc.state_dict())


                    torch.save(best_airl_dis, directory+'best_AIRL_discrim.pkl')
                    torch.save(best_airl_dis_optim, directory+'best_AIRL_discrim_optim.pkl')
                    
                elif args.discrim_type == "GAIL":
                    best_gail_dis = copy.deepcopy(discrim.state_dict())
                    best_gail_dis_optim = copy.deepcopy(discrim_optim.state_dict())

                    torch.save(best_gail_dis, directory+'best_GAIL_discrim.pkl')
                    torch.save(best_gail_dis_optim, directory+'best_GAIL_discrim_optim.pkl')

            



        # Save the latest models so far
        if args.train_RL:
            torch.save(actor.state_dict(), directory+'latest_actor.pkl')
            torch.save(critic.state_dict(), directory+'latest_critic.pkl')

            torch.save(actor_optim.state_dict(), directory+'latest_actor_optim.pkl')
            torch.save(critic_optim.state_dict(), directory+'latest_critic_optim.pkl')

        if args.train_IRL:
            if args.discrim_type == "AIRL":
                torch.save(AIRL_trainer.disc.state_dict(), directory+'latest_AIRL_discrim.pkl')
                torch.save(AIRL_trainer.optim_disc.state_dict(), directory+'latest_AIRL_discrim_optim.pkl')

            elif args.discrim_type == "GAIL":
                torch.save(discrim.state_dict(), directory+'latest_GAIL_discrim.pkl')
                torch.save(discrim_optim.state_dict(), directory+'latest_GAIL_discrim_optim.pkl')

                


        # Training the discriminator    
        if train_discrim_flag and args.train_IRL: # disc training is for enabling and disabling the discriminator

            if discriminator_delay == args.discriminator_delay_value:
                if Train_With_AIRL: # Train with AIRL discriminator

                    AIRL_trainer.disc.eval()
                    learner_acc_eval, expert_acc_eval  = AIRL_trainer.evaluate_disc(memory=memory, expert_data=expert_data, actor=actor)
                    

                    AIRL_trainer.disc.train()
                    learner_acc_train, expert_acc_train, loss_pi, loss_exp  = AIRL_trainer.update(memory=memory, expert_data=expert_data, Expert_mini_batch=args.batch_size, actor=actor)
                


                    Policy_change = learner_acc_train - learner_acc_eval

                    Expert_change = expert_acc_train - expert_acc_eval


                    # print(f"AIRL - Expert: %.2f%% | Learner: %.2f%%," % (, learner_acc * 100))

                    print(f"AIRL - Expert: {expert_acc_eval*100:.2f}% | Learner: {learner_acc_eval*100:.2f}% - Expert change: {Expert_change*100:.2f}% | Learner: {Policy_change*100:.2f}%")



                    Disc_acc_pause_criterium = (expert_acc_eval > args.suspend_accu_exp and learner_acc_eval > args.suspend_accu_gen)

                    if Disc_acc_pause_criterium: #  and iter>3
                        train_discrim_flag = False
                        discriminator_delay = 0


                        if args.train_IRL:
                            torch.save(AIRL_trainer.disc.state_dict(), directory+'latest_AIRL_discrim.pkl')
                            torch.save(AIRL_trainer.optim_disc.state_dict(), directory+'latest_AIRL_discrim_optim.pkl')

                else:
                    #GAIL training
                    discrim.eval()
                    learner_acc_eval, expert_acc_eval  = discrim.evaluate_disc(memory=memory, expert_data=expert_data,args=args, actor=actor)
                    

                    discrim.train()
                    learner_acc_train, expert_acc_train = train_discrim(discrim, memory, discrim_optim, expert_data, args)



                    Policy_change = learner_acc_train - learner_acc_eval

                    Expert_change = expert_acc_train - expert_acc_eval


                    # print(f"AIRL - Expert: %.2f%% | Learner: %.2f%%," % (, learner_acc * 100))

                    print(f"GAIL - Expert: {expert_acc_eval*100:.2f}% | Learner: {learner_acc_eval*100:.2f}% - Expert change: {Expert_change*100:.2f}% | Learner: {Policy_change*100:.2f}%")


                    if learner_acc_eval > args.suspend_accu_exp and expert_acc_eval > args.suspend_accu_gen:
                        train_discrim_flag = False
                        discriminator_delay = 0


            if args.train_IRL:
                writer.add_scalar('IRL/expert_accuracy', float(expert_acc_eval), iter)
                writer.add_scalar('IRL/trainer_accuracy', float(learner_acc_eval), iter)



        # When this is commented out, then the IRL reward will never come back after being disabled
        if discriminator_delay < args.discriminator_delay_value:
            discriminator_delay += 1


        if discriminator_delay == args.discriminator_delay_value:
            train_discrim_flag = True



        # get IRL reward if args specifies it.
        if args.use_IRL_reward:
            if Train_With_AIRL:
                memory = get_IRL_reward(memory=memory, AIRL_trainer=AIRL_trainer, normalization_val=None)
            else:
                memory = get_GAIL_reward(memory=memory, discrim=discrim, normalization_val=None)
                

        IRL_reward_average, IRL_rewards_average_end = get_rewards(memory=memory) # IRL should have converted reward to IRL reward if active
        # from tensor to float
        IRL_reward_average = float(IRL_reward_average)
        IRL_rewards_average_end = float(IRL_rewards_average_end)


        
        if args.train_RL or args.record_reward_data:
            writer.add_scalar('environment/reward_env_end', float(rewards_average_end), iter)
            writer.add_scalar('environment/reward_env_avg', float(reward_average), iter)
            writer.add_scalar('environment/reward IRL avg', float(IRL_reward_average), iter)



        # Training the actor and critic
        if args.train_RL:

            actor.train(), critic.train()


            actor_loss, critic_loss, entropy, ratio = train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args, expert_data)

            writer.add_scalar('RL/actor_loss', actor_loss, iter)
            writer.add_scalar('RL/critic_loss', critic_loss, iter)
            writer.add_scalar('RL/PPO_entropy', entropy, iter)
            writer.add_scalar('RL/ratio: ', ratio, iter)

            print(f"{iter}:: {episodes}, env avg: {reward_average:.2f} -- env END: {rewards_average_end:.2f} -- IRL avg: {IRL_reward_average:.2f}" )
            print("\n")


        # print(f"{iter}:: {episodes}, environment reward {scores_env_done_reward_avg} - IRL reward {IRL_rewards_average}")




        if args.train_RL or args.train_IRL:

            #Measure Immitation performance compared to expert demonstration
            with torch.no_grad():

                # Make a sample of the expert data to calculate the log probability of the expert actions
                states_exp_sample, actions_exp_sample, _, _, _ = process_expert_data(expert_data, batch_size=args.batch_size)
                mu_exp_sample, std_exp_sample = actor(states_exp_sample)

                log_pis_exp = log_prob_density(actions_exp_sample, mu_exp_sample, std_exp_sample)
                print("Mean log probability value: ", log_pis_exp.mean())


                mahala_val = mahalanobis_distance(actions_exp_sample, mu_exp_sample, std_exp_sample).mean().item()


                action_policy = get_action(mu_exp_sample, std_exp_sample)


                euclidean_sampled_distance = euclidean_distance(action=actions_exp_sample, action2=action_policy).mean().item()

                euclidean_distance_mean_based = euclidean_distance(action=actions_exp_sample, action2=mu_exp_sample).mean().item()


    

            writer.add_scalar('IRL/log_prob_expert_actions', log_pis_exp.mean(), iter)
            writer.add_scalar('IRL/average_std_actor', std_exp_sample.mean(), iter)
            writer.add_scalar('IRL/mahalanobis_distance', mahala_val, iter)
            writer.add_scalar('IRL/euclidean_sampled_distance', euclidean_sampled_distance, iter)
            writer.add_scalar('IRL/euclidean_mean_distance', euclidean_distance_mean_based, iter)



        time_trained = time.time() - training_start_time
        if time_trained > 3600*42: # 3600*[time in hours]
            print("Training time limit reached")
            print("Current time is: ", time.strftime("%y-%m-%d_%H-%M-%S"))
            break

    

    if env_args.expert_demonstration:
        env.save_expert_data()



    # Ensure the directory exists before saving the .npy file
    directory = path+ "/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist


    if args.train_RL:
        torch.save(actor.state_dict(), directory+'latest_actor.pkl')
        torch.save(critic.state_dict(), directory+'latest_critic.pkl')

        torch.save(actor_optim.state_dict(), directory+'latest_actor_optim.pkl')
        torch.save(critic_optim.state_dict(), directory+'latest_critic_optim.pkl')


        torch.save(best_actor, directory+'best_actor.pkl')
        torch.save(best_critic, directory+'best_critic.pkl')

        torch.save(best_actor_optim, directory+'latest_actor_optim.pkl')
        torch.save(best_critic_optim, directory+'latest_critic_optim.pkl')



    if args.train_IRL:

        if args.discrim_type == "AIRL":
            torch.save(AIRL_trainer.disc.state_dict(), directory+'latest_AIRL_discrim.pkl')
            torch.save(AIRL_trainer.optim_disc.state_dict(), directory+'latest_AIRL_discrim_optim.pkl')



            torch.save(best_airl_dis, directory+'best_AIRL_discrim.pkl')
            torch.save(best_airl_dis_optim, directory+'best_AIRL_discrim_optim.pkl')

        elif args.discrim_type == "GAIL":
            torch.save(discrim.state_dict(), directory+'latest_GAIL_discrim.pkl')
            torch.save(discrim_optim.state_dict(), directory+'latest_GAIL_discrim_optim.pkl')

            torch.save(best_gail_dis, directory+'best_GAIL_discrim.pkl')
            torch.save(best_gail_dis_optim, directory+'best_GAIL_discrim_optim.pkl')


    
