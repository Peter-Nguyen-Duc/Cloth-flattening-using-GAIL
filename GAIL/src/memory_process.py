


import torch
import numpy as np

from collections import deque

from GAIL.utils.utils import get_entropy, log_prob_density

import copy

def process_mem(memory, batch_size=None):
    """
    SUMMARY: Process the memory into variables in order (if batch size is given, then randomize a batch of variables.).
    INPUT:
            memory: The memory in list format for quicker access
            batch_size: The size of the batch to sample from the memory (If None, the entire memory is used in order)
    OUTPUT:
            states: Processed states from the memory
            actions: Processed actions from the memory
            rewards: Processed rewards from the memory
            masks: Processed done flags from the memory
            log_pis: Log probabilities of the actions computed by the actor
            next_states: Processed next states from the memory
    """

    observation_space = len(memory[0][0])

    if batch_size is not None:
        if batch_size > len(memory):
            print("WARNING: Memory is smaller than desired batch size")
            batch_size = len(memory)

        random_indices = np.random.choice(len(memory), size=batch_size, replace=False)

    
        # Load only the randomly selected samples from each array in the dataset
        states = torch.vstack([memory[i][0] for i in random_indices])
        actions = torch.vstack([memory[i][1] for i in random_indices])
        rewards = torch.tensor([memory[i][2] for i in random_indices], dtype=torch.float32) 
        dones = torch.tensor([memory[i][3] for i in random_indices], dtype=torch.float32)
        log_pis = torch.tensor([memory[i][4] for i in random_indices], dtype=torch.float32)
        next_states = torch.tensor(np.array([memory[i][5] for i in random_indices])).view(-1, observation_space)


    else:
        # If batch_size is not provided, use the entire memory in order without randomization
        batch_size = len(memory)


        states = torch.vstack([entry[0] for entry in memory]) # Stack states vertically
        actions = torch.vstack([entry[1] for entry in memory])  # Convert actions to array
        rewards = torch.tensor([entry[2] for entry in memory])  # Convert rewards to array
        dones = torch.tensor([entry[3] for entry in memory])    # Convert masks to array	  
        log_pis = torch.tensor([entry[4] for entry in memory])    # Convert masks to array
        next_states = torch.tensor(np.array([entry[5] for entry in memory])).type(dtype=torch.float32).view(-1, observation_space)    # Convert masks to array





    return states, actions, rewards, dones, log_pis, next_states





def Generate_memory_from_expert_data(expert_data):
    torch.set_num_threads(1)
    
    expert_data = copy.deepcopy(expert_data)

    memory_output = deque()


    for steps in range(len(expert_data)):

        #state = torch.tensor(memory[steps][0], dtype=torch.float32).view(1, -1)
        state = expert_data[steps][0]
        action = torch.tensor(expert_data[steps][1], dtype=torch.float32).view(1, -1)
        reward = expert_data[steps][2]
        done = expert_data[steps][3]
        log_pis = 0
        next_state = expert_data[steps][5]



        if done:    
            mask = 1
        else:
            mask = 0


        memory_output.append([state, action, reward, mask, log_pis, next_state])



    return memory_output




def process_expert_data(expert_data, actor=None, batch_size=64):
    """
    SUMMARY: Process expert data into a format suitable for training.

    INPUT: 
            expert_data: The expert data in list format for quicker access
            actor: The actor model used to compute log probabilities of actions   
            batch_size: The size of the batch to sample from the expert data
    OUTPUT:
            states_exp: Processed states from the expert data
            actions_exp: Processed actions from the expert data
            mask_exp: Processed done flags from the expert data
            log_pis_exp: Log probabilities of the expert actions computed by the actor
            next_states_exp: Processed next states from the expert data 
    """



    # Pull expert data with size of the policy data (The expert size is assumed to be larger than the policy data)
    if len(expert_data[0]) < batch_size:
        print("WARNING: Expert data is smaller than desired batch size")

    random_indices = np.random.choice(len(expert_data[0]), size=batch_size, replace=False)


    # Load only the randomly selected samples from each array in the dataset
    states_exp = torch.stack([expert_data[0][i] for i in random_indices])
    actions_exp = torch.stack([expert_data[1][i] for i in random_indices])
    dones_exp = torch.tensor([expert_data[3][i] for i in random_indices])

    # Ensure the tensors are in the correct format and context
    if actor is not None:
        with torch.no_grad():
            mu, std = actor(states_exp)
            log_pis_exp = log_prob_density(actions_exp, mu, std)

    else:
        log_pis_exp = torch.zeros_like(actions_exp)

    next_states_exp = torch.stack([expert_data[5][i] for i in random_indices])


    mask_exp = []
    for val in dones_exp:
        
        if val.item():   
            mask_exp.append(1)
        else:
            mask_exp.append(0)


    mask_exp = torch.tensor(mask_exp).to(torch.int8)  



    return states_exp, actions_exp, mask_exp, log_pis_exp, next_states_exp

