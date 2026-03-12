import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
import copy 
from GAIL.network.utils import build_mlp
import time
import math
# Calculating the log_probability (log_pis), of the actions taken using the GAIL repo implementation
from GAIL.utils.utils import get_entropy, log_prob_density
from GAIL.src.memory_process import process_mem, process_expert_data
from torchviz import make_dot



def entropy_regularizer(prob):
    return - (prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8)).mean()


def check_nan_in_model(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            return True
        else:
            return False
        
# @torch.no_grad()
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.uniform_(m.weight, a=-1, b=0)
#         m.bias.data.fill_(0.01)


class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape,action_shape,state_only, gamma,
                 value_shaping = True,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.LeakyReLU(inplace=True),
                 hidden_activation_v=nn.LeakyReLU(inplace=True)):
        

        super().__init__()
        # This variable determines if the reward function is state only, or state action based
        self.state_only = state_only

        if state_only:
            # Build linear reward function
            self.g = build_mlp(
                input_dim=state_shape[0],
                output_dim=1,
                hidden_units=hidden_units_r,
                hidden_activation=None,
            )
        else:
            self.g = build_mlp(
                input_dim=state_shape[0] + action_shape[0],
                output_dim=1,
                hidden_units=hidden_units_r,
                hidden_activation=hidden_activation_r,
            )

        self.value_shaping = value_shaping

        if self.value_shaping: 
            self.h = build_mlp(
                input_dim=state_shape[0],
                output_dim=1,
                hidden_units=hidden_units_v,
                hidden_activation=hidden_activation_v,
            )

        self.gamma = gamma




    def f(self, states, dones, next_states, actions):

        if self.state_only:
            rs = torch.flatten(self.g(next_states))
        else:
            rs = torch.flatten(self.g(torch.concatenate((states, actions), 1)))

        # rs = torch.clamp(rs, -10, 10)  # optional, keeps reward bounded


        # With reward shaping a value function into the reward function

        if self.value_shaping:
            dones = dones.cpu().to(dtype=torch.int)
            vs = torch.flatten(self.h(states))
            next_vs = torch.flatten(self.h(next_states))
            
            val = rs + self.gamma*(1 - dones)*next_vs - vs  

        else:
            
            val = rs

        


        return val
    

    def forward(self, states, dones, next_states, actions):
        # get the shapred reward output

        r_theta = self.f(states, dones, next_states, actions) 
        return r_theta
    


    def get_logits(self, states, dones, log_pis, next_states, actions):
        # Discriminator's output is sigmoid(f - log_pi).

        return self.forward(states, dones, next_states, actions) - log_pis
    






class AIRL:

    def __init__(self, state_shape, action_shape, device, seed,
                 state_only=True,
                 value_shaping=True,
                 gamma=0.995,
                 batch_size=64, lr_disc=3e-3,
                 units_disc_r=(512, 512), units_disc_v=(512, 512), epoch_disc=15, weight_decay_L2= 1e-3):
        
        print("AIRL device: ", device)


        if state_only:
            print("AIRL: State only reward function")
        else:
            print("AIRL: State action reward function")
            
        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            state_only=state_only,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True),
            value_shaping=value_shaping
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc, weight_decay=weight_decay_L2)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc    
        self.learning_steps = 0
        self.weight_decay_L2 = weight_decay_L2




    def update(self, memory, expert_data, Expert_mini_batch, actor):
        """
        SUMMARY: this method updates the discriminator
        INPUT: memory: the memory buffer of the agent
               expert_data: the expert data in list format for quicker access
        OUTPUT: acc_pi: the accuracy of the discriminator on the agent data
                acc_exp: the accuracy of the discriminator on the expert data
        """
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = process_mem(memory)

        
        n = len(dones) # It is assumed that the policy and the expert data have the same size at this point

        # states_exp =    torch.tensor(np.array([entry[0] for entry in expert_memory])).cuda().to(torch.float32)  # Stack states vertically
        # dones_exp =     torch.tensor(np.array([entry[3] for entry in expert_memory])).cuda().to(torch.int8)  # Convert actions to array
        # log_pis_exp =   torch.tensor(np.array([entry[4].detach().cpu().numpy() for entry in expert_memory])).cuda()  # Convert actions to array
        # next_states_exp = torch.tensor(np.array([entry[5] for entry in expert_memory])).cuda().to(torch.float32)  # Convert actions to array
        arr = np.arange(n)


        loss_pi_list = []
        loss_exp_list = []

        for epoch_counter in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # randomize the batch of expert datasets
            states_exp, actions_exp, mask_exp, log_pis_exp, next_states_exp = process_expert_data(expert_data, actor, batch_size=len(dones))

            np.random.shuffle(arr)



            for i in range(n // Expert_mini_batch): 
                batch_index = arr[Expert_mini_batch * i : Expert_mini_batch * (i + 1)]
                batch_index = torch.LongTensor(batch_index)


                states_minibatch = states[batch_index]
                actions_minibatch = actions[batch_index]
                dones_minibatch = dones[batch_index]
                log_pis_minibatch = log_pis[batch_index]
                next_states_minibatch = next_states[batch_index]



                states_exp_mini_batch = states_exp[batch_index]
                actions_exp_mini_batch = actions_exp[batch_index] 
                mask_exp_mini_batch = mask_exp[batch_index]
                new_log_pis_mini_batch = log_pis_exp[batch_index]
                next_states_exp_mini_batch = next_states_exp[batch_index]


                


                acc_pi, acc_exp, loss_pi, loss_expert =     self.update_disc(
                                            states_minibatch,       actions_minibatch,     dones_minibatch,         log_pis_minibatch,          next_states_minibatch, 
                                            states_exp_mini_batch, actions_exp_mini_batch, mask_exp_mini_batch,     new_log_pis_mini_batch,     next_states_exp_mini_batch
                                        )
                
                loss_pi_list.append(loss_pi)
                loss_exp_list.append(loss_expert)


        acc_pi = torch.mean(torch.tensor(acc_pi))
        acc_exp = torch.mean(torch.tensor(acc_exp))


        loss_pi_list = torch.tensor(loss_pi_list)
        loss_exp_list = torch.tensor(loss_exp_list)

        loss_pi_mean = torch.sum(loss_pi_list)
        loss_exp_mean = torch.sum(loss_exp_list)

        return acc_pi, acc_exp, loss_pi_mean, loss_exp_mean
    


    def evaluate_disc(self, memory, expert_data, actor):

        states, actions, rewards, dones, log_pis, next_states = process_mem(memory)

        states_exp = expert_data[0]
        actions_exp = expert_data[1]
        dones_exp = expert_data[3]
        # log_pis_exp = torch.zeros((len(states_exp))) # Zero pad the expert log probability

        with torch.no_grad():
            mu, std = actor(states_exp)
            log_pis_exp = log_prob_density(actions_exp, mu, std)


        next_states_exp = expert_data[5]



        mask_exp = []
        for val in dones_exp:
            
            if val.item():   
                mask_exp.append(1)
            else:
                mask_exp.append(0)

        mask_exp = torch.tensor(mask_exp).to(torch.int8)  


        acc_pi, acc_exp =     self.get_accuracies(
                                        states,actions, dones, log_pis, next_states, states_exp,actions_exp, 
                                        mask_exp, log_pis_exp, next_states_exp
                                        )
        


        
        return acc_pi, acc_exp



    def get_accuracies(self, states, actions, dones, log_pis, next_states,
                        states_exp, actions_exp, dones_exp, log_pis_exp, next_states_exp):
        
        
        with torch.no_grad():
            # D given trainer
            logit_pi = self.disc.get_logits(states, dones, log_pis, next_states, actions)


            # D given expert
            logit_exp = self.disc.get_logits(states_exp, dones_exp, log_pis_exp, next_states_exp, actions_exp)

            # Discriminator's accuracies.

            threshold_limit = 0.0

            acc_pi = (logit_pi < threshold_limit).float().mean().item()
            acc_exp = (logit_exp > threshold_limit).float().mean().item()



        return acc_pi, acc_exp


    def get_reward(self, states, actions, dones, next_states):
        
        rs_val_with_reward_shaping = self.disc(states, dones, next_states, actions)


        for val in rs_val_with_reward_shaping.cpu().numpy():
            if math.isnan(val):
                print("D: ", rs_val_with_reward_shaping)
                print("states: ", states)
                print("actions: ", actions)
                print("dones: ", dones)
                print("next_states: ", next_states)
                input()

        return rs_val_with_reward_shaping.cpu().numpy()


    

    def update_disc(self, states, actions, dones, log_pis, next_states,
                           states_exp, actions_exp, dones_exp, log_pis_exp,next_states_exp):
        # Output of discriminator is (-inf, inf), not [0, 1].
        

        logit_pi = self.disc.get_logits(states, dones, log_pis, next_states, actions)

        # D given trainer



        # D given expert
        logit_exp = self.disc.get_logits(states_exp, dones_exp, log_pis_exp, next_states_exp, actions_exp)
        

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].


        # loss_pi = -F.logsigmoid(-logits_pi).mean() # Simplified to -log(1 - D)

        
        # loss_exp = -F.logsigmoid(logits_exp).mean() # Simplified to -log(D)

        # loss_exp = -torch.log(D_expert + 1e-8).mean()
        # loss_pi =  -torch.log(1-D_policy + 1e-8).mean()

        # Anonymous tip for nummerical stability calculation of loss
        loss_exp = F.softplus(-logit_exp).mean()
        loss_pi = F.softplus(logit_pi).mean()





        loss_disc =  loss_exp + loss_pi 




        self.optim_disc.zero_grad()

        loss_disc.backward()

        self.optim_disc.step()

        if check_nan_in_model(self.disc):
            print(" ---------------------- discrimiantor model is NAN ---------------------- ")
            print("expert_loss: ", loss_exp)
            print("policy_loss: ", loss_pi)
            input()


        # Discriminator's accuracies.
        with torch.no_grad():
            threshold_limit = 0.0

            acc_pi = (logit_pi < threshold_limit).float().mean().item()
            acc_exp = (logit_exp > threshold_limit).float().mean().item()
        


        return acc_pi, acc_exp, loss_pi, loss_exp