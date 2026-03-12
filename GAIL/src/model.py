import torch
import torch.nn as nn
import numpy as np

from GAIL.network.utils import build_mlp, reparameterize, evaluate_lop_pi, calculate_log_pi

from GAIL.src.memory_process import process_expert_data, process_mem

# custom actor_critic design


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()

        #hidden_size = int(args.hidden_size/8)
        hidden_size = args.hidden_size
        self.fa1 = nn.Linear(num_inputs, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, num_outputs)
        #self.fa4 = nn.Linear(hidden_size, num_outputs)
        # self.log_std_layer = nn.Linear(num_outputs, num_outputs)


        #self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)

        self.log_std = nn.Parameter(torch.zeros(1, num_outputs))
     
        # For actor training
        self.entropy_gain = args.entropy_gain_PPO
        self.critic_gain = args.critic_gain_PPO
    


    def forward(self, x):
        x = torch.tanh(self.fa1(x))
        
        x = torch.tanh(self.fa2(x))
        #x = torch.tanh(self.fa3(x))

        mu = self.fa3(x)

        # output = self.fa3(x)
        # mu, std_val = output.chunk(2, dim=-1)
        #mu = self.fc3(x)

        # std = torch.relu(std_val)
        # std = torch.clamp(std, min=0.01, max=20)

        std = torch.exp(self.log_std).expand_as(mu)  # Ensure std has the same shape as mu

        return mu, std
    

    def generate_log_probs(self, states):

        mu, std = self.forward(states)
        _, log_prob = reparameterize(mu, torch.log(std))
        return log_prob

    def reparameterize(self, mu, log_std):
        return reparameterize(mu, log_std)

    # def evaluate_log_pi(self, states, actions):
    #     mu, std = self.forward(states)
    #     return evaluate_lop_pi(mu, self.logstd, actions) # The tanh, is a quick fix, not sure it works on the long run

    def evaluate_log_pi(self, states, actions):
        mu, std = self.forward(states)
        return evaluate_lop_pi(mu, torch.log(std), actions)



class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()

        hidden_size = args.hidden_size
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)


        #self.fc3 = nn.Linear(hidden_size, hidden_size)
        #self.fc4 = nn.Linear(hidden_size, 1)
        
        # self.fc4.weight.data.mul_(0.1)
        # self.fc4.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # x = torch.tanh(self.fc3(x))

        v = self.fc3(x)
        return v






class Discriminator(nn.Module):
    def __init__(self, num_inputs, args):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob
    

    def get_reward(self, state, action):
        reward_net = self.forward(state, action)

        with torch.no_grad():
            return -torch.log(1 - reward_net)



    def evaluate_disc(self, memory, expert_data, args, actor=None):
        """
        Evaluate the discriminator's performance on the memory and expert data.
        """

        # sample the memory and expert data
        states, actions, _, _, _, _ = process_mem(memory, batch_size=args.batch_size)
        states_exp, actions_exp, _, _, _ = process_expert_data(expert_data, actor, batch_size=args.batch_size)

        # Calculate the discriminator's predictions
        learner_acc = (self(states, actions) < 0.5).float().mean()
        expert_acc = (self(states_exp, actions_exp) > 0.5).float().mean()

        return learner_acc.item(), expert_acc.item()