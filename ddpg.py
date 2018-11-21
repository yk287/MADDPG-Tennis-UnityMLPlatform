# individual network settings for each actor + critic pair
# see networkforall for details

import networkforall
from utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np

# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents, seed=0, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()


        self.actor = networkforall.Actor(state_size, action_size).to(device)
        self.critic = networkforall.Critic(state_size, action_size, num_agents, seed=seed).to(device)

        self.target_actor = networkforall.Actor(state_size, action_size).to(device)
        self.target_critic = networkforall.Critic(state_size, action_size, num_agents, seed=seed).to(device)

        self.noise = OUNoise(action_size, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, obs, noise=0.0):

        obs = obs.to(device)
        self.actor.eval() #Sets the module in evaluation mode. Seems to get rid of error you get from batchnorm without it
        #convert to cpu() since noise is in cpu()
        action = self.actor(obs).cpu().data.numpy() + noise * self.noise.noise()

        #np.clip to make the action lie between -1 and 1
        return np.clip(action, -1, 1)

    def target_act(self, obs, noise=0.0):

        obs = obs.to(device)
        self.target_actor.eval()
        # convert to cpu() since noise is in cpu()
        action = self.target_actor(obs).cpu().data.numpy() + noise * self.noise.noise()

        # np.clip to make the action lie between -1 and 1
        return np.clip(action, -1, 1)
