# individual network settings for each actor + critic pair
# see networkforall for details

import PytorchModel
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, in_actor, out_actor, in_critic, seed, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()

        self.actor = PytorchModel.Actor(in_actor, out_actor).to(device)
        self.critic = PytorchModel.Critic(in_actor, out_actor, in_critic, seed=seed).to(device)

        self.target_actor = PytorchModel.Actor(in_actor, out_actor).to(device)
        self.target_critic = PytorchModel.Critic(in_actor, out_actor, in_critic, seed=seed).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)


    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        self.actor.eval()
        action = self.actor(obs).cpu().data.numpy() + noise*self.noise.noise()
        return np.clip(action, -1, 1)

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs).cpu().data.numpy() + noise*self.noise.noise()
        return np.clip(action, -1, 1)

