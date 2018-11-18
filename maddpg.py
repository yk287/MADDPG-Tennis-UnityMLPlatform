# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, state_size, action_size, num_agents, seed, discount_factor=0.99, tau=0.001):
        super(MADDPG, self).__init__()
        self.num_agents = num_agents
        self.maddpg_agent = [DDPGAgent(state_size, action_size,  num_agents, seed),
                             DDPGAgent(state_size, action_size,  num_agents, seed)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = []

        for i in range(self.num_agents):
            obs = obs_all_agents[i, :].view(1, -1)
            action = self.maddpg_agent[i].act(obs, noise).squeeze()
            actions.append(action)

        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """

        next_actions = []

        for i in range(self.num_agents):

            next_obs = obs_all_agents[:,i,:]
            next_action = self.maddpg_agent[i].target_act(next_obs, noise).to(device)
            next_actions.append(next_action)

        return next_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        obs, obs_full, action, reward, next_obs, next_obs_full, done =  samples

        batch_size = obs_full.shape[0]

        target_actions = self.target_act(next_obs.view([batch_size, self.num_agents, -1]))

        #get the agent that corresponds to the agent number
        agent = self.maddpg_agent[agent_number]

        agent.critic_optimizer.zero_grad()

        target_actions = torch.cat(target_actions, dim=1)

        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full, target_actions)

        y = reward[:, agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[:, agent_number].view(-1, 1))

        q = agent.critic(obs_full, action.view(batch_size, -1))

        #we can use huber_loss instead but I'm using MSE
        #huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = F.mse_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [self.maddpg_agent[i].actor(obs.view([batch_size, self.num_agents, -1])[:, i, :]) if i == agent_number else self.maddpg_agent[i].actor(obs.view([batch_size, self.num_agents, -1])[:, i, :]).detach() for i in range(self.num_agents)]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic


        # get the policy gradient
        actor_loss = -agent.critic(obs_full, q_input).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()

        #soft update the models
        self.update_targets()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




