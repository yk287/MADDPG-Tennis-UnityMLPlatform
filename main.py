# main function that sets up environments
# perform training loop

from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
from UnityWrapper import Env

from collections import deque
from util import raw_score_plotter, plotter
import os



def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

model_dir = os.getcwd() + "/model_dir"

os.makedirs(model_dir, exist_ok=True)

seeding()
env = Env()
env_name = 'Tennis'

# number of training episodes.
# change this to higher number to experiment. say 30000.
number_of_episodes = 5000
episode_length = 1000
batchsize = 128
# how many episodes to save policy and gif
save_interval = 1000
t = 0

# amplitude of OU noise
# this slowly decreases to 0
noise = 1
noise_reduction = 0.9999

# how many episodes before update
episode_per_update = 2

# keep 5000 episodes worth of replay
buffer = ReplayBuffer(int(500000), batchsize, seed=0)

# initialize policy and critic
state_size, action_size, num_agents = env.get_info()

torch.set_num_threads(num_agents * 2)

maddpg = MADDPG(state_size, action_size, num_agents, seed=12345, discount_factor=0.95, tau=0.02)

PRINT_EVERY = 5
scores_deque = deque(maxlen=100)
threshold = 0.50

avg_last_100 = []
scores = []


for episode in range(number_of_episodes):

    obs, obs_full, env_info = env.reset()

    episode_reward = 0

    agent0_reward = 0
    agent1_reward = 0

    for agent in maddpg.maddpg_agent:
        agent.noise.reset()

    for i in range(episode_length):

        actions = maddpg.act(torch.tensor(obs, dtype=torch.float), noise=noise)

        actions_for_env = torch.stack(actions).detach().numpy()

        # step forward one frame

        next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)

        # add data to buffer
        buffer.add(obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)

        agent0_reward += rewards[0]
        agent1_reward += rewards[1]

        obs = next_obs
        obs_full = next_obs_full

        noise *= noise_reduction

        if len(buffer) > batchsize and episode % episode_per_update == 0:
            for a_i in range(num_agents):
                samples = buffer.sample(batchsize)
                maddpg.update(samples, a_i)
            maddpg.update_targets()

        if np.any(dones):
            #if any of the agents are done break
            break

    #We take the max rewards between agents

    episode_reward = max(agent0_reward, agent1_reward)
    scores.append(episode_reward)
    scores_deque.append(episode_reward)
    avg_last_100.append(np.mean(scores_deque))

    #scores.append(episode_reward)

    if episode % PRINT_EVERY == 0:
        print('\rEpisode {}\tAverage Score: {:.4f}\tScore: {:.4f}'.format(episode, avg_last_100[-1], episode_reward))

    if avg_last_100[-1] >= threshold:
        # If the mean of last 100 rewards is greater than the threshold, save the model and plot graphs
        # saving model
        save_dict_list = []

        for i in range(num_agents):
            save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
                         'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                         'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
                         'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
            save_dict_list.append(save_dict)

            torch.save(save_dict_list,
                       os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

        break


raw_score_plotter(scores)
plotter(env_name, len(scores), avg_last_100, threshold)
