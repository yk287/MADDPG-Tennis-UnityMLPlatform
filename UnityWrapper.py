from unityagents import UnityEnvironment
import numpy as np

class Env:
    def __init__(self):
        '''
        A Wrapper class used to make main.py more readable
        '''

        self.env = UnityEnvironment(file_name="Tennis.x86_64")

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        states, full_state, env_info = self.reset(True)

        # number of agents
        self.num_agents = len(env_info.agents)

        # size of each action
        self.action_size = self.brain.vector_action_space_size

        # examine the state space
        self.state_size = states.shape[-1]


    def get_info(self):
        '''
        returns information about the environment
        :return: state_size, action_size, num_agents
        '''
        return self.state_size, self.action_size, self.num_agents

    def flatten_states(self, states):
        '''
        flattens Narrays to 1 dimensional arrays
        :param x:
        :return:
        '''
        return np.hstack((states[0], states[1]))

    def reset(self, train_mode=True):
        '''
        resets the environment and returns the initial state, flattened state env_info
        :param train_mode:
        :return:
        '''
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        states = env_info.vector_observations
        full_state = self.flatten_states(states)
        return states, full_state, env_info

    def step(self, actions):
        '''
        Tells the agents to take the given actions and get the values for the next_states, rewards, and done status
        :param actions:
        :return:
        '''
        env_info = self.env.step(actions)[self.brain_name]  # send all actions to the environment
        next_states = env_info.vector_observations
        next_state_full = self.flatten_states(next_states)
        rewards = np.array(env_info.rewards)
        dones = np.array(env_info.local_done)
        return next_states, next_state_full, rewards, dones, env_info

