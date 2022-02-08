"""
MADDPG agent.

Using the ddpg single agent for the agents of the environment. However just some function of it are use the rest like the step function and buffer is taken from th mddpg agent.  

"""

import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import DDPG.ddpg_agent as DDPG

GAMMA = 0.99            # discount factor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MADDPG():
    """Meta agent that contains the two DDPG agents and shared replay buffer."""

    def __init__(self, seed=0,
                 n_agents=4,
                 buffer_size=int(1e6),
                 batch_size=256,
                 gamma=0.99,
                 update_every=2,
                 noise_start=1.0,
                 noise_decay=0.9993,
                 action_dimension_size = 3 ):
        """
        Params
        ======
            seed (int): Random seed
            n_agents (int): number of distinct agents
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            noise_start (float): initial noise weighting factor
            noise_decay (float): noise decay rate
            update_every (int): how often to update the network
        """

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.n_agents = n_agents
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0
   
        self.agents = [DDPG(observation=336,  action_size=4),   # Golie Red 
                       DDPG(observation=336,  action_size=4),   # Golie Blue
                       DDPG(observation=336,  action_size=6),   # Striker Red
                       DDPG(observation=336,  action_size=6)]   # Striker Blue

    def step(self, states, actions, rewards, next_states, dones):  
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for agent, state, action, reward, next_state, done in zip(self.agents, states, actions, rewards, next_states, dones)):
            agent.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, get random subset and learn
        for agent in self.agents:
            if len(agent.memory) > agent.batch_size:
                # each agent does it's own sampling from the replay buffer
                experiences = agent.memory.sample()
                agent.learn(experiences, GAMMA)
                
    def act(self, states):
        # pass each agent's state from the environment and calculate it's action
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act( state )
            actions.append(action)
        # flatten vector into 1d
        return np.concatenate(actions).reshape(1, -1)