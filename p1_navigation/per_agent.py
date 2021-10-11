import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PERDDQNAgent():
    '''Prioritzed Experience Replay & Double Q-Learning Agent'''

    def __init__(self, state_size, action_size,seed,                
                    e = 1e-1, alpha = 0.5, beta = 1.0 ):
        '''Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        '''    
        self.alpha = alpha
        self.beta = beta
        self.e = e
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBufferPrioritized(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # self.memory.add(state, action, reward, next_state, done)
        # self.memory.update_priorities(indices, priorities)
        
        
        
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample_prioritized(self.e, 
                                                             self.alpha,self.beta)
                self.learn(experiences, GAMMA)
                 
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval() # For Evaluation. Turn Off certain layers, like drop off, normalization....
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())  # copies the tensor to the CPU and numpy changes the tensor in the memorie to an array
        else:
            return random.choice(np.arange(self.action_size))            

    def learn(self, experiences, gamma):
        '''Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''
        states, actions, rewards, next_states, dones, indices, weights = experiences
 
        "*** YOUR CODE HERE ***"
        # Double Q-Learning
        next_actions_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        ## compute and minimize the loss
        Q_targets_next = self.qnetwork_target(next_states).gather(1,next_actions_local).detach()
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        loss = (Q_expected - Q_targets).pow(2)*weights
        # Compute TD-Error for PER
        priorities = loss.detach().numpy()*1.0 
        
        loss = loss.mean()       
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # PER Update TD-Error
        self.memory.update_priorities(indices, priorities)
        
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

                

class ReplayBufferPrioritized():
    '''Fixed-size buffer to store experience tuples with prioritized replay.'''
    # This code part is strongly inspired by this Repository: https://github.com/p-serna/rainbow-dqn/blob/master/dqn_agent.py

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''Initialize a ReplayBuffer with Prioritized replay object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        self.priorities = deque(maxlen=buffer_size) 
        self.max_priority = 1.0
        
    def sample_prioritized(self, e, alpha, beta):
        '''Randomly sample a batch of experiences from memory with priority.'''
        
        # Calculating probabilities for priorities
        tderrors = np.asarray(self.priorities, dtype = np.float32).flatten()
        pis = (np.abs(tderrors)+e)**alpha
        pis = pis/pis.sum() 
        
        # Random indices with probabilities pis
        indices = np.random.choice(len(self.memory), size=self.batch_size, p = pis)
        
        # Selecting episodes from memory
        states = torch.from_numpy(np.vstack([self.memory[idx].state for idx in indices])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[idx].action for idx in indices])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx].reward for idx in indices])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[idx].next_state for idx in indices])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[idx].done for idx in indices]).astype(np.uint8)).float().to(device)
        
        # Importance sampling 
        weights = 1.0/(len(tderrors)*pis[indices])**(beta)
        # Reshape is needed because the flattening at the beginning screw dimensions
        weights = (weights/weights.max()).reshape(weights.shape[0],1)
        weights = torch.from_numpy(weights).float().to(device)
        
        
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory) 
    
    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to memory.'''
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(self.max_priority)   # First income allocated with max Prio so that all incomes are seen at least once
    
    def update_priorities(self, indices, tderrors):
        '''Update priorities array with new tderrors'''
        self.max_priority = np.max([self.max_priority,tderrors.max()])
        for idx, tde in zip(indices,tderrors):
            self.priorities[idx] = tde
     