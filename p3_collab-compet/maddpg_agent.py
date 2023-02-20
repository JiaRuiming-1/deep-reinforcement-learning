import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 10        # learning timestep interval
LEARN_NUM = 10          # learning times
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 2.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_DECAY = 0.996

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.eps = EPS_START
        self.timestep = 0

        # Actors Network
        self.actors_local = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actors_target = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]

        # Critic Network (w/ Target Network)
        self.critics_local = [Critic(num_agents, state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.critics_target = [Critic(num_agents, state_size, action_size, random_seed).to(device) for _ in range(num_agents)]

        # Init Optimizers
        self.actor_optimizers = [optim.Adam(actor_local.parameters(), lr=LR_ACTOR)
                                 for actor_local in self.actors_local]
        self.critic_optimizers = [optim.Adam(critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
                                  for critic_local in self.critics_local]

        # Noise process
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

        # record loss
        self. critic_loss_list = []
        self. actor_loss_list = []

    def act(self, states, add_noise=True):
        """Returns actions for both agents as per current policy, given their respective states."""
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        for i, state in enumerate(states):
            self.actors_local[i].eval()
            with torch.no_grad():
                actions[i, :] = self.actors_local[i](state).cpu().data.numpy()
            self.actors_local[i].train()
        # add noise to actions
        if add_noise:
            actions += self.eps * self.noise.sample()
        # actions = np.clip(actions, -1, 1)
        return actions

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.timestep += 1
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory and at learning interval settings
        if len(self.memory) > BATCH_SIZE and self.timestep % LEARN_EVERY == 0:
            self.reset(EPS_DECAY)
            for  _  in range(LEARN_NUM):
                experiences = self.memory.sample()
                critic_loss, actor_loss = self.learn(experiences, GAMMA, _)
                self.critic_loss_list.append(critic_loss)
                self.actor_loss_list.append(actor_loss)


    def learn(self, experiences, gamma, learn_times):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get predicted next-state actions and Q values from target models
        next_actions = torch.zeros((BATCH_SIZE, self.num_agents * self.action_size),
                                   dtype=torch.float32, device=device)
        for i, next_state in enumerate(next_states):
            next_actions_tmp = [self.actors_target[agent_num](next_state[agent_num])
                                for agent_num in range(self.num_agents)]
            next_actions[i, :] = torch.cat(next_actions_tmp)

        # review all gents message
        next_states_flatten = next_states.view(BATCH_SIZE, -1)
        states_flatten = states.view(BATCH_SIZE, -1)
        actions_flatten = actions.view(BATCH_SIZE, -1)
        rewards = rewards - rewards.mean(dim=0)

        # Compute each agent Q targets
        critic_loss_all = []
        actor_loss_all = []
        for agent_num in range(self.num_agents):
            # ---------------------------- update critic ---------------------------- #
            Q_targets_next = self.critics_target[agent_num](next_states_flatten, next_actions)
            Q_targets = rewards[:, agent_num].unsqueeze(1) + \
                        (gamma * Q_targets_next * (1-dones[:, agent_num].unsqueeze(1))) #(batch, 1)

            # Compute each critic loss
            Q_expected = self.critics_local[agent_num](states_flatten, actions_flatten)
            critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
            critic_loss_all.append(critic_loss.item())

            # Minimize the each critic loss
            self.critic_optimizers[agent_num].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_local[agent_num].parameters(), 1)
            self.critic_optimizers[agent_num].step()

            # ---------------------------- update actor ---------------------------- #
            # Compute each actor loss
            pred_actions = actions.clone()
            pred_actions[:,agent_num,:] = self.actors_local[agent_num](states[:,agent_num,:])

            pred_actions_flatten = pred_actions.view(BATCH_SIZE, -1)
            actor_loss = -self.critics_local[agent_num](states_flatten, pred_actions_flatten).mean()
            actor_loss_all.append(actor_loss.item())

            # Minimize the each actor loss
            self.actor_optimizers[agent_num].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_num].step()

        # ----------------------- update target networks ----------------------- #

        for agent_num in range(self.num_agents):
            self.soft_update(self.critics_local[agent_num], self.critics_target[agent_num], TAU)
            self.soft_update(self.actors_local[agent_num], self.actors_target[agent_num], TAU)

        return np.mean(critic_loss_all), np.mean(actor_loss_all)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def reset(self, esp_decay):
        self.noise.reset()
        self.eps *= esp_decay
        self.eps = max(0.1, self.eps)

    def save_model(self):
        torch.save({
            'actor': [actor.state_dict() for actor in self.actors_local],
            'critic': [critic.state_dict() for critic in self.critics_local]
        }, 'assets/checkpoint.pth')

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process.
        Params
        ======
            mu (float)    : long-running mean
            theta (float) : speed of mean reversion
            sigma (float) : volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        #agent_nums = [e.agent_num for e in experiences if e is not None] np.vstack

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)