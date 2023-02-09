import torch
import torch.nn.functional as F
import random
import numpy as np

#from reinforcement_learning.Unity_MLAgent.p2_continuous_control.replay_buffer import ReplayBuffer, OUNoise
#from reinforcement_learning.Unity_MLAgent.p2_continuous_control.model import DDPGActorCriticNet
from replay_buffer import ReplayBuffer, OUNoise
from model import DDPGActorCriticNet


BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
UPDATE_INTERVAL = 4 # how many step update target network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_dim, action_dim, random_seed=1234):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(random_seed)

        self.network = lambda: DDPGActorCriticNet(
        self.state_dim, self.action_dim,
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=LR_ACTOR),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=LR_CRITIC, weight_decay=WEIGHT_DECAY))

        self.network_local = self.network().to(device)
        self.network_target = self.network().to(device)
        self.network_target.load_state_dict(self.network_local.state_dict())

        # Noise process
        self.noise = OUNoise(action_dim, random_seed)
        # Replay memory
        self.memory = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, random_seed)
        # step_num init 0
        self.step_num = 0
        # actor and critic loss record
        self.actor_loss = []
        self.critic_loss = []

    # def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    #     list_of_images = np.asarray(images)  # (2,210,160,3)
    #     if len(list_of_images.shape) < 5:
    #         list_of_images = np.expand_dims(list_of_images, 1)  # (2,1,210,160,3)
    #     # subtract bkg and crop (2,1,80,80) pick pixel interval 2 and mean remove the color dimension
    #     list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color,
    #                                     axis=-1) / 255.
    #     batch_input = np.swapaxes(list_of_images_prepro, 0, 1)  # (1,2,80,80)
    #     return torch.from_numpy(batch_input).float().to(device)

    def step(self, states, actions, rewards, next_states, dones, add_noise=True):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.step_num += 1
        if add_noise == True:
            for i in range(len(states)):
                actions[i] = np.add(actions[i], self.noise.sample())
                self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        else:
            for i in range(len(states)):
                self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > len(states) * BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
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

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.network_target.forward(next_states)
        Q_targets_next = self.network_target.critic(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_targets = Q_targets.detach()

        # Compute critic loss
        Q_expected = self.network_local.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.network_local.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network_local.parameters(), 0.5)
        self.network_local.critic_opt.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.network_local.forward(states)
        actor_loss = -self.network_local.critic(states, actions_pred).mean()
        # Minimize the loss
        self.network_local.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network_local.parameters(), 0.5)
        self.network_local.actor_opt.step()

        # ----------------------- update target networks ----------------------- #
        if self.step_num % UPDATE_INTERVAL == 0:
            # record network loss
            self.actor_loss.append(actor_loss)
            self.critic_loss.append(critic_loss)
            self.soft_update(self.network_target, self.network_local)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def act(self, states):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.network_local.eval()
        with torch.no_grad():
            actions = self.network_local.forward(states).cpu().data.numpy()
        self.network_local.train()
        return np.clip(actions, 0, 1)

    def reset(self):
        self.noise.reset()

