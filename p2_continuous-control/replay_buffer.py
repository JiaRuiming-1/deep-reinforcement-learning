import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# collect Reacher trajectories for 20 parallelEnv object
def collect_trajectories(envs, agent, max_step=1000, nrand = 1):
    brain_name = envs.brain_names[0]
    brain = envs.brains[brain_name]
    action_size = brain.vector_action_space_size
    # reset envs
    env_info = envs.reset(train_mode=True)[brain_name]
    # initialize returning lists and start the game!
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []
    value_list = []
    terminate_list = []
    # number of parallel instances
    num_agents = len(env_info.agents)

    for _ in range(nrand):
        actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = envs.step(actions)[brain_name]

    last_state = env_info.vector_observations
    for t in range(max_step):
        # get the current state (for each agent)
        '''
        return {'action': actions,       #(n, 20, 4)
                'log_pi_a': log_prob,    #(20, 1, 4)
                'entropy': entropy_loss, #(20, 4)
                'mean': mean,            #(n, 20, 4)
                'v': estimated_values}   #(n, 20, 1)
        '''
        # states = torch.from_numpy(env_info.vector_observations).float().to(device)
        states = env_info.vector_observations
        prediction = agent.act(states)

        # excute action to env
        actions = prediction['action'].cpu().numpy()
        env_info = envs.step(actions)[brain_name]

        log_probs = prediction['log_pi_a'] # tensor
        estimate_value = prediction['v'] # tensor
        rewards = env_info.rewards  # no tensor
        dones = np.array(env_info.local_done)  # no tesnsor

        # store the result
        # tensor
        state_list.append(states)  # (1000,20, 33)
        prob_list.append(log_probs)  # (1000, 20, 1)
        action_list.append(actions)  # (1000, 20, 4)
        value_list.append(estimate_value)  # (1000,20,1)
        # no tensor
        reward_list.append(rewards)  # (1000, 20)
        terminate_list.append(1.0 - dones)  # (1000, 20)

        # stop if any of the trajectories is done
        if dones.any():
            last_state = env_info.vector_observations #(20,33)
            break

    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, action_list,\
            reward_list, value_list, terminate_list, last_state