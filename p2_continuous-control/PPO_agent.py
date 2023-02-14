import numpy as np
import random
from collections import deque

from model import Policy_Gaussian
# from reinforcement_learning.Unity_MLAgent.p2_continuous_control.model import Policy_Gaussian

import torch
import torch.optim as optim

BATCH_SIZE = 128        # minibatch size
SGD_EPOCH = 5           # number of learning
GAMMA = 0.99            # discount factor
TAU = 0.95              # TD-error estimate weight
BETA = 0.01             # entropy to calculate loss
LR = 1e-4               # common learning rate
LR_ACTOR = 3e-4         # divide learning rate of actor
LR_CRITIC = 1e-3        # divide learning rate of critic
WEIGHT_DECAY = 0        # L2 weight decay
EPSILON = 0.2           # gradient descend limit
EPSILON_DECAY = 0.999   # epsilon decay weight

GAE = True              # if use gae
COMMON_LEARN = False    # if actor and critec use same optimizer or deal with separately

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, env_num, state_size, action_size, random_seed=0):
        """Initialize an Agent object.

        Params
        ======
            envs(object): agent live enviroment
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON
        self.beta = BETA

        # record tkl and loss value by each SDG time
        self.tkl = []
        self.agent_loss_a = []
        self.agent_loss_c = []
        self.common_loss = []

        # number of work
        self.env_num = env_num

        # PPO network
        self.policy = Policy_Gaussian(state_size, action_size, random_seed, 400, 300).to(device)

        # common optim
        if COMMON_LEARN:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        else:
            # divide optim
            actor_opt_fn = lambda params: optim.Adam(params, lr=LR_ACTOR)
            self.optimizer_a = actor_opt_fn(self.policy.actor_params)
            self.optimizer_c = optim.Adam(self.policy.critic_body.parameters(), lr=LR_CRITIC)


    def step(self, old_probs, states, actions, rewards, es_values, masks, last_states):
        """Train Model one episode.
        Params
        ======
            ------------------------ is tensor
            old_probs (1000, 20, 1)
            es_values (1000, 20 ,1)
            ------------------------ no tensor
            states    (1000, 20, 33)
            actions   (1000, 20, 4)
            rewards   (1000, 20)
            masks     (1000, 20)
            last_state (20,33)
            ------------------------------------model return
            'action': actions,       #(n, 20, 4)
            'log_pi_a': log_prob,    #(n, 20, 1)
            'entropy': entropy_loss, #(n, 20, 1)
            'mean': mean,            #(n, 20, 4)
            'v': estimated_values}   #(n, 20, 1)
        """
        assert len(rewards) == len(states)
        i_max = len(states)

        # got last prediction
        prediction_last = self.act(last_states)
        states.append(last_states)
        old_probs.append(prediction_last['log_pi_a'])
        actions.append(prediction_last['action'].cpu().numpy())
        es_values.append(prediction_last['v'])

        # process all trajectories data to tensor
        old_probs = torch.stack(old_probs)
        old_values = torch.stack(es_values)
        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.from_numpy(np.array(actions)).float().to(device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(-1) #(1000, 20)->(1000, 20, 1)
        masks = torch.from_numpy(np.array(masks)).float().to(device).unsqueeze(-1) #(1000, 20)->(1000, 20, 1)

        #reward nomalized
        rewards = rewards - rewards.mean()

        # calculate the advatages
        returns = old_values[-1].detach()
        processed_rollout_adv = [None] * i_max
        processed_rollout_exp = [None] * i_max
        advantages = torch.Tensor(np.zeros((self.env_num, 1))).to(device)
        for i in reversed(range(i_max)):
            returns = rewards[i] + GAMMA * masks[i] * returns
            if GAE:
                expect_reward = rewards[i] + GAMMA * masks[i] * old_values[i+1].detach()
                td_error = expect_reward - old_values[i].detach()
                advantages = advantages * TAU * GAMMA * masks[i] + td_error
            else:
                advantages = returns - old_values[i].detach()

            processed_rollout_adv[i] = advantages
            processed_rollout_exp[i] = returns

        advantage_batch = torch.stack(processed_rollout_adv).squeeze(2) # (1000,20,1) -> (1000,20)
        advantage_batch = (advantage_batch - advantage_batch.mean()) / advantage_batch.std()
        returns_batch = torch.stack(processed_rollout_exp) # (1000,20,1)
        #returns_batch = (returns_batch - returns_batch.mean()) / (returns_batch.std() + 1.e-10)

        for _ in range(SGD_EPOCH):
            # combine all mini batch indices
            sampler = self.random_sample(np.arange(len(rewards)), BATCH_SIZE)
            # total loss
            loss_ac = []
            loss_cr = []
            loss_common = []
            for batch_indices in sampler:
                # calculate loss of mini batch
                policy_loss, v, approx_kl = self.clipped_surrogate(self.policy,
                                                old_probs[batch_indices],
                                                states[batch_indices],
                                                actions[batch_indices],
                                                advantage_batch[batch_indices]
                                                             )
                value_loss = 0.5 * (returns_batch[batch_indices] - v).pow(2).mean()

                # if approx_kl < 1.5 * TKL:
                #     self.beta = self.beta * 2
                # elif approx_kl > TKL/1.5:
                #     self.beta = self.beta / 2
                if COMMON_LEARN:
                    self.optimizer.zero_grad()
                    ## only actor loss
                    # policy_loss.backward()
                    ## actor + critic loss
                    (policy_loss + value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 2)
                    self.optimizer.step()

                    loss_common.append(policy_loss.item())
                else:
                    self.optimizer_a.zero_grad()
                    policy_loss.backward()
                    self.optimizer_a.step()

                    self.optimizer_c.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.critic_body.parameters(), 2)
                    self.optimizer_c.step()

                    loss_ac.append(policy_loss.item())
                    loss_cr.append(value_loss.item())

            if COMMON_LEARN:
                self.common_loss.append(np.mean(loss_common))
            else:
                self.agent_loss_a.append(np.mean(loss_ac))
                self.agent_loss_c.append(np.mean(loss_cr))

        # paramter decay
        self.epsilon *= EPSILON_DECAY
        self.beta *= EPSILON_DECAY


    def clipped_surrogate(self, policy, old_probs, states, actions, advantages_normalized):
        """Loss of Model to Learn and Update.
        Params
        ======
            old_probs (batch, 20, 1)
            states    (batch, 20, 33)
            actions   (batch, 20, 4)
            advantages(batch, 20)
            ------------------------------------model return
            'actions': actions,      #(batch, 20, 4)
            'log_pi_a': log_prob,    #(batch, 20, 1)
            'entropy': entropy_loss, #(batch, 20, 1)
            'mean': mean,            #(batch, 20, 4)
            'v': estimated_values}   #(batch, 20, 1)
        """
        prediction = policy(states, actions)

        ratio = (prediction['log_pi_a'] - old_probs).exp()

        # clipped function
        clip = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

        clipped_surrogate = torch.min(ratio * advantages_normalized[:, :, np.newaxis],
                                      clip * advantages_normalized[:, :, np.newaxis])

        approx_kl = (old_probs - prediction['log_pi_a']).mean()

        return -torch.mean(clipped_surrogate + self.beta * prediction['entropy']), prediction['v'], approx_kl


    def act(self, states):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.policy.eval()
        with torch.no_grad():
            prediction = self.policy(states)
        self.policy.train()
        return prediction

    def random_sample(self, indices, batch_size):
        indices = np.asarray(np.random.permutation(indices))
        batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
        for batch in batches:
            yield batch
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]


