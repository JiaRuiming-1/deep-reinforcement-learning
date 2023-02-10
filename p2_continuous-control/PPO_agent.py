import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

# collect Reacher trajectories for 20 parallelEnv object
def collect_trajectories(envs, policy, max_step=1000, nrand = 1):
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
    # number of parallel instances
    num_agents = len(env_info.agents)

    for _ in range(nrand):
        actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = envs.step(actions)[brain_name]

    for t in range(max_step):
        # get the current state (for each agent)
        states = torch.from_numpy(env_info.vector_observations).float().to(device)
        prediction = policy(states).cpu().detach().numpy()

        actions = prediction['action']
        log_probs = prediction['log_pi_a']

        env_info = envs.step(actions)[brain_name]
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished

        # store the result
        state_list.append(states)  # [(1000, 33),...]
        reward_list.append(rewards)  # [r,...]
        prob_list.append(log_probs)  # [[n],...]
        action_list.append(actions)  # [(n),...]

        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if dones.any():
            break

    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, \
           action_list, reward_list
