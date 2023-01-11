import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.spisode_step = 1
        self.alpha = 0.04
        #{0:'down', 1:'up', 2:'right', 3:'left', 4:'pick', 5:'drop'}

    def select_action(self, state, action_mask):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = max(round(1/self.spisode_step, 4), 0.003)
        self.spisode_step += 0.002
        max_index = np.argmax(self.Q[state])
        
        if action_mask[max_index] == 0:
            policy_s = action_mask.copy() / np.count_nonzero(action_mask)              
        else:
            policy_s = action_mask.copy() * epsilon / np.count_nonzero(action_mask) 
            policy_s[max_index] = 1 - epsilon + epsilon / np.count_nonzero(action_mask) 
        return np.random.choice(self.nA, p=policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if action == 4 and reward == -1:
            reward = 0
        self.Q[state][action] = (1-self.alpha) * self.Q[state][action] + self.alpha * (reward + max(self.Q[next_state]))
            
        # self.Q[state][action] += 1
        return self.Q
