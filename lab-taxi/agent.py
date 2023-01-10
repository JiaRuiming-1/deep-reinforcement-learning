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
        self.passenger_locindex = {[0,0],[0,4],[4,0],[4,3]}

    def select_action(self, state):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = 0.05
        #optimal rule of pick action
        state_decode = list(env.decode(state))
        policy_s = np.ones(self.nA) * epsilon / self.nA
        if state_decode[:2] not in self.passenger_locindex:
            policy_s[-2:] = 0
            policy_s[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA) * 3
        else:
            if loc_coordinate_set[state_decode[2]]==[-1,-1]:
                policy_s[4] = 0
                policy_s[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA) * 2
            else:
                policy_s[5] = 0
                policy_s[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA) * 2

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
        self.Q[state][action] = 0.98 * self.Q[state][action] + 0.03 * (reward + max(self.Q[next_state]))
        # self.Q[state][action] += 1
        return self.Q
