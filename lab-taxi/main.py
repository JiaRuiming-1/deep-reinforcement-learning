from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
# agent = Agent()
# avg_rewards, best_avg_reward = interact(env, agent)

action_name_set = {0:'down', 1:'up', 2:'right', 3:'left', 4:'pick', 5:'drop'}
passenger_loc_set = {0:'Red', 1:'Green', 2:'Yellow', 3:'Blue', 4:'in taxi'}
des_set = {0:'Red', 1:'Green', 2:'Yellow', 3:'Blue'}
loc_coordinate_set = {0:[0,0], 1:[0,4], 2:[4,0], 3:[4,3], 4:[-1,-1]}

'''
train agent
'''
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=20000)

'''
plot average reward
'''
import matplotlib.pyplot as plt
avg_scores2 = [x for ii,x in enumerate(avg_rewards) if ii>9999]
plt.plot(np.linspace(0,20000,len(avg_rewards),endpoint=False), np.asarray(avg_rewards))
plt.show()
plt.plot(np.linspace(10000,20000,len(avg_scores2),endpoint=False), np.asarray(avg_scores2))
plt.show()

'''
chek optimal perfomance of agent 
'''
import time
env = gym.make("Taxi-v3", render_mode="human")
state = env.reset()[0]
time.sleep(1)
score = 0
qtable = agent.Q
while True:
    env.render()
    time.sleep(0.5)
    action = np.argmax(qtable[state])
    next_state, reward, done, _, info = env.step(action)
    score += reward
    
    state = next_state
    if done:
        print(score)
        break
env.close()


