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


state = env.reset()
# state[0] = [(taxi_row*5 + taxi_col)*5 + pass_loc] * 4 + dest_idx
print(state)
print(f'current location:{list(env.decode(state[0]))[:2]}')
print(f'passenger {loc_coordinate_set[list(env.decode(state[0]))[2]]}->{loc_coordinate_set[list(env.decode(state[0]))[3]]}')


# next_state, reward, done, _ 
action = env.action_space.sample()
next_state, reward, done, _, info = env.step(4)
print((next_state, reward, done, _, info))
print(action, action_name_set[action])
print(f'current location:{list(env.decode(next_state))[:2]}')
print(f'passenger {loc_coordinate_set[list(env.decode(next_state))[2]]}->{loc_coordinate_set[list(env.decode(next_state))[3]]}')
