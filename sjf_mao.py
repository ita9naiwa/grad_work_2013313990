import gym
import numpy as np
from src.deeprm.parameters import Parameters
from src.deeprm.environment import Env

from time import sleep
from tqdm import tqdm
from src.utils import *
num_episodes = 100
max_episode_length = 2000

class SJF_model(object):
    def __init__(self):
        pass

pa = Parameters()
pa.episode_max_length = max_episode_length
pa.simu_len = 200
pa.num_ex = 10
pa.num_nw = 10
pa.num_seq_per_batch = 20
pa.output_freq = 50
pa.new_job_rate = 0.5
pa.compute_dependent_parameters()
env = Env(pa, end='all_done')
model = SJF_model()
slowdowns = []

sd = []

for i_episode in range(10):
    ep_reward = 0.0
    ep_ave_max_q = 0.0
    #print("episode %d" % i_episode)
    s = env.observe()
    env.reset()
    rewards = []
    for curr_ep_len in range(max_episode_length):
        action = get_sjf_action(env.machine, env.job_slot)
        #env.render()
        s2, reward, done, info = env.step(action)
        #print(s.shape)
        ep_reward += reward
        if done:
            break
        s = s2
    rewards.append(ep_reward)
    slowdown = get_avg_slowdown(info)
    #print(slowdown)




    sd.append(get_avg_slowdown(info))
print("avg tot rewards: %0.2f" % np.mean(rewards))
print("avg avg slowdown", np.mean(sd))
