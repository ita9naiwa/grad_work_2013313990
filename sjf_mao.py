import pickle
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
pa.episode_max_length=200
pa.compute_dependent_parameters()
"""
try:
    te_nw_len_seqs, te_nw_size_seqs = generate_sequence_work(pa, seed=45)
    with open('test_env.pickle', 'rb') as p:
        te_env = pickle.load(f)

except:
"""
def generate_sequence_work(pa, seed=42):
    np.random.seed(seed)
    simu_len = pa.simu_len * pa.num_ex
    nw_dist = pa.dist.bi_model_dist
    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.zeros((simu_len, pa.num_res), dtype=int)
    for i in range(simu_len):
        if np.random.rand() < pa.new_job_rate:  # a new job comes
            nw_len_seq[i], nw_size_seq[i, :] = nw_dist()
    nw_len_seq = np.reshape(nw_len_seq, [pa.num_ex, pa.simu_len])
    nw_size_seq = np.reshape(nw_size_seq, [pa.num_ex, pa.simu_len, pa.num_res])
    return nw_len_seq, nw_size_seq

te_nw_len_seqs, te_nw_size_seqs = generate_sequence_work(pa, seed=42)
print('write new test data')
te_env = Env(pa, nw_len_seqs=te_nw_len_seqs, nw_size_seqs=te_nw_size_seqs, end='all_done')

with open('test_env.pickle', 'wb') as p:
    pickle.dump(te_env, p)
with open('test_env.pickle', 'rb') as f:
    te_env = pickle.load(f)

model = SJF_model()
slowdowns = []
rewards = []
sd = []

for i_episode in range(pa.num_ex):
    te_env.num_ex = i_episode
    ep_reward = 0.0
    ep_ave_max_q = 0.0
    #print("episode %d" % i_episode)
    te_env.reset()
    s = te_env.observe()
    for curr_ep_len in range(max_episode_length):
        action = get_sjf_action(te_env.machine, te_env.job_slot)
        #env.render()
        s2, reward, done, info = te_env.step(action)
        #print(s.shape)
        ep_reward += reward
        if done:
            break
        s = s2
    rewards.append(ep_reward)
    slowdown = get_avg_slowdown(info)
    sd.append(slowdown)

print("avg tot rewards: %0.2f" % np.mean(rewards))
print("avg avg slowdown", np.mean(sd))
