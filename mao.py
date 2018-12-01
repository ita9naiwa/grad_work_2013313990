import copy
import src.models.REINFORCE as rl
import src.models.REINFORCE_PPO as rl_ppo
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
from src.utils import *
import argparse

discount = 0.99

def get_trajs(self, seq_no, batch_size):
    model = self.model
    env = self.env
    baseline = np.zeros(shape=(batch_size, 1000))
    ep_buffer = []
    for cnt_now in range(batch_size):
        list_s, list_a, list_r, list_y = [], [], [], []
        s = env.reset(seq_no=seq_no)
        for ep_len in range(1000):
            action = model.get_action(s)
            s2, r, done, info = env.step(action)
            list_s.append(s)
            list_a.append(action)
            list_r.append(r)
            if done:
                disc_vec = np.array(list_r, dtype=float)
                for i in range(1, len(list_y)):
                    disc_vec[i:] *= discount
                for i in range(len(list_r)):
                    y_i = np.sum(disc_vec[i:]) / (discount ** i)
                    list_y.append(y_i)
                baseline[cnt_now, :len(list_y)] = list_y
                ep_buffer.append([list_s, list_a, list_y])
                break
            s = s2
    baseline = np.mean(baseline, axis=0)
    S, ADV, A = [], np.empty(shape=(0,)), np.empty(shape=(0,), dtype=int)
    for list_s, list_a, list_y in ep_buffer:
        adv = list_y - baseline[:len(list_y)]
        S += list_s
        ADV = np.hstack([ADV, adv])
        A = np.hstack([A, list_a])
    return S, A, ADV

def __main__():
    #trajWorkers = [traj_worker(model) for _ in range(num_train_seq)]
    for iter in range(1000):
        S, A, ADV = get_trajs(model, env)
        #loss = model.train(S, A, ADV)




if __name__ == "__main__":
    __main__()
