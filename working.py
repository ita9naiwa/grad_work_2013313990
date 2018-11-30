from tqdm import tqdm
from time import sleep
import json
from time import sleep
import gym
import numpy as np
from src.utils import (
    get_env, get_avg_slowdown, get_sjf_action,
    get_possible_actions, finisihed_job_cnt)


import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from src.models.pointer_network import pointer_networks

with open("configs/env.json", 'r') as f:
    config = json.load(f)

env = get_env("configs/env.json")

sds = []
cnt = []

observation = env.reset(seq_no=0 % 50)


discount = 0.99
max_job_length = config['max_job_length']
n_resources = len(config['n_resource_slot_capacities'])
input_size = max_job_length * n_resources + 1
state_size = np.prod(observation['machine'].shape)
embedding_size = hidden_size = 16
num_sample_batch = 15

from collections import deque
import random

ptr_net = pointer_networks(
    state_size, input_size, embedding_size, hidden_size,
    max_job_length=max_job_length)

adam = optim.Adam(ptr_net.parameters(), lr=1e-3)
for iter in range(1000):
    state = env.reset(seq_no=(iter % 50))
    ob_obs = []
    ac_actions = []
    _ys = []
    baseline = np.zeros(shape=(num_sample_batch, config['ep_force_stop']), dtype=np.float32)
    batch_size = 16
    ent_mean = []
    for sample_batch in tqdm(range(num_sample_batch)):
        obs = []
        actions = []
        rewards = []
        list_y = []
        usages = np.empty(shape=(0, 3))
        state = env.reset(seq_no=(iter % 50))
        for t in range(config['ep_force_stop']):
            #env.render()
            #
            action, action_indices, ent = ptr_net.get_action(state)
            ent_mean.append(ent)
            #print(action)
            job_matrix = []
            job_indices = []
            job_id_to_indice = dict()
            idx = 0
            state2, reward, done, info = env.step(action)
            usage = (100 - env.machine.avbl_slot[0]) / 100.0
            usages = np.vstack([usages, usage])
            obs.append(state)
            actions.append(action_indices)
            #rew_mat[sample_batch, t] = reward
            rewards.append(reward)
            if done:
                break
            state = state2
        #print("usage mean", np.mean(usages, axis=0))
        disc_vec = np.array(rewards, dtype=float)
        for i in range(1, len(rewards)):
            disc_vec[i:] *= discount

        for i in range(len(rewards)):
            y_i = np.sum(disc_vec[i:]) / (discount ** i)
            list_y.append(y_i)
        _ys.append(list_y)
        baseline[sample_batch, :len(list_y)] = list_y
        ob_obs.append(obs)
        ac_actions.append(actions)
        #print("entropy mean:%0.2f" % np.mean(ent_mean))
    baseline = np.mean(baseline, axis=0)


    for jjj in range(1):
        loss = 0
        cnt = 0
        adam.zero_grad()
        for sb in range(num_sample_batch):
            obs = ob_obs[sb]
            actions = ac_actions[sb]
            ys = _ys[sb]
            #print(len(ys))
            for t in range(len(ys)):
                action = actions[t]
                if len(action) == 0:
                    continue
                ob = obs[t]
                adv = ys[t] - baseline[t]
                t_loss = ptr_net.train_single(ob, action, torch.autograd.Variable(torch.as_tensor(adv, dtype=torch.float32)))
                loss += t_loss
                cnt += 1
        loss /= cnt
        loss.backward()
        adam.step()
        print("loss : ", loss.detach().numpy())


    if (iter >= 3) and (iter % 3 == 0):
    #if True:
        usages = np.empty(shape=(0, 3))
        ep_lengths = []
        slowdowns = []
        rews = []
        ents = []
        fjc = []
        for i in range(10):
            s = env.reset(seq_no=30 + i)
            rr = []
            action = []
            for ep_len in range(config['ep_force_stop']):

                action, action_indices, entropy = ptr_net.get_action(s, argmax=True)
                #pa = get_possible_actions(env)
                #sj = get_sjf_action(env)
                #print("possibles:", pa,"sjf:", sj, "chosen:",  action)
                usage = (100 - env.machine.avbl_slot[0]) / 100.0
                usages = np.vstack([usages, usage])
                s2, r, done, info = env.step(action)
                ents.append(entropy)
                rr.append(r)
                if done:
                    break
                s = s2
            fjc.append(finisihed_job_cnt(info))
            rews.append(np.mean(rr))
            slowdowns.append(get_avg_slowdown(info))

        print("[iter %d] avg rewards: %0.3f\tavg slowdowns: %0.3f, etnropies %0.2f, fjc: %d" %
                (iter, np.mean(rews), np.mean(slowdowns), np.mean(ents), np.sum(fjc)))
        print("avg usages", np.mean(usages, axis=0))
