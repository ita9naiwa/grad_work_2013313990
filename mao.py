import json
import copy
import src.models.a2c as a2c
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
from src.utils import *
import argparse
from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

discount = 0.99
env = get_env("configs/env.json")

sds = []
cnt = []
with open("configs/env.json", 'r') as f:
    config = json.load(f)

observation = env.reset(seq_no=0 % 50)
discount = 1.00
max_job_length = config['max_job_length']
capa = config['n_resource_slot_capacities']
n_resources = len(config['n_resource_slot_capacities'])
input_size = max_job_length * n_resources + 1
state_size = np.prod(observation['machine'].shape)
embedding_size = hidden_size = 24
num_sample_batch = 5


def get_trajs(model, seq_no=5):
    ep_buffer = []

    list_s, list_a, list_r, list_y = [], [], [], []
    s = env.reset(seq_no=seq_no)
    for ep_len in range(5):
        action, _ = model.get_action(s)
        s2, r, done, info = env.step(action)
        list_s.append(s)
        list_a.append(action)
        list_r.append(r)
        if done:
            break
        s = s2
    return list_s, list_a, list_r

def __main__():
    #trajWorkers = [traj_worker(model) for _ in range(num_train_seq)]
    model = a2c.model(
        discount, state_size, input_size, embedding_size, hidden_size,
        max_job_length, capa)
    for iter in range(1):
        traj = get_trajs(model)
        model.train(traj)

        #loss = model.train(S, A, ADV)




if __name__ == "__main__":
    __main__()
