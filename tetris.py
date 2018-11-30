from tqdm import tqdm
from time import sleep
import json
from time import sleep
import gym
import numpy as np
from src.utils import *


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
max_job_length = 15
input_size = 2 + max_job_length + 1
state_size = np.prod(observation['machine'].shape)
embedding_size = hidden_size = 128

from collections import deque
import random

rews = []
slowdowns = []
fjc = []

usages = np.empty(shape=(0, 3))
for i in range(10):
    s = env.reset(seq_no=30 + i)
    rr = []
    for ep_len in range(config['ep_force_stop']):
        action = get_tetris_action(env)
        usage = (100 - env.machine.avbl_slot[0]) / 100.0
        #print(usage)
        usages = np.vstack([usages, usage])

        s2, r, done, info = env.step(action)
        rr.append(r)
        if done:
            break
        s = s2
    rews.append(np.mean(rr))
    fjc.append(finisihed_job_cnt(info))
    slowdowns.append(get_avg_slowdown(info))


print("[SJF] avg rewards: %0.3f\tavg slowdowns: %0.3f  fjc: %d" %
        (np.mean(rews), np.mean(slowdowns), np.sum(fjc)))
print(np.mean(usages, axis=0))
