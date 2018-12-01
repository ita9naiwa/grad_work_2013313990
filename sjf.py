from tqdm import tqdm
from time import sleep
import json
from time import sleep
import gym
import numpy as np
from src.utils import *

import math
import numpy as np
from src.models.pointer_network import pointer_networks

with open("configs/env.json", 'r') as f:
    config = json.load(f)
env = get_env("configs/env.json")
sds = []
cnt = []

observation = env.reset(seq_no=0 % 50)

rews = []
slowdowns = []
fjc = []

usages = np.empty(shape=(0, 3))
for i in range(30):
    s = env.reset(seq_no=100 + i)
    rr = []
    for ep_len in range(config['ep_force_stop']):
        action = get_sjf_action(env)
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


rews = []
slowdowns = []
fjc = []

usages = np.empty(shape=(0, 3))
for i in range(30):
    s = env.reset(seq_no=100 + i)
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


print("[TETRIS] avg rewards: %0.3f\tavg slowdowns: %0.3f  fjc: %d" %
        (np.mean(rews), np.mean(slowdowns), np.sum(fjc)))
print(np.mean(usages, axis=0))
