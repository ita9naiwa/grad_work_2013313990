from time import sleep
import json
from time import sleep
import gym
import numpy as np
from src.utils import (
    get_env, get_avg_slowdown, get_sjf_action,
    get_possible_actions)


with open("configs/env.json", 'r') as f:
    config = json.load(f)

env = get_env("configs/env.json")

sds = []
for i_episode in range(30):
    observation = env.reset(seq_no=i_episode)
    for t in range(config['ep_force_stop']):
        #env.render()
        p = get_possible_actions(env)
        #print(p)
        action = get_sjf_action(env)
        observation, reward, done, info = env.step(action)
        #print(info)
        #sleep(0.1)
        if done:
            break
        #sleep(0.5)
    slowdown = get_avg_slowdown(info)
    #print(slowdown)
    sds.append(slowdown)

with open("configs/test_env.pkl", "wb") as f:
    import pickle
    pickle.dump(env, f)
print(np.mean(sds))

#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
