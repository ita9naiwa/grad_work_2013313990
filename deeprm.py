import src.models.REINFORCE as rl
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import src.envs.Mao as mao
from src.utils import *
from time import sleep

env = train_env = get_env("configs/env.json", 1541)
test_env = get_env("configs/env.json", None)

sess = tf.Session()

ob = env.reset()
state_dim = np.prod(ob.shape)
action_dim = env.action_space.n
discount = 0.8
num_episodes = 1000
lr = 0.0001
seed = 1234
num_job_sets = 100
num_iter = 500
batch_size = 20

with open("configs/deeprm_env.json", 'r') as f:
    import json
    model_config = json.load(f)

model = rl.model(
    sess, state_dim, action_dim, lr,
    network_widths=model_config['network_widths'])


aspace = np.arange(action_dim)

def get_trajs(model, env, seq_no, batch_cnt):
    baseline = np.zeros(shape=(batch_cnt, 400))
    ep_buffer = []
    for cnt_now in range(batch_cnt):
        list_s, list_a, list_r, list_y = [], [], [], []
        s = env.reset(seq_no=seq_no)
        s = flatten(s)
        for ep_len in range(400):
            a = model.get_action_dist(s)
            action = np.random.choice(aspace, p=a)
            s2, r, done, info = env.step(action)
            s2 = flatten(s2)
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
        baseline = np.mean(baseline, axis=0)
        S, ADV, A = np.empty(shape=(0, state_dim)), np.empty(shape=(0,)), np.empty(shape=(0,), dtype=int)
        for list_s, list_a, list_y in ep_buffer:
            adv = list_y - baseline[:len(list_y)]
            S = np.vstack([S, list_s])
            ADV = np.hstack([ADV, adv])
            A = np.hstack([A, list_a])
        return S, A, ADV






def __main__():
    sess.run(tf.initializers.global_variables())
    for ite in range(num_iter):
        S, ADV, A = np.empty(shape=(0, state_dim)), np.empty(shape=(0,)), np.empty(shape=(0,), dtype=int)
        for current_job_cnt in range(5):
            seq_no = current_job_cnt
            s, a, adv = get_trajs(model, env, seq_no, 5)
            S = np.vstack([S, s])
            A = np.hstack([A, a])
            ADV = np.hstack([ADV, adv])

        loss = model.train(S, A, ADV)
        print(loss)

if __name__ == "__main__":
    __main__()
