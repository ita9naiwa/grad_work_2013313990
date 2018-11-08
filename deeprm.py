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
batch_size = 50


with open("configs/deeprm_env.json", 'r') as f:
    import json
    model_config = json.load(f)

model = rl.model(
    sess, state_dim, action_dim, lr,
    network_widths=model_config['network_widths'])

aspace = np.arange(action_dim)

def __main__():
    sess.run(tf.initializers.global_variables())
    for current_job_cnt in range(num_job_sets):
        job_buffer = []
        rewards = []
        slowdowns = []
        seq_no = current_job_cnt

        for current_ep in range(batch_size):
            ep_reward = 0.0

            s = env.reset(seq_no=seq_no)
            s = flatten(s)
            list_s, list_a, list_r, list_y = [], [], [], []
            for ep_len in range(400):
                a = model.get_action_dist(s)
                action = np.random.choice(aspace, p=a)
                s2, r, done, info = env.step(action)
                s2 = flatten(s2)
                list_s.append(s)
                list_a.append(action)
                list_r.append(r)
                ep_reward += r
                if done:
                    disc_vec = np.array(list_r, dtype=float)
                    for i in range(1, len(list_y)):
                        disc_vec[i:] *= discount
                    for i in range(len(list_r)):
                        y_i = np.sum(disc_vec[i:]) / (discount ** i)
                        list_y.append(y_i)
                    job_buffer.append((list_s, list_a, list_r, list_y))
                    rewards.append(ep_reward)
                    slowdowns.append(get_avg_slowdown(info))
                    break

                s = s2

        # compute baseline
        n_samples = len(job_buffer)
        max_job_length = np.max([len(x[0]) for x in job_buffer])
        baseline = np.zeros(shape=(n_samples, max_job_length), dtype=float)
        ep_lengths = []
        for i in range(n_samples):
            current_length = len(job_buffer[i][3])
            ep_lengths.append(current_length)
            baseline[i, :current_length] = job_buffer[i][3]
        baseline = np.mean(baseline, axis=0)

        ss, aa, vv = [], [], []

        for t in range(max_job_length):
            for i in range(n_samples):
                if ep_lengths[i] <= t:
                    continue
                s = job_buffer[i][0][t]
                a = job_buffer[i][1][t]
                y = job_buffer[i][3][t]
                b = baseline[t]
                var = y - b
                ss.append(s)
                aa.append(a)
                vv.append(var)

        loss = model.train(np.array(ss), np.array(aa), np.array(vv))
        print(loss)
        print("[jobset %d] avg episode length : %0.2f" % (current_job_cnt, np.mean(ep_lengths)), "avg episode reward : %0.2f" % np.mean(rewards), "slowdown %0.2f" % np.mean(slowdowns))

if __name__ == "__main__":
    __main__()
