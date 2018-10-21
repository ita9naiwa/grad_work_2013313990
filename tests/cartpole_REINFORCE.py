import test_base

import src.models.REINFORCE as rl
from src.models.buffer import ReplayBuffer
from src.noise import OrnsteinUhlenbeckActionNoise

import numpy as np
import tensorflow as tf
import gym
import scipy.signal

env = gym.make('CartPole-v1')
sess = tf.Session()
ob = env.reset()

state_dim = 4
action_dim = env.action_space.n
episode_max_length = 500
discount = 0.99
batch_size = 20
num_job_sets = 1000
render = True
buffer_size = 100000
lr = 0.001
seed = 1234

def __main__():
    model = rl.model(sess, state_dim, action_dim, lr, network_widths=[300, 200, 30])
    sess.run(tf.initializers.global_variables())
    sigma = np.diag(0.3 * np.ones(action_dim, dtype=np.float32))

    for current_job_cnt in range(num_job_sets):
        job_buffer = []
        rewards = []
        for current_ep in range(batch_size):
            ep_reward = 0.0
            ep_ave_max_q = 0.0
            s = env.reset()
            list_s = []
            list_a = []
            list_r = []
            list_y = []
            for ep_len in range(episode_max_length):
                a = model.get_action_dist(s)
                if current_job_cnt <= 50:
                    a = np.random.multivariate_normal(a, sigma)
                action = np.argmax(a)
                s2, r, done, info = env.step(action)
                list_s.append(s)
                list_a.append(action)
                list_r.append(r)
                ep_reward += r
                if done:
                    disc_vec = np.array(list_r, dtype=np.float32)
                    for i in range(1, len(list_y)):
                        disc_vec[i:] *= discount
                    for i in range(len(list_r)):
                        y_i = np.sum(disc_vec[i:]) / (discount ** i)
                        list_y.append(y_i)
                    job_buffer.append((list_s, list_a, list_r, list_y))
                    rewards.append(ep_len)
                    break
                s = s2
        # compute baseline
        n_samples = len(job_buffer)
        max_job_length = np.max([len(x[0]) for x in job_buffer])
        baseline = np.zeros(shape=(n_samples, max_job_length), dtype=np.float32)
        ep_lengths = []
        for i in range(n_samples):
            current_length = len(job_buffer[i][3])
            ep_lengths.append(current_length)
            baseline[i, :current_length] = job_buffer[i][3]
        baseline = np.mean(baseline, axis=0)
        ss = []
        aa = []
        vv = []

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
        model.train(np.array(ss), np.array(aa), np.array(vv))

        print("[jobset %d] avg episode length : %0.2f" % (current_job_cnt, np.mean(ep_lengths)), "avg episode reward : %0.2f" % np.mean(rewards))

if __name__ == "__main__":
    __main__()
