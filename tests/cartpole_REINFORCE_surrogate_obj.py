import test_base
import copy
import src.models.REINFORCE as rl
import src.models.REINFORCE_PPO as reinforce_pro

from src.models.buffer import ReplayBuffer
from src.noise import OrnsteinUhlenbeckActionNoise
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
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
lr = 0.0001
seed = 1234
aspace = np.arange(action_dim, dtype=int)

class traj_worker():
    def __init__(self, model, env):
        self.model = model
        self.env = gym.make('CartPole-v1')

    def get_trajs(self, seq_no, batch_cnt):
        model = self.model
        env = self.env
        baseline = np.zeros(shape=(batch_cnt, 1000))
        ep_buffer = []
        for cnt_now in range(batch_cnt):
            list_s, list_a, list_r, list_y = [], [], [], []
            #s = env.reset(seq_no=seq_no)
            s = env.reset()
            #s = flatten(s)
            for ep_len in range(1000):
                a = model.get_action_dist(s)
                action = np.random.choice(aspace, p=a)
                s2, r, done, info = env.step(action)
                #s2 = flatten(s2)
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
        S, ADV, A = np.empty(shape=(0, state_dim)), np.empty(shape=(0,)), np.empty(shape=(0,), dtype=int)
        for list_s, list_a, list_y in ep_buffer:
            adv = list_y - baseline[:len(list_y)]
            S = np.vstack([S, list_s])
            ADV = np.hstack([ADV, adv])
            A = np.hstack([A, list_a])
        return S, A, ADV

def __main__():
    model = reinforce_pro.model(sess, state_dim, action_dim, lr,
                network_widths=[50, 40], update_step=15)
    sess.run(tf.initializers.global_variables())
    trajWorkers = [traj_worker(model, copy.copy(env)) for _ in range(10)]
    with ThreadPoolExecutor(max_workers=4) as exec:

        for iter in range(1000):

            futures = []
            S, ADV = np.empty(shape=(0, state_dim)), np.empty(shape=(0,))
            A = np.empty(shape=(0,), dtype=int)
            for seq_no in range(10):
                future = exec.submit(trajWorkers[seq_no].get_trajs, seq_no, 10)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                s, a, adv = future.result()
                S = np.vstack([S, s])
                A = np.hstack([A, a])
                ADV = np.hstack([ADV, adv])

            loss = model.train(S, A, ADV)
            s = env.reset()
            ep_lengths = []

            for i in range(10):
                s = env.reset()
                for ep_len in range(episode_max_length):
                    a = model.get_action_dist(s)
                    #action = np.random.choice(aspace, p=a)
                    action = np.argmax(a)
                    s2, r, done, info = env.step(action)
                    if done:
                        break
                    s = s2
                ep_lengths.append(ep_len)
            print(loss)
            print("[iter %d] avg episode length : %d" % (iter, np.mean(ep_lengths)))

if __name__ == "__main__":
    __main__()
