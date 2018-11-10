from tqdm import tqdm
import copy
import src.models.REINFORCE_RNN as rnn_model
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
from src.utils import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', metavar='model', type=str)
parser.add_argument('--filter_output', metavar='filter_output', type=bool)

args = parser.parse_args()

test_env = get_env("configs/env.json", None)
test_env.reset()
sess = tf.Session()
st, seq = test_env.observation_to_rnn_sequence()
action_dim = test_env.action_space.n
episode_max_length = 500
discount = 0.99
batch_size = 20
lr = 0.001
seed = 1234
num_train_seq = 5
aspace = np.arange(action_dim, dtype=int)
machine_dim = len(st)
seq_length = len(seq)
seq_size = len(seq[0])


model = rnn_model.model(sess, machine_dim, seq_length, seq_size, action_dim, learning_rate=lr)

FILTER_OUTPUT = False
if args.filter_output is True:
    FILTER_OUTPUT = True
    print("filter_output is on")


class traj_worker():
    def __init__(self, model):
        self.model = model
        self.env = get_env("configs/env.json", 1541)
    def get_trajs(self, seq_no, batch_size):
        model = self.model
        env = self.env
        baseline = np.zeros(shape=(batch_size, 1000))
        ep_buffer = []
        for cnt_now in tqdm(range(batch_size)):
            list_s, list_machine, list_a, list_r, list_y = [], [], [], [], []
            env.reset(seq_no=seq_no)
            for ep_len in range(1000):
                machine, seq = env.observation_to_rnn_sequence()
                a = model.get_action(machine, seq)
                if FILTER_OUTPUT:
                    possible_actions = get_possible_actions(env)
                    p = np.zeros_like(aspace)
                    p[possible_actions] = 1
                    p[-1] = 1
                    a = a * p
                a /= np.sum(a)
                machine, seq = env.observation_to_rnn_sequence(one_hot=False)
                action = np.random.choice(aspace, p=a)
                s2, r, done, info = env.step(action)
                list_machine.append(machine)
                list_s.append(seq)
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
                    ep_buffer.append([list_s, list_machine, list_a, list_y])
                    break
                s = s2
        baseline = np.mean(baseline, axis=0)
        M = np.empty(shape=(0, machine_dim))
        S = np.empty(shape=(0, seq_length, seq_size))
        ADV = np.empty(shape=(0,), dtype=int)
        A = np.empty(shape=(0, ), dtype=int)

        for list_s, list_machine, list_a, list_y in ep_buffer:
            adv = list_y - baseline[:len(list_y)]
            S = np.vstack([S, list_s])
            M = np.vstack([M, list_machine])
            ADV = np.hstack([ADV, adv])
            A = np.hstack([A, list_a])
        return S, M, A, ADV

def __main__():
    sess.run(tf.initializers.global_variables())
    trajWorkers = [traj_worker(model) for _ in range(num_train_seq)]
    with ThreadPoolExecutor(max_workers=8) as exec:

        for iter in range(1000):
            futures = []
            M = np.empty(shape=(0, machine_dim))
            S = np.empty(shape=(0, seq_length, seq_size))
            ADV = np.empty(shape=(0,), dtype=int)
            A = np.empty(shape=(0, ), dtype=int)
            for seq_no in range(num_train_seq):
                future = exec.submit(trajWorkers[seq_no].get_trajs, seq_no, batch_size)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                s, m, a, adv = future.result()
                M = np.vstack([M, m])
                S = np.vstack([S, s])
                A = np.hstack([A, a])
                ADV = np.hstack([ADV, adv])
            loss = model.train((M, S), A, ADV)
            s = test_env.reset()
            ep_lengths = []
            rewards = []
            slowdowns = []
            print(loss)

            for i in tqdm(range(10)):
                rew = 0
                test_env.reset(seq_no=i)
                s = test_env._observe()
                for ep_len in range(episode_max_length):
                    machine, seq = test_env.observation_to_rnn_sequence()
                    a = model.get_action(machine, seq)
                    if FILTER_OUTPUT:
                        possible_actions = get_possible_actions(test_env)
                        p = np.zeros_like(aspace)
                        p[possible_actions] = 1
                        p[-1] = 1
                        a = a * p
                    a /= np.sum(a)
                    action = np.random.choice(aspace, p=a)
                    #action = np.argmax(a)
                    s2, r, done, info = test_env.step(action)
                    rew += r
                    if done:
                        break
                    s = s2
                ep_lengths.append(ep_len)
                rewards.append(rew)
                slowdowns.append(get_avg_slowdown(info))

            print("[iter %d] avg episode length : %0.1f avg total rewards : %0.2f avg slowdowns: %0.2f" %
                (iter, np.mean(ep_lengths), np.mean(rewards), np.mean(slowdowns)))

if __name__ == "__main__":
    __main__()
