from tqdm import tqdm
import numpy as np
import time
import tensorflow as tf
from src.deeprm import pg_network

import src.models.ddpg as ddpg
from src.models.buffer import ReplayBuffer
from src.summary import summary
from src.utils import *
from src.deeprm import parameters
from src.deeprm import environment
import src.models.REINFORCE_PPO as reinforce
import queue
from multiprocessing import Manager
import threading
from tqdm import tqdm
sess = tf.Session()
pa = parameters.Parameters()
###############
pa.num_ex = 50
pa.num_seq_per_batch = 20
###############
EP_MAX = 100
EP_LEN = 500
N_WORKER = 8                # parallel workers
MIN_BATCH_SIZE = 8
pa.compute_dependent_parameters()
state_dim = (pa.network_input_width * pa.network_input_height)
action_dim = pa.num_nw + 1
pg_learner = reinforce.model(sess, state_dim, action_dim,
                             learning_rate=0.01, network_widths=[20], update_step=30)
def generate_sequence_work(pa, seed=42):
    np.random.seed(seed)
    simu_len = pa.simu_len * pa.num_ex
    nw_dist = pa.dist.bi_model_dist
    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.zeros((simu_len, pa.num_res), dtype=int)
    for i in range(simu_len):
        if np.random.rand() < pa.new_job_rate:  # a new job comes
            nw_len_seq[i], nw_size_seq[i, :] = nw_dist()
    nw_len_seq = np.reshape(nw_len_seq, [pa.num_ex, pa.simu_len])
    nw_size_seq = np.reshape(nw_size_seq, [pa.num_ex, pa.simu_len, pa.num_res])
    return nw_len_seq, nw_size_seq

nw_len_seqs, nw_size_seqs = generate_sequence_work(pa, seed=42)

def get_env():
    return environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs, end='all_done', reward_type='delay')

QUEUE = queue.Queue()           # workers putting data in this queue

def init_accums(pg_learner):  # in rmsprop
    accums = []
    params = pg_learner.get_params()
    for param in params:
        accum = np.zeros(param.shape, dtype=param.dtype)
        accums.append(accum)
    return accums


def rmsprop_updates_outside(grads, params, accums, stepsize, rho=0.9, epsilon=1e-9):

    assert len(grads) == len(params)
    assert len(grads) == len(accums)
    for dim in range(len(grads)):
        accums[dim] = rho * accums[dim] + (1 - rho) * grads[dim] ** 2
        params[dim] += (stepsize * grads[dim] / np.sqrt(accums[dim] + epsilon))


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def get_entropy(vec):
    entropy = - np.sum(vec * np.log(vec))
    if np.isnan(entropy):
        entropy = 0
    return entropy


def get_traj(agent, env, episode_max_length):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs = []
    acts = []
    rews = []
    entropy = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):
        ob = flatten(ob)
        act_prob = agent.get_one_act_prob(ob)
        csprob_n = np.cumsum(act_prob)
        a = (csprob_n > np.random.rand()).argmax()

        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)
        entropy.append(get_entropy(act_prob))

        if done: break

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'entropy': entropy,
            'info': info
            }


def concatenate_all_ob(trajs, pa):

    timesteps_total = 0
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros(
        (timesteps_total, state_dim))

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob


def concatenate_all_ob_across_examples(all_ob, pa):
    num_ex = len(all_ob)
    total_samp = 0
    for i in range(num_ex):
        total_samp += all_ob[i].shape[0]

    all_ob_contact = np.zeros(
        (total_samp, state_dim))

    total_samp = 0

    for i in range(num_ex):
        prev_samp = total_samp
        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp : total_samp, :] = all_ob[i]
    return all_ob_contact



def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


def update():
    global COORD, GLOBAL_UPDATE_COUNTER, UPDATE_EVENT, ROLLING_EVENT

    while not COORD.should_stop():
        if GLOBAL_EP < EP_MAX:
            UPDATE_EVENT.wait()                     # wait until get batch of data
            print("Training begin!")
            data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
            data = np.vstack(data)
            s, a, r = data[:, :state_dim], data[:, state_dim: state_dim + 1].ravel(), data[:, -1:].ravel()
            pg_learner.train(s, a, r)
            UPDATE_EVENT.clear()        # updating finished
            GLOBAL_UPDATE_COUNTER = 0   # reset counter
            ROLLING_EVENT.set()         # set roll-out available


class traj_worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = get_env()
        self.learner = pg_learner
        self.pa = pa
    def get_traj_worker(self):
        pa = self.pa
        global COORD, GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            self.env.seq_no = np.random.randint(0, self.pa.num_ex)
            trajs = []
            b = False
            for i in range(pa.num_seq_per_batch):
                if not ROLLING_EVENT.is_set():
                    ROLLING_EVENT.wait()
                    b = True
                    break

                traj = get_traj(self.learner, self.env, pa.episode_max_length)
                trajs.append(traj)
            if b is True:
                continue

            all_ob = concatenate_all_ob(trajs, pa)
            # Compute discounted sums of rewards
            rets = [discount(traj["reward"], pa.discount) for traj in trajs]
            maxlen = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)

            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action = np.concatenate([traj["action"] for traj in trajs])
            all_adv = np.concatenate(advs)

            all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
            all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths

            # All Job Stat
            enter_time, finish_time, job_len = process_all_info(trajs)
            finished_idx = (finish_time >= 0)
            all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

            all_entropy = np.concatenate([traj["entropy"] for traj in trajs])
            bs, ba, br = np.vstack(all_ob), np.vstack(all_action), np.vstack(all_adv)
            GLOBAL_UPDATE_COUNTER += 1
            QUEUE.put(np.hstack((bs, ba, br)))
            if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                ROLLING_EVENT.clear()       # stop collecting data
                UPDATE_EVENT.set()          # globalPPO update


            GLOBAL_EP += 1
            if GLOBAL_EP >= EP_MAX:
                COORD.request_stop()

            print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|avg slowdown: %.2f' % np.mean(all_slowdown))


import pickle
with open('test_env.pickle', 'rb') as f:
    te_env = pickle.load(f)

f = open('multi_ref_mao.txt', 'w')
f.close()
def test(i):
    slowdowns = []
    entropies = []
    for ex in range(50):
        s = te_env.reset()
        s = flatten(s)
        te_env.seq_no = ex
        for ep_len in range(pa.episode_max_length):
            a = pg_learner.get_one_act_prob(s)
            entropies.append(calc_entropy(a))
            action = np.random.choice(np.arange(action_dim), p=a.ravel())
            #action = np.argmax(a)
            s2, r, done, info = te_env.step(action)
            s2 = flatten(s2)
            if done:
                break
            s = s2

        slowdown = get_avg_slowdown(info)
        slowdowns.append(slowdown)
        with open('ppo_mao_res.txt', 'a') as f:
            print("[test res at %d ]\tAvg slowdown of test dataset: %0.2f, Avg entropy %0.2f" %
                (i, np.mean(slowdowns), np.mean(entropies)), file=f)
    print("[test res at %d ]\tAvg slowdown of test dataset: %0.2f, Avg entropy %0.2f" % (i, np.mean(slowdowns), np.mean(entropies)))



if __name__ == '__main__':
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    test(0)
    for i in range(1, 100):
        UPDATE_EVENT.clear()            # not update now
        ROLLING_EVENT.set()             # start to roll out
        workers = [traj_worker(wid=i) for i in range(N_WORKER)]

        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
        GLOBAL_RUNNING_R = []
        COORD = tf.train.Coordinator()
        QUEUE = queue.Queue()           # workers putting data in this queue
        threads = []
        for worker in workers:          # worker threads
            t = threading.Thread(target=worker.get_traj_worker, args=())
            t.start()                   # training
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=update,))
        threads[-1].start()
        COORD.join(threads)
        test(i+1)
