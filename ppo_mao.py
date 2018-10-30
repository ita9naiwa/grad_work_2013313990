"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]
Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.
The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow 1.8.0
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from src.deeprm.parameters import Parameters
from src.deeprm.environment import Env
from src.utils import *

EP_MAX = 500
EP_LEN = 1000
N_WORKER = 8                # parallel workers
GAMMA = 0.99                # reward discount factor
A_LR = 0.001               # learning rate for actor
C_LR = 0.001               # learning rate for critic
MIN_BATCH_SIZE = 32         # minimum batch size for updating PPO
UPDATE_STEP = 15            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective

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

pa = Parameters()
pa.compute_dependent_parameters()
nw_len_seqs, nw_size_seqs = generate_sequence_work(pa, seed=35)

def get_env_with_rand_seq():
    env = Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs, end='all_done', reward_type='delay')
    #env.seq_no = 0
    env.seq_no = np.random.randint(0, pa.num_ex)
    return env

env = get_env_with_rand_seq()

S_DIM = pa.network_input_width * pa.network_input_height
A_DIM = pa.num_nw + 1


class PPONet(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')

        # critic
        #w_init = tf.random_normal_initializer(0., .1)
        w_init = tf.contrib.layers.xavier_initializer()
        lc = tf.layers.dense(self.tfs, 200, tf.nn.relu, kernel_initializer=w_init, name='lc')
        rc = tf.one_hot(self.tfa, A_DIM, name='rc0')
        rc = tf.layers.dense(rc, 100, tf.nn.relu, kernel_initializer=w_init, name='rc')
        lc = tf.concat([lc, rc], axis=1)
        lc = tf.layers.dense(lc, 100, tf.nn.relu, kernel_initializer=w_init, name='lc_tot')

        self.v = tf.layers.dense(lc, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)   # shape=(None, )
        oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )
        ratio = pi_prob/oldpi_prob
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1].ravel(), data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfa: a, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfa: a, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            w_init = tf.contrib.layers.xavier_initializer()
            l_a = tf.layers.dense(self.tfs, 30, tf.nn.relu, trainable=trainable, kernel_initializer=w_init)
            l_a = tf.layers.dense(l_a, 30, tf.nn.relu, trainable=trainable, kernel_initializer=w_init)
            a_prob = tf.layers.dense(l_a, A_DIM, tf.nn.softmax, trainable=trainable, kernel_initializer=w_init)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                      p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def get_v(self, s, a ):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s, self.tfa: [a]})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = get_env_with_rand_seq()
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():

            self.env.reset()
            self.env.seq_no = np.random.randint(0, pa.num_ex)
            s = self.env.observe()
            s = flatten(s)
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, info = self.env.step(a)
                s_ = flatten(s)
                #if done: r = -10
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    v_s_ = self.ppo.get_v(s_, self.ppo.choose_action(s_))
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, None]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

                    if done:
                        break
                slowdown = get_avg_slowdown(info)
                ep_len = t
            print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,
            'slowdown: %0.2f' % slowdown, "ep_len: %d" % ep_len)
            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1



import pickle
with open('test_env.pickle', 'rb') as f:
    te_env = pickle.load(f)

def test(i):
    slowdowns = []
    for ex in range(100):
        s = te_env.reset()
        s = flatten(s)
        te_env.seq_no = ex
        for ep_len in range(pa.episode_max_length):
            a = GLOBAL_PPO.choose_action(s)
            #action = np.argmax(a)
            s2, r, done, info = te_env.step(a)
            s2 = flatten(s2)
            if done:
                break
            s = s2
        slowdown = get_avg_slowdown(info)
        slowdowns.append(slowdown)

    print("[test res at %d ]\tAvg slowdown of test dataset: %0.2f" % (i, np.mean(slowdowns)))

if __name__ == '__main__':
    GLOBAL_PPO = PPONet()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    test(0)
    for i in range(30):
        UPDATE_EVENT.clear()            # not update now
        ROLLING_EVENT.set()             # start to roll out
        workers = [Worker(wid=i) for i in range(N_WORKER)]

        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
        GLOBAL_RUNNING_R = []
        COORD = tf.train.Coordinator()
        QUEUE = queue.Queue()           # workers putting data in this queue
        threads = []
        for worker in workers:          # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.start()                   # training
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=GLOBAL_PPO.update,))
        threads[-1].start()
        COORD.join(threads)
        # plot reward change and test

        test(1+i)