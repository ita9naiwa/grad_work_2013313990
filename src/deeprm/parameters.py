import numpy as np
import math

import numpy as np


class Dist:

    def __init__(self, num_res, max_nw_size, job_len):
        self.num_res = num_res
        self.max_nw_size = max_nw_size
        self.job_len = job_len

        self.job_small_chance = 0.8

        self.job_len_big_lower = job_len * 2 / 3
        self.job_len_big_upper = job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len / 5

        self.dominant_res_lower = max_nw_size / 2
        self.dominant_res_upper = max_nw_size

        self.other_res_lower = 1
        self.other_res_upper = max_nw_size / 5

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension

        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = np.random.randint(1, self.max_nw_size + 1)

        return nw_len, nw_size

    def bi_model_dist(self):

        # -- job length --
        if np.random.rand() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)

        nw_size = np.zeros(self.num_res)

        # -- job resource request --
        dominant_res = np.random.randint(0, self.num_res)
        for i in range(self.num_res):
            if i == dominant_res:
                nw_size[i] = np.random.randint(self.dominant_res_lower,
                                               self.dominant_res_upper + 1)
            else:
                nw_size[i] = np.random.randint(self.other_res_lower,
                                               self.other_res_upper + 1)

        return nw_len, nw_size


def generate_sequence_work(pa, seed=1541):

    np.random.seed(seed)

    simu_len = pa.simu_len * pa.num_ex

    nw_dist = pa.dist.bi_model_dist

    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.zeros((simu_len, pa.num_res), dtype=int)

    for i in range(simu_len):

        if np.random.rand() < pa.new_job_rate:  # a new job comes

            nw_len_seq[i], nw_size_seq[i, :] = nw_dist()

    nw_len_seq = np.reshape(nw_len_seq,
                            [pa.num_ex, pa.simu_len])
    nw_size_seq = np.reshape(nw_size_seq,
                             [pa.num_ex, pa.simu_len, pa.num_res])

    return nw_len_seq, nw_size_seq

class Parameters:
    def __init__(self):

        self.output_filename = 'data/tmp'

        self.num_epochs = 10000         # number of training epochs
        self.simu_len = 50             # length of the busy cycle that repeats itself
        self.num_ex = 100                # number of sequences
        self.output_freq = 50          # interval for output and store parameters
        self.num_seq_per_batch = 20    # number of sequences to compute baseline
        self.episode_max_length = 2000  # enforcing an artificial terminal
        self.num_res = 2               # number of resources in the system
        self.num_nw = 10                # maximum allowed number of work in the queue
        self.time_horizon = 20         # number of time steps in the graph
        self.max_job_len = 15          # maximum duration of new jobs
        self.res_slot = 10             # maximum number of available resource slots
        self.max_job_size = 10         # maximum resource request of new work
        self.backlog_size = 60         # backlog queue size
        self.max_track_since_new = 10  # track how many time steps since last new jobs
        self.job_num_cap = 40          # maximum number of distinct colors in current work graph
        self.new_job_rate = 0.3        # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor

        # distribution for new job arrival
        self.dist = Dist(self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size // float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = int(int(self.res_slot + self.max_job_size * self.num_nw) * self.num_res + self.backlog_width + 1)  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(self.backlog_size // self.time_horizon)
        self.network_input_height = self.time_horizon
        self.network_input_width = int((self.res_slot + self.max_job_size * self.num_nw) * self.num_res + self.backlog_width + 1)

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action
