import numpy as np
import scipy
def flatten(m):
    state_dim = np.prod(m.shape)
    return np.reshape(m, newshape=(state_dim,))

def get_sjf_action(machine, job_slot):
    sjf_score = 0
    act = len(job_slot.slot)  # if no action available, hold

    for i in range(len(job_slot.slot)):
        new_job = job_slot.slot[i]
        if new_job is not None:  # there is a pending job

            avbl_res = machine.avbl_slot[:new_job.len, :]
            res_left = avbl_res - new_job.res_vec

            if np.all(res_left[:] >= 0):  # enough resource to allocate

                tmp_sjf_score = 1 / float(new_job.len)

                if tmp_sjf_score > sjf_score:
                    sjf_score = tmp_sjf_score
                    act = i
    return act

def get_avg_slowdown(info):
    return np.mean([float(job.finish_time - job.enter_time) / float(job.len) for (job_id, job) in info.record.items() if job.finish_time > 0])

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i+1]

    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

class Reward(object):

    def __init__(self, factor, gamma):
        # Reward parameters
        self.factor = factor
        self.gamma = gamma

    # Set step rewards to total episode reward
    def total(self, ep_batch, tot_reward):
        for step in ep_batch:
            step[2] = tot_reward * self.factor
        return ep_batch

    # Set step rewards to discounted reward
    def discount(self, ep_batch):
        x = ep_batch[:,2]

        discounted = scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]
        discounted *= self.factor

        for i in range(len(discounted)):
            ep_batch[i,2] = discounted[i]

        return ep_batch
