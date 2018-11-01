import numpy as np
import scipy

def calc_entropy(p):
    p = p + 0.00001
    ret = -np.sum(np.log(p) * p)
    if np.isinf(ret):
        return 0
    else:
        return ret



def flatten(m):
    state_dim = np.prod(m.shape)
    return np.reshape(m, newshape=(state_dim,))


def get_avg_slowdown(info):
    sds = []
    for job in info.values():
        if job is None:
            continue
        if job.finish_time < 0:
            continue
        sds.append((job.finish_time - job.enter_time) / job.len)
    return np.mean(sds)

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
