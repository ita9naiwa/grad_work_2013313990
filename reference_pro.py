from tqdm import tqdm
import numpy as np
import time
import tensorflow as tf
import src.models.REINFORCE_PPO as reinforce
from src.models.buffer import ReplayBuffer
from src.summary import summary
from src.utils import *
from src.deeprm import parameters
from src.deeprm import environment

sess = tf.Session()
pa = parameters.Parameters()
###############
pa.num_ex = 50
pa.num_seq_per_batch = 20
###############
pa.compute_dependent_parameters()
state_dim = (pa.network_input_width * pa.network_input_height)
action_dim = pa.num_nw + 1
act_list = np.arange(action_dim)

pg_learner = reinforce.model(sess, state_dim, action_dim,
                             learning_rate=0.0001, network_widths=[20])

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma * out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def get_entropy(vec):
    entropy = - np.sum(vec * np.log(vec))
    if np.isnan(entropy):
        entropy = 0
    return entropy

def get_traj(model, env, episode_max_length, render=False):
    """
    Run model-environment loop for one whole episode (trajectory)
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
        ob = np.reshape(ob, (state_dim,))
        act_prob = model.get_one_act_prob(ob)
        a = np.random.choice(act_list, p=act_prob)
        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)
        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)
        entropy.append(get_entropy(act_prob))
        if done:
            break
        if render:
            env.render()

    ob = np.reshape(ob, (state_dim,))
    obs.append(ob)

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




def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    env = environment.Env(pa, render=render, repre=repre, end=end, reward_type='delay')
    # ----------------------------
    print("Preparing for data...")
    # ----------------------------
    timer_start = time.time()
    for iteration in range(pa.num_epochs):
        all_ob = []
        all_action = []
        all_adv = []
        all_eprews = []
        all_eplens = []
        all_slowdown = []
        all_entropy = []
        all_ys = []
        # go through all examples
        for ex in tqdm(range(pa.num_ex)):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajs = []
            for i in range(pa.num_seq_per_batch):
                traj = get_traj(pg_learner, env, pa.episode_max_length)
                ep_len = len(traj['reward'])

                rews = traj['reward']
                obs = traj['ob']
                acts = traj['action']
                #for i in range(ep_len):
                #    replay_buffer.add(obs[i], acts[i], rews[i], i == (ep_len - 1), obs[i + 1])
                trajs.append(traj)

            # roll to next example
            env.seq_no = (env.seq_no + 1) % env.pa.num_ex

            all_ob.append(concatenate_all_ob(trajs, pa))
            # Compute discounted sums of rewards
            rets = [discount(traj["reward"], pa.discount) for traj in trajs]
            maxlen = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]
            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)
            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action.append(np.concatenate([traj["action"] for traj in trajs]))
            all_adv.append(np.concatenate(advs))
            all_eprews.append(np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs]))  # episode total rewards
            all_eplens.append(np.array([len(traj["reward"]) for traj in trajs]))  # episode lengths


            # All Job Stat
            enter_time, finish_time, job_len = process_all_info(trajs)
            finished_idx = (finish_time >= 0)
            all_slowdown.append((finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx])
            # Action prob entropy
            all_entropy.append(np.concatenate([traj["entropy"]]))

        all_ob = concatenate_all_ob_across_examples(all_ob, pa)
        all_action = np.concatenate(all_action)
        all_adv = np.concatenate(all_adv)

        #batch = replay_buffer.sample_batch(replay_buffer.count)
        #replay_buffer.clear()
        loss = pg_learner.train(all_ob, all_action, all_adv)

        eprews = np.concatenate(all_eprews)  # episode total rewards
        eplens = np.concatenate(all_eplens)  # episode lengths

        all_slowdown = np.concatenate(all_slowdown)

        all_entropy = np.concatenate(all_entropy)

        timer_end = time.time()

        print("-----------------")
        print("Iteration: \t %i" % iteration)
        print("NumTrajs: \t %i" % len(eprews))
        print("NumTimesteps: \t %i" % np.sum(eplens))
        #print("Loss:     \t %s" % loss)
        #print("MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews]))
        print("MeanRew: \t %s +- %s" % (eprews.mean(), eprews.std()))
        print("MeanSlowdown: \t %s" % np.mean(all_slowdown))
        print("MeanLen: \t %s +- %s" % (eplens.mean(), eplens.std()))
        print("MeanEntropy \t %s" % (np.mean(all_entropy)))
        print("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
        print("-----------------")

        timer_start = time.time()


def main():

    pg_resume = None
    render = False
    launch(pa, pg_resume, render, repre='image', end='all_done')

if __name__ == '__main__':
    main()
re