import time
import gym
import numpy as np
import envs.Mao as mao
import models.model
import models.baselines
DEBUG = True

def get_discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    y = np.ones_like(x, dtype=np.float32)
    for i in range(1, len(x)):
        y[i:] *= gamma
    y *= x
    return y


def get_entropy(dist):

    entropy = -(np.sum(dist * np.log(dist + 1e-07)))
    if np.isnan(entropy):
        entropy = 0
    return entropy




def concatenate_all_observations(trajectories, input_height, input_width, n_resource_slot_capacities):
    timesteps_total = 0
    for i in range(len(trajectories)):
        timesteps_total += len(trajectories[i]['rewards'])

    all_observations = np.zeros(
        shape=(timesteps_total, input_height, input_width), dtype=np.float32)
    timesteps = 0
    for traj in trajectories:
        for j in range(len(traj['rewards'])):
            all_observations[timesteps, :, :] = traj['observations'][j]
            timesteps += 1
    return all_observations


def concatenate_all_ob_accross_examples(all_observations):

    num_examples = len(all_observations)
    num_tot_samples = 0
    input_height, input_width = all_observations[0].shape[1], all_observations[0].shape[2]

    for i in range(num_examples):
        num_tot_samples += all_observations[i].shape[0]

    all_ob_concat = np.zeros(
        shape=(num_tot_samples, input_height, input_width), dtype=np.float32)

    j = 0
    k = 0

    for i in range(num_examples):
        j = k
        k += all_observations[i].shape[0]
        all_ob_concat[j:k, :, :] = all_observations[i]
    return all_ob_concat


def process_all_info(trajectories, calc_only_finished=True):
    enter_time = []
    finish_time = []
    job_length = []

    for traj in trajectories:
        #print(traj['info'][-1])
        for key, record in traj['info'][-1].items():
            if (calc_only_finished is True) and (record.finish_time < 0):
                continue
            enter_time.append(record.enter_time)
            finish_time.append(record.finish_time)
            job_length.append(record.len)

    enter_time = np.array(enter_time)
    finish_time = np.array(finish_time)
    job_length = np.array(job_length)

    return enter_time, finish_time, job_length


def get_traj(agent, env, max_episode_length=1024, render=False):
    env.reset()
    observations = []
    actions = []
    rewards = []
    policy_dists = []
    entropies = []
    infos = []

    observation = env.observation()
    for _ in range(max_episode_length):
        policy_dist = agent.p_given_state(observation)
        # whoa, this is nice!
        action = (np.cumsum(policy_dist) > np.random.rand()).argmax()

        observations.append(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        policy_dists.append(policy_dist)
        entropies.append(get_entropy(policy_dist))
        infos.append(info)

        if done is True:
            break
        if render is True:
            env.render()

    return {
        "observations": np.array(observations),
        "rewards": np.array(rewards),
        "actions": np.array(actions),
        "info": np.array(infos),
        "policy_dists": np.array(policy_dists),
        "entropies": np.array(entropies)}




def sample_batches_given_agent(agent=None, num_slots=2, time_horizon=10, n_resource_slot_capacities=(7, 7), backlog_size=50,
    num_epochs=10,num_examples=3, num_sequences_per_batch=3, discount=0.8, max_episode_length=1024,
    render=False):

    #print(np.sum(num_slots * np.array(n_resource_slot_capacities)))
    env = mao.ClusteringEnv(
        num_slots=num_slots,
        time_horizon=time_horizon,
        backlog_size=backlog_size,
        n_resource_slot_capacities=n_resource_slot_capacities)

    input_height = time_horizon
    input_width = (1 + num_slots) * np.sum(n_resource_slot_capacities) + 1 + (backlog_size // time_horizon)

    if agent is None:
        learner = random_action_agent(num_slots)
    else:
        learner = agent

    for _iter in range(num_epochs):
        print("epoch: %d" % _iter)
        train_dict = {
            "observations": [],
            "actions": [],
            "advantages": [],
            "rewards": [],
            "episode_lengths": [],
            "avg_slowdowns": [],
            "entropy": []
        }

        begin_time = time.time()

        for _example in range(num_examples):
            trajectories = []
            for i in range(num_sequences_per_batch):
                traj = get_traj(learner, env, max_episode_length=max_episode_length, render=render)
                trajectories.append(traj)
            discounted_rewards = [get_discount(traj['rewards'], discount) for traj in trajectories]
            max_len = max(len(r) for r in discounted_rewards)
            padded = [np.concatenate([ret, [0 for _ in range(max_len - len(ret))]]) for ret in discounted_rewards]
            baseline = np.mean(padded, axis=0)
            advantages = [ret - baseline[:len(ret)] for ret in discounted_rewards]
            train_dict['observations'].append(concatenate_all_observations(trajectories, input_height, input_width, n_resource_slot_capacities))

            train_dict['actions'].append(
                    np.concatenate([traj['actions'] for traj in trajectories]))

            train_dict['advantages'].append(
                    np.concatenate(advantages))
            train_dict['rewards'].append(
                np.array([get_discount(traj['rewards'], discount)[0] for traj in trajectories]))
            train_dict['episode_lengths'].append([len(traj['rewards']) for traj in trajectories])

            enter_time, finish_time, job_length = process_all_info(trajectories, calc_only_finished=True)
            train_dict['avg_slowdowns'].append((finish_time - enter_time) / job_length)
            #train_dict['entropy'].append(
            #    np.concatenate([traj['entropy'] for traj in trajectories]))

        train_dict['observations'] = \
            concatenate_all_ob_accross_examples(train_dict['observations'])
        train_dict['actions'] = np.concatenate(train_dict['actions'])
        train_dict['advantages'] = np.concatenate(train_dict['advantages'])
        train_dict['rewards'] = np.concatenate(train_dict['rewards'])
        train_dict['episode_lengths'] = np.concatenate(train_dict['episode_lengths'])
        train_dict['avg_slowdowns'] = np.concatenate(train_dict['avg_slowdowns'])
        #train_dict['entropy'] = np.concatenate(train_dict['entropy'])

    sampling_time = time.time() - begin_time
    print("sampling time : %0.2fsec" % sampling_time)


if __name__ == "__main__":
    # parameters
    num_slots = 2
    time_horizon = 10
    n_resource_slot_capacities = (7, 7)
    input_height = time_horizon
    backlog_size = 50
    input_width = (1 + num_slots) * np.sum(n_resource_slot_capacities) + 1 + (backlog_size // time_horizon)
    batch_size = 5
    num_epochs = 50
    num_examples = 5
    discount = 0.8
    max_episode_length = 1024
    #learner = models.model.actor_network(sess=None,
    #            num_slots + 1, input_width, input_height, 0.001, 0.01)
    learner = models.baselines.random_action_model(num_slots)

    sample_batches_given_agent(agent=learner, discount=discount,
        num_slots=num_slots, time_horizon=time_horizon, n_resource_slot_capacities=n_resource_slot_capacities, backlog_size=backlog_size,
        num_epochs=num_epochs, num_examples=num_examples, num_sequences_per_batch=batch_size, max_episode_length=max_episode_length)
