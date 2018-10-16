import sys
sys.path.append('../')
import gym
import numpy as np
import envs.Mao as mao
from time import sleep
import models.model
import models.baselines

import episodes_sampler
DEBUG = True


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

    episodes_sampler.sample_batches_given_agent(agent=learner, discount=discount,
        num_slots=num_slots, time_horizon=time_horizon, n_resource_slot_capacities=n_resource_slot_capacities, backlog_size=backlog_size,
        num_epochs=num_epochs, num_examples=num_examples, num_sequences_per_batch=batch_size, max_episode_length=max_episode_length)
