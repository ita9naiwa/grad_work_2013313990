import numpy as np

class random_action_model(object):
    def __init__(self, num_slots):
        self.num_slots = num_slots

    def p_given_state(self, observation):
        return np.random.random(self.num_slots + 1)


class Fifo_model(object):
    def __init__(self, action_dim):
        self._last_observaton = None
        self.action_dim = action_dim

    def p_given_state(self, observation):
        self._last_observaton = observation
        pass


class open_shortest_job_first(object):
    def __init__(self, action_dim):
        self._last_observaton = None
        self.action_dim = action_dim

    def p_given_state(self, observation):
        self._last_observaton = observation
        pass
