import numpy as np
import heapq
from .variables import Event

class dist:
    def __init__(self,
        episode_max_size=1024,
        new_job_rate=0.3,
        num_resources=2,
        max_resource_usage=10,
        job_len=10,
        p_small_job=0.8):

        self.episode_max_size = episode_max_size
        self.new_job_rate = new_job_rate
        self.num_resources = num_resources
        self.max_resource_usage = max_resource_usage
        self.job_max = job_len
        self.p_small_job = p_small_job

        self.job_max_large = (10, 15 + 1)
        self.job_max_small = (1, 3 + 1)

        self.dominant_resource_usage = (max_resource_usage //2, max_resource_usage + 1)
        self.secondary_resource_usage = (1, max_resource_usage // 5 + 1)

    def bi_model_dist(self):

        if np.random.rand() <= self.p_small_job:
            # small job
            _job_length = self.job_max_small
        else:
            # large job
            _job_length = self.job_max_large

        work_length = np.random.randint(_job_length[0], _job_length[1])
        dominant_resource = np.random.randint(0, self.num_resources)

        work_size = np.zeros(self.num_resources, dtype=np.int32)
        for i in range(self.num_resources):
            if i == dominant_resource:
                work_size[i] = np.random.randint(
                    self.dominant_resource_usage[0], self.dominant_resource_usage[1], dtype=np.int32)
            else:
                work_size[i] = np.random.randint(
                    self.secondary_resource_usage[0], self.secondary_resource_usage[1], dtype=np.int32)
        return work_length, work_size

    def no_jobs(self):
        return 0, np.zeros(self.num_resources)

    def generate_work_sequence(self):
        ret = [None for _ in range(self.episode_max_size)]
        job_lengths = []
        job_sizes = []

        for i in range(self.episode_max_size):
            if np.random.rand() < self.new_job_rate:
                # new job arrives!
                length, size = self.bi_model_dist()
                job_lengths.append(length)
                job_sizes.append(size)
                ret[i] = {"duration": length, "size": size}
        return ret

    def generate_work_sequences_heap(self):
        pass

pass