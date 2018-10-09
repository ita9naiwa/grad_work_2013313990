import numpy as np
import collections
from queue import Queue
class Event:
    def __init__(self, timestep, priority, name, detail):
        self.timestep = timestep
        self.priority = priority
        self.name = name
        self.detail = detail
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "Event occured at %d\n, prioritiy %d\n, name %s\n" % (self.timestep, self.priority, self.name) + str(self.detail) + "\n"

    def __lt__(self, other):
        return (self.timestep * 1000 + self.priority) < (other.timestep * 1000 + other.priority)

    def __cmp__(self, other):
        return (self.timestep * 1000 + self.priority) < (other.timestep * 1000 + other.priority)

class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1


class JobSlot:
    def __init__(self, num_slots):
        self.slot = [None for _ in range(num_slots)]

    def num_allocated_jobs(self):
        return sum([x is not None for x in self.slot])

    def num_empty_slots(self):
        return sum([x is None for x in self.slot])

    def __getitem__(self, key):
        try:
            return self.slot[key]
        except:
            return None

    def release(self, i):
        self.slot[i] = None

    def assign(self, job):
        for i in range(len(self.slot)):
            if self.slot[i] is None:
                self.slot[i] = job
                return i

    def repr(self):
        enter_times = []
        res_vecs = []
        lens = []
        for job in self.slot:
            enter_time = None
            res_vec = None
            job_length = None

            if job is not None:
                enter_time = job.enter_time
                res_vec = job.res_vec
                job_length = job.len
            enter_times.append(enter_time)
            res_vecs.append(res_vec)
            lens.append(job_length)
        return {
            "lengths": lens,
            "resource_vectors": res_vecs,
            "enter_times": enter_times}





class JobBacklog:
    def __init__(self, backlog_size):
        self.backlog = Queue(maxsize=backlog_size)
        self.curr_size = 0
        self.backlog_max_size = backlog_size

    def append_job(self, job):
        self.backlog.put(job)
        self.curr_size += 1

    def get_job(self):

        if self.curr_size == 0:
            raise IndexError

        job = self.backlog.get()
        self.curr_size -= 1

        return job

    def num_jobs(self):
        return self.curr_size

class JobRecord:
    def __init__(self):
        self.record = {}

class Machine:
    def __init__(self, num_resources, time_horizon, resource_slot):
        self.num_resources = num_resources
        self.time_horizon = time_horizon
        self.res_slot = resource_slot
        self.avbl_slot = np.ones((self.time_horizon, self.num_resources)) * self.res_slot
        self.current_timestep = 0
        self.running_jobs = []
        self.graphical_view = [np.ones(shape=(self.time_horizon, self.res_slot[slot_size])) for slot_size in range(self.num_resources)]

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(30), 1, 1 / float(30))
        np.random.shuffle(self.colormap)



    def allocate_job(self, job, curr_time):
        allocated = False
        for t in range(0, self.time_horizon - job.len):
            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec
            if np.all(new_avbl_res[:] >= 0):
                allocated = True
                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len
                self.running_jobs.append(job)
                # update graphical representation

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                new_color = self.colormap[np.random.randint(0, len(self.colormap))]
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time
                for res in range(self.num_resources):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.graphical_view[res][i, :] == 0)[0]
                        #self.graphical_view[res][i, avbl_slot[: job.res_vec[res]]] = new_color


                break
        return allocated

    def time_proceed(self, curr_timestep):
        self.current_timestep = curr_timestep
        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot

        durations = []
        taken_times = []
        for job in self.running_jobs:
            if job.finish_time <= self.current_timestep:
                self.running_jobs.remove(job)
                durations.append(job.len)
                taken_times.append(job.finish_time - job.enter_time)
        return durations, taken_times

    def get_num_unfinished_jobs(self):
        return len(self.running_jobs)

    def get_job_info(self):
        durations, entering_times, start_times = [], [], []
        for job in self.running_jobs:
            durations.append(job.len)
            entering_times.append(job.enter_time)
            start_times.append(job.start_time)

    def repr(self):
        return self.avbl_slot



class ExtraInfo:
    def __init__(self, max_track_since_new=1000):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1
