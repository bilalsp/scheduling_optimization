import copy
import numpy as np


class Utils:
    
    @staticmethod
    def next_job(cur_job, J):
        """return the next job"""
        next_job = None
        try:
            next_job = np.argwhere(J[cur_job,:]==1).item()
        except:
            pass
        return next_job
    
    @staticmethod
    def traverse_pairwise(cur_job, J):
        """traverse over job sequence"""
        next_job = Utils.next_job(cur_job, J)
        while next_job:
            yield cur_job, next_job
            cur_job = next_job
            next_job = Utils.next_job(cur_job, J)

    @staticmethod
    def connect_sequence(sequence, schedule, machine):
        """connect job sequence to schedule"""
        J, M = schedule.J, schedule.M
        for job_pair in sequence:
            if job_pair[1] is not None:
                J[job_pair] = 1
                M[machine, job_pair[0]] = 1
                M[machine, job_pair[1]] = 1
            else:
                M[machine, job_pair[0]] = 1

        return schedule

    @staticmethod
    def get_total_strength(schedule, lambda_):
        "total strength of the schedule"
        return np.sum(schedule.J*lambda_)

    