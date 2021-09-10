import numpy as np

from algorithm.components.schedule import *

class Objective:
    
    def __init__(self, n_machines, n_jobs, **kwargs):
        """ 
        Args:
            n_jobs (int): number of jobs which need to be assigned to machines.
            n_machines (int): number of machines on which jobs need to be scheduled.
            **kwargs: Arbitrary keyword arguments.
        """
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        # setup time between each pair
        self.beta = kwargs['beta']
        # shade difference between each pair
        self.s_diff = kwargs['s_diff']

    def setup_time(self, schedule):
        """ """
        M, J = schedule.M, schedule.J 
        return np.multiply(self.beta, J).sum()

    def shade_consistency(self, schedule):
        """ """
        M, J = schedule.M, schedule.J 
        return np.multiply(self.s_diff, J).sum()
