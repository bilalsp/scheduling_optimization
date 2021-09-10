from __future__ import annotations
import numpy as np

from algorithm.components.schedule import *	
from algorithm.components.utils import *

class Population:
    
    def __init__(self, n_machines, n_jobs, **kwargs):
        """ 
        Args:
            n_jobs (int): number of jobs which need to be assigned to machines.
            n_machines (int): number of machines on which jobs need to be scheduled.
            **kwargs: Arbitrary keyword arguments.
        """
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        # ------------parameters setting --------------
        # strength between each pair
        self.lambda_ = kwargs['lambda_']
        
    def generate(self, size: int) -> List[Schedule]:
        """generate the base population of schedules for genetic algorithm.
        
        Args:
            size: number of schedules/chromosome in the population.

        Returns:
            list of schedule
        """
        init_population = []
        for _ in range(size):
            init_population.append(self._generate_schedule())
        
        return init_population


    def _generate_schedule(self):
        """ Greedy approach to construct the schedule """
        # create machines and jobs sequence
        n_machines, n_jobs = self.n_machines, self.n_jobs
        machines = np.arange(n_machines)
        jobs = np.arange(n_jobs)
        np.random.shuffle(jobs)

        # initialize the M and J binary matrices 
        M = np.zeros((self.n_machines, self.n_jobs), dtype=np.bool8)
        J = np.zeros((self.n_jobs, self.n_jobs), dtype=np.bool8)

        # assign single job to each machine
        M[machines[:min(n_machines,n_jobs)],jobs[:min(n_machines,n_jobs)]] = 1

        # number of assigned jobs
        n_assigned = min(self.n_machines,self.n_jobs)
  
        assert M.sum() == n_assigned

        # assign the remaining jobs to machines based on the pair strength
        for cur_job in jobs[n_assigned:]:
            # strength between cur_job and already assigned jobs
            pair_strength = self.lambda_[jobs[:n_assigned],cur_job] 

            assert pair_strength.shape == (n_assigned,)

            # find job with highest pair strength with current job
            h_job, strength = max(zip(jobs[:n_assigned], pair_strength), key=lambda x: x[1])
            assert M[:,h_job].sum() == 1

            # assign cur_job to the same machine as h_job
            assert np.dot(M[:,h_job], machines) < n_machines 
            assert M[np.dot(M[:,h_job], machines), cur_job] == 0
            M[np.dot(M[:,h_job], machines), cur_job] = 1

            # if h_job already has connected pair
            
            is_exist = np.any(J[h_job] == 1)
            if is_exist:
                q_job = np.argwhere(J[h_job]==1)[0][0]
                assert J[h_job][q_job] == 1
                J[h_job][q_job] = 0 #break the pair between h_job and q_job  
                assert J[cur_job][q_job] == 0
                J[cur_job][q_job] =  1 #create new pair between cur_job and q_job

            # create a pair between h_job and cur_job 
            assert J[h_job,cur_job] == 0
            J[h_job,cur_job] = 1 

            n_assigned += 1
        
      
        assert M.sum() == n_jobs

        return Schedule(M, J)

    def _generate_schedule_v2(self):
        """ Greedy approach to construct the schedule """
        # create machines and jobs sequence
        n_machines, n_jobs = self.n_machines, self.n_jobs
        machines = np.arange(n_machines)
        jobs = np.arange(n_jobs)
        np.random.shuffle(jobs)

        # initialize the M and J binary matrices 
        M = np.zeros((self.n_machines, self.n_jobs), dtype=np.bool8)
        J = np.zeros((self.n_jobs, self.n_jobs), dtype=np.bool8)

        # assign single job to each machine
        M[machines[:min(n_machines,n_jobs)],jobs[:min(n_machines,n_jobs)]] = 1

        # number of assigned jobs
        n_assigned = min(self.n_machines,self.n_jobs)

        last_job_vector = jobs[:min(n_machines,n_jobs)]

        
        for kk, cur_job in enumerate(jobs[n_assigned:]):
            pair_strength = self.lambda_[last_job_vector,cur_job] 

            h_job, strength = max(zip(last_job_vector, pair_strength), key=lambda x: x[1])
            
            assert np.dot(M[:,h_job], machines) < n_machines 
            h_machine = np.dot(M[:,h_job], machines)

            J[h_job, cur_job] = 1
            M[h_machine, cur_job] = 1
            last_job_vector[h_machine] = cur_job

        return Schedule(M, J)  