from __future__ import annotations

import copy
import numpy as np

from algorithm.components.schedule import *

class Mutation:
    
    def __init__(self, n_machines, n_jobs, **kwargs):
        """ 
        Args:
            n_jobs (int): number of jobs which need to be assigned to machines.
            n_machines (int): number of machines on which jobs need to be scheduled.
            **kwargs: Arbitrary keyword arguments.
        """
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        # strength between each pair
        self.lambda_ = kwargs['lambda_']
      
    def apply(self, parent_pool, prob_mut) -> List[Schedule]:
        mut_pool = []
        for schedule in parent_pool:
            r = np.random.random()
            if r <= prob_mut:
                mut_pool.append(self.apply_mutation(schedule))
        return mut_pool


    def apply_mutation(self, S: Schedule, type='within'):
        """apply mutation of schedule"""

        if type == 'within':
            S, mut_count = self._apply_within(copy.deepcopy(S))
        
        if type == 'across':
            raise NotImplementedError
        
        return S
        
    def _apply_within(self, S: Schedule) -> Schedule:
        """apply mutation on each job sequence of schedule"""
        
        mut_count = 0
        machines = np.arange(self.n_machines, dtype=np.int8)
        M, J = S.M, S.J
        
        for j in machines:
            # size of scheduling job sequence of machine, j
            sigma_j = M[j,:].sum()
            
            if sigma_j < 2:
                continue
            
            # first and last job schedule on machine-j
            first_job = np.argmax(M[j,:]*(1 - np.dot(M[j,:], J)))
            last_job = np.argmax(M[j,:]*(1 - np.dot(M[j,:], J.T)))
            
            # find job-pair with weakest strength
            try:
                i1, i2 = self._find_pair(J, first_job)
            except Exception:
                continue
            
            if self.lambda_[i1, i2] < self.lambda_[last_job, first_job]:
                # #break the pair between jobs i1 and i2
                J[i1, i2] = 0
                # create a pair between last_job and first_job 
                J[last_job, first_job] = 1
                
                mut_count += 1
                
        return Schedule(M, J), mut_count
        
        
    def _find_pair(self, J, first_job):
        """find job-pair with weakest strength"""
        
        def next(cur_job):
            """return the next job"""
            next_job = None
            try:
                next_job = np.argwhere(J[cur_job,:]==1).item()
            except:
                pass
            return next_job
        
        job_seq = []
        cur_job = first_job
        while next(cur_job):
            next_job = next(cur_job)  
            job_seq.append((cur_job, next_job))
            cur_job = next_job  

        i1, i2 = min(job_seq, key = lambda job_pair: self.lambda_[job_pair])

        return i1, i2