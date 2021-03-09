from __future__ import annotations
import copy 

from algorithm.components.schedule import *
from algorithm.components.utils import *

class CrossOver:
    
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
        
    def apply(self, s1: Schedule, s2: Schedule) -> List[Schedule]:
        partition_1, partition_2 = self._get_partition(s1)
        # jobs in partition 1 and 2
        job_sequence1 = np.argwhere(partition_1.M)[:,1] 
        job_sequence2 = np.argwhere(partition_2.M)[:,1]
        
        subschedule1 = self._get_isubschedule(s1, job_sequence1)
        subschedule2 = self._get_isubschedule(s1, job_sequence2)
        
        offspring1 = self.merge(partition_1, subschedule2)
        offspring2 = self.merge(partition_2, subschedule1)
        
        return offspring1, offspring2
    
    
    def _get_partition(self, schedule):
        n_machines, n_jobs = self.n_machines, self.n_jobs
        M, J = schedule.M, schedule.J
        machines = np.arange(n_machines, dtype=np.int8)

        #create empty partitions
        M_ = np.zeros((n_machines, n_jobs), dtype=np.bool8)
        J_ = np.zeros((n_jobs, n_jobs), dtype=np.bool8)
        partition_1 = Schedule(M_, J_)
        partition_2 = copy.deepcopy(partition_1)
        
        for j in machines:
            first_job = np.argmax(M[j,:]*(1 - np.dot(M[j,:], J)))
            
            try:
                sequence = list(Utils.traverse_pairwise(first_job, J))
                # weakest job pair
                weakest_pair = min(sequence, 
                                   key = lambda job_pair: self.lambda_[job_pair])

                flag = True
                for job_pair in sequence:
                    if list(job_pair) == list(weakest_pair):
                        partition_1.M[j, job_pair[0]] = 1
                        flag = False
                    else:
                        if flag:
                            # Partition-1
                            partition_1.M[j, job_pair[0]] = 1
                            partition_1.J[job_pair] = 1
                        else:
                            # Partition-2
                            partition_2.M[j, job_pair[0]] = 1
                            partition_2.J[job_pair] = 1

                last_pair =  sequence[-1]
                partition_2.M[j, last_pair[1]]= 1

            except Exception as e:
                partition_1.M[j, first_job] = 1

        return partition_1, partition_2
    
    def _get_isubschedule(self, schedule, job_sequence): 
        """Return induced subschedule"""
        induced_subschedule = []
        M, J = schedule.M, schedule.J
        
        n_machines, n_jobs = self.n_machines, self.n_jobs
        M, J = schedule.M, schedule.J
        machines = np.arange(n_machines, dtype=np.int8)
        
        for j in range(n_machines):
            first_job = np.argmax(M[j,:]*(1 - np.dot(M[j,:], J)))
            cur_job = first_job
            
            temp = []
            while cur_job is not None:
                next_job = Utils.next_job(cur_job, J)
                if next_job is not None:
                    if cur_job in job_sequence and next_job in job_sequence:
                        temp.append((cur_job, next_job))
                    elif cur_job in job_sequence:
                        temp.append((cur_job,None))
                        induced_subschedule.append(temp)
                        temp = []
                else:
                    if cur_job in job_sequence:
                        temp.append((cur_job,None))
                        induced_subschedule.append(temp)
                        temp = []
                cur_job = next_job
                
        return induced_subschedule
    
     
    def merge(self, partition, subschedule):
        """merge parition and subschdule to generate the offspring"""
        offspring = copy.deepcopy(partition)
        
        if subschedule is None:
            return offspring
        
        M, J = offspring.M, offspring.J
        # assign the longest component to empty machines
        for empty_machine in np.where(M.sum(axis=1)==0)[0]:
            longest_comp = max(subschedule, key=lambda x: len(x))
            subschedule.pop(subschedule.index(longest_comp))
            offspring = Utils.connect_sequence(longest_comp, offspring, empty_machine)
                   
        # connect the component of subshedule with offspring
        machines = np.arange(self.n_machines, dtype=np.int8)
        for sequence in subschedule:
            M, J = offspring.M, offspring.J
            #last job of each machines
            last_jobs = []
            for j in machines:
                last_job = np.argmax(M[j,:]*(1 - np.dot(M[j,:], J.T)))
                last_jobs.append((j,last_job))
            
            # find best matching 
            first_job = sequence[0][0]
            match = max(last_jobs, key=lambda x: self.lambda_[x[1],first_job])
            
            # connect best match with sequence
            J[match[1],first_job] = 1
            machine = match[0]            
            offspring = Utils.connect_sequence(sequence, offspring, machine)
            
        return offspring    
