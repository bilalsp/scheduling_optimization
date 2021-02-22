import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class Population:
    
    def __init__(self, n_jobs, n_machines, **kwargs):
        """ 
        Args:
            n_jobs (int): number of jobs which need to be assigned to machines.
            n_machines (int): number of machines on which jobs need to be scheduled.
            **kwargs: Arbitrary keyword arguments.
        """
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        # ------------parameters setting --------------
        np.random.seed(kwargs.get('seed',2021))
        # setup time required between jobs 
        self.beta = np.random.uniform(10.0, 50.0, size=(self.n_jobs, self.n_jobs))
        # shade difference
        s = np.random.uniform(10.0, 50.0, size=self.n_jobs)
        self.s_diff = np.abs(s - s.reshape(-1,1))
        # compute strength between each pair
        self.lambda_ = np.reciprocal(stats.hmean(np.array([self.beta,self.s_diff])))
                
    def generate(self, size: int) -> list:
        """generate the base population of schedules for genetic algorithm.
        
        Args:
            size: number of schedules/chromosome in the population.

        Returns:
            list of M and J binary matrix (i.e, representative of schedule).
        """
        init_population = []
        for _ in range(size):
            init_population.append(self._generate_schedule())
        
        return init_population
        
    def _generate_schedule(self):
        """ Greedy approach to construct the schedule """
        # create machines and jobs sequence
        machines = np.arange(self.n_machines, dtype=np.int8)
        jobs = np.arange(self.n_jobs, dtype=np.int8)
        np.random.shuffle(jobs)

        # initialize the M and J binary matrices 
        M = np.zeros((self.n_machines, self.n_jobs), dtype=np.bool8)
        J = np.zeros((self.n_jobs, self.n_jobs), dtype=np.bool8)

        # assign single job to each machine
        M[machines,jobs[:min(self.n_machines,self.n_jobs)]] = 1

        # number of assigned jobs
        n_assigned = min(self.n_machines,self.n_jobs)

        # assign the remaining jobs to machines based on the pair strength
        for cur_job in jobs[n_assigned:]:
            # strength between cur_job and already assigned jobs
            pair_strength = self.lambda_[jobs[:n_assigned],cur_job] 

            # find job with highest pair strength with current job
            h_job, strength = max(zip(jobs[:n_assigned], pair_strength), key=lambda x: x[1])

            # assign cur_job to the same machine as h_job
            M[np.dot(M[:,h_job], machines), cur_job] = 1

            # if h_job already has connected pair
            is_exist = np.any(J[h_job] == 1)
            if is_exist:
                q_job = np.argwhere(J[h_job]==1)[0][0]
                J[h_job][q_job] = 0 #break the pair between h_job and q_job  
                J[cur_job][q_job] =  1 #create new pair between cur_job and q_job

            # create a pair between h_job and cur_job 
            J[h_job,cur_job] = 1 
            
            # increment the number of jobs assigned to the machines.
            n_assigned += 1
        
        return M, J