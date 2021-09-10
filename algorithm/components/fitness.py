import numpy as np

from algorithm.components.schedule import *
from algorithm.components.objective import *

class Fitness:
    
    def __init__(self, n_machines, n_jobs, **kwargs):
        """ """
        self.objective = Objective( n_machines, n_jobs, **kwargs)

    def assignment(self, population):
        """ """
        total_fitness = 0
        for schedule in population:
            schedule.score = self.compute_fitness(schedule)
            total_fitness += schedule.score

        # normalize
        for schedule in population:
            schedule.fitness = schedule.score/total_fitness

    
    def compute_fitness(self, schedule):
        # number of objectives
        K = 2
        # weights for objective
        u = np.random.uniform(0, 1, size=K)
        w = u.sum()/u
        # fitness score
        obj = np.array(
            [
                self.objective.setup_time(schedule),
                self.objective.shade_consistency(schedule),
            ]
        )
        fitness = np.dot(w,obj)

        return fitness
            


        
        
        