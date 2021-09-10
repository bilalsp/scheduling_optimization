import numpy as np
import operator

from algorithm.components.schedule import *
from algorithm.components.fitness import *

class Selection:
    
    def __init__(self, n_machines, n_jobs, **kwargs):
        """ """
        self.fit = Fitness(n_machines, n_jobs, **kwargs)

    def apply(self, population, method='rws'):
        """ Apply selection method on given population.
        
        Args:
            population (list): schedules of current generation.
            method (str): selection method:
                - rws (Roulette Wheel Selection)
        
        Returns:
            selected schedules (list): most fit schedules of the current population.
        """
        
        if method == "rws":
            return self.roulette_wheel_selection(population)
        else:
            raise NotImplementedError


    def roulette_wheel_selection(self, population):
        """Apply Roulette Wheel Selection method.
        
        Args:
            population (list): schedules of current generation.
        Returns:
            selected schedules (list): most fit schedules of the current population.
        """
    
        pool = []

        while len(pool) < len(population):
            relative_probability = 0.0
            r = np.random.uniform(0, 1)
            for schedule in population:
                relative_probability += schedule.fitness
                if relative_probability >= r:
                    pool.append(schedule)
                    break

        return pool

    def survivor_selection(self, population, size):
        """ """
        score = map(self.fit.compute_fitness, population)
        return list(map(
            operator.itemgetter(0),
            sorted(zip(population,score), key=lambda x: x[1])[:size]
        ))  

        