from abc import ABC, abstractmethod
import random
import bisect
import logging

from .solution_database import Solution, SolutionDatabase

logger = logging.getLogger(__name__)

class Sampler(ABC):
    def __init__(self, solution_db: SolutionDatabase):
        self.solution_db = solution_db

    @abstractmethod
    def sample(self) -> Solution:
        pass


class PowerLawSampler(Sampler):
    def __init__(self, solution_db: SolutionDatabase, alpha: float=0.0):
        super().__init__(solution_db)
        self.alpha = alpha

    def sample(self):
        solutions = self.solution_db.solutions.items
        n = len(solutions)
        if n == 1:
            return solutions[0]
        
        cdf = [0] * n
        total = 0
        
        for i in range(n):
            total += (i + 1) ** self.alpha
            cdf[i] = total
        
        r = random.uniform(0, total)
        idx = bisect.bisect_left(cdf, r)
        
        logger.info(f"sampled solution {solutions[idx].id}")
        solution = solutions[idx]
        solution = self.solution_db.load_solution(solution)
        return solution
