import asyncio
from typing import List, Optional
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
import logging

from .task import Task
from .solution_database import Solution, SolutionDatabase
from .goo import GooConfig, parse_goo_json
from .planner import Planner
from .sampler import Sampler
from .mutator import Mutator

logger = logging.getLogger(__name__)

class Runner():
    def __init__(
            self,
            task: Task,
            solution_db: SolutionDatabase,
            goo_config: GooConfig,
            planner: Planner,
            sampler: Sampler,
            mutator: Mutator,
            max_max_concurrent_goo_runs: int=1
            ):
        self.task = task
        self.solution_db = solution_db
        self.goo_config = goo_config
        self.planner = planner
        self.sampler = sampler
        self.mutator = mutator
        self.run_semaphore = asyncio.Semaphore(max_max_concurrent_goo_runs)

        self.train_samples, self.test_samples = self.task.load_dataset()
    
    async def run(
            self,
            score_target: float,
            max_iterations: int=100,
            throw_exceptions: bool=False
            ) -> Solution:
        if self.solution_db.solutions.items == []:
            draft_plan = await self.planner.draft_plan(self.task)
            formal_plan = await self.planner.plan_task(self.task, draft_plan)
            goo_json = await self.planner.plan_to_goo_json(formal_plan)
            logger.debug("Root solution planned")
            solution = await self.run_solution(draft_plan, formal_plan, goo_json)
            logger.debug("Root solution generated")
            self.solution_db.add_root(solution)

            if self.solution_db.best_solution.score >= score_target:
                return self.solution_db.best_solution

        for _ in tqdm(range(max_iterations), desc="iteration"):
            try:
                solution = self.sampler.sample()
                logger.debug(f"Solution sampled (score: {solution.score})")

                logger.debug("Mutating solution...")
                new_draft_plan, new_formal_plan, new_goo_json = await self.mutator.mutate(solution, self.train_samples, self.task)

                logger.debug("Running solution...")
                solution = await self.run_solution(new_draft_plan, new_formal_plan, new_goo_json, [solution])
                self.solution_db.add_solution(solution)

                if self.solution_db.best_solution.score >= score_target:
                    break
            except Exception as e:
                logger.error(f"runner iteration failed: {e}")
                if throw_exceptions:
                    raise e

        self.solution_db.load_solution(self.solution_db.best_solution)
        return self.solution_db.best_solution
    
    async def run_solution(
            self,
            plan_draft: str,
            plan_formal: str,
            goo_json: dict,
            parents: Optional[List[Solution]]=None
            ) -> Solution:
        tasks = [self.run_sample(goo_json, sample) for sample in self.train_samples]
        results = await tqdm_async.gather(*tasks, desc="running solution")
        goos, outputs, sample_infos = zip(*results)
        goos, outputs, sample_infos = list(goos), list(outputs), list(sample_infos)
            
        score, sample_scores = await self.task.evaluate_dataset(self.train_samples, outputs, sample_infos)
        return Solution(
            goo_json,
            score,
            "", #TODO: add global feedback
            plan_draft,
            plan_formal,
            goos,
            sample_scores,
            parents
        )
    
    async def run_test(
        self,
        solution: Solution
    ) -> float:
        solution = self.solution_db.load_solution(solution)
        tasks = [self.run_sample(solution.goo_json, sample) for sample in self.test_samples]
        results = await tqdm_async.gather(*tasks, desc="running solution")
        _, outputs, sample_infos = zip(*results)
        outputs, sample_infos = list(outputs), list(sample_infos)
        return (await self.task.evaluate_dataset(self.test_samples, outputs, sample_infos))[0]

    async def run_sample(self, goo_json, sample):
        async with self.run_semaphore:
            goo = parse_goo_json(goo_json, self.goo_config)
            sample_kwargs = self.task.get_goo_kwargs(sample)
            output = await goo.run(**sample_kwargs)
            sample_info = await self.task.evaluate(sample, output)
            return goo, output, sample_info
