import sys
sys.path.append("../../AGoTI")

from abc import ABC, abstractmethod
import asyncio
from typing import Optional, Tuple, List, Dict
import logging
from tqdm.asyncio import tqdm as tqdm_async
import json
import re
import numpy as np
import random
import copy

from AGoTI.model import LLM
from AGoTI.goo import GraphOfOperations

from .prompts import EDIT_EVALUATE_PROMPT_RU, EDIT_MUTATE_PROMPT_RU, PLAN_MUTATE_PROMPT, AGGREGATE_FEEDBACK_PROMPT_RU, \
    AGGREGATE_CRITIQUE_PROMPT_RU
from .goo import GooConfig, get_io_description, get_op_description, get_op_class_description
from .task import Task
from .solution_database import Solution
from .planner import Planner

logger = logging.getLogger(__name__)

class Mutator(ABC):
    @abstractmethod
    async def mutate(self, parent: Solution, samples: List[Dict], task: Task) -> Tuple[str, str, Dict]:
        pass


async def aggrigate_critique(
        model: LLM,
        task_description: str,
        operation_description: str,
        critiques: List[str],
        prompt: str=AGGREGATE_CRITIQUE_PROMPT_RU
        ) -> str:
    prompt = prompt\
        .replace("{description}", task_description)\
        .replace("{operation_description}", operation_description)\
        .replace("{critiques}", "\n===\n".join(critiques))
    messages = [{"role": "user", "content": prompt}]

    response = await model.generate(messages)
    matches = re.findall(r'(\`\`\`|""")\s?(?:json|text)?\s*\n?(.*?)\n?\1', response, re.DOTALL)
    if len(matches) > 0:
        return matches[-1][1]
    else:
        logger.error("LLM failed to follow prompt, while trying to aggrigate critique. " \
            f"Messages:\n\"\"\"\n{messages}\n\"\"\"\nResponse:\n\"\"\"\n{response}\n\"\"\"")
        return None

async def aggrigate_feedback(
        model: LLM,
        task_description: str,
        feedbacks: List[str],
        prompt: str=AGGREGATE_FEEDBACK_PROMPT_RU
        ) -> str:
    prompt = prompt\
        .replace("{description}", task_description)\
        .replace("{feedbacks}", "\n===\n".join(feedbacks))
    messages = [{"role": "user", "content": prompt}]

    response = await model.generate(messages)
    matches = re.findall(r'(\`\`\`|""")\s?(?:json|text)?\s*\n?(.*?)\n?\1', response, re.DOTALL)
    if len(matches) > 0:
        return matches[-1][1]
    else:
        logger.error("LLM failed to follow prompt, while trying to aggrigate feedback. " \
            f"Messages:\n\"\"\"\n{messages}\n\"\"\"\nResponse:\n\"\"\"\n{response}\n\"\"\"")
        return None


class EditMutator(Mutator):
    def __init__(
            self,
            model: LLM,
            goo_config: GooConfig,
            evaluate_prompt: str=EDIT_EVALUATE_PROMPT_RU,
            mutate_prompt: str=EDIT_MUTATE_PROMPT_RU,
            worst_n=3,
            max_changes=3
            ):
        self.model = model
        self.goo_config = goo_config
        self.evaluate_prompt = evaluate_prompt
        self.edit_prompt = mutate_prompt
        self.worst_n = worst_n
        self.max_changes = max_changes

    async def mutate(self, parent: Solution, samples: List[Dict], task: Task):
        goo_results = list(zip(parent.scores, parent.goos, samples))
        goo_results.sort(key=lambda x: x[0])
        bottom_n_pairs = goo_results[:self.worst_n]

        worst_goos = []
        worst_samples = []
        for _, goo, sample in bottom_n_pairs:
            worst_goos.append(goo)
            worst_samples.append(sample)

        tasks = []
        for id, op_json in parent.goo_json["nodes"].items():
            id = int(id)
            tasks.append(self.evaluate_op(id, parent.plan_formal, op_json, worst_samples, worst_goos, task))
        all_evaluations = await tqdm_async.gather(*tasks, desc="EditMutator: evaluating operations")

        samples_score_sums = [0] * len(worst_samples)
        for op_evalution in all_evaluations:
            for i, sample_evalution in enumerate(op_evalution):
                samples_score_sums[i] += 10 - sample_evalution[0] + 0.1

        op_scores = []
        for op_evalution in all_evaluations:
            op_score = 0
            for i, sample_evalution in enumerate(op_evalution):
                op_score += (10 - sample_evalution[0] + 0.1) / samples_score_sums[i]
            op_scores.append(op_score / len(worst_samples))
        op_to_score = {id: score for id, score in zip(parent.goo_json["nodes"].keys(), op_scores)}
        logger.debug(f"Edit scores: {op_to_score}")
            
        worst_op_idx = []
        if len(all_evaluations) <= self.max_changes:
            worst_op_idx = range(len(all_evaluations))
        else:    
            weights = np.array(op_scores)
            weights = weights / weights.sum()
            
            worst_op_idx = np.random.choice(
                len(all_evaluations),
                size=self.max_changes,
                replace=False,
                p=weights
            ).tolist()
        worst_op_ids = [list(parent.goo_json["nodes"].keys())[i] for i in worst_op_idx]
        logger.debug(f"Editing operation: {worst_op_ids}")
        
        tasks = []
        for i, id in zip(worst_op_idx, worst_op_ids):
            op_json = parent.goo_json["nodes"][id]
            op_config = self.goo_config.name_to_config[op_json["class"]]
            critiques = [critique for _, critique in all_evaluations[i]]
            operation_description = get_op_description(op_json, id, self.goo_config)
            # agg_critique = "\n".join(f"{j}: {critique}" for j, critique in enumerate(critiques, start=1))
            agg_critique = await aggrigate_critique(self.model, task.description, operation_description, critiques)
            tasks.append(self.edit_op(
                get_op_class_description(op_config),
                ", ".join(op_config.mutable_kwargs),
                operation_description,
                agg_critique,
                task
                ))
        edits = await tqdm_async.gather(*tasks, desc="EditMutator: generating edits")
        edit = {f"{id}": edit for id, edit in zip(worst_op_ids, edits)}

        mutated_goo_json = self.apply_edit(parent.goo_json, edit)
        return parent.plan_draft, parent.plan_formal, mutated_goo_json

    async def evaluate_op(
            self,
            id: int,
            plan_formal: str,
            op_json: Dict,
            samples: List,
            goos: List[GraphOfOperations],
            task: Task
        ) -> List[Tuple[float, str]]:
        class_description = get_op_class_description(self.goo_config.name_to_config[op_json["class"]])
        operation_description = get_op_description(op_json, id, self.goo_config)

        tasks = []
        for sample, goo in zip(samples, goos):
            tasks.append(self.evaluate_op_sample(
                id,
                class_description,
                operation_description,
                goo,
                plan_formal,
                sample,
                task
                ))
        operation_evaluations = await asyncio.gather(*tasks)
        return operation_evaluations

    async def evaluate_op_sample(
            self,
            id: int,
            class_description: str,
            operation_description: str,
            goo: GraphOfOperations,
            plan: str,
            sample: Dict,
            task: Task
            ) -> Tuple[float, str]:
        def truncate_if_long(value: str, max_len: int = 3000) -> str:
            if len(value) <= max_len:
                return value
            half = max_len // 2
            return value[:half] + "\n...\n" + value[-half:]
        
        sample_info = "\n".join(
            [f"- {k}: {truncate_if_long(str(v))}" 
             for k, v in task.get_goo_kwargs(sample)["replace"].items()]
        )
        feedback = await task.generate_feedback(sample, goo.get_output())
        feedback = await aggrigate_feedback(self.model, task.description, [feedback])
        input_output = get_io_description(goo.id_to_node[id])

        prompt = self.evaluate_prompt\
            .replace("{description}", task.description)\
            .replace("{plan}", plan)\
            .replace("{sample_info}", sample_info)\
            .replace("{feedback}", feedback)\
            .replace("{class_description}", class_description)\
            .replace("{operation_description}", operation_description)\
            .replace("{input_output}", input_output)
        messages = [{"role": "user", "content": prompt}]
        response = await self.model.generate(messages)

        match = re.search(r"<CRITIQUE>\s*(.+?)\s*</CRITIQUE>\s*<SCORE>\s*(.+?)\s*</SCORE>", response, re.DOTALL)
        try:
            critique = match.group(1)
            score = float(match.group(2))
        except Exception as e:
            logger.error(f"LLM failed to follow output format. Response:\n```\n{response}\n/```")
            raise e
        return score, critique

    async def edit_op(self, class_description: str, mutable_args: str, operation_description: str, critique: str, task: Task) -> ...:
        prompt = self.edit_prompt\
            .replace("{description}", task.description)\
            .replace("{class_description}", class_description)\
            .replace("{mutable_args}", mutable_args)\
            .replace("{operation_description}", operation_description)\
            .replace("{critique}", critique)
        messages = [{"role": "user", "content": prompt}]
        response = await self.model.generate(messages)

        match = re.search(r">\s*CHANGE\s*({.+?})\s*<\s*CHANGE", response, re.DOTALL)
        edit_str = match.group(1)
        try:
            edit = json.loads(edit_str)
        except Exception as e:
            logger.error(f"LLM failed to follow output format. Response:\n```\n{response}\n/```")
            raise e
        return edit

    def apply_edit(
            self,
            goo_json: Dict,
            edit: List[Dict],
            ) -> Dict:
        mutated_goo_json = copy.deepcopy(goo_json)
        for id, args in edit.items():
            mutated_goo_json["nodes"][str(id)]["args"].update(args)
        return mutated_goo_json


class PlanMutator(Mutator):
    def __init__(
            self,
            model: LLM,
            goo_config: GooConfig,
            planner: Planner,
            plan_prompt: str=PLAN_MUTATE_PROMPT,
            worst_n=3
        ):
        self.model = model
        self.goo_config = goo_config
        self.planner = planner
        self.plan_prompt = plan_prompt
        self.worst_n = worst_n

    async def mutate(self, parent, samples, task):
        logger.debug(f"generating new plan")
        
        goo_results = list(zip(parent.scores, parent.goos, samples))
        goo_results.sort(key=lambda x: x[0])
        bottom_n_pairs = goo_results[:self.worst_n]

        tasks = [task.generate_feedback(sample, goo.get_output()) for _, goo, sample in bottom_n_pairs]
        feedbacks = await asyncio.gather(*tasks)
        # feedbacks = "\n===\n".join([f"{i}: {feedback}" for i, feedback in enumerate(feedbacks, start=1)])
        feedbacks = await aggrigate_feedback(self.model, task.description, feedbacks)

        messages = [{
            "role": "user",
            "content": self.plan_prompt\
                .replace("{description}", task.description)\
                .replace("{plan_draft}", parent.plan_draft)\
                .replace("{plan_formal}", parent.plan_formal)\
                .replace("{feedback}", feedbacks)
        }]
        response = await self.model.generate(messages)

        try:
            match = re.search(r">\s*DRAFT\s*(.+?)\s*<\s*DRAFT", response, re.DOTALL)
            plan_draft = match.group(1).strip()
            match = re.search(r">\s*PLAN\s*(.+?)\s*<\s*PLAN", response, re.DOTALL)
            plan_formal = match.group(1).strip()
        except Exception as e:
            logger.error(f"LLM failed to follow output format: {e}")
            raise e
            #TODO: add retry

        mutated_goo_json = await self.planner.plan_to_goo_json(plan_formal)
        return plan_draft, plan_formal, mutated_goo_json

class RandomMutator(Mutator):
    def __init__(self, mutators: List[Mutator], weights: Optional[List[float]]=None, names: Optional[List[str]]=None):
        self.mutators = mutators
        self.weights = weights
        self.names = names if names else [mutator.__class__.__name__ for mutator in self.mutators]
    
    def mutate(self, parent, samples, task):
        mutator_name, chosen_mutator = random.choices(
            population=list(zip(self.names, self.mutators)),
            weights=self.weights,
            k=1
        )[0]
        logger.debug(f"Mutation: {mutator_name}")
        return chosen_mutator.mutate(parent, samples, task)
