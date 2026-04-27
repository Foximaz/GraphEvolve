import sys
sys.path.append("../../AGoTI")

import logging
from abc import ABC, abstractmethod
import asyncio
from typing import List, Dict, Optional
import re
import json

from .task import Task
from .goo import GooConfig
from .prompts import DRAFT_PLAN_PROMPT_RU, THREE_STEP_PLAN_TASK_PROMPT_RU, COMPLITION_PROMPT_RU

from AGoTI.model import LLM

logger = logging.getLogger(__name__)

class Planner(ABC):
    @abstractmethod
    async def draft_plan(self, task: Task) -> str:
        pass

    @abstractmethod
    async def plan_task(self, task: Task, plan_draft: str) -> Optional[str]:
        pass
    
    @abstractmethod
    async def plan_to_goo_json(self, plan: str) -> Optional[Dict]:
        pass


def parse_plan(plan: str):
    nodes = []
    answers = {}

    if "Ответы:" in plan:
        nodes_text, answers_text = plan.rsplit("Ответы:", 1)
    else:
        logger.error(f"LLM failed to follow the prompt! No answers section in the plan:\n{plan}")
        raise Exception(f"LLM failed to follow the prompt! No answers section in the plan")
    node_pattern = r'(\d+)\.\s+"(.+)"\s+"(.+)"\s+"(.*)"\s+входы:\s+\[(.*)\]\s+выход:\s+(.+)'
    
    for match in re.finditer(node_pattern, nodes_text):
        node_id = int(match.group(1))
        node_type = match.group(2)
        node_name = match.group(3)
        node_description = match.group(4)
        parents_str = match.group(5).strip()
        thought_tag = match.group(6).strip().strip('"')

        parent_ids = []
        if parents_str:
            for pid in parents_str.split(','):
                pid = pid.strip()
                if pid:
                    parent_ids.append(int(pid))
        
        nodes.append({
            "id": node_id,
            "type": node_type,
            "name": node_name,
            "description": node_description,
            "parent_ids": parent_ids,
            "thought_tag": thought_tag
        })
    
    if answers_text:
        answers = json.loads(answers_text.strip())
    else:
        raise Exception(f"LLM failed to follow prompt! No answer specifications in plan:\n{plan}")
    nodes.sort(key=lambda x: x["id"])
     
    if set(answers.values()).intersection(set([node["id"] for node in nodes])) != set(answers.values()):
        raise Exception(f"LLM failed to follow prompt! Not every node id in output is present in the plan:\n{plan}")

    return nodes, answers

async def parsed_plan_to_goo_json(
        model: LLM,
        nodes_info: List[Dict],
        outputs: Dict,
        goo_config: GooConfig,
        complition_prompt: str=COMPLITION_PROMPT_RU,
        root_operations: List[str] = ["root", "корень"],
        throw_exceptions: bool=True
        ) -> Dict:
    
    async def process_node(node_info: Dict):
        """Process a single node asynchronously"""
        op_config = goo_config.name_to_config.get(node_info["type"])
        if op_config is None:
            if throw_exceptions:
                raise Exception(f"No such operation ({node_info['type']}) in GoO config")
            return None, None, None
        
        is_root = node_info["type"] in root_operations
        
        node_complition_prompt = complition_prompt\
            .replace("{name}", node_info["name"])\
            .replace("{description}", node_info["description"])\
            .replace("{op_description}", op_config.description)
        prompt = [{"role": "user", "content": node_complition_prompt}]
        
        response = await model.generate(prompt)
        
        match = re.search(r">\s*ARGS\s*(\{.+\})\s*<\s*ARGS", response, re.DOTALL)
        if match:
            try:        
                params = json.loads(match.group(1))
            except Exception as e:
                if throw_exceptions:
                    logger.error(f"{e}\nLLM response was:\"{response}\"")
                    raise e
                return None, None, None
        else:
            if throw_exceptions:
                raise Exception(f"LLM failed to follow the format: \"{repr(response)}\"")
            return None, None, None
        
        node_data = {
            "class": node_info["type"],
            "parents": node_info["parent_ids"],
            "args": params | {"thought_tag": node_info["thought_tag"], "name": node_info["name"], "description": node_info["description"]}
        }
        
        return node_info["id"], node_data, is_root
    
    tasks = [process_node(node_info) for node_info in nodes_info]
    
    results = await asyncio.gather(*tasks, return_exceptions=not throw_exceptions)
    
    nodes = {}
    roots = []
    
    for result in results:
        if isinstance(result, Exception):
            if throw_exceptions:
                raise result
            return None
        node_id, node_data, is_root = result
        if node_id is not None:
            nodes[str(node_id)] = node_data
            if is_root:
                roots.append(node_id)
    
    return {"nodes": nodes, "roots": roots, "outputs": outputs}


class ThreeStepPlanner(Planner):
    def __init__(
            self,
            model: LLM,
            goo_config: GooConfig,
            draft_plan_prompt: str=DRAFT_PLAN_PROMPT_RU,
            plan_task_prompt: str=THREE_STEP_PLAN_TASK_PROMPT_RU
            ):
        self.model = model
        self.goo_config = goo_config
        self.draft_plan_prompt = draft_plan_prompt
        self.plan_task_prompt = plan_task_prompt

    async def draft_plan(self, task: Task) -> str:
        prompt = [{"role": "user", "content": 
            self.draft_plan_prompt.replace("{description}", task.description)}]
        response = await self.model.generate(prompt)
        return response

    async def plan_task(self, task, plan_draft):
        prompt = [{"role": "user", "content": 
            self.plan_task_prompt\
                .replace("{description}", task.description)\
                .replace("{plan_draft}", plan_draft)}]
        response = await self.model.generate(prompt)
        return response

    async def plan_to_goo_json(self, plan):
        nodes_info, outputs = parse_plan(plan)
        return await parsed_plan_to_goo_json(
            self.model, nodes_info, outputs, self.goo_config)
