import sys
sys.path.append("../../AGoTI")

from typing import Dict, Any, Tuple, List, Optional
import logging
import re
import json

import AGoTI.operations as agoti_ops
from AGoTI.thoughts import Thought
from AGoTI.model import LLM

logger = logging.getLogger(__name__)

def _parse_json_matches(matches):
    results = []
    for match in matches:
        try:
            results += list(map(str, json.loads(match[1])))
        except Exception as e:
            logger.warning(f"Failed to parse json from LLM ({e}): \"{match}\"")
    return results

PATTERNS = {
        "boxed": (r"\\boxed\{(.+?)\}", lambda x: x),
        "quoted": (r'(\`\`\`|""")\s?(?:json|text)?\s*\n?(.*?)\s*?\1', lambda x: list(map(lambda y: y[1], x))),
        "json": (r'(\`\`\`|""")\s*(?:json)?\s*(\[.*?\])\s*?\1', _parse_json_matches)
    }

def basic_parse_generation(text: str, parsing: str, thought_tag: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    if parsing == "none":
        return [thought_tag], [text]

    pattern, processer = PATTERNS[parsing]    
    matches = re.findall(pattern, text, re.DOTALL)
    parsed = processer(matches)
    return [thought_tag] * len(parsed), parsed


class InitialGenerator(agoti_ops.SimplePromptGenerator):
    DEFAULT_NAME = "InitialGenerator"

    def __init__(
            self,
            model: LLM,
            prompt: str,
            parsing: str="none",
            thought_tag="none",
            **kwargs
            ):
        self.parsing = parsing
        self.thought_tag = {"replace": thought_tag} if thought_tag != "none" else {}
        messages = [{"role": "user", "content": prompt}]
        super().__init__(model, messages, **kwargs)
    
    def parse_generation(self, text: str):
        return basic_parse_generation(text, self.parsing, self.thought_tag)

class FeedForwardGenerator(agoti_ops.Generator):
    DEFAULT_NAME = "FeedForwardGenerator"

    def __init__(
            self,
            model: LLM,
            prompt: str,
            parsing: str="none",
            thought_tag="none",
            **kwargs
            ):
        super().__init__(model, **kwargs)
        self.parsing = parsing
        self.thought_tag = {"replace": thought_tag} if thought_tag != "none" else {}
        self.prompt = prompt
    
    async def thought_collector(self):
        thought_generator = agoti_ops.any_thought_waiter(self.subscribtions, self.subscribtions.keys())        
        async for thought in thought_generator:
            yield ([thought], {"text": thought.text, "tag": thought.tags["replace"]})

    def make_prompt(self, text: str, tag: str):
        return [{"role": "user", "content": self.prompt.replace(f"{{{tag}}}", text)}]
    
    def parse_generation(self, text: str):
        return basic_parse_generation(text, self.parsing, self.thought_tag)

    async def run(self, **kwargs):
        replace = kwargs.get("replace", {})
        for src, dst in replace.items():
            self.prompt = self.prompt.replace(src, dst)
        await super().run(**kwargs)


class Aggregator(agoti_ops.Generator):
    DEFAULT_NAME = "Aggregator"

    def __init__(
            self,
            model: LLM,
            prompt: str,
            parsing: str="none",
            thought_tag="none",
            **kwargs
            ):
        super().__init__(model, **kwargs)
        self.parsing = parsing
        self.thought_tag = {"replace": thought_tag} if thought_tag != "none" else {}
        self.prompt = prompt
    
    async def check_condition(self):
        return await agoti_ops.parents_finished(self.parents)

    def parse_generation(self, text: str):
        return basic_parse_generation(text, self.parsing, self.thought_tag)

    async def thought_collector(self):
        thoughts = await agoti_ops.all_operations_waiter(self.subscribtions, self.parents)
        texts = {}
        for thought in thoughts:
            tag = thought.tags.get("replace", None)
            if tag:
                if tag not in texts.keys():
                    texts[tag] = []
                texts[tag].append(thought.text)
        yield (thoughts, {"texts": texts})

    def make_prompt(self, texts: List[str]):
        prompt = self.prompt
        for tag, tag_texts in texts.items():
            prompt = prompt.replace(tag, "\n---\n".join(tag_texts))
        return [{"role": "user", "content": prompt}]
    
    async def run(self, **kwargs):
        replace = kwargs.get("replace", {})
        for src, dst in replace.items():
            self.prompt = self.prompt.replace(src, dst)
        await super().run(**kwargs)


class Filter(agoti_ops.BasicOperation):
    DEFAULT_NAME = "Filter"

    def __init__(
        self,
        model: LLM,
        prompt: str,
        true_strs: List[str]=["yes", "да"],
        thought_tag="none",
        **kwargs
        ):
        super().__init__(**kwargs)
        self.model = model
        self.prompt = prompt
        self.true_strs = set(true_strs)
        self.thought_tag = {"replace": thought_tag} if thought_tag != "none" else {}

    async def thought_collector(self):
        thought_generator = agoti_ops.any_thought_waiter(self.subscribtions, self.subscribtions.keys())        
        async for thought in thought_generator:
            yield ([thought], {"text": thought.text, "tag": thought.tags["replace"]})

    def make_prompt(self, text: str, tag: str):
        return [{"role": "user", "content": self.prompt.replace(f"{{{tag}}}", text)}]

    def parse_generation(self, text: str) -> str:
        matches = re.findall(PATTERNS["boxed"], text)
        if matches:
            return matches[-1].lower().strip()
        else:
            return "нет"

    async def operation_task(self, parents, replace: Optional[Dict[str, str]]=None, **kwargs):
        messages = self.make_prompt(**kwargs)

        if replace:
            for src, dst in replace.items():
                for message in messages:
                    message["content"] = message["content"].replace(src, dst)

        response = await self.model.generate(messages)
        if response is None:
            return
        
        original_thought = Thought(response, parents=parents, prompt=messages, tags={"no_send": True, "parsed": False})
        thoughts = [original_thought]
        response = self.parse_generation(response)
        if response in self.true_strs:
            thought = Thought(parents[0].text, parents=[original_thought], prompt=messages, tags=self.thought_tag | {"parsed": True})
            thoughts.append(thought)
        return thoughts

    async def run(self, **kwargs):
        replace = kwargs.get("replace", {})
        for src, dst in replace.items():
            self.prompt = self.prompt.replace(src, dst)
        await super().run(**kwargs)
