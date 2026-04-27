import sys
sys.path.append("../../AGoTI")

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from datasets import Dataset, load_dataset

from AGoTI.model import LLM

class Task(ABC):
    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def get_goo_kwargs(self, sample) -> Dict:
        pass

    @abstractmethod
    async def evaluate_dataset(self, samples, outputs, infos) -> Tuple[float, Any]:
        pass

    @abstractmethod
    async def evaluate(self, sample, output) -> Any:
        pass

    @abstractmethod
    async def generate_feedback(self, sample, output) -> str:
        pass


class SimpleTask(Task):
    @abstractmethod
    def dataset_args(self) -> Dict:
        pass

    def process_dataset(self, dataset: Dataset) -> Dataset:
        return dataset

    @abstractmethod
    def train_split(self) -> str:
        pass

    @abstractmethod
    def test_split(self) -> str:
        pass

    def train_count(self) -> int:
        return 100

    def test_count(self) -> int:
        return 100

    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        dataset = load_dataset(**self.dataset_args())
        train_dataset = self.process_dataset(dataset[self.train_split()])
        test_dataset = self.process_dataset(dataset[self.test_split()])

        train_dataset = train_dataset.select(range(min(self.train_count(), len(train_dataset))))
        test_dataset = test_dataset.select(range(min(self.test_count(), len(test_dataset))))
        return (train_dataset, test_dataset)

    async def evaluate_dataset(self, samples, outputs, scores):
        score = await self.aggregate(scores)
        return score, scores

    @abstractmethod
    async def aggregate(self, scores: List[float]):
        pass


async def llm_as_judge(model: LLM, prompt: str, output: Dict[str, List[str]]):
    for key, operation_output in output:
        text = "\n".join(operation_output)
        prompt = prompt.replace(key, text)
    return await model.generate({"role": "user", "content": prompt})
    