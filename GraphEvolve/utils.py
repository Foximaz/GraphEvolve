import sys
sys.path.append("../../AGoTI")

from typing import Any, List
import bisect

from AGoTI.utils import Message

class SortedList:
    def __init__(self):
        self._data = []

    def add(self, value: Any, key: Any) -> None:
        pos = bisect.bisect_left(self._data, (key, value))
        self._data.insert(pos, (key, value))

    @property
    def items(self) -> List[Any]:
        return [item[1] for item in self._data]


def prompt_to_str(prompt: List[Message]):
    return "\n\n".join([f"[{message['role']}]:\n{message['content']}" for message in prompt])
