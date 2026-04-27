import sys
sys.path.append("../../AGoTI")

from typing import Iterable, List, Dict, Any, Optional
import logging

from .operations import InitialGenerator, FeedForwardGenerator, Aggregator, Filter
from .prompts import OP_CLASS_DESCRIPTION_PROMPT, OP_DESCRIPTION_PROMPT

from AGoTI.operations import Operation
from AGoTI.model import LLM
from AGoTI.goo import parse_goo_json as _parse_goo_json
from AGoTI.goo import GraphOfOperations

logger = logging.getLogger(__name__)

class OperationConfig:
    REQ_KWARGS = ["id", "name", "description", "thought_tag"]

    def __init__(
            self,
            operation: type[Operation],
            name: str,
            description: str,
            mutable_kwargs: List[str],
            default_kwargs: Dict[str, Any],
        ):
        self.operation = operation
        self.name = name
        self.description = description
        self.mutable_kwargs = mutable_kwargs
        self.default_kwargs = default_kwargs
    
    def instantiate(self, **kwargs) -> Operation:
        k_diff = set(kwargs.keys()).difference(set(self.mutable_kwargs + self.REQ_KWARGS))
        if k_diff:
            logger.warning(f"Trying to instantiate an operation with wrong arguments: {", ".join(list(k_diff))}")
            for k in k_diff:
                del kwargs[k]
        return self.operation(**self.default_kwargs, **kwargs)


class GooConfig:
    def __init__(self, operation_configs: Iterable[OperationConfig]):
        self.name_to_config = {config.name: config for config in operation_configs}

PARSING_DESCRIPTION = """\
- parsing :str - тип парсинга ответа:
   - none - передает сгенерированный текст полностью. Подходит в случаях, когда парсинг не нужен
   - boxed - возвращает только текст внутри \\boxed{}. Подходит для обработки числовых значений или коротких строк
   - quoted - возвращает только текст внутри тройных кавычек (\"\"\"пример\"\"\"). Подходит для обработки многострочных строк
   - json - возвращает *множество* элементов внутри json списка (обязательно с кавычками!):
     ``` json
       [...]
     ```
     Подходит для обработки множества сложных структур.
   (для извлечения *множества* выходов с помощью boxed или quoted, каждый элемент множества должен быть обёрнут *отдельно*. Например: \"\\boxed{1.0}, \\boxed{2.5}, \\boxed{-1.3}\" или ```пример 1```, ```пример 2```, ```пример 3```)
"""

def get_default_config(model: LLM):
    operation_configs = [
        OperationConfig(
            InitialGenerator,
            "корень",
            "Генерирует ответ на основе промпта.\n" \
            "Параметры:\n" \
            "- prompt :str - промпт, передающийся исполнителю (не забывай учитывать тип парсинга). Объем не более 3 абзацев.\n" \
            + PARSING_DESCRIPTION,
            ["prompt", "parsing"],
            {"model": model}
        ),
        OperationConfig(
            FeedForwardGenerator,
            "генерация",
            "Генерирует ответ на основе промпта и дополнительной информации (обозначаемой строками типа {INFO_NAME}).\n" \
            "Параметры:\n" \
            "- prompt :str - промпт, передающийся исполнителю (со строками типа {INFO_NAME}) (не забывай учитывать тип парсинга). Объем не более 3 абзацев.\n" \
            + PARSING_DESCRIPTION,
            ["prompt", "parsing"],
            {"model": model}
        ),
        OperationConfig(
            Aggregator,
            "агрегация",
            "Генерирует ответ на основе промпта и дополнительной информации (обозначаемой строками типа {INFO_NAME}). Объем не более 3 абзацев.\n" \
            "Параметры:\n" \
            "- prompt :str - промпт, передающийся исполнителю (со строками типа {INFO_NAME}) (не забывай учитывать тип парсинга)\n" \
            + PARSING_DESCRIPTION,
            ["prompt", "parsing"],
            {"model": model}
        ),
        OperationConfig(
            Filter,
            "фильтр",
            "Проверяет, некоторое утвреждение, заданное промптом и дополнительной информацией (обозначаемой строками типа {INFO_NAME}).\n" \
            "Формат - после рассуждений необходимо дать ответ в виде \\boxed{да} или \\boxed{нет} (тип парсинга зафиксирован).\n"
            "Параметры:\n" \
            "- prompt :str - промпт, передающийся исполнителю (со строками типа {INFO_NAME}). Объем не более 3 абзацев.\n",
            ["prompt"],
            {"model": model}
        )
    ]
    return GooConfig(operation_configs)

def parse_goo_json(
        goo_json: Dict,
        goo_config: GooConfig,
        got_json: Optional[Dict]=None,
        throw_exceptions: bool=True
        ) -> GraphOfOperations:
    name_to_class = {name: config.instantiate for name, config in goo_config.name_to_config.items()}
    try:
        return _parse_goo_json(goo_json, name_to_class, got_json, throw_exceptions)
    except Exception as e:
        logger.debug(f"failed to parse goo_json: {e}\ngoo_json:\n```\n{goo_json}\n```" + f"\ngot_json:\n```\n{got_json}\n```" if got_json else "")
        raise e

def get_op_class_description(op_config: OperationConfig, prompt: str=OP_CLASS_DESCRIPTION_PROMPT):
    return prompt\
        .replace("{name}", op_config.name)\
        .replace("{description}", op_config.description)

def get_op_classes_description(goo_config: GooConfig, prompt: str=OP_CLASS_DESCRIPTION_PROMPT):
    res = []
    for op_config in goo_config.name_to_config.values():
        res.append(get_op_class_description(op_config, prompt))
    return "\n\n".join(res)

def get_op_description(operation_json: Dict, id: str|int, goo_config: GooConfig, prompt: str=OP_DESCRIPTION_PROMPT):
    op_config = goo_config.name_to_config[operation_json["class"]]
    args_description = []
    for arg_name in op_config.mutable_kwargs:
        arg_description = repr(operation_json["args"][arg_name])
        if isinstance(operation_json["args"][arg_name], str):
            arg_description = f"\"{arg_description}\""
        args_description.append(f"- {arg_name}: {arg_description}")
    args_description = "\n".join(args_description)
    return prompt\
        .replace("{id}", str(id))\
        .replace("{class}", operation_json["class"])\
        .replace("{name}", operation_json["args"]["name"])\
        .replace("{description}", operation_json["args"]["description"])\
        .replace("{parents}", str(operation_json["parents"]))\
        .replace("{args}", args_description)

def get_goo_description(goo_json: Dict, goo_config, prompt: str=OP_DESCRIPTION_PROMPT) -> str:
    res = []
    for id, node in goo_json["nodes"].items():
        res.append(get_op_description(node, id, goo_config, prompt))
    return "\n\n".join(res)

def get_io_description(
    operation: Operation,
    max_generations: int = 20,
    half_generations: int = 10,
    max_items: int = 20,
    half_items: int = 10
) -> str:
    if not operation.thoughts:
        return "Ни одного выхода не было сгенерировано! (возможно ничего не пришло на вход?)"
    
    generations = [t for t in operation.thoughts if not t.tags.get("parsed", True)]
    if not generations:
        return "Нет генераций, требующих описания."

    if len(generations) > max_generations:
        blocks = []
        for i, thought in enumerate(generations[:half_generations], start=1):
            blocks.append(_format_thought_block(thought, i, max_items, half_items))
        blocks.append("\n...\n")
        start_idx = len(generations) - half_generations + 1
        for i, thought in enumerate(generations[-half_generations:], start=start_idx):
            blocks.append(_format_thought_block(thought, i, max_items, half_items))
        return "\n".join(blocks)
    else:
        return "\n".join(
            _format_thought_block(t, i, max_items, half_items)
            for i, t in enumerate(generations, start=1)
        )

def _format_thought_block(thought, index: int, max_items: int, half_items: int) -> str:
    block = f"# Генерация №{index}\n"

    parents = list(thought.parents)
    block += _format_items(
        parents,
        max_items=max_items,
        half=half_items,
        item_formatter=lambda j, p: (
            f"## Вход №{j}: {p.tags.get('replace', '')}\n"
            f'"""\n{p.text}\n"""\n'
        )
    )

    block += f'## Выход\n"""\n{thought.text}\n"""\n'

    if not thought.tags.get("parsed", True):
        block += "## После парсинга:\n"
        children = list(thought.children)
        if not children:
            block += "\nНи одного выхода не сгенерировано!\n"
        else:
            block += _format_items(
                children,
                max_items=max_items,
                half=half_items,
                item_formatter=lambda j, ch: f'## Выход №{j}\n"""\n{ch.text}\n"""\n'
            )
    return block

def _format_items(items, max_items: int, half: int, item_formatter) -> str:
    if not items:
        return ""
    if len(items) > max_items:
        result = []
        for j, item in enumerate(items[:half], start=1):
            result.append(item_formatter(j, item))
        result.append("\n...\n")
        start_last = len(items) - half + 1
        for j, item in enumerate(items[-half:], start=start_last):
            result.append(item_formatter(j, item))
        return "\n".join(result)
    else:
        return "\n".join(
            item_formatter(j, item) for j, item in enumerate(items, start=1)
        )
