from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import List, Literal
Role = Literal['system', 'user', 'assistant']

@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict:
        return {'role': self.role, 'content': self.content}

@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    extra: dict = field(default_factory=dict)

class BaseLLM(abc.ABC):

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def generate(self, system: str, messages: List[Message]) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}({self.config.model})'
