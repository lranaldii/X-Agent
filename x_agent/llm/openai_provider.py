from __future__ import annotations
import os
from typing import List
from .base import BaseLLM, LLMConfig, Message

class OpenAIProvider(BaseLLM):

    def __init__(self, config: LLMConfig, api_key: str | None=None) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The 'openai' package is required. Install it with `pip install openai`.") from exc
        key = api_key or os.environ.get('OPENAI_API_KEY')
        if not key:
            raise ValueError('No OpenAI API key found. Set OPENAI_API_KEY or pass api_key.')
        self._client = OpenAI(api_key=key)

    def generate(self, system: str, messages: List[Message]) -> str:
        payload = [{'role': 'system', 'content': system}]
        payload += [m.as_dict() for m in messages]
        response = self._client.chat.completions.create(model=self.config.model, messages=payload, temperature=self.config.temperature, max_tokens=self.config.max_tokens, top_p=self.config.top_p, **self.config.extra)
        return response.choices[0].message.content.strip()
