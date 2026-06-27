from __future__ import annotations
import os
from typing import List
from .base import BaseLLM, LLMConfig, Message

class GoogleProvider(BaseLLM):

    def __init__(self, config: LLMConfig, api_key: str | None=None) -> None:
        super().__init__(config)
        key = api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not key:
            raise ValueError('No Google API key found. Set GOOGLE_API_KEY or pass api_key.')
        self._key = key
        self._mode = None
        try:
            from google import genai
            self._client = genai.Client(api_key=key)
            self._mode = 'genai'
        except ImportError:
            try:
                import google.generativeai as legacy
                legacy.configure(api_key=key)
                self._legacy = legacy
                self._mode = 'legacy'
            except ImportError as exc:
                raise ImportError("Install the Google SDK with `pip install google-genai`.") from exc

    @staticmethod
    def _to_role(role: str) -> str:
        return 'model' if role == 'assistant' else 'user'

    def generate(self, system: str, messages: List[Message]) -> str:
        if self._mode == 'genai':
            from google.genai import types
            contents = [{'role': self._to_role(m.role), 'parts': [{'text': m.content}]} for m in messages]
            response = self._client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                ),
            )
            return response.text.strip()
        model = self._legacy.GenerativeModel(model_name=self.config.model, system_instruction=system)
        contents = [{'role': self._to_role(m.role), 'parts': [m.content]} for m in messages]
        response = model.generate_content(contents, generation_config={'temperature': self.config.temperature, 'max_output_tokens': self.config.max_tokens, 'top_p': self.config.top_p})
        return response.text.strip()
