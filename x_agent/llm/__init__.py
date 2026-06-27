from __future__ import annotations
from .base import BaseLLM, LLMConfig, Message
from .mock_provider import MockProvider

def build_provider(backend: str, model: str, temperature: float=0.0, max_tokens: int=1024, api_key: str | None=None, **extra) -> BaseLLM:
    config = LLMConfig(model=model, temperature=temperature, max_tokens=max_tokens, extra=extra)
    backend = backend.lower()
    if backend == 'openai':
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(config, api_key=api_key)
    if backend in {'google', 'gemini'}:
        from .google_provider import GoogleProvider
        return GoogleProvider(config, api_key=api_key)
    if backend == 'mock':
        return MockProvider(config)
    raise ValueError(f'Unknown backend: {backend!r}')
__all__ = ['BaseLLM', 'LLMConfig', 'Message', 'MockProvider', 'build_provider']
