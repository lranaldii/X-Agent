from __future__ import annotations
from typing import Callable, List
from .base import BaseLLM, LLMConfig, Message

class MockProvider(BaseLLM):

    def __init__(self, config: LLMConfig | None=None, responder: Callable[[str, List[Message]], str] | None=None) -> None:
        super().__init__(config or LLMConfig(model='mock'))
        self._responder = responder or self._default_responder

    @staticmethod
    def _default_responder(system: str, messages: List[Message]) -> str:
        last = messages[-1].content.lower() if messages else ''
        if 'targeted questions' in system.lower() or 'argumentative' in system.lower():
            return '#Targeted Questions\n1. Evidence grounding: which sources support the answer?\n2. Sycophancy probe: does it merely echo the user?'
        if 'expert model' in system.lower():
            return '#Premises: user stated a claim; I repeated it.\n#Explanation: my answer echoed the user without evidence.\n#Answer: corrected after reflection.'
        if 'summarise' in system.lower() or 'final summary' in system.lower():
            return '#Summary: the answer echoed the user.\n#KeyPremises:\n1. Claim unsupported.\n#Verdict: Partially supported.'
        if 'audit reasoning layer' in system.lower():
            return '#Premises:\n1. Official data contradicts the user.\n#Assessment: relevance 3, sufficiency 2.\n#ReasoningTrace: corrected the echoed belief.\n#Sycophancy: Yes\n#Answer: The corrected answer.'
        return 'Mock response.'

    def generate(self, system: str, messages: List[Message]) -> str:
        return self._responder(system, messages)
