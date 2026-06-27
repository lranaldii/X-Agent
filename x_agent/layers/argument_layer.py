from __future__ import annotations
from typing import List, Tuple
from ..llm.base import BaseLLM, Message
from ..prompts import templates
from ..types import DebateRound, Example

class ArgumentReasoningLayer:

    def __init__(self, responder: BaseLLM, auditor: BaseLLM, rounds: int=3) -> None:
        self.responder = responder
        self.auditor = auditor
        self.rounds = rounds

    def _probe(self, question: str, current_answer: str) -> str:
        system = templates.ARGUMENT_PROBE_SYSTEM
        user = templates.ARGUMENT_PROBE_USER.format(question=question, model_answer=current_answer)
        return self.auditor.generate(system, [Message('user', user)])

    def _respond(self, question: str, current_answer: str, probes: str) -> str:
        system = templates.RESPONDER_SYSTEM
        user = templates.RESPONDER_USER.format(question=question, model_answer=current_answer, probes=probes)
        return self.responder.generate(system, [Message('user', user)])

    @staticmethod
    def _extract_answer(explanation: str) -> str:
        marker = '#Answer'
        if marker.lower() in explanation.lower():
            idx = explanation.lower().rfind(marker.lower())
            return explanation[idx + len(marker):].lstrip(':').strip()
        return explanation.strip()

    def run(self, example: Example, initial_answer: str) -> Tuple[List[DebateRound], str, str]:
        question = example.perturbed_question or example.question
        current_answer = initial_answer
        rounds: List[DebateRound] = []
        last_explanation = ''
        for k in range(1, self.rounds + 1):
            probe = self._probe(question, current_answer)
            explanation = self._respond(question, current_answer, probe)
            rounds.append(DebateRound(index=k, probe=probe, response=explanation))
            last_explanation = explanation
            current_answer = self._extract_answer(explanation) or current_answer
        summary = self._summarise(question, rounds, last_explanation)
        return (rounds, summary, current_answer)

    def _summarise(self, question: str, rounds: List[DebateRound], explanation: str) -> str:
        dialogue = '\n'.join((f'[R{r.index}] PROBE: {r.probe}\n[R{r.index}] REPLY: {r.response}' for r in rounds))
        system = templates.ARGUMENT_SUMMARY_SYSTEM
        user = templates.ARGUMENT_SUMMARY_USER.format(question=question, dialogue=dialogue, explanation=explanation)
        return self.auditor.generate(system, [Message('user', user)])
