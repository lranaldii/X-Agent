from __future__ import annotations
from typing import List, Optional
from .layers import ArgumentReasoningLayer, AuditReasoningLayer, EvidenceTool
from .llm.base import BaseLLM, Message
from .sycophancy import SycophancyJudge, perturb
from .types import AuditMode, Example, OversightResult, ReasoningTrace

class XAgent:

    def __init__(self, responder: BaseLLM, auditor: BaseLLM, judge: SycophancyJudge, argument_rounds: int=3, mode: AuditMode=AuditMode.ANALYTIC, tool: Optional[EvidenceTool]=None) -> None:
        self.responder = responder
        self.auditor = auditor
        self.judge = judge
        self.mode = mode
        self.argument_layer = ArgumentReasoningLayer(responder=responder, auditor=auditor, rounds=argument_rounds)
        self.audit_layer = AuditReasoningLayer(auditor=auditor, mode=mode, tool=tool)

    def _initial_answer(self, question: str) -> str:
        return self.responder.generate('Answer the question concisely.', [Message('user', question)])

    def oversee(self, example: Example, stance: Optional[str]=None, initial_answer: Optional[str]=None) -> OversightResult:
        if example.perturbed_question is None and stance is not None:
            example.perturbed_question = perturb(example, stance)
        question = example.perturbed_question or example.question
        y0 = initial_answer or self._initial_answer(question)
        syc_pre = self.judge.is_sycophantic(example.question, y0, stance or '', example.target)
        rounds, summary, refined = self.argument_layer.run(example, y0)
        dialogue = '\n'.join((f'[R{r.index}] PROBE: {r.probe}\n[R{r.index}] REPLY: {r.response}' for r in rounds))
        trace, final_answer, _syc_in_trace = self.audit_layer.run(question=example.question, dialogue=dialogue, justification=summary)
        syc_post = self.judge.is_sycophantic(example.question, final_answer, stance or '', example.target)
        return OversightResult(example=example, initial_answer=y0, rounds=rounds, argument_summary=summary, audit_trace=trace, final_answer=final_answer or refined, syc_pre=syc_pre, syc_post=syc_post, mode=self.mode)

    def extract_reasoning_traces(self, results: List[OversightResult]) -> List[ReasoningTrace]:
        traces: List[ReasoningTrace] = []
        for res in results:
            question = res.example.perturbed_question or res.example.question
            traces.append(ReasoningTrace(question=question, trace=res.audit_trace))
        return traces
