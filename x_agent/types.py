from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

class TaskType(str, Enum):
    CLOSED_FORM_QA = 'closed_form_qa'
    OPEN_ENDED_KNOWN = 'open_ended_known'
    OPEN_ENDED_NO_GT = 'open_ended_no_gt'

class AuditMode(str, Enum):
    ANALYTIC = 'analytic'
    AGENTIC = 'agentic'

@dataclass
class Example:
    question: str
    task_type: TaskType
    target: Optional[str] = None
    perturbed_question: Optional[str] = None
    options: Optional[List[str]] = None
    meta: dict = field(default_factory=dict)

@dataclass
class DebateRound:
    index: int
    probe: str
    response: str

@dataclass
class OversightResult:
    example: Example
    initial_answer: str
    rounds: List[DebateRound]
    argument_summary: str
    audit_trace: str
    final_answer: str
    syc_pre: bool
    syc_post: bool
    mode: AuditMode

    @property
    def mitigated(self) -> bool:
        return self.syc_pre and (not self.syc_post)

    def transcript_text(self) -> str:
        lines = [f'USER ANSWER (initial): {self.initial_answer}']
        for r in self.rounds:
            lines.append(f'[Round {r.index}] AUDITOR PROBE: {r.probe}')
            lines.append(f'[Round {r.index}] MODEL REPLY: {r.response}')
        return '\n'.join(lines)

@dataclass
class ReasoningTrace:
    question: str
    trace: str

    def to_chat_pair(self) -> Tuple[dict, dict]:
        return ({'role': 'user', 'content': self.question}, {'role': 'assistant', 'content': self.trace})
