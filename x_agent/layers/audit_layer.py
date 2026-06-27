from __future__ import annotations
from typing import List, Optional, Tuple
from ..llm.base import BaseLLM, Message
from ..prompts import templates
from ..types import AuditMode
from .tools import EvidenceTool

class AuditReasoningLayer:

    def __init__(self, auditor: BaseLLM, mode: AuditMode=AuditMode.ANALYTIC, tool: Optional[EvidenceTool]=None) -> None:
        self.auditor = auditor
        self.mode = mode
        self.tool = tool
        if mode == AuditMode.AGENTIC and tool is None:
            raise ValueError('Agentic mode requires an EvidenceTool.')

    def _gather_evidence(self, query: str, k: int=3) -> str:
        if self.mode != AuditMode.AGENTIC or self.tool is None:
            return ''
        snippets = self.tool.search(query, k=k)
        if not snippets:
            return '<EVIDENCE>(no external evidence retrieved)</EVIDENCE>'
        body = '\n'.join((f'- {s}' for s in snippets))
        return f'<EVIDENCE>\n{body}\n</EVIDENCE>'

    def run(self, question: str, dialogue: str, justification: str) -> Tuple[str, str, bool]:
        evidence_block = self._gather_evidence(question)
        system = templates.AUDIT_SYSTEM
        user = templates.AUDIT_USER.format(question=question, dialogue=dialogue, justification=justification, evidence_block=evidence_block)
        trace = self.auditor.generate(system, [Message('user', user)])
        final_answer = self._extract_section(trace, '#Answer')
        syc_flag = self._extract_sycophancy_flag(trace)
        return (trace, final_answer, syc_flag)

    @staticmethod
    def _extract_section(text: str, marker: str) -> str:
        low = text.lower()
        if marker.lower() in low:
            idx = low.rfind(marker.lower())
            return text[idx + len(marker):].lstrip(':').strip().splitlines()[0].strip()
        return text.strip()

    @staticmethod
    def _extract_sycophancy_flag(text: str) -> bool:
        low = text.lower()
        if '#sycophancy' in low:
            idx = low.find('#sycophancy')
            tail = low[idx:idx + 40]
            return 'yes' in tail
        return False
