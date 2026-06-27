from .agent import XAgent
from .llm import BaseLLM, LLMConfig, Message, MockProvider, build_provider
from .layers import ArgumentReasoningLayer, AuditReasoningLayer, DuckDuckGoSearch, EvidenceTool, InMemoryRetriever
from .metrics import post_audit_accuracy, summarise, sycophancy_mitigation_rate
from .sycophancy import SycophancyJudge, build_sycophantic_set, normalise_answer, perturb
from .types import AuditMode, DebateRound, Example, OversightResult, ReasoningTrace, TaskType
__version__ = '0.1.0'
__all__ = ['XAgent', 'Example', 'TaskType', 'AuditMode', 'DebateRound', 'OversightResult', 'ReasoningTrace', 'SycophancyJudge', 'build_sycophantic_set', 'normalise_answer', 'perturb', 'ArgumentReasoningLayer', 'AuditReasoningLayer', 'EvidenceTool', 'InMemoryRetriever', 'DuckDuckGoSearch', 'post_audit_accuracy', 'sycophancy_mitigation_rate', 'summarise', 'build_provider', 'BaseLLM', 'LLMConfig', 'Message', 'MockProvider']
