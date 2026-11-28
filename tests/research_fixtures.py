"""Shared offline fixtures for the research-gap pipeline tests (demo mode, no network)."""

from src.knowledge_base.retriever import QuantumRetriever
from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.llm.client import ConfigurableLLMClient
from src.llm.extractor import ProposalExtractor
from src.research import ExperimentPlanner, GapFinder, PaperLibrary, ResearchPipeline
from src.verification.engine import VerificationEngine


def build_pipeline(load_samples: bool = True) -> ResearchPipeline:
    llm = ConfigurableLLMClient(demo_mode=True)
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    library = PaperLibrary(ProposalExtractor(llm), kb)
    if load_samples:
        library.load_samples()
    return ResearchPipeline(library, GapFinder(llm, kb), ExperimentPlanner(llm),
                            VerificationEngine(retriever=QuantumRetriever(kb), llm_client=llm))
