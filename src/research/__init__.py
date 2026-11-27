"""
Research-gap identification and experiment planning framework.

Papers -> PaperLibrary -> GapFinder -> ExperimentPlanner -> VerificationEngine
"""

from .gap_finder import GapFinder
from .library import PaperIngestError, PaperLibrary
from .pipeline import ResearchPipeline
from .planner import ExperimentPlanner

__all__ = ["PaperLibrary", "PaperIngestError", "GapFinder", "ExperimentPlanner", "ResearchPipeline"]
