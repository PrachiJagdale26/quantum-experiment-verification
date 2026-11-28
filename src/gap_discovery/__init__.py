"""Gap discovery and novelty assessment (Phase 4)."""

from .gap_generator import GapGenerator, load_gap, load_gaps, save_gaps
from .gap_ranker import DEFAULT_WEIGHTS, RANK_DIMENSIONS, score_gap, weighted_total
from .novelty_checker import CoveragePolicy, NoveltyChecker, load_assessment
from .schema import CLASSIFICATION_LABELS, ORIGIN_LABELS, Gap, GapDimension, GapEvidenceItem

__all__ = ["GapGenerator", "load_gap", "load_gaps", "save_gaps", "DEFAULT_WEIGHTS", "RANK_DIMENSIONS", "score_gap",
           "weighted_total", "CoveragePolicy", "NoveltyChecker", "load_assessment", "CLASSIFICATION_LABELS",
           "ORIGIN_LABELS", "Gap", "GapDimension", "GapEvidenceItem"]
