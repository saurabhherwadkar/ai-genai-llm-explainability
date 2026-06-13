# Perturbation strategies for SHAP/LIME analysis
"""Defines different ways to perturb prompt text for perturbation-based explanation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class PerturbationStrategy(ABC):
    """Abstract base class for text perturbation strategies."""

    @abstractmethod
    def segment_text(self, text: str) -> list[str]:
        """Split text into segments that can be independently toggled."""
        ...

    @abstractmethod
    def apply_mask(self, segments: list[str], mask: list[bool]) -> str:
        """Reconstruct text from segments using the given inclusion mask."""
        ...


class SegmentPerturbation(PerturbationStrategy):
    """Sentence-level perturbation strategy that splits text into clauses."""

    # Pattern to split text at sentence boundaries
    _SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*")
    # Pattern to split at clause boundaries (commas, semicolons)
    _CLAUSE_PATTERN = re.compile(r"(?<=[,;])\s+")

    def segment_text(self, text: str) -> list[str]:
        """Split text into meaningful segments (sentences or clauses)."""
        # First try to split by sentences
        sentences = self._SENTENCE_PATTERN.split(text)
        # Filter out empty segments
        sentences = [s.strip() for s in sentences if s.strip()]
        # If we got too few segments, split by clauses too
        if len(sentences) <= 3:
            segments = []
            for sentence in sentences:
                # Split each sentence at clause boundaries
                clauses = self._CLAUSE_PATTERN.split(sentence)
                segments.extend(c.strip() for c in clauses if c.strip())
            return segments
        # If we got too many segments, merge small ones
        if len(sentences) > 20:
            return self._merge_small_segments(sentences, target_count=15)
        # Return the sentence-level segments
        return sentences

    def apply_mask(self, segments: list[str], mask: list[bool]) -> str:
        """Reconstruct text by including only segments where mask is True."""
        # Collect segments that are included by the mask
        included_segments = [
            seg for seg, include in zip(segments, mask) if include
        ]
        # Join included segments with spaces
        return " ".join(included_segments)

    def _merge_small_segments(
        self, segments: list[str], target_count: int
    ) -> list[str]:
        """Merge adjacent small segments to reach target count."""
        # Calculate how many segments to merge together
        merge_factor = max(1, len(segments) // target_count)
        # Merge segments in groups
        merged: list[str] = []
        for i in range(0, len(segments), merge_factor):
            # Combine this group of segments
            group = segments[i : i + merge_factor]
            merged.append(" ".join(group))
        # Return the merged segments
        return merged


class WordPerturbation(PerturbationStrategy):
    """Word-level perturbation strategy for fine-grained analysis."""

    def segment_text(self, text: str) -> list[str]:
        """Split text into individual words."""
        # Split on whitespace to get individual words
        words = text.split()
        # Return non-empty words
        return [w for w in words if w]

    def apply_mask(self, segments: list[str], mask: list[bool]) -> str:
        """Reconstruct text by including only words where mask is True."""
        # Include words where mask is True
        included = [word for word, include in zip(segments, mask) if include]
        # Join with spaces
        return " ".join(included)
