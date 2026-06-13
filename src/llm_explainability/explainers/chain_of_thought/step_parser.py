# Reasoning step parser for chain-of-thought analysis
"""Parses free-text reasoning into structured discrete steps."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ReasoningStep:
    """Represents a single step in a chain-of-thought reasoning sequence."""

    # The text content of this reasoning step
    text: str
    # Type of step: premise, inference, conclusion, assumption, observation
    step_type: str
    # Position index within the reasoning chain
    position: int
    # Confidence that this step was correctly parsed
    parse_confidence: float = 1.0


class StepParser:
    """Parses LLM response text into structured reasoning steps."""

    # Pattern to match numbered steps (1., 2., Step 1:, etc.)
    _NUMBERED_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:(?:Step\s+)?(\d+)[.:)\s])\s*(.*?)(?=\n\s*(?:(?:Step\s+)?\d+[.:)\s])|$)",
        re.DOTALL | re.IGNORECASE,
    )
    # Pattern to match bullet points
    _BULLET_PATTERN = re.compile(r"(?:^|\n)\s*[-*•]\s+(.*?)(?=\n\s*[-*•]|$)", re.DOTALL)
    # Keywords that indicate step type
    _PREMISE_KEYWORDS = frozenset({"given", "assume", "known", "fact", "premise"})
    _INFERENCE_KEYWORDS = frozenset({
        "therefore", "thus", "hence", "so", "implies", "means", "because",
        "since", "as a result", "consequently",
    })
    _CONCLUSION_KEYWORDS = frozenset({
        "conclusion", "finally", "in summary", "answer", "result",
        "in conclusion", "overall",
    })

    def __init__(self, max_steps: int = 20) -> None:
        """Initialize the parser with maximum step limit."""
        # Maximum number of steps to extract from a response
        self._max_steps = max_steps

    def parse(self, text: str) -> list[ReasoningStep]:
        """Parse response text into a list of reasoning steps."""
        # Try numbered format first (most structured)
        steps = self._parse_numbered(text)
        # Fall back to bullet format
        if not steps:
            steps = self._parse_bullets(text)
        # Fall back to sentence-based splitting
        if not steps:
            steps = self._parse_sentences(text)
        # Classify each step's type based on content
        classified_steps = self._classify_steps(steps)
        # Limit to max_steps
        return classified_steps[: self._max_steps]

    def _parse_numbered(self, text: str) -> list[ReasoningStep]:
        """Parse numbered steps from the text (e.g., '1. First step')."""
        # Find all numbered step matches
        matches = self._NUMBERED_PATTERN.findall(text)
        # Convert matches to ReasoningStep objects
        steps: list[ReasoningStep] = []
        for i, match in enumerate(matches):
            # Extract the step text (second group)
            step_text = match[1].strip() if len(match) > 1 else match[0].strip()
            # Skip empty steps
            if not step_text:
                continue
            # Create a reasoning step with position
            steps.append(ReasoningStep(
                text=step_text,
                step_type="inference",  # Default, will be classified later
                position=i,
                parse_confidence=0.9,
            ))
        # Return parsed steps
        return steps

    def _parse_bullets(self, text: str) -> list[ReasoningStep]:
        """Parse bullet-pointed steps from the text."""
        # Find all bullet point matches
        matches = self._BULLET_PATTERN.findall(text)
        # Convert to ReasoningStep objects
        steps: list[ReasoningStep] = []
        for i, match in enumerate(matches):
            step_text = match.strip()
            if not step_text:
                continue
            steps.append(ReasoningStep(
                text=step_text,
                step_type="inference",
                position=i,
                parse_confidence=0.8,
            ))
        # Return parsed steps
        return steps

    def _parse_sentences(self, text: str) -> list[ReasoningStep]:
        """Fall back to sentence-level parsing for unstructured text."""
        # Split text at sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Filter out very short sentences
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 10]
        # Convert to ReasoningStep objects
        steps: list[ReasoningStep] = []
        for i, sentence in enumerate(meaningful):
            steps.append(ReasoningStep(
                text=sentence,
                step_type="inference",
                position=i,
                parse_confidence=0.6,  # Lower confidence for sentence-split
            ))
        # Return parsed steps
        return steps

    def _classify_steps(self, steps: list[ReasoningStep]) -> list[ReasoningStep]:
        """Classify each step's type based on its textual content."""
        # Classify each step based on keywords
        for i, step in enumerate(steps):
            step_lower = step.text.lower()
            # Check if this is the first step (likely premise)
            if i == 0:
                step.step_type = "premise"
            # Check if this is the last step (likely conclusion)
            elif i == len(steps) - 1:
                step.step_type = "conclusion"
            # Check for premise keywords
            elif any(kw in step_lower for kw in self._PREMISE_KEYWORDS):
                step.step_type = "premise"
            # Check for conclusion keywords
            elif any(kw in step_lower for kw in self._CONCLUSION_KEYWORDS):
                step.step_type = "conclusion"
            # Check for inference keywords
            elif any(kw in step_lower for kw in self._INFERENCE_KEYWORDS):
                step.step_type = "inference"
            else:
                # Default to observation for unclassified steps
                step.step_type = "observation"
        # Return the classified steps
        return steps
