# Unit tests for the step parser
"""Tests for parsing reasoning text into structured steps."""

from __future__ import annotations

import pytest

from llm_explainability.explainers.chain_of_thought.step_parser import StepParser


@pytest.fixture
def parser() -> StepParser:
    """Provide a StepParser instance."""
    return StepParser(max_steps=10)


class TestStepParser:
    """Tests for the StepParser class."""

    def test_parse_numbered_steps(self, parser: StepParser) -> None:
        """Verify parsing of numbered step format."""
        text = "1. First, we identify the problem.\n2. Then, we analyze the data.\n3. Therefore, the answer is X."
        steps = parser.parse(text)
        assert len(steps) >= 2
        assert steps[0].position == 0

    def test_parse_bullet_points(self, parser: StepParser) -> None:
        """Verify parsing of bullet point format."""
        text = "- The premise is clear.\n- We can infer that X is true.\n- In conclusion, Y follows."
        steps = parser.parse(text)
        assert len(steps) >= 2

    def test_parse_sentence_fallback(self, parser: StepParser) -> None:
        """Verify fallback to sentence-level parsing."""
        text = "The earth is round. Therefore gravity pulls objects down. This means everything falls."
        steps = parser.parse(text)
        assert len(steps) >= 2

    def test_first_step_classified_as_premise(self, parser: StepParser) -> None:
        """Verify first step is classified as premise."""
        text = "1. Given that X is true.\n2. We can conclude Y."
        steps = parser.parse(text)
        assert steps[0].step_type == "premise"

    def test_last_step_classified_as_conclusion(self, parser: StepParser) -> None:
        """Verify last step is classified as conclusion."""
        text = "1. We know A.\n2. This implies B.\n3. In conclusion, C is true."
        steps = parser.parse(text)
        assert steps[-1].step_type == "conclusion"

    def test_max_steps_limit(self, parser: StepParser) -> None:
        """Verify that steps are limited to max_steps."""
        # Create text with more than 10 steps
        text = "\n".join(f"{i}. Step number {i}." for i in range(1, 20))
        steps = parser.parse(text)
        assert len(steps) <= 10

    def test_empty_text_returns_empty(self, parser: StepParser) -> None:
        """Verify empty text returns empty list."""
        steps = parser.parse("")
        assert steps == []

    def test_inference_keyword_classification(self, parser: StepParser) -> None:
        """Verify steps with inference keywords are classified correctly."""
        text = "1. X is given.\n2. Therefore Y must be true.\n3. The answer is Z."
        steps = parser.parse(text)
        # Middle step with 'therefore' should be inference
        middle_steps = [s for s in steps if s.step_type == "inference"]
        assert len(middle_steps) >= 1
