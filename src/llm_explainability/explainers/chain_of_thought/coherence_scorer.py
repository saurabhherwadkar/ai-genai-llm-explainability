# Coherence scorer for chain-of-thought reasoning
"""Scores the quality and coherence of reasoning chains across multiple dimensions."""

from __future__ import annotations

from dataclasses import dataclass

from llm_explainability.explainers.chain_of_thought.step_parser import ReasoningStep


@dataclass
class CoherenceSubScores:
    """Breakdown of coherence into specific quality dimensions."""

    # How well conclusions follow logically from premises [0, 1]
    logical_validity: float = 0.0
    # Whether the reasoning covers all necessary aspects [0, 1]
    completeness: float = 0.0
    # How relevant each step is to the original question [0, 1]
    relevance: float = 0.0
    # Whether steps are consistent with each other (no contradictions) [0, 1]
    consistency: float = 0.0

    def overall_score(self) -> float:
        """Calculate weighted average of all sub-scores."""
        # Weight logical validity highest as it's most important
        weights = {
            "logical_validity": 0.35,
            "completeness": 0.25,
            "relevance": 0.20,
            "consistency": 0.20,
        }
        # Compute weighted sum
        total = (
            self.logical_validity * weights["logical_validity"]
            + self.completeness * weights["completeness"]
            + self.relevance * weights["relevance"]
            + self.consistency * weights["consistency"]
        )
        # Return the weighted average
        return total


class CoherenceScorer:
    """Scores reasoning chain coherence across multiple quality dimensions."""

    def __init__(self, threshold: float = 0.7) -> None:
        """Initialize the scorer with minimum acceptable coherence threshold."""
        # Store the threshold for identifying weak reasoning
        self._threshold = threshold

    def score(
        self, steps: list[ReasoningStep], flow_graph: dict[int, list[int]]
    ) -> CoherenceSubScores:
        """Score the reasoning chain across all quality dimensions."""
        # Score logical validity based on step connections
        logical_validity = self._score_logical_validity(steps, flow_graph)
        # Score completeness based on step coverage
        completeness = self._score_completeness(steps)
        # Score relevance based on step coherence
        relevance = self._score_relevance(steps)
        # Score consistency by checking for contradictions
        consistency = self._score_consistency(steps)
        # Return the complete sub-scores
        return CoherenceSubScores(
            logical_validity=logical_validity,
            completeness=completeness,
            relevance=relevance,
            consistency=consistency,
        )

    def identify_weaknesses(
        self, steps: list[ReasoningStep], flow_graph: dict[int, list[int]]
    ) -> list[str]:
        """Identify specific logical weaknesses in the reasoning chain."""
        # Collect all identified weaknesses
        weaknesses: list[str] = []
        # Check for unsupported jumps (steps with no clear basis)
        unsupported = self._find_unsupported_steps(steps, flow_graph)
        for step_pos in unsupported:
            weaknesses.append(
                f"Step {step_pos + 1} lacks clear logical support from prior steps"
            )
        # Check for missing conclusion
        if not self._has_conclusion(steps):
            weaknesses.append("Reasoning chain lacks a clear conclusion")
        # Check for overly short chains
        if len(steps) < 2:
            weaknesses.append("Reasoning is too brief to constitute a logical chain")
        # Check for circular reasoning
        if self._has_circular_reasoning(flow_graph):
            weaknesses.append("Potential circular reasoning detected")
        # Check for repeated content
        repetitions = self._find_repetitions(steps)
        if repetitions:
            weaknesses.append(f"Repetitive reasoning in steps: {repetitions}")
        # Return all identified weaknesses
        return weaknesses

    def _score_logical_validity(
        self, steps: list[ReasoningStep], flow_graph: dict[int, list[int]]
    ) -> float:
        """Score how well conclusions follow from premises."""
        # Handle empty or single-step chains
        if len(steps) <= 1:
            return 0.3
        # Count steps with at least one logical connection
        connected_steps = 0
        total_edges = 0
        for targets in flow_graph.values():
            total_edges += len(targets)
            if targets:
                connected_steps += 1
        # Calculate connectivity ratio
        max_possible_edges = len(steps) - 1
        edge_ratio = total_edges / max_possible_edges if max_possible_edges > 0 else 0
        # Check for proper premise-to-conclusion flow
        has_premise = any(s.step_type == "premise" for s in steps)
        has_conclusion = any(s.step_type == "conclusion" for s in steps)
        # Combine factors into final score
        structure_bonus = 0.2 if (has_premise and has_conclusion) else 0.0
        score = min(1.0, edge_ratio * 0.8 + structure_bonus)
        # Return the logical validity score
        return score

    def _score_completeness(self, steps: list[ReasoningStep]) -> float:
        """Score whether the reasoning covers sufficient ground."""
        # Minimum steps for a complete argument
        min_steps = 3
        # Score based on step count (diminishing returns)
        if len(steps) >= min_steps:
            count_score = min(1.0, len(steps) / (min_steps * 2))
        else:
            count_score = len(steps) / min_steps
        # Check for diversity of step types
        step_types = set(s.step_type for s in steps)
        type_diversity = len(step_types) / 4.0  # 4 possible types
        # Combine count and diversity
        return (count_score * 0.6) + (type_diversity * 0.4)

    def _score_relevance(self, steps: list[ReasoningStep]) -> float:
        """Score how relevant each step is to the overall reasoning."""
        # Handle empty chains
        if not steps:
            return 0.0
        # Use average parse confidence as a proxy for relevance
        avg_confidence = sum(s.parse_confidence for s in steps) / len(steps)
        # Check for very short steps (likely irrelevant filler)
        substantive_steps = sum(1 for s in steps if len(s.text) > 20)
        substantive_ratio = substantive_steps / len(steps)
        # Combine confidence and substance
        return (avg_confidence * 0.5) + (substantive_ratio * 0.5)

    def _score_consistency(self, steps: list[ReasoningStep]) -> float:
        """Score internal consistency (absence of contradictions)."""
        # Handle empty or single-step chains
        if len(steps) <= 1:
            return 1.0
        # Simple contradiction detection via negation patterns
        contradiction_count = 0
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                if self._steps_contradict(steps[i], steps[j]):
                    contradiction_count += 1
        # Calculate consistency as inverse of contradiction ratio
        max_pairs = len(steps) * (len(steps) - 1) / 2
        contradiction_ratio = contradiction_count / max_pairs if max_pairs > 0 else 0
        # Return consistency score (1.0 = no contradictions)
        return max(0.0, 1.0 - contradiction_ratio * 2.0)

    def _steps_contradict(self, step_a: ReasoningStep, step_b: ReasoningStep) -> bool:
        """Check if two steps contain contradictory statements."""
        # Simple negation-based contradiction detection
        text_a = step_a.text.lower()
        text_b = step_b.text.lower()
        # Check for direct negation patterns
        negation_pairs = [
            ("is not", "is"),
            ("cannot", "can"),
            ("will not", "will"),
            ("does not", "does"),
            ("impossible", "possible"),
            ("incorrect", "correct"),
            ("false", "true"),
        ]
        for neg, pos in negation_pairs:
            # Check if one step affirms and the other negates
            if (neg in text_a and pos in text_b) or (pos in text_a and neg in text_b):
                return True
        # No contradiction detected
        return False

    def _find_unsupported_steps(
        self, steps: list[ReasoningStep], flow_graph: dict[int, list[int]]
    ) -> list[int]:
        """Find steps that have no logical support from prior steps."""
        # Collect all step positions that are targets of edges
        supported: set[int] = {0}  # Step 0 is always "supported" (starting point)
        for targets in flow_graph.values():
            supported.update(targets)
        # Find steps not in the supported set
        unsupported = [
            s.position for s in steps
            if s.position not in supported and s.position > 0
        ]
        # Return unsupported step positions
        return unsupported

    def _has_conclusion(self, steps: list[ReasoningStep]) -> bool:
        """Check if the reasoning chain contains a conclusion step."""
        # Check for explicit conclusion step type
        return any(s.step_type == "conclusion" for s in steps)

    def _has_circular_reasoning(self, flow_graph: dict[int, list[int]]) -> bool:
        """Detect cycles in the reasoning flow graph."""
        # Use DFS-based cycle detection
        visited: set[int] = set()
        in_stack: set[int] = set()

        def has_cycle(node: int) -> bool:
            """DFS helper to detect back edges indicating cycles."""
            visited.add(node)
            in_stack.add(node)
            for neighbor in flow_graph.get(node, []):
                if neighbor in in_stack:
                    return True
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
            in_stack.discard(node)
            return False

        # Check each unvisited node
        for node in flow_graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        # No cycle found
        return False

    def _find_repetitions(self, steps: list[ReasoningStep]) -> list[int]:
        """Find steps that substantially repeat content from earlier steps."""
        # Track positions of repetitive steps
        repetitive: list[int] = []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                # Check if steps have high word overlap
                words_i = set(steps[i].text.lower().split())
                words_j = set(steps[j].text.lower().split())
                if not words_i or not words_j:
                    continue
                overlap = len(words_i & words_j) / min(len(words_i), len(words_j))
                # Flag as repetitive if >80% word overlap
                if overlap > 0.8:
                    repetitive.append(j + 1)
        # Return unique repetitive step numbers
        return sorted(set(repetitive))
