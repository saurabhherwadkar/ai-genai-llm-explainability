# Logic flow analyzer for chain-of-thought reasoning
"""Builds dependency graphs between reasoning steps to identify logical flow."""

from __future__ import annotations

import re

from llm_explainability.explainers.chain_of_thought.step_parser import ReasoningStep


class LogicFlowAnalyzer:
    """Analyzes logical dependencies and connections between reasoning steps."""

    # Words that indicate reference to a previous step
    _REFERENCE_INDICATORS = frozenset({
        "this", "that", "these", "those", "it", "they",
        "above", "previous", "earlier", "mentioned",
        "from step", "as stated", "as shown",
    })
    # Causal connectors indicating dependency
    _CAUSAL_CONNECTORS = frozenset({
        "therefore", "thus", "hence", "so", "because",
        "since", "as a result", "consequently", "given that",
        "it follows", "which means", "implies",
    })

    def analyze(self, steps: list[ReasoningStep]) -> dict[int, list[int]]:
        """Build a directed dependency graph between reasoning steps."""
        # Initialize the adjacency list (step_index -> list of dependent step indices)
        flow_graph: dict[int, list[int]] = {}
        # Initialize all steps with empty dependency lists
        for step in steps:
            flow_graph[step.position] = []
        # Analyze each step for references to prior steps
        for i, step in enumerate(steps):
            # Skip the first step (no prior steps to reference)
            if i == 0:
                continue
            # Find which prior steps this step depends on
            dependencies = self._find_dependencies(step, steps[:i])
            # Add edges from dependency to this step
            for dep_idx in dependencies:
                flow_graph[dep_idx].append(step.position)
        # Add sequential fallback for orphan steps
        flow_graph = self._add_sequential_fallbacks(flow_graph, steps)
        # Return the complete dependency graph
        return flow_graph

    def _find_dependencies(
        self, step: ReasoningStep, prior_steps: list[ReasoningStep]
    ) -> list[int]:
        """Identify which prior steps the current step depends on."""
        # List to collect dependency positions
        dependencies: list[int] = []
        step_text_lower = step.text.lower()
        # Check for explicit causal connectors (strong signal)
        has_causal = any(conn in step_text_lower for conn in self._CAUSAL_CONNECTORS)
        # Check for reference indicators (weaker signal)
        has_reference = any(ref in step_text_lower for ref in self._REFERENCE_INDICATORS)
        # If causal language found, link to most recent relevant step
        if has_causal or has_reference:
            # Find the most relevant prior step by content overlap
            best_match = self._find_best_match(step, prior_steps)
            if best_match is not None:
                dependencies.append(best_match)
        # Check for explicit step number references
        explicit_refs = self._find_explicit_references(step.text)
        dependencies.extend(explicit_refs)
        # If no dependencies found, default to immediately prior step
        if not dependencies and prior_steps:
            dependencies.append(prior_steps[-1].position)
        # Remove duplicates while preserving order
        seen: set[int] = set()
        unique_deps: list[int] = []
        for dep in dependencies:
            if dep not in seen:
                seen.add(dep)
                unique_deps.append(dep)
        # Return the unique dependency list
        return unique_deps

    def _find_best_match(
        self, step: ReasoningStep, prior_steps: list[ReasoningStep]
    ) -> int | None:
        """Find the prior step most related to the current step by word overlap."""
        # Get content words from current step (exclude common words)
        step_words = self._get_content_words(step.text)
        # Score each prior step by word overlap
        best_score = 0.0
        best_position: int | None = None
        for prior in prior_steps:
            prior_words = self._get_content_words(prior.text)
            # Calculate word overlap score
            overlap = len(step_words & prior_words)
            # Weight more recent steps slightly higher
            recency_bonus = 0.1 * (prior.position / max(len(prior_steps), 1))
            score = overlap + recency_bonus
            # Track the best matching prior step
            if score > best_score:
                best_score = score
                best_position = prior.position
        # Return the position of the best match
        return best_position

    def _find_explicit_references(self, text: str) -> list[int]:
        """Find explicit references to step numbers in the text."""
        # Pattern for "step N", "point N", "#N" references
        pattern = re.compile(r"(?:step|point|#)\s*(\d+)", re.IGNORECASE)
        matches = pattern.findall(text)
        # Convert matched numbers to zero-indexed positions
        references: list[int] = []
        for match in matches:
            # Convert to zero-indexed step position
            step_num = int(match) - 1
            if step_num >= 0:
                references.append(step_num)
        # Return the list of referenced positions
        return references

    def _get_content_words(self, text: str) -> set[str]:
        """Extract meaningful content words from text, excluding stop words."""
        # Common stop words to exclude
        stop_words = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "of", "in", "to", "for", "with", "on", "at", "by", "from",
            "and", "or", "but", "not", "this", "that", "it", "its",
        })
        # Split text into lowercase words
        words = set(re.findall(r"\b[a-z]+\b", text.lower()))
        # Remove stop words
        return words - stop_words

    def _add_sequential_fallbacks(
        self, graph: dict[int, list[int]], steps: list[ReasoningStep]
    ) -> dict[int, list[int]]:
        """Add sequential edges for steps that have no incoming connections."""
        # Find steps with no incoming edges (orphan steps)
        all_targets: set[int] = set()
        for targets in graph.values():
            all_targets.update(targets)
        # For each orphan step (except step 0), add edge from previous step
        for step in steps:
            if step.position > 0 and step.position not in all_targets:
                # Add edge from the immediately preceding step
                prev_pos = step.position - 1
                if prev_pos in graph and step.position not in graph[prev_pos]:
                    graph[prev_pos].append(step.position)
        # Return the updated graph
        return graph
