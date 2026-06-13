# Chain-of-thought analysis engine orchestrator
"""Coordinates parsing, flow analysis, and coherence scoring of LLM reasoning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llm_explainability.config.settings import ChainOfThoughtConfig
from llm_explainability.explainers.base import BaseExplainer, ExplanationResult
from llm_explainability.explainers.chain_of_thought.coherence_scorer import (
    CoherenceScorer,
    CoherenceSubScores,
)
from llm_explainability.explainers.chain_of_thought.logic_flow import (
    LogicFlowAnalyzer,
)
from llm_explainability.explainers.chain_of_thought.step_parser import (
    ReasoningStep,
    StepParser,
)
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import (
    GenerationRequest,
    GenerationResponseWithMetadata,
)


@dataclass
class ChainOfThoughtResult:
    """Complete result of chain-of-thought analysis."""

    # Parsed reasoning steps from the response
    steps: list[ReasoningStep] = field(default_factory=list)
    # Adjacency list of logical dependencies between steps
    flow_graph: dict[int, list[int]] = field(default_factory=dict)
    # Overall coherence score [0.0, 1.0]
    coherence_score: float = 0.0
    # Breakdown of coherence into sub-dimensions
    sub_scores: CoherenceSubScores | None = None
    # List of identified logical weaknesses
    weaknesses: list[str] = field(default_factory=list)
    # Data for rendering the reasoning flow diagram
    visualization_data: dict[str, object] = field(default_factory=dict)


class ChainOfThoughtEngine(BaseExplainer):
    """Orchestrates chain-of-thought analysis of LLM reasoning responses."""

    def __init__(self, config: ChainOfThoughtConfig | None = None) -> None:
        """Initialize the engine with its configuration."""
        # Use default config if none provided
        self._config = config or ChainOfThoughtConfig()
        # Initialize the step parser component
        self._step_parser = StepParser(max_steps=self._config.max_steps)
        # Initialize the logic flow analyzer
        self._flow_analyzer = LogicFlowAnalyzer()
        # Initialize the coherence scorer
        self._coherence_scorer = CoherenceScorer(
            threshold=self._config.coherence_threshold
        )

    async def explain(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
    ) -> ExplanationResult:
        """Analyze the chain-of-thought reasoning in the LLM response."""
        # Record start time for performance tracking
        start_time = time.perf_counter()
        try:
            # Get the response text to analyze
            response_text = response.text
            # If response lacks explicit reasoning, request a CoT version
            if not self._has_reasoning_markers(response_text):
                response_text = await self._request_cot_response(prompt, provider)
            # Parse the response into discrete reasoning steps
            steps = self._step_parser.parse(response_text)
            # Analyze logical flow between steps
            flow_graph = self._flow_analyzer.analyze(steps)
            # Score the coherence of the reasoning chain
            sub_scores = self._coherence_scorer.score(steps, flow_graph)
            # Identify logical weaknesses
            weaknesses = self._coherence_scorer.identify_weaknesses(steps, flow_graph)
            # Calculate overall coherence as weighted average of sub-scores
            overall_score = sub_scores.overall_score()
            # Build visualization data for the flow diagram
            viz_data = self._build_visualization(steps, flow_graph, sub_scores)
            # Build the complete analysis result
            cot_result = ChainOfThoughtResult(
                steps=steps,
                flow_graph=flow_graph,
                coherence_score=overall_score,
                sub_scores=sub_scores,
                weaknesses=weaknesses,
                visualization_data=viz_data,
            )
            # Calculate execution duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            # Return as ExplanationResult
            return ExplanationResult(
                technique_name="chain_of_thought",
                success=True,
                summary=self._generate_summary(cot_result),
                data={
                    "steps": [
                        {"text": s.text, "type": s.step_type, "position": s.position}
                        for s in steps
                    ],
                    "flow_graph": flow_graph,
                    "coherence_score": overall_score,
                    "sub_scores": {
                        "logical_validity": sub_scores.logical_validity,
                        "completeness": sub_scores.completeness,
                        "relevance": sub_scores.relevance,
                        "consistency": sub_scores.consistency,
                    },
                    "weaknesses": weaknesses,
                },
                visualization_data=viz_data,
                confidence=overall_score,
                metadata={"duration_ms": duration_ms, "num_steps": len(steps)},
            )
        except Exception as exc:
            # Return failed result with error details
            return ExplanationResult(
                technique_name="chain_of_thought",
                success=False,
                error_message=str(exc),
                metadata={"duration_ms": (time.perf_counter() - start_time) * 1000},
            )

    def get_technique_name(self) -> str:
        """Return the human-readable name of this technique."""
        return "Chain-of-Thought Analysis"

    def get_supported_providers(self) -> list[str]:
        """Return list of providers this technique works with."""
        # Works with all providers (can request CoT via prompting)
        return ["all"]

    def _has_reasoning_markers(self, text: str) -> bool:
        """Check if the response text contains explicit reasoning markers."""
        # Define common reasoning indicators
        markers = [
            "step 1", "first,", "therefore", "because", "let me think",
            "reasoning:", "let's break", "to solve this", "my reasoning",
            "1.", "2.", "3.",
        ]
        # Check if any marker is present in the text
        text_lower = text.lower()
        return any(marker in text_lower for marker in markers)

    async def _request_cot_response(
        self, prompt: str, provider: BaseLLMProvider
    ) -> str:
        """Request a chain-of-thought version of the response from the LLM."""
        # Create a CoT-inducing prompt
        cot_prompt = (
            f"Please answer the following question step by step, "
            f"showing your reasoning clearly:\n\n{prompt}"
        )
        # Generate the CoT response
        request = GenerationRequest(
            prompt=cot_prompt,
            system_message="You are a helpful assistant that always shows step-by-step reasoning.",
            max_tokens=2048,
            temperature=0.0,
        )
        response = await provider.generate(request)
        # Return the text with explicit reasoning
        return response.text

    def _generate_summary(self, result: ChainOfThoughtResult) -> str:
        """Generate a human-readable summary of the CoT analysis."""
        # Build summary from analysis results
        num_steps = len(result.steps)
        score = result.coherence_score
        # Classify coherence level
        if score >= 0.8:
            quality = "strong"
        elif score >= 0.6:
            quality = "moderate"
        else:
            quality = "weak"
        # Build weakness note if any
        weakness_note = ""
        if result.weaknesses:
            weakness_note = f" Weaknesses found: {'; '.join(result.weaknesses[:3])}."
        # Return formatted summary
        return (
            f"The reasoning chain contains {num_steps} steps with {quality} "
            f"coherence (score: {score:.2f}).{weakness_note}"
        )

    def _build_visualization(
        self,
        steps: list[ReasoningStep],
        flow_graph: dict[int, list[int]],
        sub_scores: CoherenceSubScores,
    ) -> dict[str, object]:
        """Build visualization data for rendering the reasoning flow diagram."""
        # Create nodes for each reasoning step
        nodes = [
            {
                "id": step.position,
                "label": step.text[:50] + "..." if len(step.text) > 50 else step.text,
                "type": step.step_type,
                "full_text": step.text,
            }
            for step in steps
        ]
        # Create edges from the flow graph
        edges = [
            {"from": source, "to": target}
            for source, targets in flow_graph.items()
            for target in targets
        ]
        # Return the complete visualization structure
        return {
            "type": "flow_diagram",
            "nodes": nodes,
            "edges": edges,
            "scores": {
                "logical_validity": sub_scores.logical_validity,
                "completeness": sub_scores.completeness,
                "relevance": sub_scores.relevance,
                "consistency": sub_scores.consistency,
            },
        }
