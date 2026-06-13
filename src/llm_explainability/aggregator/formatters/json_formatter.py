# JSON output formatter for explanation results
"""Formats aggregated explanations as structured JSON for API consumption."""

from __future__ import annotations

from typing import Any

from llm_explainability.aggregator.aggregator import AggregatedExplanation


class JsonFormatter:
    """Formats explanation results as structured JSON dictionaries."""

    def format(self, explanation: AggregatedExplanation) -> dict[str, Any]:
        """Convert an aggregated explanation into a JSON-serializable dictionary."""
        # Build the top-level output structure
        output: dict[str, Any] = {
            "summary": explanation.summary,
            "overall_confidence": explanation.overall_confidence,
            "correlations": explanation.correlations,
            "total_duration_ms": explanation.total_duration_ms,
            "techniques": {},
            "metadata": explanation.metadata,
        }
        # Add each technique's results to the output
        for technique_name, result in explanation.technique_results.items():
            output["techniques"][technique_name] = {
                "success": result.success,
                "summary": result.summary,
                "confidence": result.confidence,
                "data": result.data,
                "visualization_data": result.visualization_data,
                "error_message": result.error_message,
                "metadata": result.metadata,
            }
        # Return the complete formatted output
        return output
