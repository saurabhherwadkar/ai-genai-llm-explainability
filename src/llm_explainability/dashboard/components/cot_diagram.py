# Chain-of-thought flow diagram visualization component
"""Renders reasoning chain as a directed flow diagram."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_cot_diagram(cot_data: dict[str, Any]) -> None:
    """Render chain-of-thought analysis visualization in Streamlit."""
    # Get visualization data from the technique result
    viz_data = cot_data.get("visualization_data", {})
    data = cot_data.get("data", {})
    # Display coherence scores as metrics
    _render_coherence_metrics(data)
    # Render the reasoning steps as a flow
    _render_steps_flow(viz_data)
    # Display any identified weaknesses
    _render_weaknesses(data)


def _render_coherence_metrics(data: dict[str, Any]) -> None:
    """Display coherence sub-scores as Streamlit metric widgets."""
    # Get sub-scores from the data
    sub_scores = data.get("sub_scores", {})
    overall = data.get("coherence_score", 0.0)
    # Display overall score prominently
    st.metric("Overall Coherence", f"{overall:.2f}", delta=None)
    # Display sub-scores in columns
    cols = st.columns(4)
    score_labels = [
        ("Logical Validity", "logical_validity"),
        ("Completeness", "completeness"),
        ("Relevance", "relevance"),
        ("Consistency", "consistency"),
    ]
    for col, (label, key) in zip(cols, score_labels):
        score = sub_scores.get(key, 0.0)
        col.metric(label, f"{score:.2f}")


def _render_steps_flow(viz_data: dict[str, Any]) -> None:
    """Render reasoning steps as a visual flow diagram."""
    # Get nodes and edges from visualization data
    nodes = viz_data.get("nodes", [])
    edges = viz_data.get("edges", [])
    if not nodes:
        st.info("No reasoning steps to display.")
        return
    # Create a network-style visualization using Plotly
    # Position nodes vertically
    node_x = [0.5] * len(nodes)
    node_y = [i / max(len(nodes) - 1, 1) for i in range(len(nodes))]
    # Map step types to colors
    type_colors = {
        "premise": "#2196F3",
        "inference": "#FF9800",
        "conclusion": "#4CAF50",
        "observation": "#9C27B0",
        "assumption": "#F44336",
    }
    # Build node colors based on step type
    node_colors = [
        type_colors.get(node.get("type", "inference"), "#FF9800")
        for node in nodes
    ]
    # Build node labels (truncated)
    node_labels = [node.get("label", f"Step {i+1}") for i, node in enumerate(nodes)]
    # Create edge traces
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for edge in edges:
        from_idx = edge.get("from", 0)
        to_idx = edge.get("to", 0)
        if from_idx < len(node_y) and to_idx < len(node_y):
            edge_x.extend([node_x[from_idx], node_x[to_idx], None])
            edge_y.extend([node_y[from_idx], node_y[to_idx], None])
    # Create the figure
    fig = go.Figure()
    # Add edges as lines
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=2, color="#888"),
            hoverinfo="none",
        ))
    # Add nodes as scatter points
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=30, color=node_colors, line=dict(width=2, color="white")),
        text=[f"Step {i+1}" for i in range(len(nodes))],
        textposition="middle right",
        hovertext=node_labels,
        hoverinfo="text",
    ))
    # Configure layout
    fig.update_layout(
        title="Reasoning Flow",
        showlegend=False,
        height=max(300, len(nodes) * 60),
        xaxis=dict(visible=False, range=[-0.5, 2]),
        yaxis=dict(visible=False, range=[-0.1, 1.1]),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    # Also display steps as expandable text
    st.subheader("Reasoning Steps")
    for node in nodes:
        step_type = node.get("type", "unknown")
        full_text = node.get("full_text", node.get("label", ""))
        with st.expander(f"Step {node.get('id', 0) + 1} ({step_type})"):
            st.write(full_text)


def _render_weaknesses(data: dict[str, Any]) -> None:
    """Display identified logical weaknesses as warnings."""
    # Get the weaknesses list from data
    weaknesses = data.get("weaknesses", [])
    if not weaknesses:
        st.success("No logical weaknesses identified in the reasoning chain.")
        return
    # Display each weakness as a warning
    st.subheader("Identified Weaknesses")
    for weakness in weaknesses:
        st.warning(weakness)
