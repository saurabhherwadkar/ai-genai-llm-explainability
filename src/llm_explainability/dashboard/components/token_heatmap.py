# Token attribution heatmap visualization component
"""Renders input text with color-coded background based on attribution scores."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_token_heatmap(attribution_data: dict[str, Any]) -> None:
    """Render a token attribution heatmap in the Streamlit dashboard."""
    # Extract the heatmap data from the technique results
    heatmap_items = attribution_data.get("visualization_data", {}).get("heatmap", [])
    # Check if there's data to display
    if not heatmap_items:
        st.warning("No token attribution data available to display.")
        return
    # Display the heatmap as colored HTML text
    _render_colored_text(heatmap_items)
    # Also render a bar chart of top tokens
    _render_importance_bar_chart(attribution_data)


def _render_colored_text(heatmap_items: list[dict[str, Any]]) -> None:
    """Render tokens as HTML spans with background color indicating importance."""
    # Build HTML with colored tokens
    html_parts: list[str] = ['<div style="line-height: 2.5; font-size: 16px;">']
    for item in heatmap_items:
        # Get the token text and score
        token = item.get("token", "")
        score = item.get("score", 0.0)
        # Map score to a red-to-green color scale
        red = int(255 * score)
        green = int(100 * (1 - score))
        blue = 50
        opacity = 0.3 + (score * 0.7)
        # Build the colored span element
        style = (
            f"background-color: rgba({red}, {green}, {blue}, {opacity}); "
            f"padding: 2px 4px; margin: 1px; border-radius: 3px; "
            f"display: inline-block;"
        )
        html_parts.append(f'<span style="{style}" title="Score: {score:.3f}">{token}</span> ')
    html_parts.append("</div>")
    # Render the HTML in Streamlit
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _render_importance_bar_chart(attribution_data: dict[str, Any]) -> None:
    """Render a horizontal bar chart of top influential tokens."""
    # Get top-k influential tokens from the data
    top_tokens = attribution_data.get("data", {}).get("top_k_influential", [])
    # Limit to top 10 for readability
    top_tokens = top_tokens[:10]
    if not top_tokens:
        return
    # Extract token names and scores
    tokens = [item.get("token", "") for item in top_tokens]
    scores = [item.get("score", 0.0) for item in top_tokens]
    # Create a horizontal bar chart with Plotly
    fig = go.Figure(go.Bar(
        x=scores,
        y=tokens,
        orientation="h",
        marker_color="rgba(55, 128, 191, 0.7)",
        marker_line_color="rgba(55, 128, 191, 1.0)",
        marker_line_width=1,
    ))
    # Configure the chart layout
    fig.update_layout(
        title="Top Influential Tokens",
        xaxis_title="Attribution Score",
        yaxis_title="Token",
        height=300,
        margin=dict(l=100, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
