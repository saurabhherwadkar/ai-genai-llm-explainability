# SHAP/LIME visualization component
"""Renders SHAP force plots and LIME waterfall charts."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_shap_plot(shap_data: dict[str, Any]) -> None:
    """Render SHAP/LIME visualization in the Streamlit dashboard."""
    # Get visualization data from the technique result
    viz_data = shap_data.get("visualization_data", {})
    plot_type = viz_data.get("type", "")
    # Route to appropriate visualization renderer
    if plot_type == "force_plot":
        _render_force_plot(viz_data)
    elif plot_type == "waterfall":
        _render_waterfall_chart(viz_data)
    else:
        # Try to render as a generic importance chart
        _render_importance_chart(shap_data)


def _render_force_plot(viz_data: dict[str, Any]) -> None:
    """Render a SHAP-style force plot showing feature contributions."""
    # Extract features and their SHAP values
    features = viz_data.get("features", [])
    base_value = viz_data.get("base_value", 0.0)
    if not features:
        st.warning("No SHAP data available.")
        return
    # Sort features by absolute value for visual impact
    sorted_features = sorted(features, key=lambda f: abs(f.get("value", 0)), reverse=True)
    # Separate positive and negative contributions
    names = [f.get("name", "")[:30] for f in sorted_features[:15]]
    values = [f.get("value", 0.0) for f in sorted_features[:15]]
    colors = ["rgba(255, 65, 54, 0.7)" if v > 0 else "rgba(55, 128, 191, 0.7)" for v in values]
    # Create the waterfall/force chart
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        marker_line_width=1,
    ))
    # Add a vertical line for the base value
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    # Configure layout
    fig.update_layout(
        title=f"SHAP Feature Importance (base value: {base_value:.3f})",
        xaxis_title="SHAP Value (impact on output)",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=200, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def _render_waterfall_chart(viz_data: dict[str, Any]) -> None:
    """Render a LIME-style waterfall chart of feature importance."""
    # Extract features from visualization data
    features = viz_data.get("features", [])
    if not features:
        st.warning("No LIME data available.")
        return
    # Get names and values (already sorted by importance)
    names = [f.get("name", "")[:30] for f in features[:15]]
    values = [f.get("value", 0.0) for f in features[:15]]
    colors = ["red" if v > 0 else "blue" for v in values]
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        opacity=0.7,
    ))
    # Configure layout
    fig.update_layout(
        title="LIME Feature Importance",
        xaxis_title="Weight (contribution to prediction)",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=200, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def _render_importance_chart(shap_data: dict[str, Any]) -> None:
    """Render a generic feature importance chart from data fields."""
    # Get feature names and scores from the data
    data = shap_data.get("data", {})
    feature_names = data.get("feature_names", [])
    scores = data.get("importance_scores", [])
    if not feature_names or not scores:
        st.warning("No feature importance data available.")
        return
    # Create a simple bar chart
    fig = go.Figure(go.Bar(
        x=scores[:15],
        y=[n[:30] for n in feature_names[:15]],
        orientation="h",
        marker_color="rgba(55, 128, 191, 0.7)",
    ))
    fig.update_layout(
        title="Feature Importance Scores",
        xaxis_title="Importance Score",
        height=350,
        margin=dict(l=200, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
