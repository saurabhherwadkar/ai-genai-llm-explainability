# Dashboard sidebar configuration component
"""Provides the configuration sidebar for selecting providers and options."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass
class SidebarConfig:
    """Configuration values collected from the sidebar widgets."""

    # Selected LLM provider name
    provider: str
    # List of selected explainability techniques
    techniques: list[str]
    # Optional system message
    system_message: str
    # Maximum tokens for LLM generation
    max_tokens: int
    # Sampling temperature
    temperature: float


def render_sidebar() -> SidebarConfig:
    """Render the configuration sidebar and return selected values."""
    # Create the sidebar section
    st.sidebar.title("Configuration")
    # Provider selection dropdown
    provider = st.sidebar.selectbox(
        "LLM Provider",
        options=["openai", "anthropic", "ollama", "huggingface"],
        index=0,
        help="Select which LLM provider to use for generation",
    )
    # Technique selection (multi-select)
    techniques = st.sidebar.multiselect(
        "Explainability Techniques",
        options=["token_attribution", "shap_lime", "chain_of_thought"],
        default=["token_attribution", "shap_lime", "chain_of_thought"],
        help="Select which explanation techniques to run",
    )
    # Use "all" if all techniques are selected
    if len(techniques) == 3:
        techniques = ["all"]
    # System message input
    system_message = st.sidebar.text_area(
        "System Message (optional)",
        value="",
        height=80,
        help="Optional system prompt to provide context to the LLM",
    )
    # Advanced options in an expander
    with st.sidebar.expander("Advanced Options"):
        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4096,
            value=1024,
            step=100,
            help="Maximum tokens in the LLM response",
        )
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Sampling temperature (0 = deterministic)",
        )
    # Return the collected configuration
    return SidebarConfig(
        provider=provider,
        techniques=techniques,
        system_message=system_message,
        max_tokens=max_tokens,
        temperature=temperature,
    )
