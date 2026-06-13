# Main Streamlit dashboard entry point
"""Entry point for the LLM Explainability interactive dashboard."""

from __future__ import annotations

import streamlit as st

from llm_explainability.dashboard.api_client import ApiClient
from llm_explainability.dashboard.components.cot_diagram import render_cot_diagram
from llm_explainability.dashboard.components.shap_plot import render_shap_plot
from llm_explainability.dashboard.components.sidebar import render_sidebar
from llm_explainability.dashboard.components.token_heatmap import render_token_heatmap


def main() -> None:
    """Run the main Streamlit dashboard application."""
    # Configure the Streamlit page settings
    st.set_page_config(
        page_title="LLM Explainability Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Display the main title
    st.title("LLM Explainability Dashboard")
    st.markdown("*Understand why LLMs produce their outputs*")
    # Render the sidebar and get configuration
    sidebar_config = render_sidebar()
    # Create the API client
    api_client = ApiClient(base_url="http://localhost:8000")
    # Main content area - prompt input
    st.header("Enter Your Prompt")
    prompt = st.text_area(
        "Prompt",
        value="",
        height=150,
        placeholder="Enter the prompt you want to explain...",
        help="Type or paste the prompt you want to analyze",
    )
    # Explain button
    if st.button("Explain", type="primary", disabled=not prompt.strip()):
        # Show a spinner while processing
        with st.spinner("Generating explanation... This may take a moment."):
            try:
                # Call the API to get the explanation
                result = api_client.explain(
                    prompt=prompt,
                    provider=sidebar_config.provider,
                    techniques=sidebar_config.techniques,
                    system_message=sidebar_config.system_message or None,
                    max_tokens=sidebar_config.max_tokens,
                    temperature=sidebar_config.temperature,
                )
                # Store result in session state for persistence
                st.session_state["last_result"] = result
                st.session_state["last_prompt"] = prompt
            except Exception as exc:
                st.error(f"Error calling API: {str(exc)}")
                st.info("Make sure the API server is running (make run-api)")
                return
    # Display results if available
    if "last_result" in st.session_state:
        _render_results(st.session_state["last_result"])


def _render_results(result: dict) -> None:
    """Render the explanation results in the dashboard."""
    # Display the LLM response
    st.header("LLM Response")
    st.markdown(f"**Provider:** {result.get('provider_used', 'unknown')} | "
                f"**Model:** {result.get('model_used', 'unknown')}")
    st.text_area("Response", value=result.get("llm_response", ""), height=100, disabled=True)
    # Display the overall summary
    st.header("Explanation Summary")
    st.info(result.get("summary", "No summary available"))
    # Display confidence and metadata
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Confidence", f"{result.get('overall_confidence', 0):.2f}")
    metadata = result.get("metadata", {})
    col2.metric("Duration", f"{metadata.get('duration_ms', 0):.0f} ms")
    col3.metric("Techniques Run", str(len(result.get("explanations", {}))))
    # Display individual technique results in tabs
    explanations = result.get("explanations", {})
    if explanations:
        st.header("Technique Results")
        tabs = st.tabs(list(explanations.keys()))
        for tab, (technique_name, technique_data) in zip(tabs, explanations.items()):
            with tab:
                _render_technique_tab(technique_name, technique_data)


def _render_technique_tab(technique_name: str, data: dict) -> None:
    """Render the results for a single explainability technique."""
    # Show success/failure status
    if not data.get("success", False):
        st.error(f"Technique failed: {data.get('error_message', 'Unknown error')}")
        return
    # Show technique summary
    st.markdown(f"**{data.get('summary', '')}**")
    st.metric("Confidence", f"{data.get('confidence', 0):.2f}")
    # Render technique-specific visualization
    if technique_name == "token_attribution":
        render_token_heatmap(data)
    elif technique_name == "shap_lime":
        render_shap_plot(data)
    elif technique_name == "chain_of_thought":
        render_cot_diagram(data)
    # Show raw data in an expander for debugging
    with st.expander("Raw Data"):
        st.json(data.get("data", {}))


# Run the app when executed directly
if __name__ == "__main__":
    main()
