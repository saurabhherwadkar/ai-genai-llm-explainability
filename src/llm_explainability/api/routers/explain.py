# Explanation endpoints
"""API endpoints for submitting prompts and receiving explanations."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from llm_explainability.aggregator.aggregator import ExplanationAggregator
from llm_explainability.api.dependencies import (
    get_aggregator,
    get_config,
    get_input_sanitizer,
    get_provider_factory,
)
from llm_explainability.api.schemas.requests import ExplainRequest, SingleTechniqueRequest
from llm_explainability.api.schemas.responses import (
    ExplainResponse,
    ResponseMetadata,
    TechniqueResult,
)
from llm_explainability.config.settings import AppSettings
from llm_explainability.providers.factory import ProviderFactory
from llm_explainability.providers.models import GenerationRequest
from llm_explainability.security.input_sanitizer import InputSanitizer
from llm_explainability.utils.validators import validate_provider_name, validate_technique_names

# Create the router for explanation endpoints
router = APIRouter(prefix="/api/v1/explain", tags=["explain"])


@router.post("", response_model=ExplainResponse)
async def explain_prompt(
    request: ExplainRequest,
    config: AppSettings = Depends(get_config),
    factory: ProviderFactory = Depends(get_provider_factory),
    aggregator: ExplanationAggregator = Depends(get_aggregator),
    sanitizer: InputSanitizer = Depends(get_input_sanitizer),
) -> ExplainResponse:
    """Submit a prompt and receive a full multi-technique explanation."""
    # Generate a unique request identifier
    request_id = str(uuid.uuid4())[:8]
    # Sanitize the user's prompt input
    clean_prompt = sanitizer.sanitize_prompt(request.prompt)
    # Validate the provider name
    provider_name = validate_provider_name(request.provider)
    # Validate the technique names
    techniques = validate_technique_names(request.techniques)
    # Create the LLM provider instance
    provider = factory.create(provider_name, config)
    # Build the generation request
    gen_request = GenerationRequest(
        prompt=clean_prompt,
        system_message=request.system_message,
        max_tokens=request.options.max_tokens,
        temperature=request.options.temperature,
        include_logprobs=True,
        include_attention=True,
    )
    # Generate the LLM response with metadata
    response_with_metadata = await provider.generate_with_metadata(gen_request)
    # Run all requested explainability techniques
    aggregated = await aggregator.explain_all(
        prompt=clean_prompt,
        response=response_with_metadata,
        provider=provider,
        techniques=techniques,
    )
    # Convert technique results to API response format
    explanations: dict[str, TechniqueResult] = {}
    for name, result in aggregated.technique_results.items():
        explanations[name] = TechniqueResult(
            success=result.success,
            summary=result.summary,
            confidence=result.confidence,
            data=result.data,
            visualization_data=result.visualization_data,
            error_message=result.error_message,
        )
    # Build the response metadata
    metadata = ResponseMetadata(
        duration_ms=aggregated.total_duration_ms,
        techniques_requested=techniques,
        techniques_completed=list(aggregated.technique_results.keys()),
        prompt_tokens=response_with_metadata.usage.prompt_tokens,
        completion_tokens=response_with_metadata.usage.completion_tokens,
    )
    # Return the complete explanation response
    return ExplainResponse(
        request_id=request_id,
        prompt=clean_prompt,
        llm_response=response_with_metadata.text,
        provider_used=provider.provider_name,
        model_used=provider.model_name,
        summary=aggregated.summary,
        overall_confidence=aggregated.overall_confidence,
        correlations=aggregated.correlations,
        explanations=explanations,
        metadata=metadata,
    )


@router.post("/token-attribution", response_model=ExplainResponse)
async def explain_token_attribution(
    request: SingleTechniqueRequest,
    config: AppSettings = Depends(get_config),
    factory: ProviderFactory = Depends(get_provider_factory),
    aggregator: ExplanationAggregator = Depends(get_aggregator),
    sanitizer: InputSanitizer = Depends(get_input_sanitizer),
) -> ExplainResponse:
    """Run only token attribution analysis on the given prompt."""
    # Delegate to the main explain logic with a single technique
    full_request = ExplainRequest(
        prompt=request.prompt,
        system_message=request.system_message,
        provider=request.provider,
        model=request.model,
        techniques=["token_attribution"],
        options=request.options,
    )
    return await explain_prompt(full_request, config, factory, aggregator, sanitizer)


@router.post("/shap-lime", response_model=ExplainResponse)
async def explain_shap_lime(
    request: SingleTechniqueRequest,
    config: AppSettings = Depends(get_config),
    factory: ProviderFactory = Depends(get_provider_factory),
    aggregator: ExplanationAggregator = Depends(get_aggregator),
    sanitizer: InputSanitizer = Depends(get_input_sanitizer),
) -> ExplainResponse:
    """Run only SHAP/LIME analysis on the given prompt."""
    # Delegate to main explain with shap_lime technique only
    full_request = ExplainRequest(
        prompt=request.prompt,
        system_message=request.system_message,
        provider=request.provider,
        model=request.model,
        techniques=["shap_lime"],
        options=request.options,
    )
    return await explain_prompt(full_request, config, factory, aggregator, sanitizer)


@router.post("/chain-of-thought", response_model=ExplainResponse)
async def explain_chain_of_thought(
    request: SingleTechniqueRequest,
    config: AppSettings = Depends(get_config),
    factory: ProviderFactory = Depends(get_provider_factory),
    aggregator: ExplanationAggregator = Depends(get_aggregator),
    sanitizer: InputSanitizer = Depends(get_input_sanitizer),
) -> ExplainResponse:
    """Run only chain-of-thought analysis on the given prompt."""
    # Delegate to main explain with chain_of_thought technique only
    full_request = ExplainRequest(
        prompt=request.prompt,
        system_message=request.system_message,
        provider=request.provider,
        model=request.model,
        techniques=["chain_of_thought"],
        options=request.options,
    )
    return await explain_prompt(full_request, config, factory, aggregator, sanitizer)
