# HuggingFace Transformers local model provider implementation
"""Implements the BaseLLMProvider interface for locally-loaded HuggingFace models."""

from __future__ import annotations

import time
from typing import Any

from llm_explainability.config.settings import HuggingFaceProviderConfig
from llm_explainability.exceptions import (
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import (
    AttentionWeights,
    GenerationRequest,
    GenerationResponse,
    GenerationResponseWithMetadata,
    ProviderFeature,
    ProviderHealthStatus,
    TokenInfo,
    TokenLogProbabilities,
    UsageStats,
)


class HuggingFaceProvider(BaseLLMProvider):
    """LLM provider for locally-loaded HuggingFace transformer models."""

    # Full set of features available with local model access
    _SUPPORTED_FEATURES = frozenset({
        ProviderFeature.LOGPROBS,
        ProviderFeature.ATTENTION_WEIGHTS,
        ProviderFeature.GRADIENT_ATTRIBUTION,
        ProviderFeature.EMBEDDINGS,
        ProviderFeature.SYSTEM_MESSAGE,
    })

    def __init__(self, config: HuggingFaceProviderConfig) -> None:
        """Initialize the HuggingFace provider with its configuration."""
        # Store the provider configuration
        self._config = config
        # Model and tokenizer are loaded lazily on first use
        self._model: Any = None
        self._tokenizer: Any = None
        # Track whether the model has been loaded
        self._is_loaded = False

    @property
    def provider_name(self) -> str:
        """Return the provider identifier string."""
        return "huggingface"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._config.model_name

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a text completion using the locally-loaded model."""
        # Ensure the model is loaded before generation
        self._ensure_model_loaded()
        try:
            # Import torch here to keep it optional at module level
            import torch

            # Build the input prompt with optional system message
            full_prompt = self._build_prompt(request)
            # Tokenize the input prompt
            inputs = self._tokenizer(
                full_prompt, return_tensors="pt", truncation=True
            )
            # Move inputs to the model's device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            # Record input length for usage stats
            input_length = inputs["input_ids"].shape[1]
            # Generate output tokens without gradient computation
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=max(request.temperature, 0.01),
                    do_sample=request.temperature > 0,
                )
            # Decode only the newly generated tokens
            generated_ids = outputs[0][input_length:]
            generated_text = self._tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            # Calculate token usage
            output_length = len(generated_ids)
            # Return the generation response
            return GenerationResponse(
                text=generated_text,
                finish_reason="stop",
                usage=UsageStats(
                    prompt_tokens=input_length,
                    completion_tokens=output_length,
                    total_tokens=input_length + output_length,
                ),
                model=self._config.model_name,
            )
        except Exception as exc:
            # Handle generation errors
            raise ProviderResponseError(
                f"Generation failed: {str(exc)}", "huggingface", details=str(exc)
            ) from exc

    async def generate_with_metadata(
        self, request: GenerationRequest
    ) -> GenerationResponseWithMetadata:
        """Generate with full metadata including attention weights and logprobs."""
        # Ensure the model is loaded before generation
        self._ensure_model_loaded()
        try:
            # Import torch for tensor operations
            import torch

            # Build the full prompt text
            full_prompt = self._build_prompt(request)
            # Tokenize the input
            inputs = self._tokenizer(
                full_prompt, return_tensors="pt", truncation=True
            )
            # Move to model device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            # Record input length
            input_length = inputs["input_ids"].shape[1]
            # Generate with attention weights and scores
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=max(request.temperature, 0.01),
                    do_sample=request.temperature > 0,
                    output_attentions=request.include_attention,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            # Decode the generated tokens
            generated_ids = outputs.sequences[0][input_length:]
            generated_text = self._tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            # Extract log probabilities from generation scores
            token_logprobs = self._extract_logprobs_from_scores(outputs, generated_ids)
            # Extract attention weights if requested
            attention_data = None
            if request.include_attention and hasattr(outputs, "attentions"):
                attention_data = self._extract_attention_weights(outputs, inputs)
            # Build token info from the generated tokens
            token_info = self._build_token_info(generated_ids, token_logprobs)
            # Calculate usage statistics
            output_length = len(generated_ids)
            # Return the full metadata response
            return GenerationResponseWithMetadata(
                text=generated_text,
                finish_reason="stop",
                usage=UsageStats(
                    prompt_tokens=input_length,
                    completion_tokens=output_length,
                    total_tokens=input_length + output_length,
                ),
                model=self._config.model_name,
                token_logprobs=token_logprobs,
                attention_weights=attention_data,
                token_info=token_info,
                raw_response={},
            )
        except Exception as exc:
            raise ProviderResponseError(
                f"Generation with metadata failed: {str(exc)}",
                "huggingface",
                details=str(exc),
            ) from exc

    async def get_token_logprobs(
        self, request: GenerationRequest
    ) -> TokenLogProbabilities:
        """Retrieve per-token log probabilities from the local model."""
        # Generate with metadata to get logprob information
        request.include_logprobs = True
        response = await self.generate_with_metadata(request)
        # Return the extracted logprobs
        return response.token_logprobs or TokenLogProbabilities()

    async def health_check(self) -> ProviderHealthStatus:
        """Check if the model can be loaded and used for inference."""
        # Record start time for latency
        start_time = time.perf_counter()
        try:
            # Attempt to load the model if not already loaded
            self._ensure_model_loaded()
            # Calculate load/check latency
            latency = (time.perf_counter() - start_time) * 1000
            # Return healthy status
            return ProviderHealthStatus(
                is_healthy=True,
                latency_ms=latency,
                provider_name="huggingface",
                model_name=self._config.model_name,
            )
        except Exception as exc:
            # Calculate latency for failure case
            latency = (time.perf_counter() - start_time) * 1000
            # Return unhealthy status
            return ProviderHealthStatus(
                is_healthy=False,
                latency_ms=latency,
                provider_name="huggingface",
                model_name=self._config.model_name,
                error_message=str(exc),
            )

    def supports_feature(self, feature: ProviderFeature) -> bool:
        """Check if this provider supports the given feature."""
        # Local models support the most features
        return feature in self._SUPPORTED_FEATURES

    def _ensure_model_loaded(self) -> None:
        """Load the model and tokenizer if not already loaded."""
        # Skip if already loaded
        if self._is_loaded:
            return
        try:
            # Import transformers for model loading
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load the tokenizer from the model repository
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_name, trust_remote_code=True
            )
            # Configure model loading kwargs based on settings
            model_kwargs: dict[str, Any] = {
                "device_map": self._config.device,
                "trust_remote_code": True,
            }
            # Add 8-bit quantization if configured
            if self._config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            # Load the model with configured options
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_name, **model_kwargs
            )
            # Mark as loaded to prevent repeated loading
            self._is_loaded = True
        except Exception as exc:
            # Raise connection error if model loading fails
            raise ProviderConnectionError(
                f"Failed to load model: {str(exc)}",
                "huggingface",
                details=str(exc),
            ) from exc

    def _build_prompt(self, request: GenerationRequest) -> str:
        """Build the full prompt string including optional system message."""
        # Start with empty parts list
        parts: list[str] = []
        # Add system message if provided
        if request.system_message:
            parts.append(f"System: {request.system_message}\n")
        # Add the user prompt
        parts.append(f"User: {request.prompt}\nAssistant:")
        # Join and return the full prompt
        return "".join(parts)

    def _extract_logprobs_from_scores(
        self, outputs: Any, generated_ids: Any
    ) -> TokenLogProbabilities:
        """Extract log probabilities from generation scores."""
        import torch

        # Check if scores are available in the output
        if not hasattr(outputs, "scores") or not outputs.scores:
            return TokenLogProbabilities()
        # Initialize result lists
        tokens: list[str] = []
        logprobs: list[float] = []
        top_alternatives: list[dict[str, float]] = []
        # Process each generation step's scores
        for i, score_tensor in enumerate(outputs.scores):
            # Apply log softmax to get log probabilities
            log_probs = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
            # Get the token that was actually generated
            token_id = generated_ids[i].item()
            token_text = self._tokenizer.decode([token_id])
            # Get the logprob of the generated token
            token_logprob = log_probs[token_id].item()
            # Get top-k alternatives
            top_k_values, top_k_indices = torch.topk(log_probs, k=5)
            alternatives: dict[str, float] = {}
            for val, idx in zip(top_k_values.tolist(), top_k_indices.tolist()):
                alt_token = self._tokenizer.decode([idx])
                alternatives[alt_token] = val
            # Append to result lists
            tokens.append(token_text)
            logprobs.append(token_logprob)
            top_alternatives.append(alternatives)
        # Return structured logprob data
        return TokenLogProbabilities(
            tokens=tokens,
            logprobs=logprobs,
            top_alternatives=top_alternatives,
        )

    def _extract_attention_weights(
        self, outputs: Any, inputs: dict[str, Any]
    ) -> AttentionWeights | None:
        """Extract attention weight matrices from model outputs."""
        # Check if attention data is available
        if not hasattr(outputs, "attentions") or not outputs.attentions:
            return None
        # Get token labels for the attention matrix
        input_ids = inputs["input_ids"][0]
        token_labels = [
            self._tokenizer.decode([tid]) for tid in input_ids.tolist()
        ]
        # Extract attention from the first generation step only (memory efficiency)
        first_step_attentions = outputs.attentions[0]
        num_layers = len(first_step_attentions)
        num_heads = first_step_attentions[0].shape[1]
        # Convert to nested lists (only first layer for memory efficiency)
        layer_weights: list[list[list[list[float]]]] = []
        for layer_attention in first_step_attentions[:4]:  # Limit to 4 layers
            # Convert tensor to nested Python lists
            layer_data = layer_attention[0].cpu().tolist()
            layer_weights.append(layer_data)
        # Return structured attention data
        return AttentionWeights(
            layer_weights=layer_weights,
            num_heads=num_heads,
            num_layers=num_layers,
            token_labels=token_labels,
        )

    def _build_token_info(
        self, generated_ids: Any, logprobs: TokenLogProbabilities
    ) -> list[TokenInfo]:
        """Build token info list from generated IDs and logprobs."""
        # Initialize the token info list
        token_info_list: list[TokenInfo] = []
        # Iterate over generated tokens
        for i, token_id in enumerate(generated_ids.tolist()):
            # Get the logprob for this position if available
            logprob = logprobs.logprobs[i] if i < len(logprobs.logprobs) else None
            # Get alternatives if available
            alternatives = (
                logprobs.top_alternatives[i]
                if i < len(logprobs.top_alternatives)
                else {}
            )
            # Create the token info object
            info = TokenInfo(
                text=self._tokenizer.decode([token_id]),
                token_id=token_id,
                logprob=logprob,
                top_alternatives=alternatives,
            )
            token_info_list.append(info)
        # Return the complete list
        return token_info_list
