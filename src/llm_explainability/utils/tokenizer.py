# Text tokenization helpers
"""Provides utilities for splitting text into tokens for analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Token:
    """Represents a single token with its text and position information."""

    # The text content of this token
    text: str
    # Starting character index in the original string
    start_index: int
    # Ending character index in the original string (exclusive)
    end_index: int
    # Position index within the token sequence
    position: int


class SimpleTokenizer:
    """Word-level tokenizer for use when provider tokenizers are unavailable."""

    # Pattern that splits on whitespace and punctuation boundaries
    _SPLIT_PATTERN = re.compile(r"(\s+|[.,!?;:\"'()\[\]{}])")

    def tokenize(self, text: str) -> list[Token]:
        """Split text into tokens preserving position information."""
        # List to accumulate token objects
        tokens: list[Token] = []
        # Track current character position in the text
        current_pos = 0
        # Track token sequence position
        token_index = 0
        # Split text using the pattern, keeping delimiters
        parts = self._SPLIT_PATTERN.split(text)
        # Process each part from the split
        for part in parts:
            # Skip empty strings from split
            if not part:
                continue
            # Skip whitespace-only parts
            if part.isspace():
                # Advance position past the whitespace
                current_pos += len(part)
                continue
            # Create a token with position metadata
            token = Token(
                text=part,
                start_index=current_pos,
                end_index=current_pos + len(part),
                position=token_index,
            )
            # Add the token to our results
            tokens.append(token)
            # Advance position past this token
            current_pos += len(part)
            # Increment the sequence index
            token_index += 1
        # Return the complete list of tokens
        return tokens

    def detokenize(self, tokens: list[Token]) -> str:
        """Reconstruct text from a list of tokens with proper spacing."""
        # Handle empty token list
        if not tokens:
            return ""
        # Build the output by joining tokens with spaces
        parts: list[str] = []
        for i, token in enumerate(tokens):
            # Add the token text
            parts.append(token.text)
            # Add space between tokens unless next is punctuation
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]
                # Skip space before punctuation marks
                if not re.match(r"^[.,!?;:\"')\]}]$", next_token.text):
                    parts.append(" ")
        # Join all parts into the final string
        return "".join(parts)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in the given text."""
        # Tokenize and return the count
        return len(self.tokenize(text))
