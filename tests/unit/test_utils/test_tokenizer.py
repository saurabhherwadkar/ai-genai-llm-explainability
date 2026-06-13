# Unit tests for the tokenizer utility
"""Tests for text tokenization and detokenization."""

from __future__ import annotations

import pytest

from llm_explainability.utils.tokenizer import SimpleTokenizer


@pytest.fixture
def tokenizer() -> SimpleTokenizer:
    """Provide a SimpleTokenizer instance."""
    return SimpleTokenizer()


class TestSimpleTokenizer:
    """Tests for the SimpleTokenizer class."""

    def test_basic_tokenization(self, tokenizer: SimpleTokenizer) -> None:
        """Verify basic word tokenization."""
        tokens = tokenizer.tokenize("hello world")
        assert len(tokens) == 2
        assert tokens[0].text == "hello"
        assert tokens[1].text == "world"

    def test_punctuation_split(self, tokenizer: SimpleTokenizer) -> None:
        """Verify punctuation is tokenized separately."""
        tokens = tokenizer.tokenize("Hello, world!")
        token_texts = [t.text for t in tokens]
        assert "," in token_texts
        assert "!" in token_texts

    def test_position_tracking(self, tokenizer: SimpleTokenizer) -> None:
        """Verify token positions are correct."""
        tokens = tokenizer.tokenize("abc def")
        assert tokens[0].start_index == 0
        assert tokens[0].end_index == 3
        assert tokens[1].start_index == 4
        assert tokens[1].end_index == 7

    def test_position_index_sequential(self, tokenizer: SimpleTokenizer) -> None:
        """Verify token position indices are sequential."""
        tokens = tokenizer.tokenize("one two three")
        for i, token in enumerate(tokens):
            assert token.position == i

    def test_empty_string(self, tokenizer: SimpleTokenizer) -> None:
        """Verify empty string returns empty list."""
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_count_tokens(self, tokenizer: SimpleTokenizer) -> None:
        """Verify token counting."""
        count = tokenizer.count_tokens("one two three")
        assert count == 3

    def test_detokenize_basic(self, tokenizer: SimpleTokenizer) -> None:
        """Verify detokenization produces readable text."""
        tokens = tokenizer.tokenize("hello world")
        result = tokenizer.detokenize(tokens)
        assert "hello" in result
        assert "world" in result
