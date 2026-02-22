"""Tests for scoring.hard_constraint."""

import pytest

from scoring.hard_constraint import contains_forbidden_word


class TestContainsForbiddenWord:
    """Test strict substring matching (default)."""

    def test_exact_lowercase(self):
        assert contains_forbidden_word("I like apple pie") is True

    def test_exact_uppercase(self):
        assert contains_forbidden_word("Apple is great") is True

    def test_mixed_case(self):
        assert contains_forbidden_word("aPpLe") is True

    def test_pineapple_strict(self):
        """With strict_substring=True (default), 'pineapple' triggers."""
        assert contains_forbidden_word("I like pineapple") is True

    def test_pineapple_word_boundary(self):
        """With strict_substring=False, 'pineapple' does NOT trigger."""
        assert (
            contains_forbidden_word("I like pineapple", strict_substring=False)
            is False
        )

    def test_no_match(self):
        assert contains_forbidden_word("I like oranges and bananas") is False

    def test_empty_string(self):
        assert contains_forbidden_word("") is False

    def test_word_embedded_in_sentence(self):
        assert contains_forbidden_word("The apple fell from the tree") is True

    def test_custom_forbidden_word(self):
        assert contains_forbidden_word("Hello world", word="hello") is True
        assert contains_forbidden_word("Hello world", word="goodbye") is False

    def test_near_miss(self):
        """'a**le' should not trigger (it's not literally 'apple')."""
        assert contains_forbidden_word("a**le") is False

    def test_plural(self):
        """'apples' contains 'apple' as a substring."""
        assert contains_forbidden_word("I ate three apples") is True

    def test_word_boundary_exact(self):
        """With strict_substring=False, exact 'apple' still triggers."""
        assert (
            contains_forbidden_word("I ate an apple", strict_substring=False) is True
        )
