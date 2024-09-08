"""
Tests for the Dixit game.
"""

from test.query_and_validate_tests import query_and_validate


def test_dixit():
    assert query_and_validate(
        question="How many cards have the players of Dixit in their hands? (Answer with the number only)",
        expected_response="6",
    )
