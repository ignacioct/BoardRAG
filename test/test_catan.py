"""
Tests for the Catan game.
"""

from test.query_and_validate_tests import query_and_validate


def test_catan():
    assert query_and_validate(
        question="In a game of Catan, how many types of trade can a player make in a turn? (Answer with the number only)",
        expected_response="2",
    )
