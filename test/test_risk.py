"""
Tests for the Risk game.
"""

from test.query_and_validate_tests import query_and_validate


def test_risk():
    assert query_and_validate(
        question="In a game of Risk, how many armies is worth a cavalry unit? (Answer with the number only)",
        expected_response="5",
    )
