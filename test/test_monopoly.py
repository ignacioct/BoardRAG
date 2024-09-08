"""
Test for the Dixit game.
"""

from test.query_and_validate_tests import query_and_validate


def test_dixit_rules():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )
