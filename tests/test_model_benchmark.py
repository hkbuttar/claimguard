from integration.model_benchmark import (
    closer_to_one_winner,
    lower_is_better_winner,
)


def test_lower_is_better_selects_material_advantage() -> None:
    assert lower_is_better_winner(1.0, 0.8) == "ML"
    assert lower_is_better_winner(0.8, 1.0) == "Traditional"


def test_negligible_difference_is_comparable() -> None:
    assert lower_is_better_winner(1.0, 1.0005) == "Comparable"


def test_calibration_winner_is_closer_to_one() -> None:
    assert closer_to_one_winner(1.01, 0.8) == "Traditional"
    assert closer_to_one_winner(1.2, 0.99) == "ML"
