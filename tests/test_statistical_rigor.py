import numpy as np
import pandas as pd

from calibration.statistical_rigor import (
    bootstrap_aggregate_ratio,
    paired_cluster_bootstrap,
    percentile_interval,
)


def test_percentile_interval_contains_constant() -> None:
    result = percentile_interval(np.ones(100))
    assert result["lower"] == 1.0
    assert result["upper"] == 1.0
    assert result["probability_below_zero"] == 0.0


def test_paired_bootstrap_preserves_cluster_rows() -> None:
    frame = pd.DataFrame(
        {"Policy": [1, 1, 2], "Actual": [1.0, 2.0, 3.0], "A": 1.0, "B": 2.0}
    )
    values = paired_cluster_bootstrap(
        frame,
        "Policy",
        lambda sample: sample["A"].mean(),
        lambda sample: sample["B"].mean(),
        repetitions=20,
    )
    assert np.all(values == 1.0)


def test_aggregate_ratio_bootstrap_is_exact_for_balanced_data() -> None:
    frame = pd.DataFrame(
        {"Policy": [1, 2], "Actual": [10.0, 20.0], "Predicted": [10.0, 20.0]}
    )
    ratios = bootstrap_aggregate_ratio(
        frame, "Policy", "Actual", "Predicted", repetitions=20
    )
    assert np.all(ratios == 1.0)
