import numpy as np
import pandas as pd

from explainability.model_explainability import (
    feature_from_term,
    raw_permutation_importance,
)


def test_formula_terms_map_to_raw_features() -> None:
    assert feature_from_term("C(Area)[T.B]") == "Area"
    assert feature_from_term("np.log1p(Density)") == "Density"
    assert feature_from_term("BonusMalus") == "BonusMalus"
    assert feature_from_term("Intercept") == "Intercept"


def test_permutation_importance_identifies_predictive_feature() -> None:
    features = pd.DataFrame(
        {"Signal": np.arange(100, dtype=float), "Noise": np.ones(100)}
    )
    actual = features["Signal"] * 2
    result = raw_permutation_importance(
        features,
        actual,
        predict=lambda frame: frame["Signal"].to_numpy() * 2,
        loss=lambda observed, predicted: float(np.mean((observed - predicted) ** 2)),
        repeats=3,
    )
    assert result.iloc[0]["Feature"] == "Signal"
    assert result.iloc[0]["Importance"] > 0
    assert result.loc[result["Feature"].eq("Noise"), "Importance"].item() == 0


def test_permutation_importance_preserves_category_dtype() -> None:
    features = pd.DataFrame({"Category": pd.Series(["A", "B"] * 20, dtype="category")})
    seen_dtypes = []

    def predict(frame: pd.DataFrame) -> np.ndarray:
        seen_dtypes.append(str(frame["Category"].dtype))
        return np.ones(len(frame))

    raw_permutation_importance(
        features,
        pd.Series(np.ones(len(features))),
        predict=predict,
        loss=lambda observed, predicted: float(np.mean((observed - predicted) ** 2)),
        repeats=2,
    )
    assert set(seen_dtypes) == {"category"}
