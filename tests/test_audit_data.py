import pandas as pd

from preprocessing.audit_data import audit_tables


def test_audit_classifies_and_reconciles_issues() -> None:
    freq = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "ClaimNb": [1, 0, 0],
            "Exposure": [1.0, 1.2, 0.0],
            "VehAge": [5, 81, 2],
            "DrivAge": [40, 96, 30],
            "BonusMalus": [50, 151, 50],
        }
    )
    sev = pd.DataFrame(
        {"IDpol": [1, 1, 99], "ClaimAmount": [100.0, 100.0, 0.0]}
    )

    results, freq_flags, sev_flags = audit_tables(freq, sev)
    counts = {(result.dataset, result.rule): result.count for result in results}

    assert counts[("freMTPL2freq", "exposure_above_one")] == 1
    assert counts[("freMTPL2freq", "nonpositive_exposure")] == 1
    assert counts[("freMTPL2sev", "unmatched_policy")] == 1
    assert counts[("freMTPL2sev", "possible_duplicate_claim")] == 2
    assert counts[("cross_table", "claim_count_mismatch")] == 1
    assert freq_flags.loc[1, "driver_age_extreme"]
    assert freq_flags.loc[1, "disposition"] == "ambiguous"
    assert freq_flags.loc[2, "disposition"] == "excluded"
    assert sev_flags.loc[2, "nonpositive_claim_amount"]
    assert sev_flags.loc[2, "disposition"] == "excluded"


def test_clean_data_has_no_exclusion_flags() -> None:
    freq = pd.DataFrame(
        {
            "IDpol": [1],
            "ClaimNb": [1],
            "Exposure": [1.0],
            "VehAge": [5],
            "DrivAge": [40],
            "BonusMalus": [50],
        }
    )
    sev = pd.DataFrame({"IDpol": [1], "ClaimAmount": [100.0]})
    results, freq_flags, sev_flags = audit_tables(freq, sev)
    exclusions = [r for r in results if r.classification == "excluded"]
    assert all(result.count == 0 for result in exclusions)
    assert freq_flags.loc[0, "disposition"] == "valid"
    assert sev_flags.loc[0, "disposition"] == "valid"
