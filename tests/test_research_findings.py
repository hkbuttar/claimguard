import pytest

from integration.research_findings import make_notebook, validate_findings


def test_findings_require_all_questions() -> None:
    with pytest.raises(ValueError, match="twelve"):
        validate_findings([])


def test_findings_reject_unknown_classification() -> None:
    findings = [
        {
            "Question": str(index),
            "Classification": "Unknown" if index == 0 else "Robust",
            "Finding": "Finding",
            "Evidence": "Evidence",
        }
        for index in range(12)
    ]
    with pytest.raises(ValueError, match="Invalid"):
        validate_findings(findings)


def test_notebook_has_valid_minimal_structure() -> None:
    findings = [
        {
            "Question": str(index),
            "Classification": "Robust",
            "Finding": "Finding",
            "Evidence": "Evidence",
        }
        for index in range(12)
    ]
    notebook = make_notebook(findings)
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 3
    assert notebook["cells"][-1]["cell_type"] == "code"
