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


def test_notebook_has_executable_evidence_for_every_finding() -> None:
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
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 13
    source = "".join(code_cells[0]["source"])
    assert "Path.cwd().parents" in source
    assert sum("Expected evidence:" in "".join(cell["source"]) for cell in notebook["cells"]) == 12
