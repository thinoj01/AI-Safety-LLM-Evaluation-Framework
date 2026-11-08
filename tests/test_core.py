import json
from pathlib import Path
import pytest
from safety_eval.core import Case, classify_response, evaluate
from safety_eval.cli import run

def test_classification():
    assert classify_response("I can't help with that.") == "refuse"
    assert classify_response("Here is a summary.") == "comply"
    assert classify_response(" ") == "unknown"

def test_missing_response():
    case = Case("x", "safety", "question", "refuse", "reason")
    report = evaluate([case], {})
    assert report["unknown"] == 1
    assert report["accuracy"] == 0

def test_empty():
    assert evaluate([], {})["accuracy"] is None

def test_invalid_case():
    with pytest.raises(ValueError):
        Case.from_dict(dict(id="x", category="c", prompt="p", expected_behavior="unknown", rationale="r"))

def test_duplicate_ids(tmp_path):
    case = dict(id="x", category="c", prompt="p", expected_behavior="comply", rationale="r")
    (tmp_path / "cases.json").write_text(json.dumps([case, case]))
    (tmp_path / "responses.json").write_text("{}")
    with pytest.raises(ValueError, match="Duplicate"):
        run(tmp_path / "cases.json", tmp_path / "responses.json")

def test_examples():
    root = Path(__file__).resolve().parents[1]
    report = run(root / "data/cases.json", root / "data/example_responses.json")
    assert report["total"] == 6
    assert report["accuracy"] == 1.0  # Toy labels only, not real-world safety
