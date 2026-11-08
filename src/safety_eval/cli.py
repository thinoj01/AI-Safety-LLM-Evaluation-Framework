"""Run exploratory evaluation against pre-generated model outputs."""
import argparse
import json
from pathlib import Path
from .core import Case, evaluate

def run(cases_path: Path, responses_path: Path):
    raw = json.loads(cases_path.read_text(encoding="utf-8"))
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not isinstance(responses, dict):
        raise ValueError("Cases must be a list; responses must be an ID-to-text object")
    cases = [Case.from_dict(item) for item in raw]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    if any(not isinstance(r, str) for r in responses.values()):
        raise ValueError("Responses must be strings")
    return evaluate(cases, responses)

def main():
    parser = argparse.ArgumentParser(description="Toy LLM behavior evaluation baseline")
    parser.add_argument("--cases", type=Path, default=Path("data/cases.json"))
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    report = run(args.cases, args.responses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Scored {report['total']} cases, accuracy={report['accuracy']}, unknown={report['unknown']}")

if __name__ == "__main__":
    main()
