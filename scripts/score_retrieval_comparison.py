"""Prepare diagnosis review and score error-level P/R/F1 after human adjudication.

This offline entry point never calls a model or infers correctness from position,
candidate count, or whether an answer has any error. One diagnostic can match at
most one adjudicated error; duplicate diagnoses count as false positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare(inputs_path, references_path, checker_run, arms):
    rows, references = read(inputs_path), read(references_path)
    ids = [str(r["id"]) for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Expected nonempty, unique input IDs")
    reference_map = {str(r["id"]): r for r in references}
    if len(reference_map) != len(references) or set(reference_map) != set(ids):
        raise ValueError("Reference IDs must match the entire frozen input set")
    run = Path(checker_run)
    manifest = read(run / "manifest.json")
    if manifest["input_sha256"] != sha(inputs_path):
        raise ValueError("Checker input hash mismatch")
    cases = []
    for row in rows:
        sid = str(row["id"])
        reference = reference_map[sid]
        case = {
            "id": sid, "question": row["question"], "prediction": row["prediction"],
            "reference_answer": reference.get("answer"),
            "reference_claims": [e["source_gt"] for e in reference["errors"]],
            # Remove false/duplicate claims, correct text and add missing real
            # errors here. IDs identify distinct physical root causes.
            "adjudicated_errors": [], "reference_review_complete": False,
            "reference_review_notes": "", "arms": {},
        }
        for arm in arms:
            key = hashlib.sha256(sid.encode()).hexdigest()[:16]
            path = run / f"arm{arm}" / key / "result.json"
            result = read(path) if path.exists() else None
            if result is not None:
                if (str(result.get("id")), result.get("arm")) != (sid, arm):
                    raise ValueError("Checker result identity mismatch")
                if result.get("configuration_sha256") != manifest["configuration_sha256"]:
                    raise ValueError("Checker result configuration mismatch")
                if result.get("model") != manifest["model"]:
                    raise ValueError("Checker model mismatch")
            status = result.get("status") if result else "missing"
            diagnostics = result.get("trace", {}).get("diagnostics") if result else None
            if status == "completed" and not isinstance(diagnostics, list):
                raise ValueError("Completed checker result lacks diagnostics")
            case["arms"][str(arm)] = {
                "status": status, "result_sha256": sha(path) if path.exists() else None,
                "diagnostics": [
                    {"diagnostic_id": f"d{i+1}", "diagnostic": d,
                     "review_complete": False, "matched_error_ids": [], "rationale": ""}
                    for i, d in enumerate(diagnostics or [])
                ],
            }
        cases.append(case)
    return {
        "schema_version": 1, "input_sha256": sha(inputs_path),
        "reference_sha256": sha(references_path),
        "checker_manifest_sha256": sha(run / "manifest.json"),
        "arms": arms, "reviewer_kind": "human", "reviewer": "",
        "note": "Operator-side diagnosis adjudication; arm labels are visible. Not a blind review or model-generated gold.",
        "cases": cases,
    }


def match_count(edges):
    """Maximum one-to-one matching, independent of candidate order."""
    owner = {}

    def augment(index, seen):
        for error in edges[index]:
            if error in seen:
                continue
            seen.add(error)
            if error not in owner or augment(owner[error], seen):
                owner[error] = index
                return True
        return False

    for index in range(len(edges)):
        augment(index, set())
    return len(owner)


def score(review, expected):
    for key in ("schema_version", "input_sha256", "reference_sha256", "checker_manifest_sha256", "arms"):
        if review.get(key) != expected[key]:
            raise ValueError("Review provenance mismatch: " + key)
    if review.get("reviewer_kind") != "human" or not str(review.get("reviewer", "")).strip():
        raise ValueError("Named human adjudication is required for physical metrics")
    cases = review.get("cases", [])
    if len(cases) != len(expected["cases"]):
        raise ValueError("Review must include every frozen case")
    totals = {str(a): {"tp": 0, "fp": 0, "fn": 0, "completed_cases": 0, "cases_with_valid_detection": 0} for a in review["arms"]}
    for case, original in zip(cases, expected["cases"]):
        for field in ("id", "question", "prediction", "reference_answer", "reference_claims"):
            if case.get(field) != original[field]:
                raise ValueError("Frozen review input changed: " + field)
        if case.get("reference_review_complete") is not True or not case.get("reference_review_notes", "").strip():
            raise ValueError("Reference review is incomplete")
        errors = case.get("adjudicated_errors", [])
        if not isinstance(errors, list) or any(not isinstance(e, dict) or not isinstance(e.get("error_id"), str) or not e["error_id"].strip() or not isinstance(e.get("error_text"), str) or not e["error_text"].strip() for e in errors):
            raise ValueError("Each adjudicated error needs an ID and physical description")
        error_ids = {e["error_id"] for e in errors}
        if len(error_ids) != len(errors):
            raise ValueError("Duplicate adjudicated error IDs")
        if set(case["arms"]) != set(original["arms"]):
            raise ValueError("Review arms changed")
        for arm, original_arm in original["arms"].items():
            reviewed_arm = case["arms"][arm]
            if original_arm["status"] != "completed":
                raise ValueError("Incomplete checker jobs: no full-batch physical score")
            if any(reviewed_arm.get(k) != original_arm[k] for k in ("status", "result_sha256")):
                raise ValueError("Checker evidence changed")
            diagnostics = reviewed_arm["diagnostics"]
            if len(diagnostics) != len(original_arm["diagnostics"]):
                raise ValueError("Diagnostics added or removed")
            edges = []
            for d, original_d in zip(diagnostics, original_arm["diagnostics"]):
                if any(d.get(k) != original_d[k] for k in ("diagnostic_id", "diagnostic")):
                    raise ValueError("Frozen diagnostic changed")
                if d.get("review_complete") is not True or not d.get("rationale", "").strip():
                    raise ValueError("Diagnosis review is incomplete")
                matches = d.get("matched_error_ids")
                if not isinstance(matches, list) or any(not isinstance(e, str) for e in matches) or len(matches) != len(set(matches)) or not set(matches) <= error_ids:
                    raise ValueError("Invalid matched error IDs")
                edges.append(matches)
            tp = match_count(edges)
            total = totals[arm]
            total["tp"] += tp
            total["fp"] += len(diagnostics) - tp
            total["fn"] += len(errors) - tp
            total["completed_cases"] += 1
            total["cases_with_valid_detection"] += int(tp > 0)
    for total in totals.values():
        tp, fp, fn = (total[k] for k in ("tp", "fp", "fn"))
        total["precision"] = tp / (tp + fp) if tp + fp else None
        total["recall"] = tp / (tp + fn) if tp + fn else None
        total["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"metric": "human_adjudicated_error_level_micro", "cases": len(cases),
            "reviewer": review["reviewer"], "arms": totals,
            "note": "Null means an undefined denominator. These are diagnosis metrics, not retrieval precision or question-level accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "score"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-review", type=Path, required=True)
    parser.add_argument("--checker-run", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("CONDA_PREFIX"):
        parser.error("Run in the project conda environment")
    if args.mode == "score" and args.review is None:
        parser.error("--review is required for scoring")
    expected = prepare(args.input, args.reference_review, args.checker_run, sorted(set(args.arms)))
    result = expected if args.mode == "prepare" else score(read(args.review), expected)
    if args.mode == "score":
        result["review_sha256"] = sha(args.review)
        result["checker_manifest_sha256"] = expected["checker_manifest_sha256"]
        result["input_sha256"] = expected["input_sha256"]
    write_new(args.output, result)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
