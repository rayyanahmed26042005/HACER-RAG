#!/usr/bin/env python3
"""Evaluate JSON reports collected from physical phone runs.

This script consumes reports written by run_device_inference.py and a labeled
query file. It computes deterministic retrieval/answer metrics and summarizes
latency by requested RAG policy. It never converts synthetic records into
physical evidence and never replaces missing labels with quality claims.
"""

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List

from evaluate_rag_offline import evaluate_rag_turn


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_query(value: str) -> str:
    return " ".join((value or "").lower().split())


def load_labels(path: str) -> Dict[str, Dict[str, Any]]:
    labels = load_json(path)
    return {normalized_query(item["query"]): item for item in labels if item.get("query")}


def report_files(runs_dir: str) -> Iterable[str]:
    if not os.path.isdir(runs_dir):
        return []
    return sorted(
        os.path.join(runs_dir, name)
        for name in os.listdir(runs_dir)
        if name.endswith(".json")
    )


def numeric(values: List[Any]) -> List[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def median_or_none(values: List[Any]):
    converted = numeric(values)
    return round(statistics.median(converted), 4) if converted else None


def mean_or_none(values: List[Any]):
    converted = numeric(values)
    return round(statistics.mean(converted), 4) if converted else None


def evaluate_report(report: Dict[str, Any], label: Dict[str, Any]) -> Dict[str, Any]:
    rag = report.get("rag") or {}
    context = rag.get("context", "")
    retrieved_ids = rag.get("selectedChunkIds", report.get("rag_selected_chunk_ids", []))
    response = report.get("response_text", report.get("answer", ""))
    requested_policy = report.get("requested_rag_policy")
    nested_policy = (report.get("research_metrics") or {}).get("policy")
    strategy = rag.get("strategy")
    policy = nested_policy or requested_policy or strategy or "unknown"
    telemetry = {
        "decode_tok_per_sec": report.get("tokens_per_second"),
        "prefill_tok_per_sec": report.get("prefill_tokens_per_second"),
        "ttft_ms": report.get("ttft_ms"),
        "total_ms": report.get("total_warm_latency_ms"),
        "prompt_tokens": report.get("prompt_tokens"),
        "gen_tokens": report.get("generated_tokens"),
        "mode": report.get("controller_mode"),
        "bottleneck": report.get("primary_bottleneck"),
        "rag_policy": policy,
    }
    result = evaluate_rag_turn(
        query_id=label.get("id", report.get("run_id", "unknown")),
        query=label["query"],
        ground_truth=label.get("ground_truth", ""),
        expected_action=label.get("expected_action", "answer"),
        retrieved_context=context,
        model_response=response,
        telemetry=telemetry,
    )
    return {
        "run_id": report.get("run_id"),
        "source_path": report.get("source_path"),
        "request_id": report.get("request_id"),
        "query_id": label.get("id"),
        "query": label["query"],
        "chat_mode": report.get("chat_mode"),
        "policy": policy,
        "requested_policy": requested_policy,
        "nested_policy": nested_policy,
        "rag_strategy": strategy,
        "evidence_status": report.get("evidence_status"),
        "answer_contract_status": (report.get("rag_contract_audit") or {}).get("status"),
        "execution_path": report.get("execution_path"),
        "benchmark": result,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    policies = sorted({row["policy"] for row in rows})
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        benchmark_rows = [row["benchmark"] for row in selected]
        telemetry = [row.get("benchmark", {}).get("telemetry", {}) for row in selected]
        def metric(name: str) -> List[Any]:
            return [item.get(name) for item in benchmark_rows]
        output[policy] = {
            "run_count": len(selected),
            "median_ttft_ms": median_or_none([item.get("ttft_ms") for item in telemetry]),
            "median_total_ms": median_or_none([item.get("total_ms") for item in telemetry]),
            "mean_context_recall": mean_or_none(metric("context_recall")),
            "mean_context_precision": mean_or_none(metric("context_precision")),
            "mean_token_f1": mean_or_none(metric("token_f1")),
            "mean_faithfulness": mean_or_none(metric("faithfulness")),
            "mean_refusal_accuracy": mean_or_none(metric("refusal_accuracy")),
            "contract_status_counts": {
                status: sum(1 for item in selected if item.get("answer_contract_status") == status)
                for status in sorted({item.get("answer_contract_status") for item in selected if item.get("answer_contract_status")})
            },
            "execution_path_counts": {
                path: sum(1 for item in selected if item.get("execution_path") == path)
                for path in sorted({item.get("execution_path") for item in selected if item.get("execution_path")})
            },
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate physical phone RAG reports")
    parser.add_argument("--runs-dir", default=os.path.join("benchmark", "results", "physical_runs"))
    parser.add_argument("--labels", default=os.path.join("benchmark", "zephyr_evaluation_queries.json"))
    parser.add_argument("--output", default=os.path.join("benchmark", "results", "physical_run_evaluation.json"))
    parser.add_argument("--strict", action="store_true",
                        help="Require request correlation, physical status, document mode, and explicit policy")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    rows: List[Dict[str, Any]] = []
    skipped = []
    for path in report_files(args.runs_dir):
        report = load_json(path)
        query = normalized_query(report.get("query", ""))
        label = labels.get(query)
        if not label:
            skipped.append({"path": path, "reason": "no_matching_query_label"})
            continue
        if args.strict and report.get("evidence_status") != "PHYSICAL_DEVICE":
            skipped.append({"path": path, "reason": "missing_physical_evidence_status"})
            continue
        if not args.strict and report.get("evidence_status") not in (None, "PHYSICAL_DEVICE"):
            skipped.append({"path": path, "reason": "not_physical_device"})
            continue
        if args.strict:
            required = ["request_id", "query_id", "chat_mode", "execution_path"]
            missing = [field for field in required if not report.get(field)]
            if missing:
                skipped.append({"path": path, "reason": "missing_correlation_fields", "fields": missing})
                continue
            if report.get("chat_mode") != "document":
                skipped.append({"path": path, "reason": "not_document_rag_run"})
                continue
            if report.get("query_id") != label.get("id"):
                skipped.append({"path": path, "reason": "query_id_mismatch",
                                "reported": report.get("query_id"), "expected": label.get("id")})
                continue
        requested_policy = report.get("requested_rag_policy")
        nested_policy = (report.get("research_metrics") or {}).get("policy")
        strategy = (report.get("rag") or {}).get("strategy")
        if requested_policy and nested_policy and requested_policy != nested_policy:
            skipped.append({"path": path, "reason": "policy_mismatch",
                            "requested": requested_policy, "reported": nested_policy})
            continue
        expected_strategy = {"ebmrag-v2": "ebmrag-v2", "mobcert-v1": "mobcert-v1"}.get(requested_policy or nested_policy)
        if expected_strategy and strategy and strategy != expected_strategy:
            skipped.append({"path": path, "reason": "strategy_mismatch",
                            "policy": requested_policy or nested_policy, "strategy": strategy})
            continue
        report["source_path"] = os.path.relpath(path, os.getcwd())
        rows.append(evaluate_report(report, label))

    result = {
        "evaluator": "physical-rag-v2-strict" if args.strict else "physical-rag-v1",
        "claims": {
            "physicalLatency": True,
            "labeledAnswerQuality": True,
            "syntheticTiming": False,
            "strictCorrelation": args.strict,
        },
        "runDirectory": os.path.relpath(os.path.abspath(args.runs_dir), os.getcwd()),
        "labelsPath": os.path.relpath(os.path.abspath(args.labels), os.getcwd()),
        "evaluatedTurnCount": len(rows),
        "skippedReports": skipped,
        "byPolicy": summarize(rows),
        "rows": rows,
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
