#!/usr/bin/env python3
"""
Automated Batch Benchmark Runner for Physical Mobile Device.
Iterates through labeled benchmark queries, tests both RAG policies
('ebmrag-v2' and 'mobcert-v1'), gathers hardware telemetry and research
parameters, and generates the consolidated comparison table.
"""

import os
import sys
import json
import time
import argparse
import uuid
from typing import List, Dict, Any

# Prevent charmap encoding errors on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure benchmark dir is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_device_inference import run_test, DEVICE_SERIAL
import run_device_inference as device_runner
from evaluate_physical_runs import evaluate_report, load_labels, summarize, normalized_query


def run_batch_evaluation(
    queries_path: str = os.path.join(SCRIPT_DIR, "zephyr_evaluation_queries.json"),
    policies: List[str] = ("ebmrag-v2", "mobcert-v1"),
    limit: int = None,
    delay_sec: float = 5.0,
    inference_mode: str = "baseline",
    repetitions: int = 1,
    counterbalance: bool = True,
    output_dir: str = None,
    profile_id: str = None,
    allow_direct_evidence: bool = False,
    model_path: str = None,
):
    print("=" * 70)
    print("🚀 AUTOMATED ON-DEVICE BATCH BENCHMARK SUITE")
    print(f"Device Serial : {DEVICE_SERIAL}")
    print(f"Policies      : {policies}")
    print(f"Queries File  : {queries_path}")
    print("=" * 70)

    if not os.path.isfile(queries_path):
        print(f"Error: Queries file not found: {queries_path}")
        return 1

    with open(queries_path, "r", encoding="utf-8") as f:
        query_records = json.load(f)

    if limit is not None:
        query_records = query_records[:limit]

    labels = load_labels(queries_path)
    all_evaluated_rows = []

    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if any(item.get("mode", "document") == "document" for item in query_records) and not profile_id:
        raise ValueError("Document physical experiments require an explicit --profile-id")
    output_dir = output_dir or os.path.join(SCRIPT_DIR, "results", "physical_runs")
    os.makedirs(output_dir, exist_ok=True)
    total_runs = len(query_records) * len(policies) * repetitions
    current_run = 0

    failures = []
    for repetition in range(repetitions):
        for item_index, item in enumerate(query_records):
            ordered_policies = list(policies)
            if counterbalance and (item_index + repetition) % 2 == 1:
                ordered_policies.reverse()
            for policy in ordered_policies:
                current_run += 1
                qid = item.get("id", "unknown")
                query = item["query"]
                mode = item.get("mode", "document")

                print(f"\n[{current_run}/{total_runs}] Repetition {repetition + 1}: running '{qid}' under {policy}...")
                print(f"Query: \"{query}\"")

                run_id = f"batch_{policy}_{qid}_r{repetition + 1}_{uuid.uuid4().hex[:8]}"
                output_path = os.path.join(output_dir, f"{run_id}.json")
                try:
                    report = run_test(
                        query=query,
                        chat_mode=mode,
                        inference_mode=inference_mode,
                        rag_policy=policy if mode == "document" else None,
                        run_id=run_id,
                        output_path=output_path,
                        query_id=qid,
                        corpus_id=os.path.splitext(os.path.basename(queries_path))[0],
                        profile_id=profile_id,
                        allow_direct_evidence=allow_direct_evidence,
                        model_path=model_path,
                    )
                except Exception as ex:
                    print(f"  ✗ Runner error: {ex}")
                    report = None
                    failures.append({"query_id": qid, "policy": policy, "repetition": repetition + 1, "error": str(ex)})

                if report:
                    norm_q = normalized_query(query)
                    label = labels.get(norm_q, item)
                    try:
                        eval_row = evaluate_report(report, label)
                        all_evaluated_rows.append(eval_row)
                        bench = eval_row.get("benchmark", {})
                        print(f"  ✓ Context Recall    : {bench.get('context_recall')}")
                        print(f"  ✓ Context Precision : {bench.get('context_precision')}")
                        print(f"  ✓ Faithfulness      : {bench.get('faithfulness')}")
                        print(f"  ✓ Token F1          : {bench.get('token_f1')}")
                        print(f"  ✓ Refusal Accuracy  : {bench.get('refusal_accuracy')}")
                        print(f"  ✓ Contract Status   : {eval_row.get('answer_contract_status')}")
                    except Exception as ex:
                        print(f"  ⚠ Failed to evaluate report row: {ex}")
                else:
                    print(f"  ✗ Test execution timed out or failed on device.")
                    failures.append({"query_id": qid, "policy": policy, "repetition": repetition + 1,
                                     "error": "no completed physical report"})

                if delay_sec > 0:
                    print(f"Cooling down device for {delay_sec}s...")
                    time.sleep(delay_sec)

    # Summarize and output final results
    if all_evaluated_rows:
        summary = summarize(all_evaluated_rows)
        out_summary_file = os.path.join(output_dir, "batch_run_summary.json")
        os.makedirs(os.path.dirname(out_summary_file), exist_ok=True)
        with open(out_summary_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": all_evaluated_rows}, f, indent=2)

        print("\n" + "=" * 70)
        print("📊 FINAL COMPARATIVE BENCHMARK SUMMARY (IEEE Research Results)")
        print("=" * 70)
        print(f"{'Metric':<30} | {'EBM-RAG-v2':<18} | {'MobCert-v1':<18}")
        print("-" * 70)

        ebm = summary.get("ebmrag-v2", {})
        mob = summary.get("mobcert-v1", {})

        metrics_to_print = [
            ("Run Count", "run_count"),
            ("Mean Context Recall", "mean_context_recall"),
            ("Mean Context Precision", "mean_context_precision"),
            ("Mean Faithfulness/Grounded", "mean_faithfulness"),
            ("Mean Token F1", "mean_token_f1"),
            ("Mean Refusal Accuracy", "mean_refusal_accuracy"),
            ("Median TTFT (ms)", "median_ttft_ms"),
            ("Median Total Latency (ms)", "median_total_ms"),
        ]

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        for label_name, key in metrics_to_print:
            val_ebm = ebm.get(key, "N/A")
            val_mob = mob.get(key, "N/A")
            print(f"{label_name:<30} | {fmt(val_ebm):<18} | {fmt(val_mob):<18}")

        print("-" * 70)
        print(f"EBM Contract Statuses   : {ebm.get('contract_status_counts', {})}")
        print(f"MobCert Contract Statuses: {mob.get('contract_status_counts', {})}")
        print("=" * 70)
        print(f"Complete results saved to: {out_summary_file}")

    if failures:
        failure_path = os.path.join(output_dir, "failures.json")
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        print(f"\nERROR: {len(failures)} experiment cells did not complete; see {failure_path}")
        return 2

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated batch evaluation on physical mobile device")
    parser.add_argument("--queries", default=os.path.join(SCRIPT_DIR, "zephyr_evaluation_queries.json"))
    parser.add_argument("--serial", default=None, help="ADB device serial; overrides ANDROID_SERIAL")
    parser.add_argument("--policies", nargs="+", default=["ebmrag-v2", "mobcert-v1"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to test (e.g. --limit 3 for quick check)")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between queries in seconds for thermal stabilization")
    parser.add_argument("--inference-mode", default="baseline", choices=["baseline", "adaptive", "fixed"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--no-counterbalance", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--allow-direct-evidence", action="store_true")
    args = parser.parse_args()
    if args.serial:
        device_runner.DEVICE_SERIAL = args.serial

    sys.exit(run_batch_evaluation(
        queries_path=args.queries,
        policies=args.policies,
        limit=args.limit,
        delay_sec=args.delay,
        inference_mode=args.inference_mode,
        repetitions=args.repetitions,
        counterbalance=not args.no_counterbalance,
        output_dir=args.output_dir,
        profile_id=args.profile_id,
        allow_direct_evidence=args.allow_direct_evidence,
        model_path=args.model_path,
    ))
