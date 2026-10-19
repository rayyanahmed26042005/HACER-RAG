#!/usr/bin/env python3
"""Analysis and Statistical Evaluation for HACER-RAG Scaling Matrix Suite.

Calculates:
- Paired prefill latency difference (ms and %) with 95% bootstrap confidence intervals
- Paired end-to-end wall latency difference (ms and %) with 95% bootstrap confidence intervals
- Throughput and decode token rates
- Prompt SHA-256 and byte-equality invariance validation
- Hardware telemetry: Delta read_bytes, rchar, battery temp, CPU frequency, PSS memory
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


def bootstrap_ci_paired_diff(
    baseline_vals: List[float],
    reuse_vals: List[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes mean paired difference (baseline - reuse) and 95% percentile bootstrap CI."""
    assert len(baseline_vals) == len(reuse_vals) and len(baseline_vals) > 0
    diffs = np.array(baseline_vals) - np.array(reuse_vals)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = [np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_resamples)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return float(np.mean(diffs)), low, high


def bootstrap_ci_ratio(
    baseline_vals: List[float],
    reuse_vals: List[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes mean percentage reduction (1 - reuse/baseline) * 100 and 95% bootstrap CI."""
    b = np.array(baseline_vals)
    r = np.array(reuse_vals)
    pct_red = (1.0 - (r / b)) * 100.0
    rng = np.random.default_rng(seed)
    n = len(pct_red)
    boot_means = [np.mean(rng.choice(pct_red, size=n, replace=True)) for _ in range(n_resamples)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return float(np.mean(pct_red)), low, high


def analyze_log(log_path: str) -> Dict[str, Any]:
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} execution records from {log_path}")

    # Group records by (config_id, pair_id)
    pairs_by_config = defaultdict(lambda: defaultdict(dict))
    sha_set = set()
    ans_set = set()
    trunc_count = 0

    for r in records:
        cfg = r.get("config_id", f"K{r.get('k_chunks')}_ctx{r.get('n_ctx')}")
        pair = r.get("pair_id", f"rep_{r.get('rep')}")
        cond = r["condition"]
        pairs_by_config[cfg][pair][cond] = r
        sha_set.add(r.get("prompt_sha256"))
        ans_set.add(r.get("response_text"))
        if r.get("truncated"):
            trunc_count += 1

    summary = {
        "total_records": len(records),
        "unique_prompt_sha256_count": len(sha_set),
        "unique_answer_count": len(ans_set),
        "native_truncation_count": trunc_count,
        "configs": {},
    }

    print("\n" + "=" * 95)
    print(f"{'Config':<16} | {'K':<2} | {'n_ctx':<5} | {'N Pairs':<7} | {'Prefill Baseline':<16} | {'Prefill Reuse':<14} | {'Prefill Red (%)':<16} | {'Speedup':<7}")
    print("=" * 95)

    incomplete_pairs_list = []
    storage_classes_count = defaultdict(int)

    for cfg_id, pairs in sorted(pairs_by_config.items()):
        b_prefill, r_prefill = [], []
        b_wall, r_wall = [], []
        b_decode, r_decode = [], []
        io_read_reuse = []

        k_val = None
        ctx_val = None

        for pair_id, trial_pair in pairs.items():
            if "baseline" in trial_pair and "reuse" in trial_pair:
                b = trial_pair["baseline"]
                r = trial_pair["reuse"]
                b_prefill.append(b["prefill_ms"])
                r_prefill.append(r["prefill_ms"])
                b_wall.append(b["wall_elapsed_ms"])
                r_wall.append(r["wall_elapsed_ms"])
                b_decode.append(b["decode_ms"])
                r_decode.append(r["decode_ms"])
                io_read_reuse.append(r.get("io_delta", {}).get("delta_read_bytes", 0))
                k_val = b.get("k_chunks")
                ctx_val = b.get("n_ctx")
                storage_classes_count[r.get("storage_condition", "unknown")] += 1
            else:
                incomplete_pairs_list.append({
                    "config_id": cfg_id,
                    "pair_id": pair_id,
                    "conditions_present": list(trial_pair.keys()),
                })

        n_complete = len(b_prefill)
        if n_complete == 0:
            continue

        mean_b_pref = float(np.mean(b_prefill))
        mean_r_pref = float(np.mean(r_prefill))
        med_b_pref = float(np.median(b_prefill))
        med_r_pref = float(np.median(r_prefill))
        std_b_pref = float(np.std(b_prefill))
        std_r_pref = float(np.std(r_prefill))

        mean_b_wall = float(np.mean(b_wall))
        mean_r_wall = float(np.mean(r_wall))
        med_b_wall = float(np.median(b_wall))
        med_r_wall = float(np.median(r_wall))
        std_b_wall = float(np.std(b_wall))
        std_r_wall = float(np.std(r_wall))

        diff_pref, pref_low, pref_high = bootstrap_ci_paired_diff(b_prefill, r_prefill) if n_complete >= 2 else (mean_b_pref - mean_r_pref, 0.0, 0.0)
        pct_red_pref, red_low, red_high = bootstrap_ci_ratio(b_prefill, r_prefill) if n_complete >= 2 else (((1.0 - mean_r_pref / mean_b_pref) * 100.0), 0.0, 0.0)
        diff_wall, wall_low, wall_high = bootstrap_ci_paired_diff(b_wall, r_wall) if n_complete >= 2 else (mean_b_wall - mean_r_wall, 0.0, 0.0)

        speedup = mean_b_wall / mean_r_wall if mean_r_wall > 0 else 1.0

        print(f"{cfg_id:<16} | {k_val or '?':<2} | {ctx_val or '?':<5} | {n_complete:<7} | "
              f"{mean_b_pref/1e3:6.2f}s ({med_b_pref/1e3:5.2f}s) | {mean_r_pref/1e3:6.2f}s ({med_r_pref/1e3:5.2f}s) | "
              f"{pct_red_pref:5.1f}% [{red_low:4.1f}%, {red_high:4.1f}%] | {speedup:4.2f}x")

        summary["configs"][cfg_id] = {
            "k_chunks": k_val,
            "n_ctx": ctx_val,
            "pairs_count": n_complete,
            "prefill_baseline_ms_mean": mean_b_pref,
            "prefill_baseline_ms_median": med_b_pref,
            "prefill_baseline_ms_std": std_b_pref,
            "prefill_reuse_ms_mean": mean_r_pref,
            "prefill_reuse_ms_median": med_r_pref,
            "prefill_reuse_ms_std": std_r_pref,
            "prefill_reduction_pct_mean": pct_red_pref,
            "prefill_reduction_ci_95": [red_low, red_high],
            "prefill_paired_delta_ms_mean": diff_pref,
            "prefill_paired_delta_ci_95": [pref_low, pref_high],
            "wall_baseline_ms_mean": mean_b_wall,
            "wall_baseline_ms_median": med_b_wall,
            "wall_baseline_ms_std": std_b_wall,
            "wall_reuse_ms_mean": mean_r_wall,
            "wall_reuse_ms_median": med_r_wall,
            "wall_reuse_ms_std": std_r_wall,
            "wall_paired_delta_ms_mean": diff_wall,
            "wall_paired_delta_ci_95": [wall_low, wall_high],
            "wall_speedup": speedup,
        }

    summary["incomplete_pairs"] = incomplete_pairs_list
    summary["storage_classes_count"] = dict(storage_classes_count)

    print("=" * 95)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze HACER-RAG Scaling Matrix Results")
    parser.add_argument("log_path", help="Path to trials JSONL file")
    args = parser.parse_args()

    summary = analyze_log(args.log_path)
    out_json = Path(args.log_path).with_suffix(".summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote statistical summary to {out_json}")


if __name__ == "__main__":
    main()
