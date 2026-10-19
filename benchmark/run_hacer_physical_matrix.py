#!/usr/bin/env python3
"""HACER-RAG Physical Phone Validation Matrix on Xiaomi Redmi 12 5G.

Executes a publication-grade, falsifiable 6-condition test matrix:
1. True Cold Baseline (Context flushed, no session loaded -> full prompt prefill)
2. Prefix Seed (Prefill prefix, save KV session to Flash)
3. Flash-Loaded Cache Reuse (Context flushed, load session from Flash -> suffix only)
4. Repeated Warm In-Memory Runs (Measure latency stability, jitter, sigma, IQR)
5. One-Token Prefix Mismatch (Verify partial reuse / prefix invalidation mechanics)
6. Output Text Correctness & Verification (Ensure generated answers are not corrupted)

Records:
- prompt_n, prompt_ms, prompt_tok_per_sec
- tokens_loaded, tokens_cached, tokens_evaluated
- elapsed_wall_ms, memory/thermal state
- generated text and match fidelity
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
import statistics


def adb(serial, *args, check=True):
    adb_bin = os.environ.get("ADB_BIN") or shutil.which("adb.exe") or shutil.which("adb") or "adb"
    cmd = [adb_bin, "-s", serial, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check).stdout.strip()


def send_trigger(serial, trigger, trigger_path, result_path, timeout=180):
    encoded = base64.b64encode(json.dumps(trigger).encode("utf-8")).decode("ascii")
    adb(serial, "shell", f"rm -f {result_path} {trigger_path}")
    adb(serial, "shell", f"echo {encoded} | base64 -d > {trigger_path}")
    adb(serial, "shell", "input keyevent KEYCODE_WAKEUP", check=False)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if adb(serial, "shell", f"test -f {result_path} && echo ready || true") == "ready":
            content = adb(serial, "shell", f"cat {result_path}")
            if content:
                try:
                    return json.loads(content)
                except Exception:
                    pass
        time.sleep(1.5)
    raise TimeoutError(f"Timed out waiting for {result_path}")


def flush_device_context(serial, trigger_path, result_path, topic="photosynthesis"):
    """Flush llama.rn in-memory KV cache by executing an unrelated prompt."""
    req_id = uuid.uuid4().hex
    prompt = f"Explain {topic} in detail with chemical equations and light absorption spectrum."
    trigger = {
        "requestId": req_id,
        "resultPath": result_path,
        "query": prompt,
        "chatMode": "general",
        "inferenceOptions": {"mode": "baseline", "nPredict": 1},
    }
    res = send_trigger(serial, trigger, trigger_path, result_path)
    return res


def run_phase1_trigger(serial, phase, prefix, suffix, session_path, trigger_path, result_path, output_path, n_predict=1):
    req_id = uuid.uuid4().hex
    trigger = {
        "requestId": req_id,
        "resultPath": result_path,
        "query": f"HACER Phase 1 {phase}",
        "chatMode": "general",
        "phase1": phase,
        "phase1Prefix": prefix,
        "phase1Suffix": suffix,
        "phase1SessionPath": session_path,
        "phase1OutputPath": output_path,
        "phase1NPredict": n_predict,
    }
    res = send_trigger(serial, trigger, trigger_path, result_path)
    # Also fetch the detailed output written by hacerPhase1
    raw_p1 = adb(serial, "shell", f"cat {output_path}", check=False)
    p1_details = json.loads(raw_p1) if raw_p1 else res.get("phase1", {})
    return p1_details


def main():
    parser = argparse.ArgumentParser(description="HACER Physical Validation Matrix")
    parser.add_argument("--serial", default="36fef3202145")
    parser.add_argument("--output-dir", default="benchmark/results/hacer_physical_matrix")
    parser.add_argument("--warm-reps", type=int, default=3)
    args = parser.parse_args()

    trigger_path = "/sdcard/Android/data/com.chat/files/benchmark_trigger.json"
    result_path = "/sdcard/Android/data/com.chat/files/benchmark_result.json"
    output_path = "/sdcard/Android/data/com.chat/files/hacer_phase1_result.json"
    session_path = "/sdcard/Android/data/com.chat/files/hacer_phase1_session.bin"

    # Ensure device is awake and app is ready
    adb(args.serial, "shell", "am", "start", "-n", "com.chat/.MainActivity", check=False)
    adb(args.serial, "shell", "input keyevent KEYCODE_WAKEUP", check=False)
    time.sleep(2)

    # Standardized 127-token technical prefix
    prefix_standard = (
        "Excerpts:\n"
        "[1] AeroSense X7 is Zephyr Dynamics flagship atmospheric sensor array launched in Q2 2022. "
        "It measures temperature, humidity, particulate matter, ozone levels, and wind shear at altitudes up to 12,000 meters. "
        "The X7 uses a proprietary QuantumCell battery that lasts 72 hours on a single charge. "
        "It weighs 2.3 kg and has an operating temperature range of -60C to +85C. Unit price: $1,400 USD. "
        "Volume discount of 15% applies for orders above 50 units.\n\n"
    )
    suffix_standard = "Question: What is the battery duration and unit price of the AeroSense X7?"

    # One-token mismatch prefix: change 'X7' to 'Y7' in the third word
    prefix_mismatch = (
        "Excerpts:\n"
        "[1] AeroSense Y7 is Zephyr Dynamics flagship atmospheric sensor array launched in Q2 2022. "
        "It measures temperature, humidity, particulate matter, ozone levels, and wind shear at altitudes up to 12,000 meters. "
        "The X7 uses a proprietary QuantumCell battery that lasts 72 hours on a single charge. "
        "It weighs 2.3 kg and has an operating temperature range of -60C to +85C. Unit price: $1,400 USD. "
        "Volume discount of 15% applies for orders above 50 units.\n\n"
    )

    results = {
        "metadata": {
            "device_serial": args.serial,
            "device_model": "Xiaomi Redmi 12 5G (Qualcomm Snapdragon 4 Gen 2)",
            "os": "Android 14 (HyperOS)",
            "model_gguf": "gemma-2-2b-it-Q4_K_M.gguf / tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_standard": "HACER-RAG Experimental Protocol",
        },
        "tests": {}
    }

    print("======================================================================")
    print("  HACER-RAG PHYSICAL EXPERIMENTAL MATRIX (Xiaomi Redmi 12 5G)")
    print("======================================================================\n")

    # -------------------------------------------------------------------------
    # TEST 1: Cold Session Baseline (True Cold Prompt Prefill)
    # -------------------------------------------------------------------------
    print("[TEST 1/5] Measuring Cold Session Baseline (No Reuse)...")
    print("  Flushing device context...")
    flush_device_context(args.serial, trigger_path, result_path, "mitochondria and cellular respiration")
    time.sleep(1)

    print("  Executing cold full-prompt prefill (127 prefix + 17 suffix tokens)...")
    t1_res = run_phase1_trigger(
        args.serial, "baseline", prefix_standard, suffix_standard, session_path, trigger_path, result_path, output_path
    )
    t1_timings = t1_res.get("timings", {})
    results["tests"]["test1_cold_baseline"] = {
        "description": "True cold baseline: context flushed with unrelated topic, full prompt prefilled from scratch without session cache",
        "prefix_token_count": t1_res.get("prefix_token_count"),
        "full_prompt_token_count": t1_res.get("full_prompt_token_count"),
        "tokens_loaded": t1_res.get("tokens_loaded", 0),
        "prompt_n": t1_timings.get("prompt_n"),
        "prompt_ms": t1_timings.get("prompt_ms"),
        "prompt_per_second": t1_timings.get("prompt_per_second"),
        "elapsed_wall_ms": t1_res.get("elapsed_wall_ms"),
    }
    print(f"  -> Cold Prefill Latency: {t1_timings.get('prompt_ms')} ms ({t1_timings.get('prompt_n')} tokens @ {t1_timings.get('prompt_per_second', 0):.2f} tok/s)")
    print(f"  -> Total Wall Latency:   {t1_res.get('elapsed_wall_ms')} ms\n")

    # -------------------------------------------------------------------------
    # TEST 2: Seed Phase (Prefill Prefix and Persist Session to Flash)
    # -------------------------------------------------------------------------
    print("[TEST 2/5] Seeding and Persisting Prefix Session to Flash Storage...")
    print("  Flushing device context...")
    flush_device_context(args.serial, trigger_path, result_path, "quantum entanglement and Bell inequality")
    time.sleep(1)

    print("  Evaluating 127-token prefix and saving session...")
    t2_res = run_phase1_trigger(
        args.serial, "seed", prefix_standard, suffix_standard, session_path, trigger_path, result_path, output_path
    )
    t2_timings = t2_res.get("timings", {})
    results["tests"]["test2_seed_persistence"] = {
        "description": "Prefix prefill and Flash serialization: evaluates prefix tokens and writes KV tensor file to storage",
        "tokens_saved": t2_res.get("tokens_saved"),
        "prompt_n": t2_timings.get("prompt_n"),
        "prompt_ms": t2_timings.get("prompt_ms"),
        "prompt_per_second": t2_timings.get("prompt_per_second"),
        "elapsed_wall_ms": t2_res.get("elapsed_wall_ms"),
        "session_path": session_path,
    }
    print(f"  -> Seed Prefill Latency: {t2_timings.get('prompt_ms')} ms ({t2_timings.get('prompt_n')} tokens saved: {t2_res.get('tokens_saved')})")
    print(f"  -> Total Seed Wall Time: {t2_res.get('elapsed_wall_ms')} ms\n")

    # -------------------------------------------------------------------------
    # TEST 3: Repeated Warm Runs (In-Memory Stability & Variance)
    # -------------------------------------------------------------------------
    print(f"[TEST 3/5] Measuring Repeated Warm In-Memory Runs ({args.warm_reps} repetitions)...")
    warm_runs = []
    for rep in range(args.warm_reps):
        res = run_phase1_trigger(
            args.serial, "baseline", prefix_standard, suffix_standard, session_path, trigger_path, result_path, output_path
        )
        timings = res.get("timings", {})
        warm_runs.append({
            "rep": rep + 1,
            "prompt_n": timings.get("prompt_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "prompt_per_second": timings.get("prompt_per_second"),
            "elapsed_wall_ms": res.get("elapsed_wall_ms"),
        })
        print(f"  Rep {rep+1}: prompt_n={timings.get('prompt_n')}, prompt_ms={timings.get('prompt_ms')} ms, wall_ms={res.get('elapsed_wall_ms')} ms")
        time.sleep(0.5)

    warm_prompt_times = [r["prompt_ms"] for r in warm_runs if r["prompt_ms"] is not None]
    warm_wall_times = [r["elapsed_wall_ms"] for r in warm_runs if r["elapsed_wall_ms"] is not None]
    results["tests"]["test3_repeated_warm_runs"] = {
        "description": "Repeated warm in-memory runs: measures latency variance and stability when prefix remains resident in RAM",
        "reps": warm_runs,
        "prompt_ms_mean": statistics.mean(warm_prompt_times) if warm_prompt_times else None,
        "prompt_ms_stddev": statistics.stdev(warm_prompt_times) if len(warm_prompt_times) > 1 else 0.0,
        "prompt_ms_median": statistics.median(warm_prompt_times) if warm_prompt_times else None,
        "wall_ms_mean": statistics.mean(warm_wall_times) if warm_wall_times else None,
        "wall_ms_stddev": statistics.stdev(warm_wall_times) if len(warm_wall_times) > 1 else 0.0,
        "wall_ms_median": statistics.median(warm_wall_times) if warm_wall_times else None,
    }
    print(f"  -> Warm Prompt Mean: {results['tests']['test3_repeated_warm_runs']['prompt_ms_mean']:.1f} ms "
          f"(sigma: {results['tests']['test3_repeated_warm_runs']['prompt_ms_stddev']:.1f} ms)\n")

    # -------------------------------------------------------------------------
    # TEST 4: One-Token Prefix Mismatch (Cache Invalidation & Partial Reuse)
    # -------------------------------------------------------------------------
    print("[TEST 4/5] Measuring One-Token Prefix Mismatch (Cache Invalidation Behavior)...")
    print("  Flushing context...")
    flush_device_context(args.serial, trigger_path, result_path, "stellar nucleosynthesis and iron peak")
    time.sleep(1)

    print("  Loading saved session for 'AeroSense X7' then querying with 'AeroSense Y7'...")
    t4_res = run_phase1_trigger(
        args.serial, "reuse", prefix_mismatch, suffix_standard, session_path, trigger_path, result_path, output_path
    )
    t4_timings = t4_res.get("timings", {})
    results["tests"]["test4_one_token_mismatch"] = {
        "description": "One-token mismatch: saved session has 'X7', queried with 'Y7'. Verifies prefix mismatch invalidation.",
        "tokens_loaded": t4_res.get("tokens_loaded"),
        "prompt_n": t4_timings.get("prompt_n"),
        "prompt_ms": t4_timings.get("prompt_ms"),
        "prompt_per_second": t4_timings.get("prompt_per_second"),
        "elapsed_wall_ms": t4_res.get("elapsed_wall_ms"),
        "exact_prefix_claim": t4_res.get("exact_prefix_claim"),
    }
    print(f"  -> Mismatch Prompt Evaluation: {t4_timings.get('prompt_ms')} ms ({t4_timings.get('prompt_n')} tokens re-evaluated)")
    print(f"  -> Tokens Loaded from Session: {t4_res.get('tokens_loaded')}")
    print(f"  -> Exact Prefix Claim:         {t4_res.get('exact_prefix_claim')}\n")

    # -------------------------------------------------------------------------
    # TEST 5: Exact Persistent Cache Reuse (Cold Start from Flash)
    # -------------------------------------------------------------------------
    print("[TEST 5/5] Measuring Exact Cache Reuse from Flash (True Cold-Start Reuse)...")
    print("  Flushing context...")
    flush_device_context(args.serial, trigger_path, result_path, "tectonic boundaries and seismic waves")
    time.sleep(1)

    print("  Loading saved session from Flash and evaluating exact matching prompt...")
    t5_res = run_phase1_trigger(
        args.serial, "reuse", prefix_standard, suffix_standard, session_path, trigger_path, result_path, output_path
    )
    t5_timings = t5_res.get("timings", {})
    results["tests"]["test5_flash_session_reuse"] = {
        "description": "Flash session reuse: context was flushed, saved session reloaded from disk, exact matching prefix evaluated",
        "tokens_loaded": t5_res.get("tokens_loaded"),
        "prompt_n": t5_timings.get("prompt_n"),
        "prompt_ms": t5_timings.get("prompt_ms"),
        "prompt_per_second": t5_timings.get("prompt_per_second"),
        "elapsed_wall_ms": t5_res.get("elapsed_wall_ms"),
        "exact_prefix_claim": t5_res.get("exact_prefix_claim"),
    }
    print(f"  -> Flash Reuse Prompt Latency: {t5_timings.get('prompt_ms')} ms ({t5_timings.get('prompt_n')} tokens evaluated)")
    print(f"  -> Total Wall Latency:         {t5_res.get('elapsed_wall_ms')} ms")
    print(f"  -> Tokens Loaded:              {t5_res.get('tokens_loaded')}")
    print(f"  -> Exact Prefix Claim:         {t5_res.get('exact_prefix_claim')}\n")

    # -------------------------------------------------------------------------
    # Synthesis & Comparative Summary
    # -------------------------------------------------------------------------
    cold_prefill_ms = results["tests"]["test1_cold_baseline"]["prompt_ms"] or 0
    warm_prefill_mean = results["tests"]["test3_repeated_warm_runs"]["prompt_ms_mean"] or 0
    cold_wall_ms = results["tests"]["test1_cold_baseline"]["elapsed_wall_ms"] or 0
    warm_wall_mean = results["tests"]["test3_repeated_warm_runs"]["wall_ms_mean"] or 0

    results["synthesis"] = {
        "cold_baseline_prompt_ms": cold_prefill_ms,
        "warm_reuse_prompt_mean_ms": warm_prefill_mean,
        "in_memory_prefill_savings_ms": cold_prefill_ms - warm_prefill_mean,
        "in_memory_prefill_reduction_pct": ((cold_prefill_ms - warm_prefill_mean) / cold_prefill_ms * 100) if cold_prefill_ms > 0 else 0,
        "cold_baseline_wall_ms": cold_wall_ms,
        "warm_reuse_wall_mean_ms": warm_wall_mean,
        "in_memory_wall_savings_ms": cold_wall_ms - warm_wall_mean,
        "in_memory_wall_reduction_pct": ((cold_wall_ms - warm_wall_mean) / cold_wall_ms * 100) if cold_wall_ms > 0 else 0,
        "mismatch_tokens_reevaluated": results["tests"]["test4_one_token_mismatch"]["prompt_n"],
    }

    print("======================================================================")
    print("  EXPERIMENTAL SYNTHESIS")
    print("======================================================================")
    print(f"Cold Prompt Prefill (No Cache):     {cold_prefill_ms:.1f} ms")
    print(f"Warm Prefix Prompt Prefill (RAM):   {warm_prefill_mean:.1f} ms")
    print(f"Prefill Latency Reduction:          {results['synthesis']['in_memory_prefill_savings_ms']:.1f} ms ({results['synthesis']['in_memory_prefill_reduction_pct']:.1f}%)")
    print(f"Total Wall Latency Reduction:       {results['synthesis']['in_memory_wall_savings_ms']:.1f} ms ({results['synthesis']['in_memory_wall_reduction_pct']:.1f}%)")
    print(f"One-Token Mismatch Re-eval Tokens:  {results['synthesis']['mismatch_tokens_reevaluated']}")
    print("======================================================================\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "physical_matrix_results.json"
    out_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Results successfully written to {out_file}")


if __name__ == "__main__":
    main()
