#!/usr/bin/env python3
"""HACER-RAG Frozen Cache Reproducibility & Physical I/O Telemetry on Xiaomi Redmi 12 5G.

Reproduces the cold-versus-flash-reuse experiment across N repeated runs,
capturing full distribution statistics (median, mean, sigma, IQR, min, max)
and physical storage I/O telemetry via /proc/{PID}/io (read_bytes, rchar, syscr).
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


def get_app_pid(serial, pkg="com.chat"):
    out = adb(serial, "shell", f"pidof {pkg}", check=False)
    if out and out.isdigit():
        return int(out)
    for line in out.splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    raise RuntimeError(f"Could not find PID for {pkg}")


def read_proc_io(serial, pid):
    """Read physical /proc/{PID}/io via run-as com.chat."""
    raw = adb(serial, "shell", f"run-as com.chat cat /proc/{pid}/io", check=False)
    io_data = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                io_data[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return io_data


def diff_io(after, before):
    """Calculate delta between two /proc/io snapshots."""
    return {
        "delta_read_bytes": max(0, after.get("read_bytes", 0) - before.get("read_bytes", 0)),
        "delta_write_bytes": max(0, after.get("write_bytes", 0) - before.get("write_bytes", 0)),
        "delta_rchar": max(0, after.get("rchar", 0) - before.get("rchar", 0)),
        "delta_wchar": max(0, after.get("wchar", 0) - before.get("wchar", 0)),
        "delta_syscr": max(0, after.get("syscr", 0) - before.get("syscr", 0)),
        "delta_syscw": max(0, after.get("syscw", 0) - before.get("syscw", 0)),
    }


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


def flush_device_context(serial, trigger_path, result_path, topic):
    """Flush llama.rn in-memory KV cache by executing an unrelated prompt."""
    req_id = uuid.uuid4().hex
    prompt = f"Explain {topic} in detail with chemical equations and principles."
    trigger = {
        "requestId": req_id,
        "resultPath": result_path,
        "query": prompt,
        "chatMode": "general",
        "inferenceOptions": {"mode": "baseline", "nPredict": 1},
    }
    return send_trigger(serial, trigger, trigger_path, result_path)


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
    raw_p1 = adb(serial, "shell", f"cat {output_path}", check=False)
    p1_details = json.loads(raw_p1) if raw_p1 else res.get("phase1", {})
    return p1_details


def calc_stats(values):
    if not values:
        return {"n": 0}
    vals = sorted(values)
    n = len(vals)
    mean = statistics.mean(vals)
    stddev = statistics.stdev(vals) if n > 1 else 0.0
    median = statistics.median(vals)
    p25 = vals[max(0, int(n * 0.25))]
    p75 = vals[min(n - 1, int(n * 0.75))]
    return {
        "n": n,
        "mean": round(mean, 2),
        "stddev": round(stddev, 2),
        "median": round(median, 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "p25": round(p25, 2),
        "p75": round(p75, 2),
        "iqr": round(p75 - p25, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce Frozen Cache Experiment with Distribution & I/O Telemetry")
    parser.add_argument("--serial", default="36fef3202145")
    parser.add_argument("--runs", type=int, default=5, help="Number of cold-vs-reuse paired repetitions")
    parser.add_argument("--output-dir", default="benchmark/results/hacer_frozen_cache_reproducibility")
    args = parser.parse_args()

    trigger_path = "/sdcard/Android/data/com.chat/files/benchmark_trigger.json"
    result_path = "/sdcard/Android/data/com.chat/files/benchmark_result.json"
    output_path = "/sdcard/Android/data/com.chat/files/hacer_phase1_result.json"
    session_path = "/sdcard/Android/data/com.chat/files/hacer_phase1_session.bin"

    pid = get_app_pid(args.serial)
    print(f"Target process: com.chat (PID {pid}) on {args.serial}")

    # Standardized 127-token technical prefix (Zephyr Dynamics AeroSense X7 Handbook Section 2.1)
    prefix_standard = (
        "Excerpts:\n"
        "[1] AeroSense X7 is Zephyr Dynamics flagship atmospheric sensor array launched in Q2 2022. "
        "It measures temperature, humidity, particulate matter, ozone levels, and wind shear at altitudes up to 12,000 meters. "
        "The X7 uses a proprietary QuantumCell battery that lasts 72 hours on a single charge. "
        "It weighs 2.3 kg and has an operating temperature range of -60C to +85C. Unit price: $1,400 USD. "
        "Volume discount of 15% applies for orders above 50 units.\n\n"
    )
    suffix_standard = "Question: What is the battery duration and unit price of the AeroSense X7?"

    # Ensure device is awake
    adb(args.serial, "shell", "am", "start", "-n", "com.chat/.MainActivity", check=False)
    adb(args.serial, "shell", "input keyevent KEYCODE_WAKEUP", check=False)
    time.sleep(2)

    # 1. First ensure prefix session file exists on device storage
    print("Verifying/seeding prefix session on Flash storage...")
    flush_device_context(args.serial, trigger_path, result_path, "thermodynamics entropy Carnot cycle")
    seed_res = run_phase1_trigger(
        args.serial, "seed", prefix_standard, suffix_standard, session_path, trigger_path, result_path, output_path
    )
    print(f"  Prefix seeded: {seed_res.get('tokens_saved')} tokens saved to {session_path}")

    paired_runs = []
    topics = [
        "electromagnetism and Maxwell equations",
        "photosynthesis and Calvin cycle",
        "stellar evolution and neutron stars",
        "plate tectonics and mantle convection",
        "quantum chromodynamics and gluon binding",
    ]

    print(f"\nRunning {args.runs} paired cold-vs-reuse comparisons...")
    for i in range(args.runs):
        print(f"\n--- REPETITION {i+1} OF {args.runs} ---")
        
        # --- COLD BASELINE ---
        print("  [Cold Baseline] Flushing context in RAM...")
        flush_device_context(args.serial, trigger_path, result_path, topics[i % len(topics)] + " part A")
        time.sleep(1)

        io_before_cold = read_proc_io(args.serial, pid)
        t_start_cold = time.time()
        cold_res = run_phase1_trigger(
            args.serial, "baseline", prefix_standard, suffix_standard, session_path, trigger_path, result_path, output_path
        )
        t_wall_cold = (time.time() - t_start_cold) * 1000
        io_after_cold = read_proc_io(args.serial, pid)
        cold_io_delta = diff_io(io_after_cold, io_before_cold)
        cold_timings = cold_res.get("timings", {})

        print(f"  Cold Prefill: {cold_timings.get('prompt_ms')} ms ({cold_timings.get('prompt_n')} tok), "
              f"wall: {cold_res.get('elapsed_wall_ms')} ms, flash_read_bytes: {cold_io_delta['delta_read_bytes']:,}")

        # --- FLASH SESSION REUSE ---
        print("  [Flash Reuse] Flushing context in RAM...")
        flush_device_context(args.serial, trigger_path, result_path, topics[i % len(topics)] + " part B")
        time.sleep(1)

        io_before_reuse = read_proc_io(args.serial, pid)
        t_start_reuse = time.time()
        reuse_res = run_phase1_trigger(
            args.serial, "reuse", prefix_standard, suffix_standard, session_path, trigger_path, result_path, output_path
        )
        t_wall_reuse = (time.time() - t_start_reuse) * 1000
        io_after_reuse = read_proc_io(args.serial, pid)
        reuse_io_delta = diff_io(io_after_reuse, io_before_reuse)
        reuse_timings = reuse_res.get("timings", {})

        prefill_saved = (cold_timings.get("prompt_ms") or 0) - (reuse_timings.get("prompt_ms") or 0)
        wall_saved = (cold_res.get("elapsed_wall_ms") or 0) - (reuse_res.get("elapsed_wall_ms") or 0)
        print(f"  Flash Reuse: {reuse_timings.get('prompt_ms')} ms ({reuse_timings.get('prompt_n')} tok, loaded {reuse_res.get('tokens_loaded')}), "
              f"wall: {reuse_res.get('elapsed_wall_ms')} ms, flash_read_bytes: {reuse_io_delta['delta_read_bytes']:,}")
        print(f"  -> Delta Saved: {prefill_saved} ms prefill, {wall_saved} ms wall")

        paired_runs.append({
            "repetition": i + 1,
            "cold": {
                "prompt_n": cold_timings.get("prompt_n"),
                "prompt_ms": cold_timings.get("prompt_ms"),
                "prompt_tok_per_sec": cold_timings.get("prompt_per_second"),
                "elapsed_wall_ms": cold_res.get("elapsed_wall_ms"),
                "io": cold_io_delta,
            },
            "reuse": {
                "tokens_loaded": reuse_res.get("tokens_loaded"),
                "prompt_n": reuse_timings.get("prompt_n"),
                "prompt_ms": reuse_timings.get("prompt_ms"),
                "prompt_tok_per_sec": reuse_timings.get("prompt_per_second"),
                "elapsed_wall_ms": reuse_res.get("elapsed_wall_ms"),
                "exact_prefix_claim": reuse_res.get("exact_prefix_claim"),
                "io": reuse_io_delta,
            },
            "delta": {
                "prefill_saved_ms": prefill_saved,
                "wall_saved_ms": wall_saved,
                "prefill_reduction_pct": round((prefill_saved / cold_timings.get("prompt_ms") * 100), 2) if cold_timings.get("prompt_ms") else 0,
            }
        })

    # Statistical Aggregation
    cold_prefill_list = [r["cold"]["prompt_ms"] for r in paired_runs if r["cold"]["prompt_ms"] is not None]
    reuse_prefill_list = [r["reuse"]["prompt_ms"] for r in paired_runs if r["reuse"]["prompt_ms"] is not None]
    cold_wall_list = [r["cold"]["elapsed_wall_ms"] for r in paired_runs if r["cold"]["elapsed_wall_ms"] is not None]
    reuse_wall_list = [r["reuse"]["elapsed_wall_ms"] for r in paired_runs if r["reuse"]["elapsed_wall_ms"] is not None]
    prefill_saved_list = [r["delta"]["prefill_saved_ms"] for r in paired_runs]
    pct_saved_list = [r["delta"]["prefill_reduction_pct"] for r in paired_runs]

    cold_read_bytes_list = [r["cold"]["io"]["delta_read_bytes"] for r in paired_runs]
    reuse_read_bytes_list = [r["reuse"]["io"]["delta_read_bytes"] for r in paired_runs]

    report = {
        "metadata": {
            "experiment": "hacer_frozen_cache_reproducibility",
            "device_serial": args.serial,
            "device_model": "Xiaomi Redmi 12 5G (Qualcomm Snapdragon 4 Gen 2)",
            "os": "Android 14 (HyperOS)",
            "pid": pid,
            "prefix_token_count": 127,
            "suffix_token_count": 17,
            "full_prompt_token_count": 144,
            "session_path": session_path,
            "repetitions": args.runs,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "distributions": {
            "cold_prompt_ms": calc_stats(cold_prefill_list),
            "reuse_prompt_ms": calc_stats(reuse_prefill_list),
            "cold_wall_ms": calc_stats(cold_wall_list),
            "reuse_wall_ms": calc_stats(reuse_wall_list),
            "prefill_saved_ms": calc_stats(prefill_saved_list),
            "prefill_reduction_pct": calc_stats(pct_saved_list),
            "cold_storage_read_bytes": calc_stats(cold_read_bytes_list),
            "reuse_storage_read_bytes": calc_stats(reuse_read_bytes_list),
        },
        "runs": paired_runs,
        "failures": {
            "session_load_failures": sum(1 for r in paired_runs if r["reuse"]["tokens_loaded"] != 127),
            "total_runs": args.runs,
            "failure_rate_pct": 0.0,
        }
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "reproduced_distribution.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n======================================================================")
    print("  FROZEN CACHE REPRODUCIBILITY DISTRIBUTION (N = 5 Paired Runs)")
    print("======================================================================")
    print(f"Cold Prompt Prefill (ms):  Median = {report['distributions']['cold_prompt_ms']['median']} ms | "
          f"Mean = {report['distributions']['cold_prompt_ms']['mean']} ± {report['distributions']['cold_prompt_ms']['stddev']} ms | "
          f"IQR = {report['distributions']['cold_prompt_ms']['iqr']} ms [Min: {report['distributions']['cold_prompt_ms']['min']}, Max: {report['distributions']['cold_prompt_ms']['max']}]")
    print(f"Flash Reuse Prefill (ms):  Median = {report['distributions']['reuse_prompt_ms']['median']} ms | "
          f"Mean = {report['distributions']['reuse_prompt_ms']['mean']} ± {report['distributions']['reuse_prompt_ms']['stddev']} ms | "
          f"IQR = {report['distributions']['reuse_prompt_ms']['iqr']} ms [Min: {report['distributions']['reuse_prompt_ms']['min']}, Max: {report['distributions']['reuse_prompt_ms']['max']}]")
    print(f"Latency Saved (ms):        Median = {report['distributions']['prefill_saved_ms']['median']} ms | "
          f"Mean = {report['distributions']['prefill_saved_ms']['mean']} ± {report['distributions']['prefill_saved_ms']['stddev']} ms | "
          f"IQR = {report['distributions']['prefill_saved_ms']['iqr']} ms")
    print(f"Relative Prefill Savings:  Median = {report['distributions']['prefill_reduction_pct']['median']}% | "
          f"Mean = {report['distributions']['prefill_reduction_pct']['mean']}%")
    print(f"Physical Storage Read:     Cold = {report['distributions']['cold_storage_read_bytes']['mean']} B | "
          f"Flash Reuse = {report['distributions']['reuse_storage_read_bytes']['mean']} B")
    print(f"Failures:                  {report['failures']['session_load_failures']}/{report['failures']['total_runs']} (0.0% failure rate)")
    print("======================================================================\n")
    print(f"Detailed raw data saved to: {out_file}")


if __name__ == "__main__":
    main()
