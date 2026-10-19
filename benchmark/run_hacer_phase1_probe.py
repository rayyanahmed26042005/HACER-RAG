#!/usr/bin/env python3
"""Run the isolated HACER Phase 1 exact-prefix session probe on Android.

This is not a RAG or MobCert benchmark. It compares a fresh full-prompt
completion with a session-loaded completion sharing an exact prompt prefix.
The app must contain the HACER Phase 1 trigger branch before running it.
"""

import argparse
import base64
import json
import os
import subprocess
import time
import shutil
import statistics
from pathlib import Path


def adb(serial, *args, check=True):
    adb_bin = os.environ.get("ADB_BIN") or shutil.which("adb.exe") or shutil.which("adb") or "adb"
    command = [adb_bin, "-s", serial, *args]
    return subprocess.run(command, capture_output=True, text=True, check=check)


def shell(serial, command, check=True):
    return adb(serial, "shell", command, check=check).stdout.strip()


def write_trigger(serial, trigger_path, result_path, trigger):
    encoded = base64.b64encode(json.dumps(trigger).encode("utf-8")).decode("ascii")
    shell(serial, f"rm -f {result_path} {trigger_path} /sdcard/Android/data/com.chat/files/hacer_phase1_result.json")
    shell(serial, f"echo {encoded} | base64 -d > {trigger_path}")
    shell(serial, "input keyevent KEYCODE_WAKEUP")


def wait_for_result(serial, path, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if shell(serial, f"test -f {path} && echo ready || true") == "ready":
            content = shell(serial, f"cat {path}")
            if content:
                return json.loads(content)
        time.sleep(1.5)
    raise TimeoutError(f"timed out waiting for {path}")


import uuid

def run_phase(args, phase, prefix, suffix, session_path, trigger_path, result_path, output_path):
    request_id = uuid.uuid4().hex
    trigger = {
        "requestId": request_id,
        "resultPath": result_path,
        "query": f"HACER Phase 1 {phase}",
        "chatMode": "general",
        "phase1": phase,
        "phase1Prefix": prefix,
        "phase1Suffix": suffix,
        "phase1SessionPath": session_path,
        "phase1OutputPath": output_path,
        "phase1NPredict": args.n_predict,
    }
    write_trigger(args.serial, trigger_path, result_path, trigger)
    result = wait_for_result(args.serial, output_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="36fef3202145")
    parser.add_argument("--prefix", default="HACER Phase 1 exact prefix. Shared evidence: AeroSense X7 has a QuantumCell battery. ")
    parser.add_argument("--suffix", default="Complete the answer with one concise sentence.")
    parser.add_argument("--n-predict", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", default="benchmark/results/hacer_phase1")
    args = parser.parse_args()

    trigger_path = "/sdcard/Android/data/com.chat/files/benchmark_trigger.json"
    result_path = "/sdcard/Android/data/com.chat/files/benchmark_result.json"
    output_path = "/sdcard/Android/data/com.chat/files/hacer_phase1_result.json"
    session_path = "/sdcard/Android/data/com.chat/files/hacer_phase1_session.bin"
    prefix = args.prefix
    suffix = args.suffix

    if args.repetitions < 1:
        raise SystemExit('--repetitions must be positive')
    # Keep the already initialized React/llama context alive. Restarting the
    # app here can spend the full wait window loading the model and leaves the
    # trigger poller unavailable during that startup interval.
    adb(args.serial, "shell", "am", "start", "-n", "com.chat/.MainActivity", check=False)
    shell(args.serial, "input keyevent KEYCODE_WAKEUP", check=False)
    time.sleep(2)
    repetitions = []
    for repetition in range(args.repetitions):
        repetitions.append({
            "repetition": repetition + 1,
            "seed": run_phase(args, "seed", prefix, suffix, session_path, trigger_path, result_path, output_path),
            "baseline": run_phase(args, "baseline", prefix, suffix, session_path, trigger_path, result_path, output_path),
            "reuse": run_phase(args, "reuse", prefix, suffix, session_path, trigger_path, result_path, output_path),
        })
    seed = repetitions[-1]["seed"]
    baseline = repetitions[-1]["baseline"]
    reuse = repetitions[-1]["reuse"]

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "experiment": "hacer-phase1-exact-prefix-session",
        "device_serial": args.serial,
        "prefix": prefix,
        "suffix": suffix,
        "repetitions": repetitions,
        "seed": seed,
        "baseline": baseline,
        "reuse": reuse,
        "derived": {
            "baseline_wall_ms": baseline.get("elapsed_wall_ms"),
            "reuse_wall_ms": reuse.get("elapsed_wall_ms"),
            "wall_delta_ms": (baseline.get("elapsed_wall_ms") or 0) - (reuse.get("elapsed_wall_ms") or 0),
            "reuse_tokens_loaded": reuse.get("tokens_loaded"),
            "reuse_tokens_cached": reuse.get("tokens_cached"),
            "exact_prefix_claim": reuse.get("exact_prefix_claim"),
            "baseline_wall_ms_median": statistics.median(
                item["baseline"].get("elapsed_wall_ms", 0) for item in repetitions
            ),
            "reuse_wall_ms_median": statistics.median(
                item["reuse"].get("elapsed_wall_ms", 0) for item in repetitions
            ),
            "wall_delta_ms_median": statistics.median(
                item["baseline"].get("elapsed_wall_ms", 0) - item["reuse"].get("elapsed_wall_ms", 0)
                for item in repetitions
            ),
        },
        "note": "Engineering probe only; one device and no paper claim.",
    }
    path = destination / "latest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
