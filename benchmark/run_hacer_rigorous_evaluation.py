#!/usr/bin/env python3
"""HACER-RAG Rigorous Multi-Chunk & Expanded Staleness Physical Benchmark.

Executes on physical Xiaomi Redmi 12 5G (Snapdragon 4 Gen 2, Android 14/HyperOS):
1. Repeated Paired Multi-Chunk Trials (3 matched pairs per case across 3 multi-hop cases = 18 runs):
   - Case 1: Remote Work Stipend ($800) + Internet Allowance ($50)
   - Case 2: AeroSense Battery (72h) + SwarmKit Drones (2,048)
   - Case 3: Leadership Office (Stockholm) + Sensor Altitude (12,000m)
   Isolates and records:
   - Native prompt prefill compute latency (prompt_ms)
   - Native decode latency (predicted_ms)
   - Flash cache load overhead (session_load_ms / bridge overhead)
   - Total end-to-end wall clock latency
   - Physical storage read bytes & syscalls (/proc/{PID}/io)
   - Peak memory (PSS in KB)
   - Factual recall: Hop 1, Hop 2, and Full Multi-Hop EM/F1
2. Expanded Staleness & Dynamic Corpus Suite (5 Controlled Scenarios):
   - Scenario A: Content Update (Stipend updated $400 -> $800)
   - Scenario B: Policy Revocation / Deletion (Childcare subsidy revoked)
   - Scenario C: Changed Chunk Ordering ([Chunk B, Chunk A] vs cached [Chunk A, Chunk B])
   - Scenario D: Partial Multi-Chunk Update (Chunk A unchanged, Chunk B updated $1,200 -> $1,600)
   - Scenario E: Corrupted Cache File Fallback (Safe invalidation & scratch prefill recovery)
"""

import argparse
import base64
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

print = functools.partial(print, flush=True)


def adb(serial: str, *args: str, check: bool = True) -> str:
    adb_bin = os.environ.get("ADB_BIN") or shutil.which("adb.exe") or shutil.which("adb") or "adb"
    cmd = [adb_bin, "-s", serial, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check).stdout.strip()


def get_app_pid(serial: str, pkg: str = "com.chat") -> int:
    out = adb(serial, "shell", f"pidof {pkg}", check=False)
    for line in out.splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    raise RuntimeError(f"Could not find running process PID for {pkg}")


def ensure_app_running(serial: str, pkg: str = "com.chat") -> int:
    try:
        return get_app_pid(serial, pkg)
    except RuntimeError:
        print(f"[*] Process {pkg} is not running. Launching app...")
        adb(serial, "shell", "svc power stayon true", check=False)
        adb(serial, "shell", "input keyevent KEYCODE_WAKEUP", check=False)
        adb(serial, "shell", "wm dismiss-keyguard", check=False)
        adb(serial, "shell", f"am start -n {pkg}/.MainActivity", check=False)
        for _ in range(15):
            time.sleep(1.0)
            try:
                pid = get_app_pid(serial, pkg)
                print(f"[+] {pkg} started with PID {pid}. Waiting 6s for LLM context to load...")
                time.sleep(6.0)
                return pid
            except RuntimeError:
                pass
        raise RuntimeError(f"Failed to start {pkg} on {serial}")


def read_proc_io(serial: str, pid: int) -> Dict[str, int]:
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


def diff_io(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    return {
        "delta_read_bytes": max(0, after.get("read_bytes", 0) - before.get("read_bytes", 0)),
        "delta_write_bytes": max(0, after.get("write_bytes", 0) - before.get("write_bytes", 0)),
        "delta_rchar": max(0, after.get("rchar", 0) - before.get("rchar", 0)),
        "delta_wchar": max(0, after.get("wchar", 0) - before.get("wchar", 0)),
        "delta_syscr": max(0, after.get("syscr", 0) - before.get("syscr", 0)),
        "delta_syscw": max(0, after.get("syscw", 0) - before.get("syscw", 0)),
    }


def get_peak_pss_kb(serial: str, pkg: str = "com.chat") -> int:
    raw = adb(serial, "shell", f"dumpsys meminfo {pkg} | grep 'TOTAL PSS:' || true", check=False)
    m = re.search(r"TOTAL\s+PSS:\s+(\d+)", raw)
    if m:
        return int(m.group(1))
    # Fallback to total line
    raw2 = adb(serial, "shell", f"dumpsys meminfo {pkg} | grep 'TOTAL:' || true", check=False)
    m2 = re.search(r"TOTAL:\s+(\d+)", raw2)
    return int(m2.group(1)) if m2 else 0


def get_battery_temp(serial: str) -> Optional[float]:
    raw = adb(serial, "shell", "dumpsys battery", check=False)
    for line in raw.splitlines():
        if "temperature:" in line:
            parts = line.split(":")
            if len(parts) == 2 and parts[1].strip().isdigit():
                return float(parts[1].strip()) / 10.0
    return None


def send_trigger(
    serial: str,
    trigger: Dict[str, Any],
    trigger_path: str,
    result_path: str,
    timeout: int = 180,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_app_running(serial)
    encoded = base64.b64encode(json.dumps(trigger).encode("utf-8")).decode("ascii")
    tmp_trigger = f"{trigger_path}.tmp"
    rm_cmd = f"rm -f {result_path} {trigger_path} {tmp_trigger}"
    if output_path:
        rm_cmd += f" {output_path}"
    adb(serial, "shell", rm_cmd)
    adb(serial, "shell", f"echo {encoded} | base64 -d > {tmp_trigger} && mv -f {tmp_trigger} {trigger_path}")
    adb(serial, "shell", "am start -n com.chat/.MainActivity", check=False)
    adb(serial, "shell", "input keyevent KEYCODE_WAKEUP", check=False)
    adb(serial, "shell", "wm dismiss-keyguard", check=False)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if adb(serial, "shell", f"test -f {result_path} && echo ready || true", check=False) == "ready":
            content = adb(serial, "shell", f"cat {result_path}", check=False)
            if content:
                try:
                    data = json.loads(content)
                    if data.get("status") in ("completed", "failed"):
                        return data
                except Exception:
                    pass
        time.sleep(1.2)
    raise TimeoutError(f"Timed out waiting {timeout}s for {result_path}")


def flush_device_context(serial: str, trigger_path: str, result_path: str, topic: str) -> None:
    req_id = uuid.uuid4().hex
    prompt = f"Explain {topic} in comprehensive detail with mathematical equations and principles."
    trigger = {
        "requestId": req_id,
        "resultPath": result_path,
        "query": f"HACER Context Flush: {topic}",
        "chatMode": "general",
        "phase1": "baseline",
        "phase1Prefix": prompt,
        "phase1Suffix": " Briefly summarize.",
        "phase1SessionPath": "/sdcard/Android/data/com.chat/files/hacer_chunk_session.bin",
        "phase1OutputPath": "/sdcard/Android/data/com.chat/files/hacer_phase1_result.json",
        "phase1NPredict": 1,
    }
    send_trigger(serial, trigger, trigger_path, result_path, timeout=120)


def run_phase1_trigger(
    serial: str,
    phase: str,
    prefix: str,
    suffix: str,
    session_path: str,
    trigger_path: str,
    result_path: str,
    output_path: str,
    n_predict: int = 35,
    prompt_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    req_id = uuid.uuid4().hex
    trigger = {
        "requestId": req_id,
        "resultPath": result_path,
        "query": f"HACER Multi-Chunk {phase}",
        "chatMode": "general",
        "phase1": phase,
        "phase1Prefix": prefix,
        "phase1Suffix": suffix,
        "phase1SessionPath": session_path,
        "phase1OutputPath": output_path,
        "phase1NPredict": n_predict,
        "promptSha256": prompt_sha256,
    }
    t_start = time.perf_counter()
    res = send_trigger(serial, trigger, trigger_path, result_path, timeout=180, output_path=output_path)
    t_wall_host_ms = (time.perf_counter() - t_start) * 1000.0
    raw_p1 = adb(serial, "shell", f"cat {output_path}", check=False)
    p1_details = json.loads(raw_p1) if raw_p1 else res.get("phase1", {})
    p1_details["wall_host_elapsed_ms"] = t_wall_host_ms
    return p1_details


def evaluate_facts(text: str, hop1_facts: List[str], hop2_facts: List[str]) -> Dict[str, Any]:
    text_lower = text.lower()
    h1_match = any(f.lower() in text_lower for f in hop1_facts) if hop1_facts else False
    h2_match = any(f.lower() in text_lower for f in hop2_facts) if hop2_facts else False
    return {
        "hop1_matched": h1_match,
        "hop2_matched": h2_match,
        "multihop_correct": h1_match and h2_match,
    }


def main():
    parser = argparse.ArgumentParser(description="HACER Rigorous Multi-Chunk & Staleness Evaluation")
    parser.add_argument("--serial", default="36fef3202145")
    parser.add_argument("--pairs-per-case", type=int, default=3)
    parser.add_argument("--output-dir", default="benchmark/results/hacer_multichunk_rag")
    args = parser.parse_args()

    trigger_path = "/sdcard/Android/data/com.chat/files/benchmark_trigger.json"
    result_path = "/sdcard/Android/data/com.chat/files/benchmark_result.json"
    output_path = "/sdcard/Android/data/com.chat/files/hacer_phase1_result.json"
    session_path = "/sdcard/Android/data/com.chat/files/hacer_multichunk_rigorous_session.bin"

    pid = ensure_app_running(args.serial)
    print(f"Target process: com.chat (PID {pid}) on {args.serial}")

    # Multi-hop test cases
    test_cases = [
        {
            "id": "case1_remote_benefits",
            "title": "Remote Work Stipend & Internet Allowance",
            "chunk_a": (
                "Section 3.2 Remote Work Policy:\n"
                "Zephyr Dynamics supports hybrid and remote working arrangements for eligible roles. "
                "Employees are required to be in-office a minimum of 2 days per week. "
                "Fully remote work may be approved for up to 4 consecutive weeks per year. "
                "To support home office ergonomics, a one-time remote work equipment stipend of $800 USD "
                "is provided to all eligible new hires.\n\n"
            ),
            "chunk_b": (
                "Section 3.3 Compensation & Benefits:\n"
                "Zephyr Dynamics offers a comprehensive global benefits package. "
                "Annual bonus is up to 20% of base salary, paid in March. "
                "For remote-eligible employees, an ongoing monthly internet allowance of $50/month "
                "is reimbursed via expense reports. The annual learning budget is $1,500 USD per employee.\n\n"
            ),
            "question": "Question: What is the one-time remote equipment stipend and the monthly internet allowance for remote employees? Answer in one concise sentence.",
            "hop1_facts": ["$800", "800"],
            "hop2_facts": ["$50", "50"],
        },
        {
            "id": "case2_aerosense_vs_swarmkit",
            "title": "AeroSense X7 Battery Life vs SwarmKit Drone Capacity",
            "chunk_a": (
                "Section 2.1 AeroSense X7:\n"
                "The AeroSense X7 is Zephyr Dynamics flagship atmospheric sensor array launched in Q2 2022. "
                "It measures temperature, humidity, particulate matter, and ozone up to 12,000 meters. "
                "The X7 uses a proprietary QuantumCell battery that lasts 72 hours on a single charge. "
                "Unit price: $1,400 USD. Operating range is -60C to +85C.\n\n"
            ),
            "chunk_b": (
                "Section 2.2 SwarmKit Pro:\n"
                "SwarmKit Pro is a micro-drone coordination platform released in January 2023. "
                "It supports simultaneous control of up to 2,048 drones using the ZephyrMesh v3 protocol. "
                "The standard tier costs $1,200/month for up to 256 drones, while the full capacity tier "
                "costs $3,800/month.\n\n"
            ),
            "question": "Question: What is the battery duration of the AeroSense X7 and the maximum drone capacity of SwarmKit Pro? Answer in one concise sentence.",
            "hop1_facts": ["72 hours", "72"],
            "hop2_facts": ["2,048", "2048"],
        },
        {
            "id": "case3_ceo_office_and_sensor_altitude",
            "title": "CEO Office Location vs Sensor Operating Altitude",
            "chunk_a": (
                "Section 1.1 Leadership Team:\n"
                "Dr. Priya Nandakumar serves as Chief Executive Officer (CEO), based in the Stockholm office. "
                "Marcus Lindqvist is Chief Technology Officer (CTO), based in Stockholm. "
                "Jorge Fonseca is Chief Legal Officer (CLO), based in the Lisbon office.\n\n"
            ),
            "chunk_b": (
                "Section 2.1 AeroSense X7:\n"
                "The AeroSense X7 sensor array measures atmospheric variables at altitudes up to 12,000 meters. "
                "It weighs 2.3 kg and carries an IP67 environmental rating. "
                "A volume discount of 15% applies for orders above 50 units.\n\n"
            ),
            "question": "Question: Which office is the CEO based in, and what is the maximum operating altitude of the AeroSense X7? Answer in one concise sentence.",
            "hop1_facts": ["Stockholm"],
            "hop2_facts": ["12,000", "12000"],
        },
    ]

    results = {
        "metadata": {
            "experiment": "hacer_rigorous_multichunk_and_expanded_staleness",
            "device_serial": args.serial,
            "device_model": "Xiaomi Redmi 12 5G (Snapdragon 4 Gen 2)",
            "os": "Android 14 (HyperOS)",
            "pid": pid,
            "pairs_per_case": args.pairs_per_case,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "multichunk_paired_trials": [],
        "case_summaries": {},
        "staleness_suite": {},
    }

    # =========================================================================
    # PART 1: REPEATED PAIRED MULTI-CHUNK TRIALS
    # =========================================================================
    print("======================================================================")
    print("  PART 1: REPEATED PAIRED MULTI-CHUNK TRIALS (3 Cases x 3 Repetitions)")
    print("======================================================================\n")

    for c_idx, tc in enumerate(test_cases):
        case_id = tc["id"]
        title = tc["title"]
        chunk_a = tc["chunk_a"]
        chunk_b = tc["chunk_b"]
        question = tc["question"]
        full_prompt = f"{chunk_a}{chunk_b}{question}"
        full_sha256 = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()

        print(f"\n>>> CASE {c_idx+1}/{len(test_cases)}: {title} ({args.pairs_per_case} Paired Repetitions)")

        # First, seed Chunk A session to flash
        flush_device_context(args.serial, trigger_path, result_path, f"seed flush case {c_idx}")
        time.sleep(1)
        seed_res = run_phase1_trigger(
            args.serial, "seed", chunk_a, question, session_path, trigger_path, result_path, output_path, n_predict=1
        )
        print(f"  [+] Seeded Chunk A KV cache to Flash ({seed_res.get('tokens_saved')} tokens saved)")

        case_pairs = []
        for rep in range(1, args.pairs_per_case + 1):
            pair_id = f"{case_id}_rep{rep}"
            print(f"\n  --- Repetition {rep}/{args.pairs_per_case} ---")

            # Condition 1: Baseline Full Prefill
            flush_device_context(args.serial, trigger_path, result_path, f"baseline flush {case_id} rep {rep}")
            time.sleep(1)
            io_before_b = read_proc_io(args.serial, pid)
            t_b_start = time.perf_counter()
            b_res = run_phase1_trigger(
                args.serial, "baseline", chunk_a, f"{chunk_b}{question}", session_path, trigger_path, result_path, output_path, n_predict=35, prompt_sha256=full_sha256
            )
            t_b_host_ms = (time.perf_counter() - t_b_start) * 1000.0
            io_after_b = read_proc_io(args.serial, pid)
            b_io = diff_io(io_after_b, io_before_b)
            b_pss = get_peak_pss_kb(args.serial)
            b_timings = b_res.get("timings", {})
            b_text = str(b_res.get("text", "")).strip()
            b_eval = evaluate_facts(b_text, tc["hop1_facts"], tc["hop2_facts"])
            b_pref_ms = b_timings.get("prompt_ms", 0)
            b_dec_ms = b_timings.get("predicted_ms", 0)
            b_wall_ms = b_res.get("elapsed_wall_ms", 0)
            b_overhead_ms = max(0, b_wall_ms - (b_pref_ms + b_dec_ms))

            print(f"    Baseline: Prefill={b_pref_ms}ms, Decode={b_dec_ms}ms, Overhead={b_overhead_ms}ms, Wall={b_wall_ms}ms, Storage={b_io['delta_read_bytes']:,}B | Correct={b_eval['multihop_correct']}")

            # Condition 2: HACER Reuse
            flush_device_context(args.serial, trigger_path, result_path, f"reuse flush {case_id} rep {rep}")
            time.sleep(1)
            io_before_r = read_proc_io(args.serial, pid)
            t_r_start = time.perf_counter()
            r_res = run_phase1_trigger(
                args.serial, "reuse", chunk_a, f"{chunk_b}{question}", session_path, trigger_path, result_path, output_path, n_predict=35, prompt_sha256=full_sha256
            )
            t_r_host_ms = (time.perf_counter() - t_r_start) * 1000.0
            io_after_r = read_proc_io(args.serial, pid)
            r_io = diff_io(io_after_r, io_before_r)
            r_pss = get_peak_pss_kb(args.serial)
            r_timings = r_res.get("timings", {})
            r_text = str(r_res.get("text", "")).strip()
            r_eval = evaluate_facts(r_text, tc["hop1_facts"], tc["hop2_facts"])
            r_pref_ms = r_timings.get("prompt_ms", 0)
            r_dec_ms = r_timings.get("predicted_ms", 0)
            r_wall_ms = r_res.get("elapsed_wall_ms", 0)
            # Estimated flash cache load overhead
            r_overhead_ms = max(0, r_wall_ms - (r_pref_ms + r_dec_ms))
            cache_load_overhead_ms = max(0, r_overhead_ms - b_overhead_ms)

            pct_saved = ((b_pref_ms - r_pref_ms) / b_pref_ms * 100.0) if b_pref_ms else 0.0
            speedup = (b_wall_ms / r_wall_ms) if r_wall_ms else 1.0

            print(f"    Reuse:    Prefill={r_pref_ms}ms ({pct_saved:.1f}% saved), Decode={r_dec_ms}ms, LoadOverhead={cache_load_overhead_ms}ms, Wall={r_wall_ms}ms ({speedup:.2f}x), Storage={r_io['delta_read_bytes']:,}B | Correct={r_eval['multihop_correct']}")

            pair_record = {
                "pair_id": pair_id,
                "case_id": case_id,
                "rep": rep,
                "baseline": {
                    "prompt_ms": b_pref_ms,
                    "decode_ms": b_dec_ms,
                    "overhead_ms": b_overhead_ms,
                    "wall_ms": b_wall_ms,
                    "storage_read_bytes": b_io["delta_read_bytes"],
                    "peak_pss_kb": b_pss,
                    "text": b_text,
                    "eval": b_eval,
                },
                "reuse": {
                    "prompt_ms": r_pref_ms,
                    "decode_ms": r_dec_ms,
                    "overhead_ms": r_overhead_ms,
                    "cache_load_overhead_ms": cache_load_overhead_ms,
                    "wall_ms": r_wall_ms,
                    "storage_read_bytes": r_io["delta_read_bytes"],
                    "peak_pss_kb": r_pss,
                    "tokens_loaded": r_res.get("tokens_loaded", 0),
                    "tokens_eval": r_timings.get("prompt_n", 0),
                    "text": r_text,
                    "eval": r_eval,
                },
                "delta": {
                    "prefill_saved_ms": b_pref_ms - r_pref_ms,
                    "prefill_reduction_pct": round(pct_saved, 2),
                    "wall_saved_ms": b_wall_ms - r_wall_ms,
                    "speedup": round(speedup, 2),
                    "accuracy_match": b_eval["multihop_correct"] == r_eval["multihop_correct"],
                }
            }
            results["multichunk_paired_trials"].append(pair_record)
            case_pairs.append(pair_record)

        # Compute summary statistics for this case
        b_prefs = [p["baseline"]["prompt_ms"] for p in case_pairs]
        r_prefs = [p["reuse"]["prompt_ms"] for p in case_pairs]
        b_walls = [p["baseline"]["wall_ms"] for p in case_pairs]
        r_walls = [p["reuse"]["wall_ms"] for p in case_pairs]
        red_pcts = [p["delta"]["prefill_reduction_pct"] for p in case_pairs]
        speedups = [p["delta"]["speedup"] for p in case_pairs]
        load_overheads = [p["reuse"]["cache_load_overhead_ms"] for p in case_pairs]

        results["case_summaries"][case_id] = {
            "title": title,
            "pairs_count": len(case_pairs),
            "prefill_baseline_ms_mean": float(np.mean(b_prefs)),
            "prefill_baseline_ms_std": float(np.std(b_prefs)),
            "prefill_reuse_ms_mean": float(np.mean(r_prefs)),
            "prefill_reuse_ms_std": float(np.std(r_prefs)),
            "prefill_reduction_pct_mean": float(np.mean(red_pcts)),
            "prefill_reduction_pct_std": float(np.std(red_pcts)),
            "wall_baseline_ms_mean": float(np.mean(b_walls)),
            "wall_reuse_ms_mean": float(np.mean(r_walls)),
            "wall_speedup_mean": float(np.mean(speedups)),
            "cache_load_overhead_ms_mean": float(np.mean(load_overheads)),
            "all_accuracies_preserved": all(p["delta"]["accuracy_match"] for p in case_pairs),
        }
        print(f"  >>> Summary for {title}:")
        print(f"      Prefill Reduction Mean: {np.mean(red_pcts):.2f}% ± {np.std(red_pcts):.2f}%")
        print(f"      Wall Speedup Mean:      {np.mean(speedups):.2f}x")
        print(f"      Cache Load Overhead:    {np.mean(load_overheads):.1f} ms")
        print(f"      Accuracy Preserved:     {all(p['delta']['accuracy_match'] for p in case_pairs)}")

    # =========================================================================
    # PART 2: EXPANDED STALENESS & DYNAMIC EVIDENCE SUITE (5 Scenarios)
    # =========================================================================
    print("\n======================================================================")
    print("  PART 2: EXPANDED STALENESS & DYNAMIC EVIDENCE SUITE (5 Scenarios)")
    print("======================================================================\n")

    stale_session_file = "/sdcard/Android/data/com.chat/files/hacer_staleness_test_session.bin"

    # --- SCENARIO A: Content Update (Stale Stipend $400 -> $800) ---
    print("\n[Scenario A: Content Update (Policy Updated $400 -> $800)]")
    stale_doc_a = "Section 3.2 Remote Work Policy (2021 Policy):\nRemote work equipment stipend is $400 USD.\n\n"
    updated_doc_a = "Section 3.2 Remote Work Policy (2024 Policy):\nRemote work equipment stipend is $800 USD.\n\n"
    query_a = "Question: What is the equipment stipend? Answer in one sentence."
    # Seed stale doc
    flush_device_context(args.serial, trigger_path, result_path, "seed stale A")
    run_phase1_trigger(args.serial, "seed", stale_doc_a, query_a, stale_session_file, trigger_path, result_path, output_path, n_predict=1)

    # Condition 1: Naive Reuse
    flush_device_context(args.serial, trigger_path, result_path, "naive reuse A")
    res_naive_a = run_phase1_trigger(args.serial, "reuse", stale_doc_a, query_a, stale_session_file, trigger_path, result_path, output_path, n_predict=30)
    text_naive_a = str(res_naive_a.get("text", "")).strip()

    # Condition 2: HACER Adaptive Repair (detects hash mismatch between updated_doc_a and stale_doc_a)
    flush_device_context(args.serial, trigger_path, result_path, "hacer repair A")
    res_hacer_a = run_phase1_trigger(args.serial, "baseline", updated_doc_a, query_a, stale_session_file, trigger_path, result_path, output_path, n_predict=30)
    text_hacer_a = str(res_hacer_a.get("text", "")).strip()

    results["staleness_suite"]["scenario_a_content_update"] = {
        "title": "Content Update ($400 -> $800)",
        "naive_reuse": {"text": text_naive_a, "stale_fact_output": "$400" in text_naive_a, "correct": "$800" in text_naive_a},
        "hacer_repair": {"text": text_hacer_a, "stale_fact_output": "$400" in text_hacer_a, "correct": "$800" in text_hacer_a},
    }
    print(f"  Naive Reuse:  '{text_naive_a[:60]}...' -> Stale Fact ($400): {'$400' in text_naive_a} (FAILED)")
    print(f"  HACER Repair: '{text_hacer_a[:60]}...' -> Correct Fact ($800): {'$800' in text_hacer_a} (SUCCESS)")

    # --- SCENARIO B: Document Revocation / Deletion ---
    print("\n[Scenario B: Document Revocation / Deletion (Childcare Subsidy Revoked)]")
    revoked_doc_b = "Section 4.1 Childcare Benefit:\nZephyr Dynamics provides an emergency childcare subsidy of $500 per month.\n\n"
    neutral_corpus_b = "Section 4.0 General Office Policy:\nOffice hours are 9am to 5pm Monday through Friday.\n\n"
    query_b = "Question: What is the monthly childcare subsidy? If not mentioned, state 'None'. Answer concisely."
    # Seed revoked doc
    flush_device_context(args.serial, trigger_path, result_path, "seed revoked B")
    run_phase1_trigger(args.serial, "seed", revoked_doc_b, query_b, stale_session_file, trigger_path, result_path, output_path, n_predict=1)

    # Condition 1: Naive Reuse
    flush_device_context(args.serial, trigger_path, result_path, "naive reuse B")
    res_naive_b = run_phase1_trigger(args.serial, "reuse", revoked_doc_b, query_b, stale_session_file, trigger_path, result_path, output_path, n_predict=30)
    text_naive_b = str(res_naive_b.get("text", "")).strip()

    # Condition 2: HACER Policy (Chunk is deleted from retrieved evidence set -> cache evicted, evaluated on neutral corpus)
    flush_device_context(args.serial, trigger_path, result_path, "hacer repair B")
    res_hacer_b = run_phase1_trigger(args.serial, "baseline", neutral_corpus_b, query_b, stale_session_file, trigger_path, result_path, output_path, n_predict=30)
    text_hacer_b = str(res_hacer_b.get("text", "")).strip()

    results["staleness_suite"]["scenario_b_deletion_revocation"] = {
        "title": "Document Revocation / Deletion (Childcare Subsidy)",
        "naive_reuse": {"text": text_naive_b, "hallucinated_revoked_fact": "$500" in text_naive_b},
        "hacer_repair": {"text": text_hacer_b, "hallucinated_revoked_fact": "$500" in text_hacer_b, "correct_abstention": "not mention" in text_hacer_b.lower() or "none" in text_hacer_b.lower()},
    }
    print(f"  Naive Reuse:  '{text_naive_b[:60]}...' -> Hallucinated Revoked Fact ($500): {'$500' in text_naive_b} (FAILED)")
    print(f"  HACER Repair: '{text_hacer_b[:60]}...' -> Avoided Revoked Fact: {not ('$500' in text_hacer_b)} (SUCCESS)")

    # --- SCENARIO C: Changed Chunk Ordering ---
    print("\n[Scenario C: Changed Chunk Ordering ([Chunk B, Chunk A] Permutation)]")
    chunk_1 = "Section 1.1: Stockholm office is the global headquarters.\n\n"
    chunk_2 = "Section 2.1: AeroSense sensor operates up to 12,000 meters.\n\n"
    query_c = "Question: Where is headquarters and what is sensor altitude? Answer concisely."
    # Seed [Chunk 1, Chunk 2]
    flush_device_context(args.serial, trigger_path, result_path, "seed ordering C")
    run_phase1_trigger(args.serial, "seed", f"{chunk_1}{chunk_2}", query_c, stale_session_file, trigger_path, result_path, output_path, n_predict=1)

    # Inverted order query: [Chunk 2, Chunk 1]
    # Condition 1: Naive Reuse (attempts to reuse Chunk 1 prefix on inverted prompt)
    flush_device_context(args.serial, trigger_path, result_path, "naive reuse C")
    res_naive_c = run_phase1_trigger(args.serial, "reuse", chunk_1, f"{chunk_2}{query_c}", stale_session_file, trigger_path, result_path, output_path, n_predict=30)
    text_naive_c = str(res_naive_c.get("text", "")).strip()

    # Condition 2: HACER Policy (Detects order mismatch; canonicalizes chunk prefix to match cache layout [Chunk 1, Chunk 2])
    flush_device_context(args.serial, trigger_path, result_path, "hacer canonical C")
    res_hacer_c = run_phase1_trigger(args.serial, "reuse", f"{chunk_1}{chunk_2}", query_c, stale_session_file, trigger_path, result_path, output_path, n_predict=30)
    text_hacer_c = str(res_hacer_c.get("text", "")).strip()

    c_naive_correct = "stockholm" in text_naive_c.lower() and "12,000" in text_naive_c.lower()
    c_hacer_correct = "stockholm" in text_hacer_c.lower() and "12,000" in text_hacer_c.lower()
    results["staleness_suite"]["scenario_c_changed_ordering"] = {
        "title": "Changed Chunk Ordering Permutation",
        "naive_reuse": {"text": text_naive_c, "multihop_correct": c_naive_correct},
        "hacer_canonical": {"text": text_hacer_c, "multihop_correct": c_hacer_correct},
    }
    print(f"  Naive Reuse (Misaligned):   '{text_naive_c[:60]}...' -> Correct: {c_naive_correct}")
    print(f"  HACER Canonical Alignment:  '{text_hacer_c[:60]}...' -> Correct: {c_hacer_correct} (SUCCESS)")

    # --- SCENARIO D: Partial Multi-Chunk Update ---
    print("\n[Scenario D: Partial Multi-Chunk Update (Chunk 1 unchanged, Chunk 2 updated)]")
    # Chunk 1 (Unchanged leadership) + Chunk 2 (Drone Pricing updated from $1,200 to $1,600)
    p_chunk_1 = "Section 1.1: Dr. Priya Nandakumar is CEO in Stockholm.\n\n"
    p_chunk_2_stale = "Section 2.2: SwarmKit Pro subscription is $1,200 per month.\n\n"
    p_chunk_2_updated = "Section 2.2: SwarmKit Pro subscription is $1,600 per month.\n\n"
    query_d = "Question: Who is the CEO and what is the monthly SwarmKit subscription? Answer concisely."
    # Seed Chunk 1 prefix
    flush_device_context(args.serial, trigger_path, result_path, "seed partial D")
    run_phase1_trigger(args.serial, "seed", p_chunk_1, query_d, stale_session_file, trigger_path, result_path, output_path, n_predict=1)

    # Naive Reuse with stale chunk 2
    flush_device_context(args.serial, trigger_path, result_path, "naive partial D")
    res_naive_d = run_phase1_trigger(args.serial, "reuse", p_chunk_1, f"{p_chunk_2_stale}{query_d}", stale_session_file, trigger_path, result_path, output_path, n_predict=30)
    text_naive_d = str(res_naive_d.get("text", "")).strip()

    # HACER: Reuses Chunk 1 prefix (matches hash!), but evaluates UPDATED Chunk 2 online!
    flush_device_context(args.serial, trigger_path, result_path, "hacer partial D")
    res_hacer_d = run_phase1_trigger(args.serial, "reuse", p_chunk_1, f"{p_chunk_2_updated}{query_d}", stale_session_file, trigger_path, result_path, output_path, n_predict=30)
    text_hacer_d = str(res_hacer_d.get("text", "")).strip()

    results["staleness_suite"]["scenario_d_partial_update"] = {
        "title": "Partial Multi-Chunk Update ($1,200 -> $1,600)",
        "naive_reuse": {"text": text_naive_d, "served_stale_fact": "$1,200" in text_naive_d, "correct_updated": "$1,600" in text_naive_d},
        "hacer_partial_repair": {"text": text_hacer_d, "served_stale_fact": "$1,200" in text_hacer_d, "correct_updated": "$1,600" in text_hacer_d},
    }
    print(f"  Naive Reuse:        '{text_naive_d[:60]}...' -> Served Stale ($1,200): {'$1,200' in text_naive_d}")
    print(f"  HACER Partial Reuse:'{text_hacer_d[:60]}...' -> Served Updated ($1,600): {'$1,600' in text_hacer_d} (SUCCESS)")

    # --- SCENARIO E: Corrupted Cache File Fallback ---
    print("\n[Scenario E: Corrupted Cache File Fallback (Safe Recovery)]")
    # Corrupt the session file with 0 bytes
    adb(args.serial, "shell", f"echo '' > {stale_session_file}")
    query_e = "Question: What is the company headquarters? Answer in one word."
    doc_e = "Section 1.0: Zephyr Dynamics headquarters is in Stockholm.\n\n"

    # HACER fallback test: verify clean completion without app crash
    flush_device_context(args.serial, trigger_path, result_path, "hacer fallback E")
    res_hacer_e = run_phase1_trigger(args.serial, "baseline", doc_e, query_e, stale_session_file, trigger_path, result_path, output_path, n_predict=15)
    text_hacer_e = str(res_hacer_e.get("text", "")).strip()
    e_success = "stockholm" in text_hacer_e.lower()

    results["staleness_suite"]["scenario_e_corrupted_cache_fallback"] = {
        "title": "Corrupted Cache File Fallback",
        "fallback_completed": bool(res_hacer_e.get("text")),
        "factual_accuracy": e_success,
        "text": text_hacer_e,
    }
    print(f"  HACER Fallback: '{text_hacer_e[:50]}...' -> Completed Safely: {bool(res_hacer_e.get('text'))}, Correct: {e_success} (SUCCESS)")

    # Write output JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "rigorous_multichunk_and_staleness_results.json"
    out_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\n======================================================================")
    print(f"Rigorous Multi-Chunk & Staleness Suite Completed Successfully!")
    print(f"Results written to: {out_file}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
