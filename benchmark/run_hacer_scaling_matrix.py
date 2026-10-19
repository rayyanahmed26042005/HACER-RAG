#!/usr/bin/env python3
"""HACER-RAG Physical Scaling Matrix Controller on Xiaomi Redmi 12 5G.

Runs the controlled 3x3 scaling matrix:
- K in {1, 2, 4} cached chunks
- n_ctx in {512, 1024, 2048} context capacity
- 9 experimental configurations (C1 to C9)
- 10 paired repetitions per configuration (90 pairs / 180 executions)
- Generation budget: n_predict = 35, temperature = 0

Strict Matched-Workload Protocol:
- The logical prompt is byte-for-byte identical (SHA-256 verified) across all configurations.
- prefix + continuation == canonical_prompt.
- Dynamic n_ctx reconfiguration on physical device.
- Full hardware telemetry: /proc/{PID}/io, thermal, CPU frequencies, PSS memory.
- Randomized execution order within each baseline/HACER pair.
"""

import argparse
import base64
import functools
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

print = functools.partial(print, flush=True)

# -------------------------------------------------------------------------
# Canonical Prompt Construction (Byte-for-Byte Invariant)
# -------------------------------------------------------------------------

CHUNK_1_OVERVIEW = (
    "Zephyr Dynamics Inc. was founded in 2011 by Dr. Priya Nandakumar and Marcus Löbell "
    "in Reykjavik, Iceland. The company specializes in atmospheric data harvesting systems "
    "and micro-drone swarm technology. Zephyr Dynamics has 847 full-time employees across "
    "6 global offices. Annual revenue for 2023 was $412 million USD."
)

CHUNK_2_LEADERSHIP = (
    "Dr. Priya Nandakumar serves as Chief Executive Officer (CEO) and Co-Founder in Reykjavik. "
    "Marcus Löbell serves as Chief Technology Officer (CTO). "
    "Saoirse Hennessey serves as Chief Financial Officer (CFO). "
    "Jorge Fonseca serves as Chief Legal Officer (CLO), having joined the executive team in 2019."
)

CHUNK_3_AEROSENSE = (
    "The AeroSense X7 is Zephyr Dynamics flagship atmospheric sensor array, launched in Q2 2022. "
    "It measures temperature, humidity, particulate matter, and wind shear at altitudes up to 12,000 meters. "
    "The X7 uses a proprietary QuantumCell battery that lasts 72 hours on a single charge. Unit price: $8,400 USD."
)

CHUNK_4_SWARMKIT = (
    "SwarmKit Pro is a micro-drone coordination platform released in January 2023. "
    "It supports simultaneous control of up to 2,048 drones using the ZephyrMesh v3 protocol. "
    "Licensing is subscription-based: $1,200/month for standard capacity, "
    "and $3,800/month for the full 2,048-drone capacity tier."
)

CANONICAL_QUESTION = (
    "Question: In what year was Zephyr Dynamics founded, who is the CLO, "
    "what is the battery duration of the AeroSense X7, and what is the monthly subscription "
    "price for the full-capacity tier of SwarmKit Pro? Answer in one concise paragraph.\nAnswer:"
)


def build_canonical_prompt() -> str:
    chunks = [
        f"[1] Company Overview:\n{CHUNK_1_OVERVIEW}",
        f"[2] Leadership Team:\n{CHUNK_2_LEADERSHIP}",
        f"[3] AeroSense X7 Specifications:\n{CHUNK_3_AEROSENSE}",
        f"[4] SwarmKit Pro Platform:\n{CHUNK_4_SWARMKIT}",
    ]
    evidence = "\n\n".join(chunks)
    return f"{evidence}\n\n{CANONICAL_QUESTION}"


def get_prefix_and_continuation(k_chunks: int) -> Tuple[str, str]:
    chunks = [
        f"[1] Company Overview:\n{CHUNK_1_OVERVIEW}",
        f"[2] Leadership Team:\n{CHUNK_2_LEADERSHIP}",
        f"[3] AeroSense X7 Specifications:\n{CHUNK_3_AEROSENSE}",
        f"[4] SwarmKit Pro Platform:\n{CHUNK_4_SWARMKIT}",
    ]
    if k_chunks == 1:
        prefix = f"{chunks[0]}\n\n"
        continuation = f"{chunks[1]}\n\n{chunks[2]}\n\n{chunks[3]}\n\n{CANONICAL_QUESTION}"
    elif k_chunks == 2:
        prefix = f"{chunks[0]}\n\n{chunks[1]}\n\n"
        continuation = f"{chunks[2]}\n\n{chunks[3]}\n\n{CANONICAL_QUESTION}"
    elif k_chunks == 4:
        prefix = f"{chunks[0]}\n\n{chunks[1]}\n\n{chunks[2]}\n\n{chunks[3]}\n\n"
        continuation = CANONICAL_QUESTION
    else:
        raise ValueError(f"Unsupported K: {k_chunks}")
    return prefix, continuation


# -------------------------------------------------------------------------
# ADB & Low-Level Mobile System Telemetry
# -------------------------------------------------------------------------

def adb(serial: str, *args, check: bool = True) -> str:
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


def get_battery_temp(serial: str) -> Optional[float]:
    raw = adb(serial, "shell", "dumpsys battery", check=False)
    for line in raw.splitlines():
        if "temperature:" in line:
            try:
                # Android reports tenths of a degree Celsius (e.g. 410 = 41.0 C)
                val = float(line.split(":", 1)[1].strip())
                return val / 10.0
            except ValueError:
                pass
    return None


def get_cpu_frequencies(serial: str) -> Dict[str, int]:
    freqs = {}
    # Snapdragon 4 Gen 2: cores 0-5 (A55 LITTLE), cores 6-7 (A78 BIG)
    for core in (0, 4, 6, 7):
        path = f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq"
        val = adb(serial, "shell", f"cat {path} 2>/dev/null || true", check=False)
        if val.strip().isdigit():
            freqs[f"cpu{core}_khz"] = int(val.strip())
    return freqs


def get_pss_kb(serial: str, pkg: str = "com.chat") -> Optional[int]:
    raw = adb(serial, "shell", f"dumpsys meminfo {pkg}", check=False)
    for line in raw.splitlines():
        if "TOTAL PSS:" in line or "TOTAL:" in line:
            parts = line.split()
            for part in parts:
                if part.isdigit():
                    return int(part)
    return None


def get_file_metadata(serial: str, path: str) -> Tuple[int, str]:
    exists = adb(serial, "shell", f"test -f {path} && echo yes || echo no", check=False)
    if exists.strip() != "yes":
        return 0, ""
    size_str = adb(serial, "shell", f"stat -c %s {path} 2>/dev/null || wc -c < {path}", check=False)
    size = int(size_str.strip()) if size_str.strip().isdigit() else 0
    sha = adb(serial, "shell", f"sha256sum {path} 2>/dev/null || true", check=False).split()
    sha_str = sha[0] if sha else ""
    return size, sha_str


# -------------------------------------------------------------------------
# Trigger Protocol & Context Controller
# -------------------------------------------------------------------------

def send_trigger(serial: str, trigger: Dict[str, Any], trigger_path: str, result_path: str, timeout: int = 180, output_path: Optional[str] = None) -> Dict[str, Any]:
    ensure_app_running(serial)
    encoded = base64.b64encode(json.dumps(trigger).encode("utf-8")).decode("ascii")
    tmp_trigger = f"{trigger_path}.tmp"
    rm_cmd = f"rm -f {result_path} {trigger_path} {tmp_trigger}"
    if output_path:
        rm_cmd += f" {output_path}"
    adb(serial, "shell", rm_cmd)
    # Atomic write via temporary file to prevent partial-read race condition in React Native
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


def reconfigure_context(serial: str, target_n_ctx: int, trigger_path: str, result_path: str) -> None:
    print(f"[*] Reconfiguring device LlamaContext to n_ctx={target_n_ctx}...")
    req_id = uuid.uuid4().hex
    trigger = {
        "requestId": req_id,
        "resultPath": result_path,
        "action": "reconfigure_context",
        "nCtx": target_n_ctx,
    }
    res = send_trigger(serial, trigger, trigger_path, result_path, timeout=180)
    if res.get("status") != "completed":
        raise RuntimeError(f"Context reconfiguration failed: {res}")
    active_ctx = res.get("active_n_ctx")
    print(f"[+] Context reconfigured. Active n_ctx on device: {active_ctx}")
    if active_ctx != target_n_ctx:
        raise RuntimeError(f"Active n_ctx ({active_ctx}) does not match requested ({target_n_ctx})")


def flush_device_context(serial: str, trigger_path: str, result_path: str, topic: str = "quantum_chromodynamics") -> None:
    req_id = uuid.uuid4().hex
    prompt = f"Explain {topic} and gluon field dynamics in physics. Answer briefly."
    trigger = {
        "requestId": req_id,
        "resultPath": result_path,
        "query": prompt,
        "chatMode": "general",
        "phase1": "baseline",
        "phase1Prefix": prompt,
        "phase1Suffix": " Complete in 5 words.",
        "phase1SessionPath": "/sdcard/Android/data/com.chat/files/flush_session.bin",
        "phase1OutputPath": "/sdcard/Android/data/com.chat/files/flush_result.json",
        "phase1NPredict": 1,
    }
    send_trigger(serial, trigger, trigger_path, result_path, timeout=180)


# -------------------------------------------------------------------------
# Paired Trial Execution
# -------------------------------------------------------------------------

def execute_single_trial(
    serial: str,
    condition: str,  # 'baseline' or 'reuse'
    canonical_prompt: str,
    prefix: str,
    continuation: str,
    session_path: str,
    n_predict: int,
    expected_n_ctx: int,
    trigger_path: str,
    result_path: str,
    output_path: str,
) -> Dict[str, Any]:
    pid = get_app_pid(serial)
    
    # Pre-trial telemetry
    io_before = read_proc_io(serial, pid)
    temp_before = get_battery_temp(serial)
    freqs_before = get_cpu_frequencies(serial)
    pss_before = get_pss_kb(serial)
    
    req_id = uuid.uuid4().hex
    prompt_sha256 = hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()
    
    if condition == "baseline":
        p1_prefix = canonical_prompt
        p1_suffix = ""
    else:
        p1_prefix = prefix
        p1_suffix = continuation

    trigger = {
        "requestId": req_id,
        "resultPath": result_path,
        "query": f"Scaling Study: {condition}",
        "chatMode": "general",
        "phase1": condition,
        "phase1Prefix": p1_prefix,
        "phase1Suffix": p1_suffix,
        "phase1SessionPath": session_path,
        "phase1OutputPath": output_path,
        "phase1NPredict": n_predict,
        "promptSha256": prompt_sha256,
        "nCtx": expected_n_ctx,
    }

    wall_start = time.monotonic()
    dev_res = send_trigger(serial, trigger, trigger_path, result_path, timeout=180, output_path=output_path)
    wall_elapsed_ms = (time.monotonic() - wall_start) * 1000.0
    if dev_res.get("status") != "completed":
        raise RuntimeError(f"Trial failed on device: {dev_res}")

    # Post-trial telemetry
    wall_post = time.monotonic()
    io_after = read_proc_io(serial, pid)
    temp_after = get_battery_temp(serial)
    freqs_after = get_cpu_frequencies(serial)
    pss_after = get_pss_kb(serial)

    io_delta = diff_io(io_after, io_before)

    # Read probe output file written by React Native
    raw_p1 = adb(serial, "shell", f"cat {output_path} 2>/dev/null || true", check=False)
    p1 = json.loads(raw_p1) if raw_p1 else dev_res.get("phase1", {})

    timings = p1.get("timings") or {}
    prefill_ms = timings.get("prompt_ms") or 0.0
    decode_ms = timings.get("predicted_ms") or 0.0
    tok_eval = p1.get("tokens_evaluated")
    tok_pred = p1.get("tokens_predicted")
    tok_loaded = p1.get("tokens_loaded", 0)
    tok_cached = p1.get("tokens_cached", 0)
    resp_text = p1.get("text", "")
    if p1.get("truncated"):
        raise RuntimeError(f"FATAL: Native llama engine reported truncated=true for {condition}! Workload exceeded context limit.")

    # Storage condition classification
    sess_size, sess_sha = get_file_metadata(serial, session_path) if condition == "reuse" else (0, "")
    storage_class = "unverified"
    if condition == "reuse" and sess_size > 0:
        if io_delta["delta_read_bytes"] >= (sess_size * 0.85):
            storage_class = "cold_verified"
        elif io_delta["delta_rchar"] >= (sess_size * 0.85):
            storage_class = "warm_candidate"

    prompt_tokens_total = p1.get("full_prompt_token_count", 395)
    tokens_loaded_from_session = p1.get("tokens_loaded", 0)
    tokens_evaluated_this_run = timings.get("prompt_n", prompt_tokens_total if condition == "baseline" else (prompt_tokens_total - tokens_loaded_from_session + 1))
    resident_kv_tokens_post_decode = tok_cached

    # Context budget headroom metrics
    # 1. Enforced safety headroom under llama.cpp's (n_ctx - 4) constraint with requested generation budget:
    safety_headroom_enforced = (expected_n_ctx - 4) - (prompt_tokens_total + n_predict)
    # 2. Raw capacity remaining under nominal n_ctx with requested generation budget:
    raw_context_remaining = expected_n_ctx - (prompt_tokens_total + n_predict)
    # 3. Post-decode safety headroom using actual generated tokens:
    post_decode_safety_headroom = (expected_n_ctx - 4) - (prompt_tokens_total + (tok_pred if tok_pred is not None else n_predict))

    return {
        "condition": condition,
        "request_id": req_id,
        "wall_elapsed_ms": wall_elapsed_ms,
        "device_elapsed_wall_ms": p1.get("elapsed_wall_ms"),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prompt_tokens_total": prompt_tokens_total,
        "tokens_loaded_from_session": tokens_loaded_from_session,
        "tokens_evaluated_this_run": tokens_evaluated_this_run,
        "resident_kv_tokens_post_decode": resident_kv_tokens_post_decode,
        "tokens_predicted": tok_pred,
        "tokens_evaluated_raw_rn": tok_eval,
        "tokens_cached_raw_rn": tok_cached,
        "truncated": p1.get("truncated", False),
        "stopped_eos": p1.get("stopped_eos", False),
        "stopped_limit": p1.get("stopped_limit", False),
        "full_prompt_tokens_native": prompt_tokens_total,
        "prefix_tokens_native": p1.get("prefix_token_count"),
        "prompt_n_evaluated": tokens_evaluated_this_run,
        "context_headroom": safety_headroom_enforced,
        "safety_headroom_enforced": safety_headroom_enforced,
        "raw_context_remaining": raw_context_remaining,
        "post_decode_safety_headroom": post_decode_safety_headroom,
        "context_headroom_raw_rn": p1.get("context_headroom"),
        "response_text": resp_text,
        "prompt_sha256": prompt_sha256,
        "active_n_ctx": dev_res.get("active_n_ctx") or p1.get("active_n_ctx"),
        "session_path": session_path,
        "session_size_bytes": sess_size,
        "session_sha256": sess_sha,
        "storage_condition": storage_class,
        "io_delta": io_delta,
        "battery_temp_c": temp_after,
        "cpu_frequencies": freqs_after,
        "pss_kb": pss_after,
    }


# -------------------------------------------------------------------------
# Seed Session Generator
# -------------------------------------------------------------------------

def ensure_session_seeded(serial: str, k_chunks: int, prefix: str, session_path: str, trigger_path: str, result_path: str, output_path: str) -> Tuple[int, str]:
    size, sha = get_file_metadata(serial, session_path)
    if size > 10000 and sha:
        print(f"[*] Session cache for K={k_chunks} already exists: {size} bytes (SHA256: {sha[:12]}...)")
        return size, sha

    print(f"[*] Seeding session cache for K={k_chunks} ({len(prefix)} chars)...")
    flush_device_context(serial, trigger_path, result_path, "session_seed_flush")

    req_id = uuid.uuid4().hex
    trigger = {
        "requestId": req_id,
        "resultPath": result_path,
        "query": f"HACER Seed K={k_chunks}",
        "chatMode": "general",
        "phase1": "seed",
        "phase1Prefix": prefix,
        "phase1Suffix": " Seed",
        "phase1SessionPath": session_path,
        "phase1OutputPath": output_path,
        "phase1NPredict": 1,
    }
    res = send_trigger(serial, trigger, trigger_path, result_path, timeout=120)
    size, sha = get_file_metadata(serial, session_path)
    if size == 0 or not sha:
        raise RuntimeError(f"Failed to generate session binary at {session_path}")
    print(f"[+] Seeded K={k_chunks} session: {size} bytes (SHA256: {sha[:12]}...)")
    return size, sha


# -------------------------------------------------------------------------
# Main Scaling Matrix Sweep Runner
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HACER-RAG Physical Scaling Matrix Suite")
    parser.add_argument("--serial", default="36fef3202145", help="ADB target serial")
    parser.add_argument("--output-dir", default="benchmark/results/hacer_scaling_matrix", help="Results directory")
    parser.add_argument("--repetitions", type=int, default=10, help="Paired repetitions per configuration")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 pair of K=1/512, K=2/1024, K=4/2048")
    parser.add_argument("--n-predict", type=int, default=35, help="Tokens to generate")
    args = parser.parse_args()

    trigger_path = "/sdcard/Android/data/com.chat/files/benchmark_trigger.json"
    result_path = "/sdcard/Android/data/com.chat/files/benchmark_result.json"
    output_path = "/sdcard/Android/data/com.chat/files/hacer_phase1_result.json"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_log_path = out_dir / ("smoke_test_trials_v2.jsonl" if args.smoke_test else "scaling_trials.jsonl")
    summary_path = out_dir / ("smoke_test_summary_v2.json" if args.smoke_test else "scaling_summary.json")

    canonical_prompt = build_canonical_prompt()
    canonical_sha256 = hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()
    print(f"=== HACER-RAG Physical Scaling Matrix Study ===")
    print(f"Device: Xiaomi Redmi 12 5G ({args.serial})")
    print(f"Canonical Prompt Length: {len(canonical_prompt)} chars (SHA256: {canonical_sha256})")
    print(f"Output File: {trials_log_path}")

    # Wake device & confirm app
    adb(args.serial, "shell", "input keyevent KEYCODE_WAKEUP", check=False)
    pid = get_app_pid(args.serial)
    print(f"Target process: com.chat (PID {pid})")

    # Define the 3x3 matrix configurations
    if args.smoke_test:
        configs = [
            {"id": "smoke_C1_K1_512", "k": 1, "n_ctx": 512, "reps": 1},
            {"id": "smoke_C5_K2_1024", "k": 2, "n_ctx": 1024, "reps": 1},
            {"id": "smoke_C9_K4_2048", "k": 4, "n_ctx": 2048, "reps": 1},
        ]
    else:
        configs = [
            {"id": "C1_K1_CTX512", "k": 1, "n_ctx": 512, "reps": args.repetitions},
            {"id": "C2_K1_CTX1024", "k": 1, "n_ctx": 1024, "reps": args.repetitions},
            {"id": "C3_K1_CTX2048", "k": 1, "n_ctx": 2048, "reps": args.repetitions},
            {"id": "C4_K2_CTX512", "k": 2, "n_ctx": 512, "reps": args.repetitions},
            {"id": "C5_K2_CTX1024", "k": 2, "n_ctx": 1024, "reps": args.repetitions},
            {"id": "C6_K2_CTX2048", "k": 2, "n_ctx": 2048, "reps": args.repetitions},
            {"id": "C7_K4_CTX512", "k": 4, "n_ctx": 512, "reps": args.repetitions},
            {"id": "C8_K4_CTX1024", "k": 4, "n_ctx": 1024, "reps": args.repetitions},
            {"id": "C9_K4_CTX2048", "k": 4, "n_ctx": 2048, "reps": args.repetitions},
        ]

    all_completed_trials = []

    # Read existing trials for crash recovery / resumption
    if trials_log_path.exists():
        with open(trials_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_completed_trials.append(json.loads(line))
                    except Exception:
                        pass
        print(f"[*] Resumed: found {len(all_completed_trials)} existing trial records.")

    for cfg in configs:
        cfg_id = cfg["id"]
        k = cfg["k"]
        n_ctx = cfg["n_ctx"]
        reps = cfg["reps"]
        prefix, continuation = get_prefix_and_continuation(k)
        session_file = f"/sdcard/Android/data/com.chat/files/hacer_v2_k{k}_ctx{n_ctx}_session.bin"

        # Verification of prompt reconstruction
        assert prefix + continuation == canonical_prompt, f"Prompt reconstruction failed for K={k}"
        assert hashlib.sha256((prefix + continuation).encode("utf-8")).hexdigest() == canonical_sha256

        print(f"\n=======================================================")
        print(f"Running Configuration {cfg_id}: K={k} chunks, n_ctx={n_ctx}, reps={reps}")
        print(f"=======================================================")

        completed_reps = sum(1 for r in range(1, reps + 1) if len([t for t in all_completed_trials if t.get("config_id") == cfg_id and t.get("rep") == r]) >= 2)
        if completed_reps >= reps:
            print(f"[*] Configuration {cfg_id} already fully completed ({completed_reps}/{reps} reps). Skipping.")
            continue

        # 1. Reconfigure active context size on device
        reconfigure_context(args.serial, n_ctx, trigger_path, result_path)

        # 2. Ensure session cache is seeded for this K and n_ctx
        ensure_session_seeded(args.serial, k, prefix, session_file, trigger_path, result_path, output_path)

        for rep in range(1, reps + 1):
            pair_id = f"{cfg_id}_rep{rep}_{uuid.uuid4().hex[:6]}"
            
            # Check if this rep was already completed in existing records
            existing_pair = [t for t in all_completed_trials if t.get("config_id") == cfg_id and t.get("rep") == rep]
            if len(existing_pair) >= 2:
                print(f"[*] Repetition {rep}/{reps} already completed. Skipping.")
                continue

            # Randomize order within pair to neutralize thermal drift
            order = ["baseline", "reuse"] if (random.random() < 0.5) else ["reuse", "baseline"]
            print(f"\n--- Repetition {rep}/{reps} (Pair: {pair_id}) | Order: {order} ---")

            pair_records = []
            for cond in order:
                # Flush context before EVERY trial to eliminate in-memory KV retention
                print(f"[{cond.upper()}] Flushing context...")
                flush_device_context(args.serial, trigger_path, result_path, f"flush_{pair_id}_{cond}")

                print(f"[{cond.upper()}] Executing {cond} inference on Redmi 12 5G...")
                trial_res = execute_single_trial(
                    serial=args.serial,
                    condition=cond,
                    canonical_prompt=canonical_prompt,
                    prefix=prefix,
                    continuation=continuation,
                    session_path=session_file,
                    n_predict=args.n_predict,
                    expected_n_ctx=n_ctx,
                    trigger_path=trigger_path,
                    result_path=result_path,
                    output_path=output_path,
                )

                trial_res["config_id"] = cfg_id
                trial_res["k_chunks"] = k
                trial_res["n_ctx"] = n_ctx
                trial_res["rep"] = rep
                trial_res["pair_id"] = pair_id
                trial_res["execution_order"] = order.index(cond) + 1
                trial_res["timestamp"] = time.time()

                # Print live telemetry
                print(f"  -> Prefill: {trial_res['prefill_ms']:.1f} ms | Decode: {trial_res['decode_ms']:.1f} ms | "
                      f"Tokens: eval_this_run={trial_res['tokens_evaluated_this_run']}, loaded={trial_res['tokens_loaded_from_session']}, total_prompt={trial_res['prompt_tokens_total']} | "
                      f"IO read={trial_res['io_delta']['delta_read_bytes']} B | Temp={trial_res['battery_temp_c']} C")

                # Stream to JSONL
                with open(trials_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trial_res) + "\n")
                all_completed_trials.append(trial_res)
                pair_records.append(trial_res)
                time.sleep(1.0)

    print("\n[+] Scaling matrix execution finished!")


if __name__ == "__main__":
    main()
