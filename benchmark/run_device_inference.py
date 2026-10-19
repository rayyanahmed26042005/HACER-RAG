#!/usr/bin/env python3
"""
Device-Native RAG Inference Profiler for Physical Mobile (Xiaomi Snapdragon 4 Gen 2).
Measures:
  - Total warm latency
  - Prefill latency (ms), Prompt tokens, Prefill throughput (tok/s)
  - Decode latency (ms), Generated tokens, Generation throughput (tok/s)
  - TTFT (ms)
  - Hardware state: CPU frequencies, thermal throttling, battery temp, peak PSS.
"""

import os
import sys
import json
import time
import subprocess
import argparse
import base64
import re
import shutil
import uuid
from datetime import datetime

# Prevent charmap encoding errors on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ADB_BIN = os.environ.get("ADB_BIN") or ("adb.exe" if shutil.which("adb.exe") else "adb")
DEVICE_SERIAL = os.environ.get("ANDROID_SERIAL", "36fef3202145")

def adb_cmd(cmd):
    if isinstance(cmd, str):
        if cmd.startswith("shell "):
            sub = cmd[6:].strip()
            if (sub.startswith("'") and sub.endswith("'")) or (sub.startswith('"') and sub.endswith('"')):
                sub = sub[1:-1]
            args = [ADB_BIN, "-s", DEVICE_SERIAL, "shell", sub]
        else:
            args = [ADB_BIN, "-s", DEVICE_SERIAL] + cmd.split()
    else:
        args = [ADB_BIN, "-s", DEVICE_SERIAL] + list(cmd)
    res = subprocess.run(args, capture_output=True, text=True)
    return res.stdout.strip()

def get_device_hardware_state():
    print("Reading physical Xiaomi hardware metrics...")
    
    # 1. CPU Frequencies
    cur_freqs = adb_cmd("shell 'cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null'").splitlines()
    max_freqs = adb_cmd("shell 'cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq 2>/dev/null'").splitlines()
    
    # 2. Battery & Temperature
    battery_dump = adb_cmd("shell 'dumpsys battery'")
    temp_line = [l for l in battery_dump.splitlines() if "temperature:" in l]
    temp_c = float(temp_line[0].split(":", 1)[1].strip()) / 10.0 if temp_line else 0.0
    level_line = [l for l in battery_dump.splitlines() if "level:" in l]
    battery_level = int(level_line[0].split(":", 1)[1].strip()) if level_line else None

    # Thermal status is an Android power/thermal signal, not proof that a
    # frequency change was caused by thermal throttling. Preserve the raw
    # service output so a later analysis can audit the parser.
    thermal_dump = adb_cmd("shell 'dumpsys thermalservice 2>/dev/null'")
    thermal_status = None
    status_match = re.search(r"Current thermal status:\s*(\w+)", thermal_dump, re.IGNORECASE)
    if status_match:
        thermal_status = status_match.group(1)

    # 3. Memory PSS of com.chat
    mem_dump = adb_cmd("shell 'dumpsys meminfo com.chat'")
    pss_lines = [l for l in mem_dump.splitlines() if "TOTAL PSS:" in l]
    total_pss_kb = 0
    if pss_lines:
        parts = pss_lines[0].split()
        if len(parts) >= 3:
            try:
                total_pss_kb = int(parts[2].replace(",", ""))
            except ValueError:
                pass

    return {
        "current_frequencies_khz": [int(f) for f in cur_freqs if f.isdigit()],
        "max_frequencies_khz": [int(f) for f in max_freqs if f.isdigit()],
        "battery_temp_c": temp_c,
        "battery_level_percent": battery_level,
        "thermal_status": thermal_status,
        "pss_mb": round(total_pss_kb / 1024.0, 2),
    }

def run_test(query="Tell me about zypher", chat_mode="document", inference_mode="baseline",
             n_threads=None, context_token_budget=None, max_output_tokens=None,
             model_path=None, rag_policy=None, run_id=None, output_path=None,
             query_id=None, corpus_id=None, profile_id=None,
             allow_direct_evidence=False, allow_legacy_global_result=False):
    print(f"\n========================================================")
    print(f" Physical Mobile Profiling: Xiaomi Redmi 12 5G (Snapdragon 4 Gen 2)")
    print(f" Query: '{query}' | Mode: {chat_mode}")
    print(f"========================================================")

    # Pre-test hardware capture
    pre_hw = get_device_hardware_state()
    print(f"Pre-test Battery Temp: {pre_hw['battery_temp_c']} °C")
    print(f"Pre-test Battery Level: {pre_hw.get('battery_level_percent')} %")
    print(f"Pre-test Thermal Status: {pre_hw.get('thermal_status', 'unavailable')}")
    print(f"Pre-test App PSS RAM : {pre_hw['pss_mb']} MB")
    if pre_hw['current_frequencies_khz']:
        print(f"CPU Frequencies      : {[round(freq / 1000.0, 1) for freq in pre_hw['current_frequencies_khz']]} MHz")
        print(f"CPU Max Frequencies  : {[round(freq / 1000.0, 1) for freq in pre_hw['max_frequencies_khz']]} MHz")

    if chat_mode == "document" and rag_policy not in ("ebmrag-v2", "mobcert-v1"):
        raise ValueError("Document physical benchmarks require an explicit --rag-policy")

    request_id = uuid.uuid4().hex
    remote_dir = "/sdcard/Android/data/com.chat/files/benchmark_requests"
    result_path = f"{remote_dir}/{request_id}.json"
    status_path = f"{remote_dir}/{request_id}.status.json"

    # Send trigger to mobile app via internal storage. The request ID and
    # request-specific result path prevent stale reports from being accepted.
    trigger_data = {
        "requestId": request_id,
        "query": query,
        "queryId": query_id,
        "corpusId": corpus_id,
        "profileId": profile_id,
        "chatMode": chat_mode,
        "inferenceMode": inference_mode,
        "allowDirectEvidenceAnswer": bool(allow_direct_evidence),
        "resultPath": result_path,
        "statusPath": status_path,
        "timestamp": time.time()
    }
    if n_threads is not None:
        trigger_data["nThreads"] = n_threads
    if context_token_budget is not None:
        trigger_data["contextTokenBudget"] = context_token_budget
    if max_output_tokens is not None:
        trigger_data["maxOutputTokens"] = max_output_tokens
    if rag_policy is not None:
        trigger_data["ragPolicy"] = rag_policy
    trigger_json = json.dumps(trigger_data)
    trigger_path = "/sdcard/Android/data/com.chat/files/benchmark_trigger.json"
    # Remove previous results
    adb_cmd(f"shell 'mkdir -p {remote_dir} && rm -f {result_path} {status_path}'")

    print(f"\nDispatching benchmark trigger to running app...")
    # Base64 avoids shell-quoting failures for questions containing quotes or
    # punctuation and makes the trigger content auditable.
    encoded_trigger = base64.b64encode(trigger_json.encode("utf-8")).decode("ascii")
    adb_cmd(f"shell 'echo {encoded_trigger} | base64 -d > {trigger_path}'")

    # Wake up screen so React Native execution doesn't sleep
    adb_cmd("shell 'input keyevent KEYCODE_WAKEUP'")

    print("Waiting for on-device inference to complete...")
    start_wait = time.time()
    found = False

    while time.time() - start_wait < 120:
        check = adb_cmd(f"shell 'ls {result_path} 2>/dev/null'")
        if request_id in check:
            found = True
            break
        time.sleep(2)
        sys.stdout.write(".")
        sys.stdout.flush()

    print()
    if not found:
        if not allow_legacy_global_result:
            print("\nERROR: request-specific benchmark result timed out; no stale fallback is accepted.")
            return None
        print("\nWARNING: using explicitly requested legacy global result; correlation is invalid for strict evaluation.")
        report_content = adb_cmd("shell 'cat /sdcard/Android/data/com.chat/files/inference_benchmark_latest.json 2>/dev/null'")
        legacy_uncorrelated = True
    else:
        report_content = adb_cmd(f"shell 'cat {result_path} 2>/dev/null'")
        legacy_uncorrelated = False

    if report_content and "{" in report_content:
        try:
            envelope = json.loads(report_content)
            if legacy_uncorrelated:
                report = envelope
                if report.get("query") != query or report.get("chat_mode") != chat_mode:
                    print("\nRefusing a mismatched legacy global result.")
                    return None
                report["evidence_status"] = "PHYSICAL_DEVICE_LEGACY_UNCORRELATED"
                report["legacy_uncorrelated"] = True
                report["requested_rag_policy"] = rag_policy if chat_mode == "document" else "not_applicable"
                report["requested_inference_mode"] = inference_mode
                report["run_id"] = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
                report["result_path"] = os.path.abspath(output_path) if output_path else None
                print("Legacy physical report captured; it is not eligible for strict correlation evaluation.")
                if output_path:
                    output_dir = os.path.dirname(output_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    if os.path.exists(output_path):
                        raise FileExistsError(f"Refusing to overwrite existing benchmark result: {output_path}")
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2)
                return report
            if envelope.get("status") != "completed":
                print(f"\nDevice returned terminal status {envelope.get('status')!r}; refusing partial data.")
                return None
            if envelope.get("request_id") != request_id:
                print("\nRefusing to save a result for a different request ID.")
                return None
            report = envelope.get("benchmark") or {}
            if not report:
                print("\nDevice result is missing the benchmark report.")
                return None
            # Never attach a stale/previous report to the current request.
            # The app report is the authoritative query/policy record; the
            # filename and CLI arguments are only collection metadata.
            report_query = report.get("query")
            report_mode = report.get("chat_mode")
            if report_query != query or report_mode != chat_mode:
                print("\nRefusing to save a mismatched physical report:")
                print(f"  requested query: {query!r}")
                print(f"  reported query : {report_query!r}")
                print(f"  requested mode : {chat_mode!r}")
                print(f"  reported mode  : {report_mode!r}")
                return None
            benchmark = report
            if benchmark.get("benchmark_request_id") != request_id:
                print("\nRefusing to save a result with missing benchmark correlation metadata.")
                return None
            if benchmark.get("benchmark_query_id") != query_id:
                print("\nRefusing to save a result with mismatched benchmark query metadata.")
                return None
            if benchmark.get("benchmark_corpus_id") != corpus_id:
                print("\nRefusing to save a result with mismatched benchmark corpus metadata.")
                return None
            if profile_id and benchmark.get("profile_id") != profile_id:
                print("\nRefusing to save a result for a different profile ID.")
                return None
            if bool(benchmark.get("allow_direct_evidence_answer")) != bool(allow_direct_evidence):
                print("\nRefusing to save a result with a different direct-answer setting.")
                return None
            reported_policy = (report.get("research_metrics") or {}).get("policy") or (report.get("rag") or {}).get("strategy")
            if rag_policy and reported_policy and reported_policy != rag_policy:
                print("\nRefusing to save a mismatched physical policy report:")
                print(f"  requested policy: {rag_policy!r}")
                print(f"  reported policy : {reported_policy!r}")
                return None
            post_hw = get_device_hardware_state()
            report["hardware"] = {
                "pre_test": pre_hw,
                "post_test": post_hw,
                "n_threads": report.get("n_threads", n_threads or 4),
                "n_gpu_layers": 0,
                "context_length": 512,
                "batch_size": 512,
                "ubatch_size": 512,
            }
            report["requested_rag_policy"] = rag_policy if chat_mode == "document" else "not_applicable"
            report["requested_inference_mode"] = inference_mode
            report["requested_allow_direct_evidence"] = bool(allow_direct_evidence)
            report["request_id"] = request_id
            report["query_id"] = query_id
            report["corpus_id"] = corpus_id
            if n_threads is not None:
                report["requested_threads"] = n_threads
            report["evidence_status"] = "PHYSICAL_DEVICE"
            report["benchmark_script"] = os.path.basename(__file__)
            if model_path:
                model_hash = adb_cmd(f"shell 'sha256sum {model_path} 2>/dev/null'")
                if model_hash:
                    report["model_sha256"] = model_hash.split()[0]

            total_lat = float(report.get('total_warm_latency_ms') or 0.0)
            rag_lat = float(report.get('rag_retrieval_ms') or 0.0)
            prefill_lat = float(report.get('prefill_ms') or 0.0)
            prefill_tps = float(report.get('prefill_tokens_per_second') or 0.0)
            decode_lat = float(report.get('decode_ms') or 0.0)
            decode_tps = float(report.get('tokens_per_second') or 0.0)
            ttft_lat = report.get('ttft_ms')

            print("\n" + "="*60)
            print("🚀 ON-DEVICE INFERENCE BENCHMARK RESULTS")
            print("="*60)
            print(f"Total Warm Query Latency   : {total_lat:.2f} ms ({total_lat/1000.0:.2f} s)")
            print(f"RAG Retrieval Latency      : {rag_lat:.2f} ms")
            print(f"Prompt Prefill Latency     : {prefill_lat:.2f} ms ({prefill_lat/1000.0:.2f} s)")
            print(f"├── Prompt Tokens          : {report.get('prompt_tokens', 0)}")
            print(f"└── Prefill Throughput     : {prefill_tps:.2f} tokens/sec")
            print(f"Token Decode Latency       : {decode_lat:.2f} ms ({decode_lat/1000.0:.2f} s)")
            print(f"├── Generated Tokens       : {report.get('generated_tokens', 0)}")
            print(f"└── Generation Throughput  : {decode_tps:.2f} tokens/sec")
            if ttft_lat is not None and isinstance(ttft_lat, (int, float)):
                print(f"TTFT (Time-To-First-Token) : {float(ttft_lat):.2f} ms ({float(ttft_lat)/1000.0:.2f} s)")
            else:
                print(f"TTFT (Time-To-First-Token) : N/A (avoided inference / instant refusal)")
            print(f"Peak PSS RAM               : {post_hw['pss_mb']} MB")
            print(f"Battery Temperature        : {post_hw['battery_temp_c']} °C")
            print(f"Thermal Status             : {post_hw.get('thermal_status', 'unavailable')}")
            print(f"Primary Bottleneck         : {report.get('primary_bottleneck', '').upper()}")
            print("="*60)
            
            # Save to results
            safe_policy = rag_policy or "ebmrag-v2"
            safe_mode = inference_mode or "baseline"
            resolved_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = output_path or os.path.join(
                os.path.dirname(__file__), "results", "physical_runs",
                f"{resolved_run_id}_{safe_policy}_{safe_mode}.json",
            )
            output_dir = os.path.dirname(out_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(out_file):
                raise FileExistsError(f"Refusing to overwrite existing benchmark result: {out_file}")
            report["run_id"] = resolved_run_id
            report["result_path"] = os.path.abspath(out_file)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Full benchmark record saved to: {out_file}")
            return report
        except Exception as e:
            print("JSON parse error:", e)

    print("\nCould not read latest benchmark json from device.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="ADB device serial; defaults to ANDROID_SERIAL or the project device")
    parser.add_argument("--query", default="Tell me about zypher")
    parser.add_argument("--mode", default="document")
    parser.add_argument("--inference-mode", choices=["baseline", "fixed", "adaptive"], default="baseline")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--context-token-budget", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--rag-policy", choices=["ebmrag-v2", "mobcert-v1"], default=None,
                        help="RAG policy; mobcert-v1 enables the evidence contract")
    parser.add_argument("--run-id", default=None,
                        help="Stable run identifier; defaults to a timestamp")
    parser.add_argument("--output", dest="output_path", default=None,
                        help="Optional exact JSON output path")
    parser.add_argument("--model-path", default=None, help="Remote GGUF path for sha256sum")
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--corpus-id", default=None)
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--allow-direct-evidence", action="store_true",
                        help="Allow MobCert direct answers; disabled by default for fair generation comparisons")
    parser.add_argument("--allow-legacy-global-result", action="store_true",
                        help="Explicitly capture the old global report path; output is not strict-evaluation eligible")
    args = parser.parse_args()
    if args.serial:
        DEVICE_SERIAL = args.serial
    run_test(
        args.query,
        args.mode,
        args.inference_mode,
        args.threads,
        args.context_token_budget,
        args.max_output_tokens,
        args.model_path,
        args.rag_policy,
        args.run_id,
        args.output_path,
        args.query_id,
        args.corpus_id,
        args.profile_id,
        args.allow_direct_evidence,
        args.allow_legacy_global_result,
    )
