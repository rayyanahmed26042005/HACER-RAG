#!/usr/bin/env python3
"""
Warm-query benchmark harness for the on-device RAG project.

--simulate creates a deterministic *synthetic* dataset for validating the
analysis pipeline. It is explicitly labeled SIMULATED in every record and
must not be used as device evidence. Physical-device measurements are
collected by run_device_inference.py through the app's trigger/report files.
"""

import os
import sys
import json
import time
import subprocess
import argparse
import statistics

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)

CONFIG_PATH = os.path.join(ROOT_DIR, "avd_configs.json")
QUERIES_PATH = os.path.join(ROOT_DIR, "queries.json")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

def run_cmd(cmd, check=True, timeout=120):
    print(f"[CMD] {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}\nStderr: {res.stderr}")
    return res

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def calculate_stats(values):
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std_dev": 0}
    mean_val = sum(values) / len(values)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median_val = sorted_vals[n // 2] if n % 2 != 0 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "std_dev": round(std_dev, 2)
    }

def benchmark_profile(profile_key, config, model_meta, emb_meta, queries, datasets, iterations=5, warmups=1, simulate=True):
    print(f"\n==================================================")
    print(f"Warm-Query Benchmark: {config['name']} ({profile_key})")
    print(f"Model: {model_meta['model_name']} ({model_meta['quantization']})")
    print(f"==================================================")

    if not simulate:
        raise RuntimeError(
            "No real AVD measurement adapter is implemented in run_benchmark.py. "
            "Use benchmark/run_device_inference.py for a physical device, or "
            "pass --simulate when validating the synthetic analysis pipeline."
        )

    print(f"[SIMULATION MODE] Generating labeled synthetic records for profile '{profile_key}'.")

    all_runs_results = []

    for doc_name, doc_text in datasets.items():
        doc_size = len(doc_text.encode('utf-8'))
        page_count = max(1, doc_size // 2500)
        chunk_count = max(1, doc_size // 400)

        for query_item in queries:
            query_runs = []
            for run_idx in range(warmups + iterations):
                is_warmup = run_idx < warmups
                run_type = "WARMUP" if is_warmup else f"MEASURED_{run_idx - warmups + 1}"

                cpu_cores = config["cpu_configuration"]["cores"]
                ram_mb = config["memory_configuration"]["ram_mb"]

                # Warm-query latency modeling (Model is ALREADY loaded in memory)
                scale_factor = (doc_size / 500.0) * (4.0 / cpu_cores) * (2048.0 / ram_mb)

                query_emb_ms = round(35 * (4.0 / cpu_cores), 2)
                vector_search_ms = round(12 + (chunk_count * 0.15), 2)
                context_const_ms = round(5.0, 2)
                tokenization_ms = round(18.0 * (4.0 / cpu_cores), 2)
                prefill_ms = round(1800 * (4.0 / cpu_cores) * (512 / 512), 2)
                ttft_ms = round(query_emb_ms + vector_search_ms + context_const_ms + tokenization_ms + prefill_ms, 2)

                gen_tokens = 50
                decode_ms = round(gen_tokens * 85 * (4.0 / cpu_cores), 2)
                total_warm_latency = round(ttft_ms + decode_ms, 2)
                gen_throughput = round(gen_tokens / (decode_ms / 1000.0), 2)
                peak_pss_mb = round(min(ram_mb * 0.75, 520 + (chunk_count * 0.5) + (model_meta['file_size_mb'] * 0.35)), 2)

                record = {
                    "device_profile": profile_key,
                    "android_version": config["android_version"],
                    "architecture": config["architecture"],
                    "cpu_configuration": f"{cpu_cores} Cores @ {config['cpu_configuration']['frequency_mhz']}MHz",
                    "memory_configuration": f"{ram_mb}MB RAM",

                    # Exact Model Metadata for Reproducibility
                    "model_name": model_meta["model_name"],
                    "model_version": model_meta["model_version"],
                    "gguf_filename": model_meta["gguf_filename"],
                    "quantization": model_meta["quantization"],
                    "parameter_count": model_meta["parameter_count"],
                    "file_size_mb": model_meta["file_size_mb"],
                    "context_length": model_meta["context_length"],
                    "n_threads": config["default_threads"],
                    "n_gpu_layers": config["gpu_layers"],
                    "backend": model_meta["backend"],

                    "document_name": doc_name,
                    "document_size": doc_size,
                    "page_count": page_count,
                    "chunk_count": chunk_count,
                    "embedding_model": emb_meta["model_name"],
                    "embedding_dimension": emb_meta["embedding_dimension"],
                    "vector_index": "cosine_json_store",

                    "query_embedding_time_ms": query_emb_ms,
                    "vector_search_time_ms": vector_search_ms,
                    "context_construction_time_ms": context_const_ms,
                    "tokenization_time_ms": tokenization_ms,
                    "prefill_time_ms": prefill_ms,
                    "ttft_ms": ttft_ms,
                    "decode_time_ms": decode_ms,
                    "generation_throughput_tokens_sec": gen_throughput,
                    "total_warm_query_latency_ms": total_warm_latency,
                    "peak_pss_ram_mb": peak_pss_mb,

                    "error": None,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "run_type": run_type,
                    "evidence_status": "SIMULATED",
                    "simulation_formula_version": "v1"
                }

                if not is_warmup:
                    query_runs.append(record)

            all_runs_results.extend(query_runs)

    return all_runs_results

def main():
    parser = argparse.ArgumentParser(description="Automated Warm-Query AVD Benchmark Runner")
    parser.add_argument("--profile", choices=["low", "medium", "high", "all"], default="all", help="Target device profile")
    parser.add_argument("--iterations", type=int, default=5, help="Measured iterations per test case (5-10 recommended)")
    parser.add_argument("--warmups", type=int, default=1, help="Warm-up iterations")
    parser.add_argument("--simulate", action="store_true", help="Generate labeled synthetic records")
    args = parser.parse_args()

    if not os.path.exists(CONFIG_PATH):
        print(f"[ERROR] Config not found at {CONFIG_PATH}")
        sys.exit(1)

    config_data = load_json(CONFIG_PATH)
    model_meta = config_data["model_metadata"]
    emb_meta = config_data["embedding_model_metadata"]
    queries = load_json(QUERIES_PATH)

    datasets = {}
    for fname in ["dataset_a_small.txt", "dataset_b_medium.txt", "dataset_c_large.txt"]:
        fpath = os.path.join(DATASETS_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                datasets[fname] = f.read()

    profiles_to_run = list(config_data["profiles"].keys()) if args.profile == "all" else [args.profile]

    if not args.simulate:
        parser.error("--simulate is required for the synthetic harness; use run_device_inference.py for physical-device measurements")

    all_results = []
    for p_key in profiles_to_run:
        p_conf = config_data["profiles"][p_key]
        results = benchmark_profile(p_key, p_conf, model_meta, emb_meta, queries, datasets, args.iterations, args.warmups, args.simulate)
        all_results.extend(results)

    latencies = [r["total_warm_query_latency_ms"] for r in all_results]
    stats = calculate_stats(latencies)
    print(f"\n==================================================")
    print(f"BENCHMARK COMPLETED ACROSS {len(all_results)} RUNS")
    print(f"Total Warm-Query Latency Stats (ms):")
    print(f"  Mean:   {stats['mean']} ms")
    print(f"  Median: {stats['median']} ms")
    print(f"  Min:    {stats['min']} ms")
    print(f"  Max:    {stats['max']} ms")
    print(f"  StdDev: {stats['std_dev']} ms")
    print(f"==================================================")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_out = os.path.join(RESULTS_DIR, "warm_query_benchmark_results.json")
    csv_out = os.path.join(RESULTS_DIR, "warm_query_benchmark_results.csv")

    save_json(json_out, all_results)

    if all_results:
        import csv
        keys = list(all_results[0].keys())
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"[SUCCESS] Warm-query results saved to {json_out} and {csv_out}")

if __name__ == "__main__":
    main()
