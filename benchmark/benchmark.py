#!/usr/bin/env python3
"""
Master Benchmark CLI for On-Device RAG Research Framework.
Usage:
  python benchmark.py --profile low
  python benchmark.py --profile medium
  python benchmark.py --profile high
  python benchmark.py --profile all
"""

import argparse
import sys
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Master On-Device RAG Benchmark CLI")
    parser.add_argument("--profile", choices=["low", "medium", "high", "all"], default="all", help="Target AVD profile")
    parser.add_argument("--iterations", type=int, default=3, help="Measured iterations per test case")
    parser.add_argument("--warmups", type=int, default=1, help="Warm-up iterations")
    parser.add_argument("--simulate", action="store_true", help="Generate labeled synthetic records for pipeline validation")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    runner_script = os.path.join(current_dir, "run_benchmark.py")
    analyzer_script = os.path.join(current_dir, "analyze_bottlenecks.py")

    print("==================================================")
    print(" STARTING ON-DEVICE RAG BENCHMARK & PROFILING SUITE")
    print("==================================================")

    # 1. Run Benchmarks
    run_cmd = [sys.executable, runner_script, "--profile", args.profile,
               "--iterations", str(args.iterations), "--warmups", str(args.warmups)]
    if args.simulate:
        run_cmd.append("--simulate")
    print(f"Executing: {' '.join(run_cmd)}")
    ret = subprocess.run(run_cmd).returncode
    if ret != 0:
        print("[ERROR] Benchmark execution failed.")
        sys.exit(ret)

    # 2. Analyze Bottlenecks & Generate Report
    analyze_cmd = [sys.executable, analyzer_script]
    print(f"Executing: {' '.join(analyze_cmd)}")
    ret = subprocess.run(analyze_cmd).returncode
    if ret != 0:
        print("[ERROR] Bottleneck analysis failed.")
        sys.exit(ret)

    print("\n==================================================")
    print(" BENCHMARK & BOTTLENECK ANALYSIS COMPLETED SUCCESSFULLY")
    print(" Results saved to benchmark/results/")
    print(" Report generated at benchmark/reports/bottleneck_report.md")
    print("==================================================")

if __name__ == "__main__":
    main()
