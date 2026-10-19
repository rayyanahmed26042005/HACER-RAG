#!/usr/bin/env python3
"""
Evaluate Physical Phone Runs from Redmi 12 5G (October 11, 2026).
"""

import json
from evaluate_rag_offline import evaluate_rag_turn, generate_ieee_latex_table

phone_experiments = [
    {
        "query_id": "q3_single_hop_bonus",
        "query": "What is the annual learning budget and gym allowance of employees?",
        "ground_truth": "The learning budget is $1,500 USD/year per employee, and the gym allowance is $60/month reimbursed via expense report.",
        "expected_action": "answer",
        "retrieved_context": "[1] (from \"offlineRag.pdf\", relevance: 0.81)\nLearning Budget: $1,500 USD/year per employee\nGym Allowance: $60/month reimbursed via expense report",
        "model_response": "The annual learning budget is $1,500 USD/year per employee and the gym allowance is $60/month reimbursed via expense report.",
        "telemetry": {
            "decode_tok_per_sec": 3.6,
            "prefill_tok_per_sec": 6.0,
            "ttft_ms": 36683.7,
            "total_ms": 45881.7,
            "prompt_tokens": 218,
            "gen_tokens": 33,
            "mode": "ADAPTIVE (4T)",
            "bottleneck": "prompt_prefill"
        }
    },
    {
        "query_id": "q2_single_hop_clo",
        "query": "Who is chief legal officer and which office are they based in ?",
        "ground_truth": "Jorge Fonseca is the Chief Legal Officer, based in the Lisbon office.",
        "expected_action": "answer",
        "retrieved_context": "[1] (from \"offlineRag.pdf\", relevance: 0.72)\n1.1 Leadership Team Name Title Office Joined Dr. Priya Nandakumar CEO & Co-Founder Reykjavik 2011 Marcus Löbell CTO & Co-Founder Reykjavik 2011",
        "model_response": "The provided text doesn't explicitly state the name of the Chief Legal Officer or their location.",
        "telemetry": {
            "decode_tok_per_sec": 3.2,
            "ttft_ms": 80880.0,
            "total_ms": 89060.0,
            "mode": "BASELINE (4T)",
            "bottleneck": "prompt_prefill"
        }
    },
    {
        "query_id": "q1_single_hop_battery",
        "query": "What is battery life and weight of aerosence X7?",
        "ground_truth": "The AeroSense X7 uses a QuantumCell battery that lasts 72 hours on a single charge and weighs 2.3 kg.",
        "expected_action": "answer",
        "retrieved_context": "[1] (from \"offlineRag.pdf\", relevance: 0.54)\nOverview Zephyr Dynamics Inc. was founded in 2011 by Dr. Priya Nandakumar and Marcus Löbell in Reykjavik, Iceland.",
        "model_response": "I don't have enough information in the uploaded documents to answer that.",
        "telemetry": {
            "decode_tok_per_sec": 3.0,
            "ttft_ms": 34030.0,
            "total_ms": 40950.0,
            "mode": "ADAPTIVE (4T)",
            "bottleneck": "prompt_prefill"
        }
    },
    {
        "query_id": "q_overview_zypher",
        "query": "What is zypher?",
        "ground_truth": "Zephyr Dynamics Inc. specializes in atmospheric data harvesting systems and micro-drone swarm technology.",
        "expected_action": "answer",
        "retrieved_context": "[1] (from \"offlineRag.pdf\", relevance: 0.74)\nZephyr Dynamics Inc. specializes in atmospheric data harvesting systems and micro-drone swarm technology. Zephyr Dynamics has 847 full-time employees across 6 offices.",
        "model_response": "Zephyr Dynamics Inc. specializes in atmospheric data harvesting systems and micro-drone swarm technology.",
        "telemetry": {
            "decode_tok_per_sec": 2.8,
            "prefill_tok_per_sec": 6.6,
            "ttft_ms": 39159.1,
            "total_ms": 46300.1,
            "prompt_tokens": 255,
            "gen_tokens": 20,
            "mode": "ADAPTIVE (4T)",
            "bottleneck": "prompt_prefill"
        }
    }
]

evaluated = []
for exp in phone_experiments:
    res = evaluate_rag_turn(
        query_id=exp["query_id"],
        query=exp["query"],
        ground_truth=exp["ground_truth"],
        expected_action=exp["expected_action"],
        retrieved_context=exp["retrieved_context"],
        model_response=exp["model_response"],
        telemetry=exp["telemetry"]
    )
    evaluated.append(res)

out_path = "benchmark/results/phone_rag_evaluation.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(evaluated, f, indent=2)

print(f"Evaluated {len(evaluated)} phone runs. Saved to {out_path}.")
print("\n=== IEEE LaTeX Table ===")
print(generate_ieee_latex_table(evaluated))
