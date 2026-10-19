#!/usr/bin/env python3
"""Regression tests for physical-report aggregation without an Android device."""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from evaluate_physical_runs import evaluate_report, load_labels, summarize


class PhysicalRunEvaluatorTests(unittest.TestCase):
    def test_report_uses_nested_rag_metrics_and_policy(self):
        labels = {
            "what is the battery life?": {
                "id": "q1",
                "query": "What is the battery life?",
                "ground_truth": "The battery lasts 72 hours.",
                "expected_action": "answer",
            }
        }
        report = {
            "run_id": "run-1",
            "evidence_status": "PHYSICAL_DEVICE",
            "requested_rag_policy": "mobcert-v1",
            "response_text": "The battery lasts 72 hours [1].",
            "ttft_ms": 1000,
            "total_warm_latency_ms": 1200,
            "rag_contract_audit": {"status": "certified"},
            "rag": {
                "strategy": "mobcert-v1",
                "context": "Battery life: 72 hours.",
                "selectedChunkIds": ["doc:1"],
            },
        }
        row = evaluate_report(report, labels["what is the battery life?"])
        self.assertEqual(row["policy"], "mobcert-v1")
        self.assertEqual(row["benchmark"]["token_f1"], 1.0)
        summary = summarize([row])
        self.assertEqual(summary["mobcert-v1"]["run_count"], 1)
        self.assertEqual(summary["mobcert-v1"]["median_ttft_ms"], 1000.0)

    def test_labels_load_normalizes_query_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "labels.json"
            path.write_text(json.dumps([{"query": "What   is this?", "id": "q"}], indent=2), encoding="utf-8")
            labels = load_labels(str(path))
            self.assertIn("what is this?", labels)

    def test_policy_mismatch_is_visible_to_evaluation_row(self):
        labels = {
            "what is this?": {
                "id": "q", "query": "What is this?", "ground_truth": "This is true.",
                "expected_action": "answer",
            }
        }
        report = {
            "requested_rag_policy": "mobcert-v1",
            "research_metrics": {"policy": "ebmrag-v2"},
            "rag": {"strategy": "ebmrag-v2", "context": "This is true."},
            "response_text": "This is true [1].",
        }
        row = evaluate_report(report, labels["what is this?"])
        self.assertEqual(row["requested_policy"], "mobcert-v1")
        self.assertEqual(row["nested_policy"], "ebmrag-v2")


if __name__ == "__main__":
    unittest.main()
