#!/usr/bin/env python3
"""Small dependency-free regression suite for the offline evaluator."""

import unittest
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from evaluate_rag_offline import (
    compute_groundedness,
    compute_retrieval_metrics,
    compute_token_f1,
)


class OfflineEvaluatorTests(unittest.TestCase):
    def test_exact_context_has_full_precision_and_recall(self):
        metrics = compute_retrieval_metrics(
            "AeroSense X7 weighs 2.3 kg.",
            "AeroSense X7 weighs 2.3 kg.",
        )
        self.assertEqual(metrics, {"context_recall": 1.0, "context_precision": 1.0})

    def test_irrelevant_content_reduces_precision_without_changing_recall(self):
        metrics = compute_retrieval_metrics(
            "AeroSense X7 weighs 2.3 kg. The unrelated weather report covers Reykjavik.",
            "AeroSense X7 weighs 2.3 kg.",
        )
        self.assertEqual(metrics["context_recall"], 1.0)
        self.assertLess(metrics["context_precision"], 1.0)

    def test_groundedness_distinguishes_supported_and_unsupported_claims(self):
        supported = compute_groundedness("The device weighs 2.3 kg.", "The device weighs 2.3 kg.")
        unsupported = compute_groundedness("The device weighs 9.9 kg.", "The device weighs 2.3 kg.")
        self.assertGreater(supported["faithfulness"], unsupported["faithfulness"])
        self.assertGreater(unsupported["hallucination_rate"], 0.0)

    def test_answer_token_f1_is_reproducible(self):
        metrics = compute_token_f1("72 hours", "The battery lasts 72 hours")
        self.assertEqual(metrics["f1"], 0.5714)

    def test_citation_ordinals_do_not_reduce_answer_f1(self):
        metrics = compute_token_f1("The battery lasts 72 hours [1].", "The battery lasts 72 hours.")
        self.assertEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
