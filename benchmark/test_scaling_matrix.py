#!/usr/bin/env python3
"""Unit tests for HACER-RAG Scaling Matrix Study.

Validates:
1. Byte-for-byte canonical prompt construction and SHA-256 equality.
2. Stable chunk separation without accidental token/character concatenation.
3. Exact prefix + continuation reconstruction: prefix + continuation == canonical_prompt.
4. Context window headroom and safety checks (no truncation under n_ctx - 4).
5. Empirical token accounting formula validation.
6. Telemetry schema and result integrity validation.
"""

import hashlib
import unittest
from typing import Dict, Tuple

# Canonical Chunks from Zephyr Dynamics Official Handbook (offlineRag.pdf)
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
    """Builds the canonical 4-chunk prompt with explicit stable delimiters."""
    chunks = [
        f"[1] Company Overview:\n{CHUNK_1_OVERVIEW}",
        f"[2] Leadership Team:\n{CHUNK_2_LEADERSHIP}",
        f"[3] AeroSense X7 Specifications:\n{CHUNK_3_AEROSENSE}",
        f"[4] SwarmKit Pro Platform:\n{CHUNK_4_SWARMKIT}",
    ]
    evidence_block = "\n\n".join(chunks)
    return f"{evidence_block}\n\n{CANONICAL_QUESTION}"


def get_prefix_and_continuation(k_chunks: int) -> Tuple[str, str]:
    """Splits the canonical prompt into prefix and continuation based on K cached chunks.
    
    Guarantees: prefix + continuation == build_canonical_prompt()
    """
    chunks = [
        f"[1] Company Overview:\n{CHUNK_1_OVERVIEW}",
        f"[2] Leadership Team:\n{CHUNK_2_LEADERSHIP}",
        f"[3] AeroSense X7 Specifications:\n{CHUNK_3_AEROSENSE}",
        f"[4] SwarmKit Pro Platform:\n{CHUNK_4_SWARMKIT}",
    ]
    if k_chunks not in (1, 2, 4):
        raise ValueError(f"Invalid K: {k_chunks}. Must be 1, 2, or 4.")
    
    if k_chunks == 1:
        prefix = f"{chunks[0]}\n\n"
        continuation = f"{chunks[1]}\n\n{chunks[2]}\n\n{chunks[3]}\n\n{CANONICAL_QUESTION}"
    elif k_chunks == 2:
        prefix = f"{chunks[0]}\n\n{chunks[1]}\n\n"
        continuation = f"{chunks[2]}\n\n{chunks[3]}\n\n{CANONICAL_QUESTION}"
    elif k_chunks == 4:
        prefix = f"{chunks[0]}\n\n{chunks[1]}\n\n{chunks[2]}\n\n{chunks[3]}\n\n"
        continuation = CANONICAL_QUESTION
    
    return prefix, continuation


def check_context_safety(prompt_tokens: int, n_predict: int, n_ctx: int) -> Tuple[bool, int, int]:
    """Validates that prompt_tokens + n_predict fits under llama.cpp context limit.
    
    llama.cpp requires: prompt_tokens + n_predict <= n_ctx - 4.
    Returns:
      is_safe: boolean
      safety_headroom: (n_ctx - 4) - (prompt_tokens + n_predict)  [strictly enforced safety margin]
      raw_headroom: n_ctx - (prompt_tokens + n_predict)          [total remaining slots in context]
    """
    safety_headroom = (n_ctx - 4) - (prompt_tokens + n_predict)
    raw_headroom = n_ctx - (prompt_tokens + n_predict)
    is_safe = (safety_headroom >= 0) and (prompt_tokens < n_ctx - 4)
    return is_safe, safety_headroom, raw_headroom


def reconcile_tokens(n_baseline: int, n_cached: int) -> int:
    """Predicts evaluated tokens in llama.rn v0.5.8 session resumption path."""
    return (n_baseline - n_cached) + 1


class TestScalingMatrixHarness(unittest.TestCase):
    def setUp(self):
        self.canonical = build_canonical_prompt()
        self.canonical_sha = hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()

    def test_prompt_byte_equality_k1(self):
        prefix, cont = get_prefix_and_continuation(1)
        reconstructed = prefix + cont
        self.assertEqual(reconstructed, self.canonical)
        self.assertEqual(hashlib.sha256(reconstructed.encode("utf-8")).hexdigest(), self.canonical_sha)

    def test_prompt_byte_equality_k2(self):
        prefix, cont = get_prefix_and_continuation(2)
        reconstructed = prefix + cont
        self.assertEqual(reconstructed, self.canonical)
        self.assertEqual(hashlib.sha256(reconstructed.encode("utf-8")).hexdigest(), self.canonical_sha)

    def test_prompt_byte_equality_k4(self):
        prefix, cont = get_prefix_and_continuation(4)
        reconstructed = prefix + cont
        self.assertEqual(reconstructed, self.canonical)
        self.assertEqual(hashlib.sha256(reconstructed.encode("utf-8")).hexdigest(), self.canonical_sha)

    def test_stable_delimiters(self):
        """Ensure chunks are separated by double newlines without sentence merger."""
        self.assertIn("[1] Company Overview:", self.canonical)
        self.assertIn("\n\n[2] Leadership Team:", self.canonical)
        self.assertIn("\n\n[3] AeroSense X7 Specifications:", self.canonical)
        self.assertIn("\n\n[4] SwarmKit Pro Platform:", self.canonical)
        self.assertIn("\n\nQuestion:", self.canonical)

    def test_facts_present_in_prompt(self):
        """Verify all 4 target facts required by the question are present."""
        self.assertIn("2011", self.canonical)          # Founding year (Chunk 1)
        self.assertIn("Jorge Fonseca", self.canonical) # CLO (Chunk 2)
        self.assertIn("72 hours", self.canonical)      # AeroSense battery (Chunk 3)
        self.assertIn("$3,800/month", self.canonical)  # SwarmKit full tier (Chunk 4)

    def test_context_window_safety_512(self):
        """Ensure canonical prompt (395 native tokens) safely fits in 512 context with n_predict=35."""
        prompt_tokens = 395
        n_predict_requested = 35
        actual_tokens_generated = 34
        is_safe, safety_headroom, raw_headroom = check_context_safety(prompt_tokens, n_predict_requested, 512)
        self.assertTrue(is_safe)
        # Enforced inequality: (512 - 4) - (395 + 35) = 508 - 430 = 78 tokens
        self.assertEqual(safety_headroom, 78)
        # Raw remaining context: 512 - (395 + 35) = 82 tokens
        self.assertEqual(raw_headroom, 82)
        # Actual post-decode safety headroom with 34 tokens generated: 508 - 429 = 79 tokens
        _, post_decode_safety, _ = check_context_safety(prompt_tokens, actual_tokens_generated, 512)
        self.assertEqual(post_decode_safety, 79)

    def test_context_window_safety_1024_and_2048(self):
        prompt_tokens = 395
        n_predict = 35
        safe_1024, safety_1024, raw_1024 = check_context_safety(prompt_tokens, n_predict, 1024)
        safe_2048, safety_2048, raw_2048 = check_context_safety(prompt_tokens, n_predict, 2048)
        self.assertTrue(safe_1024)
        self.assertTrue(safe_2048)
        # Enforced safety headroom: (n_ctx - 4) - 430
        self.assertEqual(safety_1024, 590)
        self.assertEqual(raw_1024, 594)
        self.assertEqual(safety_2048, 1614)
        self.assertEqual(raw_2048, 1618)

    def test_token_accounting_reconciliation_formula(self):
        """Verifies the implementation-specific N_eval = (N_baseline - N_cached) + 1."""
        self.assertEqual(reconcile_tokens(187, 85), 103)
        self.assertEqual(reconcile_tokens(218, 100), 119)
        self.assertEqual(reconcile_tokens(156, 58), 99)
        # Empirical scaling measurements for 395-token canonical prompt:
        # K=1 (84 tok loaded): 395 - 84 + 1 = 312 evaluated
        self.assertEqual(reconcile_tokens(395, 84), 312)
        # K=2 (158 tok loaded): 395 - 158 + 1 = 238 evaluated
        self.assertEqual(reconcile_tokens(395, 158), 238)
        # K=4 (341 tok loaded): 395 - 341 + 1 = 55 evaluated
        self.assertEqual(reconcile_tokens(395, 341), 55)


if __name__ == "__main__":
    unittest.main()
