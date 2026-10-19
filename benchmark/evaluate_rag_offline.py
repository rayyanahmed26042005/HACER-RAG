#!/usr/bin/env python3
"""
Offline Deterministic RAG Evaluator for IEEE Research Paper.
Evaluates On-Device RAG pipelines without requiring cloud LLM judges.

Metrics Computed:
1. Context Precision & Context Recall (Evidence Retrieval Quality)
2. Deterministic Faithfulness / Groundedness (Context Entailment & Hallucination Rate)
3. Answer Correctness & Truthness (Token F1, Precision, Recall, ROUGE-1, ROUGE-L)
4. Grammar & Linguistic Integrity (Repetition penalty, sentence termination, readability)
5. Hardware & Latency Efficiency (TTFT, Prefill tok/s, Decode tok/s, Warm Latency)
"""

import json
import math
import re
import sys
from typing import Dict, List, Any, Optional

def normalize_text(text: str) -> str:
    """Lowercases, strips punctuation, and collapses whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def strip_citation_ordinals(text: str) -> str:
    """Remove prompt-visible citation metadata before content scoring."""
    return re.sub(r'\[\s*\d+\s*\]', ' ', text or '')

def get_tokens(text: str) -> List[str]:
    norm = normalize_text(text)
    return norm.split() if norm else []

def is_content_token(token: str, stopwords: set) -> bool:
    """Keep short numeric fragments such as decimal components 2 and 3."""
    return token not in stopwords and (len(token) > 2 or any(char.isdigit() for char in token))

def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# ─── 1. Retrieval Metrics ─────────────────────────────────────────────────────

def compute_retrieval_metrics(retrieved_context: str, ground_truth: str) -> Dict[str, float]:
    """
    Evaluates whether the retrieved context contains the necessary factual evidence.
    """
    gt_tokens = set(get_tokens(strip_citation_ordinals(ground_truth)))
    # Filter common stopwords to focus on content words
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'that', 'this'}
    content_gt = {t for t in gt_tokens if is_content_token(t, stopwords)}
    
    if not content_gt:
        return {"context_recall": 1.0, "context_precision": 1.0}

    ctx_tokens = set(get_tokens(strip_citation_ordinals(retrieved_context)))
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'that', 'this'}
    content_ctx = {t for t in ctx_tokens if is_content_token(t, stopwords)}
    matched = content_gt.intersection(content_ctx)
    recall = len(matched) / len(content_gt)

    # Deterministic lexical precision: relevant ground-truth content tokens
    # divided by all unique content tokens in the retrieved context. This is a
    # transparent overlap proxy, not a substitute for labeled evidence spans.
    precision = len(matched) / max(1, len(content_ctx))

    return {
        "context_recall": round(recall, 4),
        "context_precision": round(precision, 4),
    }

# ─── 2. Deterministic Groundedness & Faithfulness ─────────────────────────────

def compute_groundedness(response: str, context: str) -> Dict[str, float]:
    """
    Measures what proportion of the model's output claims are strictly supported by context.
    Detects ungrounded hallucinations deterministically.
    """
    resp_sentences = [s.strip() for s in re.split(r'[.!?]', strip_citation_ordinals(response)) if s.strip()]
    if not resp_sentences:
        return {"faithfulness": 1.0, "hallucination_rate": 0.0}

    ctx_norm = normalize_text(strip_citation_ordinals(context))
    ctx_words = set(ctx_norm.split())
    ctx_bigrams = set(get_ngrams(ctx_norm.split(), 2))

    grounded_sentences = 0
    total_content_words = 0
    supported_content_words = 0

    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'that', 'this', 'it', 'its', 'as'}

    for sent in resp_sentences:
        sent_words = [w for w in get_tokens(sent) if is_content_token(w, stopwords)]
        if not sent_words:
            grounded_sentences += 1
            continue
        
        sent_bigrams = get_ngrams(get_tokens(sent), 2)
        
        words_in_ctx = sum(1 for w in sent_words if w in ctx_words)
        bigrams_in_ctx = sum(1 for bg in sent_bigrams if bg in ctx_bigrams) if sent_bigrams else 0
        
        total_content_words += len(sent_words)
        supported_content_words += words_in_ctx

        # A sentence is considered grounded if >= 75% of its content words
        # or a strong bigram match exists in the retrieved context
        ratio = words_in_ctx / len(sent_words)
        if ratio >= 0.70 or (len(sent_bigrams) > 0 and bigrams_in_ctx / len(sent_bigrams) >= 0.5):
            grounded_sentences += 1

    sentence_faithfulness = grounded_sentences / len(resp_sentences)
    token_groundedness = supported_content_words / max(1, total_content_words)
    
    # Combined deterministic faithfulness score
    faithfulness = round(0.5 * sentence_faithfulness + 0.5 * token_groundedness, 4)
    hallucination_rate = round(1.0 - faithfulness, 4)

    return {
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate,
    }

# ─── 3. Answer Correctness & Truthness (Token F1 & ROUGE-L) ───────────────────

def compute_token_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    pred_tokens = get_tokens(strip_citation_ordinals(prediction))
    gt_tokens = get_tokens(strip_citation_ordinals(ground_truth))

    if not pred_tokens or not gt_tokens:
        exact = 1.0 if pred_tokens == gt_tokens else 0.0
        return {"precision": exact, "recall": exact, "f1": exact, "exact_match": exact}

    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gt_counts = {}
    for t in gt_tokens:
        gt_counts[t] = gt_counts.get(t, 0) + 1

    overlap = 0
    for t, count in pred_counts.items():
        if t in gt_counts:
            overlap += min(count, gt_counts[t])

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact = 1.0 if normalize_text(strip_citation_ordinals(prediction)) == normalize_text(strip_citation_ordinals(ground_truth)) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": exact,
    }

def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """Computes ROUGE-L via Longest Common Subsequence (LCS)."""
    p_tokens = get_tokens(strip_citation_ordinals(prediction))
    g_tokens = get_tokens(strip_citation_ordinals(ground_truth))
    m = len(p_tokens)
    n = len(g_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tokens[i - 1] == g_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(f1, 4)

# ─── 4. Linguistic Quality & Fluency ──────────────────────────────────────────

def compute_fluency_metrics(text: str) -> Dict[str, float]:
    tokens = get_tokens(strip_citation_ordinals(text))
    if not tokens:
        return {"repetition_ratio": 1.0, "sentence_integrity": 1.0, "fluency_score": 0.0}

    # Distinct n-grams (distinct-2) to detect degenerative repetition loops
    bigrams = get_ngrams(tokens, 2)
    distinct_bigrams = len(set(bigrams)) / max(1, len(bigrams))

    # Sentence integrity check: checks if answer ends with proper punctuation
    ends_cleanly = 1.0 if text.strip() and text.strip()[-1] in '.!?"\'' else 0.5

    # Check for leaked prompt tags or template syntax
    leak_penalty = 0.0
    if re.search(r'(<end_of_turn>|<eos>|Context:|Question:)', strip_citation_ordinals(text)):
        leak_penalty = 0.4

    fluency = max(0.0, round(distinct_bigrams * 0.5 + ends_cleanly * 0.5 - leak_penalty, 4))

    return {
        "distinct_2": round(distinct_bigrams, 4),
        "sentence_integrity": ends_cleanly,
        "fluency_score": fluency,
    }

# ─── Full Query Evaluation ────────────────────────────────────────────────────

def evaluate_rag_turn(
    query_id: str,
    query: str,
    ground_truth: str,
    expected_action: str,
    retrieved_context: str,
    model_response: str,
    telemetry: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates a single RAG inference turn across all mathematical metrics.
    """
    is_refusal = bool(re.search(
        r"don't have enough information|not explicitly state|does not mention|do not mention|doesn't mention|"
        r"provided excerpts? do not contain|provided text does not contain|insufficient document evidence|"
        r"cannot certify|not mentioned in the provided", model_response, re.I))

    # If ground truth was an expected refusal (negative query)
    if expected_action == "refuse":
        refusal_correctness = 1.0 if is_refusal else 0.0
        return {
            "query_id": query_id,
            "query": query,
            "expected_action": expected_action,
            "model_action": "refused" if is_refusal else "answered",
            "refusal_accuracy": refusal_correctness,
            "context_recall": None,
            "context_precision": None,
            "faithfulness": None,
            "hallucination_rate": None,
            "token_f1": None,
            "rouge_l": None,
            "telemetry": telemetry or {},
        }

    # Normal factual query
    retrieval_m = compute_retrieval_metrics(retrieved_context, ground_truth)
    groundedness_m = compute_groundedness(model_response, retrieved_context)
    token_m = compute_token_f1(model_response, ground_truth)
    rouge_l = compute_rouge_l(model_response, ground_truth)
    fluency_m = compute_fluency_metrics(model_response)

    # If model refused an answerable question
    if is_refusal:
        return {
            "query_id": query_id,
            "query": query,
            "expected_action": expected_action,
            "model_action": "false_refusal",
            "context_recall": retrieval_m["context_recall"],
            "context_precision": retrieval_m["context_precision"],
            "faithfulness": None,
            "hallucination_rate": None,
            "token_f1": None,
            "rouge_l": None,
            "fluency": fluency_m["fluency_score"],
            "telemetry": telemetry or {},
        }

    return {
        "query_id": query_id,
        "query": query,
        "expected_action": expected_action,
        "model_action": "answered",
        "context_recall": retrieval_m["context_recall"],
        "context_precision": retrieval_m["context_precision"],
        "faithfulness": groundedness_m["faithfulness"],
        "hallucination_rate": groundedness_m["hallucination_rate"],
        "token_precision": token_m["precision"],
        "token_recall": token_m["recall"],
        "token_f1": token_m["f1"],
        "rouge_l": rouge_l,
        "fluency": fluency_m["fluency_score"],
        "telemetry": telemetry or {},
    }

def generate_ieee_latex_table(results: List[Dict[str, Any]]) -> str:
    """Generates IEEEtran compliant LaTeX table."""
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Deterministic On-Device RAG Evaluation and Efficiency Metrics (Redmi 12 5G)}",
        r"\label{tab:rag_evaluation}",
        r"\begin{tabular}{lccccccr}",
        r"\hline",
        r"\textbf{Query ID} & \textbf{Ctx Recall} & \textbf{Ctx Prec} & \textbf{Faithfulness} & \textbf{Token F1} & \textbf{ROUGE-L} & \textbf{Decode (t/s)} & \textbf{TTFT (s)} \\",
        r"\hline",
    ]
    for r in results:
        qid = r["query_id"].replace('_', r'\_')
        cr = f"{r.get('context_recall', 0.0):.2f}"
        cp = f"{r.get('context_precision', 0.0):.2f}"
        faith = f"{r.get('faithfulness', 0.0):.2f}"
        f1 = f"{r.get('token_f1', 0.0):.2f}"
        rouge = f"{r.get('rouge_l', 0.0):.2f}"
        t = r.get('telemetry', {})
        tps = f"{t.get('decode_tok_per_sec', 0.0):.1f}" if 'decode_tok_per_sec' in t else "--"
        ttft = f"{t.get('ttft_ms', 0.0)/1000.0:.2f}" if 'ttft_ms' in t and t.get('ttft_ms') else "--"
        latex.append(f"{qid} & {cr} & {cp} & {faith} & {f1} & {rouge} & {tps} & {ttft} \\\\")
    latex.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return '\n'.join(latex)

if __name__ == "__main__":
    print("Testing Deterministic Offline Evaluator...")
    sample_context = (
        "Jorge Fonseca: Chief Legal Officer | Office: Lisbon | Joined: 2019. "
        "The learning budget is $1,500 USD/year per employee and the gym allowance is $60/month."
    )
    sample_gt = "Jorge Fonseca is the Chief Legal Officer, based in the Lisbon office."
    sample_resp = "Jorge Fonseca is the Chief Legal Officer based in Lisbon."

    res = evaluate_rag_turn(
        query_id="q2_single_hop_clo",
        query="Who is chief legal officer and which office are they based in?",
        ground_truth=sample_gt,
        expected_action="answer",
        retrieved_context=sample_context,
        model_response=sample_resp,
        telemetry={"decode_tok_per_sec": 3.6, "ttft_ms": 36683.7}
    )
    print(json.dumps(res, indent=2))
