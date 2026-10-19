# Efficient Mobile RAG: Research Protocol and Current Evidence

Status: implementation and experiment protocol, not a claim of completed
multi-device results.

## Research question

Can an evidence-budgeted, source-contract-aware RAG pipeline reduce user-visible
latency and unsupported answers on low-cost Android phones while preserving
answer coverage, compared with a budgeted fixed-top-k retriever?

The current project can answer this question for the retrieval and answer
generation path. It cannot yet claim the stronger HACER-RAG KV-cache contribution:
independent chunk KV import, split-precision KV storage, or selective native
recomputation are not implemented in the shipped app.

## System variants

Every experiment must identify one of these variants explicitly:

1. `budgeted_top_k`: fixed ranked chunks, subject to the same prompt-fit budget.
2. `ebmrag-v2`: hybrid dense/lexical retrieval, evidence-unit selection,
   overlap suppression, contradiction preservation, and prompt budgeting.
3. `mobcert-v1-generation`: source-contract retrieval and contract audit, but
   generation is forced for an apples-to-apples LLM comparison.
4. `mobcert-v1-direct`: the product optimization that may answer from a
   complete numeric/unit certificate without invoking the LLM. This is reported
   separately and is never pooled with generated latency.
5. `general-inference`: no retrieval; a separate CPU inference sanity set.

The physical runner now disables direct evidence answers by default. Use an
explicit opt-in when collecting the separate `mobcert-v1-direct` condition.

## Falsifiable hypotheses

- **H1 latency:** EBM-RAG-v2 reduces prompt tokens and/or retrieval-to-first-
  token latency relative to `budgeted_top_k` at equal output and context limits.
- **H2 coverage:** EBM-RAG-v2 retains or improves gold evidence coverage on
  multi-demand, enumeration, typo, numeric, negation, and distractor queries.
- **H3 integrity:** MobCert contract generation lowers unsupported protected-value
  claims and improves correct abstention relative to unconstrained generation.
- **H4 direct path:** `mobcert-v1-direct` lowers end-to-end latency for complete
  certificates, but its benefit is an execution-path optimization rather than a
  fair generation-speed comparison.

Reject a hypothesis when the paired confidence interval crosses zero for the
primary metric, when a quality guardrail regresses, or when the run is missing
correlation/provenance metadata.

## Required query set

The current Zephyr nine-query file is a regression/pilot set. A paper-grade
extension must add labeled cases for:

- single fact and multi-demand questions;
- bridge/entity-property retrieval;
- enumerations and long answers;
- numeric values, units, dates, and versions;
- negation and conflicting source values;
- spelling/Unicode variants and distractor chunks;
- unanswerable questions and expected abstentions.

Each record must contain an immutable `id`, exact query, mode, corpus ID,
relevant chunk IDs or source atoms, reference answer, and expected action.

## Physical experiment design

For each device, query, policy, and repetition:

1. Verify the package, APK/build, model, embedding model, index, corpus, and
   profile identity.
2. Wait for the battery/temperature readiness gate.
3. Run one unmeasured warm-up under the same execution path.
4. Run the measured request with a unique request ID.
5. Store the request-specific terminal result; never read a global latest file.
6. Cool down using the declared policy and record telemetry samples.

Use paired, counterbalanced order. For example, odd query IDs run EBM first and
even query IDs run MobCert first; the next repetition reverses the order. Keep
model, output cap, context cap, thread count, prompt template, index, and query
order fixed within a block.

Minimum useful pilot: 5 measured repetitions per query/condition/device after
one warm-up. A final report must retain every failure and timeout rather than
silently dropping it. Report median, IQR, p95, per-query values, paired deltas,
and bootstrap confidence intervals. Do not claim energy savings from battery
percentage or temperature; external current/voltage instrumentation is needed
for energy/query.

## Primary metrics

### System

- retrieval latency;
- prompt construction latency;
- prompt tokens and output tokens;
- TTFT and total warm latency;
- prefill/decode milliseconds and tokens/sec;
- peak PSS only when sampled during the request;
- CPU frequency and thermal samples with timestamps;
- failure, timeout, and invalid-telemetry counts.

### Retrieval and answer quality

- evidence recall/precision at the selected frontier;
- reciprocal rank and selected chunk count;
- answer token F1/ROUGE-L as secondary lexical metrics;
- source-unit support labels for a manually reviewed subset;
- unsupported protected-value claim rate;
- correct answer/refusal action;
- contract status and execution path.

Direct and generated execution paths must be separate strata. The primary
generation comparison uses `allowDirectEvidenceAnswer=false` for every policy.

## Provenance requirements

Every report in a final experiment directory must include:

- `experiment_id`, `request_id`, `query_id`, `corpus_id`;
- exact `chat_mode`, policy, strategy, execution path, and direct-answer flag;
- effective thread/context/output settings;
- APK, source revision, model, embedding model, index, corpus, and manifest
  hashes where available;
- device serial/model/Android build and timestamped telemetry;
- terminal status and error if not completed.

The current `run_device_inference.py` protocol writes request-specific result
paths and rejects stale, mismatched, partial, or policy-mismatched reports.
`evaluate_physical_runs.py --strict` rejects reports lacking correlation fields
and excludes general-mode rows from the document-RAG summary.

## Current evidence and limitations

The existing Redmi 12 5G pilot measured general inference at about 22.7 seconds
for `What is AI?`, with approximately 8.7 seconds prefill and 13.8 seconds
decode. It is an engineering measurement, not a final paper result: the phone
was at 9% battery and the original 131-second artifact is unavailable.

The existing 18-report policy snapshot is retained as historical engineering
evidence only. It mixes direct/refusal and generated execution paths, has too few
repetitions, includes general-mode rows, and lacks complete provenance. It must
not be used as the final policy claim without the protocol above.

## HACER-RAG boundary

The HACER-RAG blueprint remains the correct next research track after efficient
mobile RAG is measured. A HACER claim requires a native correctness harness that
compares full-prefill logits with exact-prefix and independently encoded chunk
reuse, records next-token KL/top-1 agreement, validates RoPE/mask state, and
falls back safely. It then requires actual KV storage formats, page accounting,
dequantization measurements, selective recomputation, and multi-device tests.

The current Phase 1 session probe demonstrates only a llama.rn session load
operation. It is not independent chunk KV reuse and must not be described as a
HACER speedup.

## Reproducible commands

Offline checks:

```bash
cd chat
npx tsc --noEmit
npm run test:rag
cd ..
python3 -m unittest discover -s benchmark -p 'test*.py'
```

Strict evaluation of a completed physical experiment:

```bash
python3 benchmark/evaluate_physical_runs.py \
  --runs-dir benchmark/results/<experiment-id> \
  --labels benchmark/zephyr_evaluation_queries.json \
  --strict \
  --output benchmark/results/<experiment-id>/evaluation.json
```

Real-phone collection requires a working `adb` client, a charged device, a
built APK containing the current Metro bundle, and an explicitly selected
profile/model/index. Missing hardware or toolchain access is a blocked run,
not a zero or synthetic result.
