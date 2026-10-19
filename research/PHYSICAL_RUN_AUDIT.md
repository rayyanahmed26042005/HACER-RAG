# Physical Run Audit and Corrected Experiment Status

## Audit date

2026-10-17

## Why the supplied comparison cannot yet be used as a paper result

The checked-in physical reports are genuine `PHYSICAL_DEVICE` artifacts, but
the first batch is not a valid matched EBM-RAG-v2 versus MobCert-v1 comparison.
The audit found three reproducibility failures:

1. Several filenames encode one query ID while the JSON content contains a
   different query. For example, the file named for the MobCert battery query
   contains the general Iceland-capital query and reports `ebmrag-v2` inside
   `research_metrics`.
2. The files labeled EBM-RAG-v2 were recorded with
   `rag.strategy: fixed-top-k`, not the evidence-budgeted `ebmrag-v2` strategy.
3. The previous evaluator scored skipped or refused answers as faithfulness
   `1.0` and sometimes treated a refusal as a perfect answer. It also missed
   refusal wording such as “the provided excerpts do not contain...”.

Consequently, the headline values claiming 100% MobCert refusal accuracy and a
32.8% TTFT reduction are not accepted as final research results. The old data
must be retained as an audit trail, but it should not be used in the paper's
matched comparison table.

## Root causes found in the implementation

### MobCert zero-selection for `What is zypher?`

The old path treated the lower-case entity token `zypher` as an ordinary
attribute-like demand. It required direct lexical support and had no broad
entity demand mode. The indexed document contains `Zephyr`, so dense retrieval
returned candidates, but the certificate saw no demand match and selected zero
atoms. The old report confirms this:

* `candidateChunkCount: 3`
* `selectedChunkCount: 0`
* `queryTermCoverage: 0`
* demand label `zypher`, status `missing`
* `failureReason: contract_missing_demands`

The fix adds broad entity demands, bounded transposition matching (`zypher` ↔
`zephyr`), a broad synthesis prompt, and a larger broad-query output/context
plan. The contract remains source-span based and does not accept arbitrary
unrelated chunks.

### False ambiguity in flattened leadership tables

The old atomizer interpreted a flattened PDF row such as:

```text
Name Title Office Joined
Jorge Fonseca Chief Legal Officer Lisbon 2019
```

as ordinary prose. It associated `2019` and an unrelated `2` from another
selected atom with both the officer and office demands. This produced false
ambiguity despite the answer being present.

The fix recognizes the common flattened table header, marks the header as
structural, and maps the row's trailing fields to `Office` and `Joined` facts.
The source span remains unchanged and the header cannot certify an answer by
itself.

### Metrics for skipped inference

The old `researchEvaluator.ts` assigned faithfulness `1.0` and hallucination
rate `0.0` to a skipped response. That meant “no answer was generated” looked
like “the answer was fully grounded.” The revised evaluator reports:

* `faithfulness: null`
* `hallucinationRate: null`
* answer correctness: `null`
* `qualityStatus: not_applicable`
* `answerGenerated: false`
* `ttft_ms: null`

Refusal correctness remains separately measurable.

## Implemented collection safeguards

`run_device_inference.py` now refuses to save a result when the app report's
query or chat mode differs from the requested query/mode. It also rejects a
policy mismatch between the requested policy and the nested application
report. Each accepted run retains a unique run ID and output path.

An explicit `--rag-policy ebmrag-v2` request now exercises the evidence-budgeted
selector even when inference uses fixed four-thread settings. The historical
`fixed-top-k` path is no longer mislabeled as EBM-RAG-v2 for new policy-tagged
runs.

## Automated validation after the fixes

```bash
cd chat
npm run test:rag
npx tsc --noEmit

cd ..
python3 -m unittest discover -s benchmark -p 'test_*.py' -v
python3 -m py_compile benchmark/*.py
python3 benchmark/evaluate_physical_runs.py \
  --runs-dir benchmark/results/physical_runs \
  --labels benchmark/zephyr_evaluation_queries.json \
  --output benchmark/results/physical_run_evaluation_corrected.json
```

Current automated status:

* **53/53** TypeScript RAG/contract tests pass.
* **8/8** Python evaluator tests pass.
* TypeScript compilation passes.
* Python benchmark compilation passes.

The corrected audit intentionally excludes stale fixed-top-k reports and
policy-mismatched files. It is diagnostic, not a new matched phone result.

## Final matched physical comparison

After rebuilding the embedded Metro bundle and installing the debug APK on the
connected Redmi 12 5G (`36fef3202145`), the final clean set contains **18/18
accepted physical reports**: nine labeled queries under each explicit policy.
The clean evaluator output is:

`benchmark/results/physical_run_evaluation_final_clean.json`

| Metric | EBM-RAG-v2 | MobCert-v1 |
|---|---:|---:|
| Runs | 9 | 9 |
| Mean context recall | 0.6429 | 0.5959 |
| Mean context precision | 0.1356 | 0.3218 |
| Mean faithfulness | 0.6888 | 0.6458 |
| Mean token F1 | 0.6373 | 0.4225 |
| Refusal accuracy | 0.5000 | 1.0000 |
| Median TTFT | 38,884.8 ms | 13,531.9 ms |
| Median total latency | 48,449.4 ms | 841.7 ms |

MobCert contract statuses were **4 certified**, **1 repaired**, and **2
abstained**; general-mode rows do not use an evidence contract. The direct
evidence path was physically verified on q1, q3, and q5, while q6 and q7
correctly abstained. The q2 leadership lookup was physically certified after
the flattened-table numeric-column fix.

The comparison is now suitable as a matched engineering result, with two
important interpretation limits: MobCert's median latency includes immediate
direct/refusal paths, and EBM-RAG-v2 still incorrectly answered q7 once, so
its refusal accuracy is 0.5 rather than 1.0. These limitations must remain
visible in any paper table.

## Lightweight direct-evidence upgrade

The app now has a narrow MobCert fast path. When every requested demand is
certified with exactly one positive numeric value and unit, the app renders a
cited answer directly from the evidence certificate instead of invoking Gemma.
Broad qualitative questions, enumerations, conflicts, negation, missing units,
and incomplete coverage continue through the existing generation/abstention
path. The behavior is recorded as `direct_evidence_answer` in the benchmark
report and remains controllable through `allowDirectEvidenceAnswer`.

## Reproduction command

To reproduce the physical comparison, rebuild and install the revised app,
re-ingest the same handbook, then run the same query list under both explicit
policies:

```bash
python3 benchmark/run_batch_benchmark.py \
  --serial "$ANDROID_SERIAL" \
  --queries benchmark/zephyr_evaluation_queries.json \
  --policies ebmrag-v2 mobcert-v1 \
  --inference-mode baseline \
  --delay 5
```

Before this run, cool the phone, record the model SHA-256, use the same profile
and index, and keep the query order matched. The resulting files should show:

* `query` equal to the requested query;
* `chat_mode` equal to the label mode;
* `requested_rag_policy` equal to `research_metrics.policy`;
* document EBM rows with `rag.strategy: ebmrag-v2`;
* MobCert rows with `rag.strategy: mobcert-v1`;
* skipped queries with N/A quality metrics rather than 100% faithfulness.

The clean evaluator command used for the final table was:

```bash
python3 benchmark/evaluate_physical_runs.py \
  --runs-dir benchmark/results/physical_runs_final \
  --labels benchmark/zephyr_evaluation_queries.json \
  --output benchmark/results/physical_run_evaluation_final_clean.json
```

## Latest request-correlated physical run

**Collection date:** 2026-10-18  
**Device:** Redmi 12 5G, model `23076RN4BI`, ADB serial `36fef3202145`  
**ADB route:** Windows ADB server at `172.25.224.1:5037`  
**APK SHA-256:** `207a9c79675b4b33513f040499c185a44b8784e5c9ca55cfff5230360cbc4fa7`  
**Model SHA-256:** `e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787`  
**Profile:** `p_1791008294421_84onr856`  
**Experiment directory:** `benchmark/results/adb_strict_matched_20261018`  
**Evaluator:** `physical-rag-v2-strict`

The newly built APK was installed successfully and its embedded bundle contains
the request-correlation markers `requestId`, `statusPath`, `resultPath`,
`benchmarkRequestId`, and `benchmark_query_id`.

The existing matched runner then completed **14/14 requested cells**: seven
document queries under `ebmrag-v2` and the same seven under `mobcert-v1`, one
measured repetition per cell, with counterbalanced policy order. The strict
evaluator accepted all 14 physical reports. It skipped only the batch summary
JSON because that file is an aggregate rather than a query report.

Correlation verification over the 14 query reports found:

- 14 unique request IDs;
- every `request_id` matched the embedded `benchmark_request_id`;
- every embedded `benchmark_query_id` matched the requested query ID;
- every report had `evidence_status: PHYSICAL_DEVICE`;
- every report was written under the request-specific result path;
- all reports included the requested model hash.

### Strict one-repetition result

| Metric | EBM-RAG-v2 | MobCert-v1 |
|---|---:|---:|
| Accepted reports | 7 | 7 |
| Mean context recall | 0.9000 | 0.8342 |
| Mean context precision | 0.1893 | 0.4506 |
| Mean faithfulness | 0.9234 | 0.8489 |
| Mean token F1 | 0.7706 | 0.7658 |
| Refusal accuracy | 0.5000 | 1.0000 |
| Median TTFT | 38,577.9 ms | 16,108.5 ms |
| Median total latency | 46,553.1 ms | 24,656.6 ms |
| Execution paths | 7 generated | 5 generated, 2 abstained |

Relative to EBM-RAG-v2, the MobCert arm's median values were approximately
`58.2%` lower for TTFT and `47.0%` lower for total latency in this block. These
are descriptive one-repetition results, not confidence-supported paper claims.

Direct evidence answering was disabled for this matched block, so MobCert's
two negative queries used the abstention path and did not invoke generation.
Those two cells must therefore remain a separate execution-path stratum rather
than being interpreted as a pure generation-speed improvement. The five other
MobCert cells were generated; their contract statuses were four `certified`,
one `repaired`, and two `abstained` overall.

The phone began at approximately 48% battery and 40 °C and reached 42 °C in the
longer block. PSS was sampled before and after each request; it is not a true
continuous peak measurement. No direct current/voltage instrumentation was
available, so this run supports latency/retrieval/answer-path observations but
not energy-per-query claims.

The detailed strict output is:

```text
benchmark/results/adb_strict_matched_20261018/evaluation_strict.json
```

This block is a valid request-correlated physical engineering result. It is not
yet the final HACER-RAG result: it evaluates the existing RAG policies and does
not implement independent KV-chunk import, HACERKV Flash pages, or selective
cross-chunk recomputation. It also has one repetition and one physical device;
the multi-tier HACER protocol remains outstanding.

## Latest full-query matched run

To avoid manually selecting only the document queries, the existing matched
runner was rerun against the complete `benchmark/zephyr_evaluation_queries.json`
list on 2026-10-18, with both explicit policies, one repetition, and the same
device/profile/model/APK provenance as the preceding block:

```text
benchmark/results/adb_strict_full_20261018
```

The runner completed all **18/18 requested cells**: nine queries under each
policy. The strict evaluator accepted the 14 document-RAG reports and skipped
the four general-mode queries plus the aggregate `batch_run_summary.json`.
The aggregate summary is not suitable for policy comparison because its
legacy run-count grouping includes general-mode MobCert rows; use
`evaluation_strict.json` as the authoritative result.

Strict evaluation output:

```text
benchmark/results/adb_strict_full_20261018/evaluation_strict.json
```

| Metric | EBM-RAG-v2 | MobCert-v1 |
|---|---:|---:|
| Accepted document reports | 7 | 7 |
| Mean context recall | 0.9000 | 0.8342 |
| Mean context precision | 0.1893 | 0.4506 |
| Mean faithfulness | 0.8909 | 0.8744 |
| Mean token F1 | 0.7718 | 0.7574 |
| Refusal accuracy | 0.5000 | 1.0000 |
| Median TTFT | 43,565.0 ms | 15,508.7 ms |
| Median total latency | 49,501.9 ms | 24,805.1 ms |

The strict block contains seven generated EBM-RAG-v2 paths. MobCert contains
five generated paths and two abstentions, with contract statuses **4
certified**, **1 repaired**, and **2 abstained**. Relative to EBM-RAG-v2, the
MobCert median values are approximately **64.4% lower TTFT** and **49.9% lower
total latency** in this one-repetition block. Because the two MobCert
abstentions have no TTFT, the TTFT comparison already reflects generated
MobCert rows only. Restricting total latency to the five generated MobCert rows
changes its median to **25,550.9 ms**, still approximately **48.4% lower** than
the EBM-RAG-v2 median. These remain descriptive engineering observations, not
confidence-supported HACER or paper claims.

The full-query run also confirms that general-mode queries must not be mixed
into the document-RAG quality table. The strict evaluator skipped q8 and q9 as
`not_document_rag_run`; this is expected and is not a failed inference. The
strict report set also preserves request correlation, physical-device evidence,
 policy labels, and model provenance for all 14 evaluated reports.

## Second matched repetition

A second full-query repetition was collected immediately afterward under the
same APK, model, profile, device, and request-correlated protocol:

```text
benchmark/results/adb_strict_full_20261018_r2
benchmark/results/adb_strict_full_20261018_r2/evaluation_strict.json
```

Again, all **18/18 requested cells** completed; the strict evaluator accepted
14 document reports and skipped the four general-mode reports plus the batch
summary. The second strict document block was:

| Metric | EBM-RAG-v2 | MobCert-v1 |
|---|---:|---:|
| Accepted document reports | 7 | 7 |
| Mean context recall | 0.9000 | 0.8342 |
| Mean context precision | 0.1893 | 0.4506 |
| Mean faithfulness | 0.9727 | 0.8814 |
| Mean token F1 | 0.7530 | 0.7309 |
| Refusal accuracy | 0.5000 | 1.0000 |
| Median TTFT | 44,368.7 ms | 15,213.3 ms |
| Median total latency | 54,042.4 ms | 25,241.9 ms |

MobCert again had five generated paths and two abstentions, but its generated
contract statuses changed to **3 certified**, **1 ambiguous**, and **1
unverifiable**. This variation is itself a reason not to treat the single
block's contract counts as stable accuracy evidence.

Across the two strict blocks (14 document reports per policy), the pooled
descriptive medians were **43,727.0 ms TTFT / 53,017.4 ms total** for
EBM-RAG-v2 and **15,361.0 ms TTFT / 25,023.5 ms total** for MobCert-v1. The
pooled MobCert medians are approximately **64.9% lower TTFT** and **52.8%
lower total latency**, but the observations are still from one physical device
and two repetitions rather than a confidence-supported result.

The device began the second block near 40 °C and reached 43 °C; CPU frequency
readings fell to 672/960 MHz late in the run. This visible thermal/frequency
state change means the two blocks should not be presented as independent
identically distributed samples. Future repetitions should record a cooling
criterion and either randomize block order more aggressively or stratify by
thermal state.
