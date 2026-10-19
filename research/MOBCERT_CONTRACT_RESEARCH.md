# MobCert-v1: Evidence Contract Review

## Scope

MobCert-v1 is an opt-in, extractive answer contract for the offline document
RAG path. It is intended to answer a narrower systems question:

> Can a small on-device controller make the evidence required by a question
> explicit, preserve conflicts instead of silently deduplicating them, and
> prevent protected numeric/polarity claims from being presented as sourced
> when the selected evidence does not support them?

This is a contract and evidence-selection mechanism. It is not a claim of
general semantic entailment, a new neural retriever, or a substitute for the
physical-phone experiment.

## Implementation

| Component | Role |
|---|---|
| `chat/utils/rag/questionDemands.ts` | Splits multi-part questions into required attribute/entity demands. |
| `chat/utils/rag/evidenceAtoms.ts` | Converts source-preserving chunks into sentence/row atoms and attribute-scoped facts. |
| `chat/utils/rag/evidenceCertificate.ts` | Selects a bounded evidence cover, records missing/ambiguous demands, renders citations, and audits protected answer claims. |
| `chat/utils/rag/ragPipeline.ts` | Exposes the opt-in `mobcert-v1` retrieval policy and preserves its certificate in per-query metrics. |
| `chat/utils/rag/promptBudget.ts` | Tells the generator to report missing evidence and disclose conflicts rather than choose silently. |
| `chat/tasks/qa.ts` | Audits the generated answer and removes sentences containing unsupported protected values in the opt-in path. |

The mechanism remains extractive: source text is never paraphrased by the
selector. Atom metadata is used for admission and auditing, while the original
source span is passed to the model.

## Contract semantics

Each demand is assigned one of these statuses:

* **`certified`** — at least one selected source span matches the requested
  entity/attribute and no scoped value or polarity conflict was observed.
* **`ambiguous`** — selected evidence contains different values or opposing
  polarity for the same demand. An answer audit cannot report this contract as
  fully certified.
* **`missing`** — no selected source span satisfies the demand.

The answer audit has a separate status:

* **`certified`**, **`partial`**, **`ambiguous`**, **`unverifiable`**, or
  **`abstained`**.

The audit checks protected numbers, units, multi-word entity labels, citation
ordinals, and polarity. It deliberately does not claim that word overlap
proves full factual entailment; answer correctness and groundedness still need
human or labeled evaluation.

## Edge cases closed in this review

1. **Unrelated numbers in one source atom** — `Weight: 2.3 kg` and
   `Manufacturing year: 2024` no longer make a weight demand ambiguous.
2. **Local polarity** — `approved but not exportable` is positive evidence for
   approval; the unrelated negation is not assigned to the approval fact.
3. **Table rows** — a row such as `Jorge Fonseca | Chief Legal Officer |
   Lisbon | 2019` inherits `Office` and `Joined` column semantics from its
   header. The header itself is structural and cannot certify a value.
4. **Qualitative conflicts** — `Office: Lisbon` and `Office: Reykjavik` are
   recorded as conflicting values, not treated as harmless duplicate text.
5. **Enumeration** — a request such as “What are the offices?” may certify
   multiple distinct office values; a singular lookup reports the conflict.
6. **Contradictory answer status** — an answer that cites one side of an
   ambiguous contract is reported as `ambiguous`, never `certified`.

## Deterministic validation

The production TypeScript modules are exercised by the local regression suite:

```bash
cd chat
npm run test:rag
npx tsc --noEmit
```

The suite currently contains 33 passing cases covering ingestion, persistence,
hybrid retrieval, token budgets, complementary evidence, conflicting numeric
values, conflicting polarity, scoped values, table semantics, enumeration,
partial answers, unsupported protected values, citations, qualitative labels,
end-to-end table retrieval, end-to-end conflict retrieval, missing-demand
abstention, and the opt-in MobCert pipeline. These are software-contract
results only; they are not phone latency or model-answer-quality results.

## Physical-phone study protocol

The physical study should compare the same document and query order under
`ebmrag-v2` and `mobcert-v1`, keeping model hash, context length, output limit,
temperature, thread policy, warm-up, and device state fixed. Record for every
query:

* selected and candidate chunk IDs;
* demand statuses and `conflictingValues`;
* context and prompt token counts;
* native prefill, TTFT, decode, and total latency;
* the generated answer and its contract-audit status;
* independent labels for retrieval evidence coverage, answer correctness,
  groundedness, and refusal correctness.

The primary comparison is a quality-constrained latency/coverage trade-off,
not raw certificate status. A faster partial or ambiguous answer is not a
successful result unless the labeled quality protocol accepts the trade-off.

The physical trigger now accepts the policy explicitly:

```bash
cd benchmark
python3 run_device_inference.py --serial "$ANDROID_SERIAL" \
  --query "What is the battery life and weight of the AeroSense X7?" \
  --mode document --inference-mode baseline \
  --rag-policy ebmrag-v2 \
  --model-path /sdcard/Android/data/com.chat/files/models/gemma-2-2b-it-Q4_K_M.gguf

python3 run_device_inference.py --serial "$ANDROID_SERIAL" \
  --query "What is the battery life and weight of the AeroSense X7?" \
  --mode document --inference-mode baseline \
  --rag-policy mobcert-v1 \
  --model-path /sdcard/Android/data/com.chat/files/models/gemma-2-2b-it-Q4_K_M.gguf
```

Use the same warm-up and query order for both commands. The resulting report
contains `requested_rag_policy`, `rag_strategy`, `answerContract`, and the
native timing fields. Keep `PHYSICAL_DEVICE` records separate from the
`SIMULATED` files generated by the computer-side harness.

Each run is now saved under `benchmark/results/physical_runs/` with a unique
timestamp/policy/mode filename. After collecting a matched block, evaluate it
with:

```bash
python3 benchmark/evaluate_physical_runs.py \
  --runs-dir benchmark/results/physical_runs \
  --labels benchmark/zephyr_evaluation_queries.json \
  --output benchmark/results/physical_run_evaluation.json
```

The evaluator groups median TTFT/total latency and labeled context recall,
context precision, answer F1, faithfulness, refusal accuracy, and contract
statuses by policy. It skips reports without matching labels and never treats
synthetic records as physical runs.

## Current research conclusion

The contract behavior is now deterministic and regression-tested, including
the false-ambiguity cases identified during review. The research claim remains
conditional until the physical phone runs establish whether MobCert-v1 changes
prefill latency, evidence coverage, refusal correctness, or answer quality in
the target deployment.

The first collected batch has now been audited in
`research/PHYSICAL_RUN_AUDIT.md`. It contains real phone measurements but is
not a valid final policy comparison because several records have query/policy
metadata mismatches and the old EBM rows used `fixed-top-k`. The new collector
rejects those mismatches; the physical experiment must be rerun after the
revised application is installed.

## Computer-side validation and insights

The repository's computer-side scripts were also exercised to validate the
measurement pipeline. These results are useful for software correctness and
metric sanity checks only:

* `benchmark/test_evaluate_rag_offline.py` and
  `benchmark/test_evaluate_physical_runs.py`: **7/7** evaluator regression
  tests pass. The evaluator now uses explicit content-token precision, retains
  short numeric fragments, and removes citation ordinals before scoring, so an
  unsupported value such as `9.9 kg` is not treated as grounded by overlap
  with `2.3 kg`, and `[1]` does not lower answer F1.
* `benchmark/check_vectors.py`: the checked-in device index contains **9
  chunks**, each with a **384-dimensional** vector and norm **1.0000**. The
  check uses only the Python standard library; it does not require NumPy.
* `benchmark/benchmark.py --iterations 3 --warmups 1 --simulate`: the wrapper
  completes **108 explicitly `SIMULATED` records** across low, medium, and high
  profiles and generates the bottleneck report. The wrapper now uses the active
  Python interpreter rather than assuming a `python` executable exists.
* The current synthetic formula produces a mean warm-query latency of
  **7,138.02 ms** and a median of **6,120.90 ms**. Its modeled component split
  is decode **69.5%**, prefill **29.4%**, embedding **0.6%**, tokenization
  **0.3%**, vector search **0.2%**, and context construction **0.1%**.

The last percentages are properties of the synthetic formula, not observations
of the Redmi 12 5G. They are useful as a pipeline smoke test and as a reminder
that a physical run must record native prefill and decode timings separately.
They must not be used as the paper's device-performance result.

The pre-existing four-case phone-evaluation fixture was also recomputed with
the corrected metrics. It shows the important failure pattern the physical
study should test: the learning-budget case has full context recall and high
answer F1, while the stored CLO and AeroSense cases have context recall of
`0.12` and `0.00` and are false refusals. This supports prioritizing retrieval
and evidence coverage before making a thread-policy claim. The fixture is
repository test data rather than a newly collected, independently verified
device run, so it remains diagnostic only.
