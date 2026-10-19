# MobiCert-RAG: Research and Publication Plan

**Review date:** 2026-10-11  
**Repository:** `Porting-LLM-s-Model-Project`  
**Status:** implemented opt-in prototype; novelty narrowed after AdaGATE and evidence-verification review; physical ablation pending

## Executive conclusion

The current repository contains a strong, testable mobile-RAG engineering
artifact. It does **not** yet justify the claim that it introduced a new RAG
algorithm. Structure-aware chunking, dense plus BM25 retrieval, reciprocal-rank
fusion, typo handling, extractive context reduction, adaptive token budgets, and
mobile telemetry each have close precedents.

The initial MobiCert-RAG proposal was too broad. Its controller core—tracking
missing information, retrieving targeted evidence, selecting by coverage per
token, and stopping under a budget—is now directly overlapped by **AdaGATE**
(arXiv:2605.05245), and related sufficiency-aware systems. It should not be
presented as a new gap-aware retrieval algorithm.

The defensible research direction is now a narrower **MobiCert-RAG contract
protocol**:

> a deterministic, zero-auxiliary-model answer contract that binds each
> requested fact and every emitted value to source spans, preserves numeric and
> polarity integrity under extractive compression, and supports certified
> partial answers when a mobile prompt budget cannot cover every demand.

The candidate distinction is therefore the coupling of **typed source-span
contracts**, **hard evidence-integrity invariants**, and **per-slot answer
validation** on a low-resource offline device. Gap-aware selection and adaptive
retrieval remain useful implementation components, but they are baselines or
supporting modules rather than the novelty claim. The central experiment is
whether a zero-neural-overhead contract layer reduces unsupported numeric,
entity, temporal, and polarity claims while preserving partial-answer utility
under exact native prompt budgets.

This is a research hypothesis, not a novelty guarantee. A paper claim is
appropriate only after implementation, a direct AdaGATE comparison, an
expanded evidence-verification comparison, and matched physical-device
experiments.

### Novelty reassessment: 2026-10-11

The new search confirms that the original “certificate-gated gap-aware
retrieval” wording would overclaim. AdaGATE explicitly provides entity-centric
gap tracking, targeted micro-queries, utility-based evidence selection, and
replacement under a token budget. FAIR-RAG, A2RAG, SURE-RAG, EvidentialRAG,
BELIEFRAG, VikingRAG, and related work further cover evidence sufficiency,
uncertainty, adaptive escalation, and conflict-aware answering.

MobiCert-RAG is worth continuing only if it is implemented and evaluated as a
**mobile answer-contract and integrity protocol**, with these necessary
properties:

1. no auxiliary LLM, verifier, cross-encoder, or cloud call;
2. exact source-span provenance for every typed evidence item;
3. hard preservation of numbers, units, temporal qualifiers, and polarity;
4. per-demand status (`certified`, `ambiguous`, or `missing`);
5. constrained generation or post-generation checking that prevents an
   unsupported requested slot from being presented as supported;
6. a measured partial-answer utility/latency tradeoff on a physical Android
   device.

If those properties do not improve the measured risk/latency frontier over
AdaGATE-style selection and a strong answer-verification baseline, the work
should be published as a reproducible mobile evaluation or engineering report,
not as a new RAG algorithm.

## 1. What is already implemented

| Repository feature | Current status | Publication position |
|---|---|---|
| Structure-aware, heading/table-preserving chunks | Implemented in `chat/utils/rag/chunker.ts` | Engineering component; not individually novel |
| Normalized dense embeddings | Implemented in `embedder.ts` and `vectorStore.ts` | Standard retrieval infrastructure |
| BM25-like lexical postings plus dense retrieval | Implemented in `vectorStore.ts` | Hybrid retrieval is established |
| Reciprocal-rank fusion | Implemented in `vectorStore.ts` | Established ranking method |
| Bounded edit-distance entity matching | Implemented in `text.ts` and `vectorStore.ts` | Useful mobile heuristic; not a new retriever |
| Extractive evidence-unit packing | Implemented in `evidenceSelector.ts` | Close to selective context reduction and prompt compression |
| Exact chat-template/output-runway checks | Implemented in `promptBudget.ts` | Important correctness guard; not sufficient as a paper contribution alone |
| Adaptive thread/context/output controller | Implemented in `inferenceController.ts` and telemetry bridge | Candidate mobile-systems study; requires physical validation |
| Atomic index persistence and profile isolation | Implemented in `vectorStore.ts` | Reliability contribution, not algorithmic novelty |
| Label-dependent quality metrics | Implemented in `metrics.ts` and benchmark scripts | Needed for a credible evaluation |
| Deterministic evidence atoms and demand slots | Prototype in `evidenceAtoms.ts` and `questionDemands.ts` | Research instrumentation; parser errors must be measured |
| Source-span certificate selection and contract audit | Opt-in `mobcert-v1` path in `evidenceCertificate.ts` and `ragPipeline.ts`; 33 deterministic production-module regressions pass, including end-to-end table/conflict/missing-demand cases | Candidate contribution; not yet physically evaluated |

The existing internal documents correctly describe EBM-RAG v2 as an engineering
policy rather than a new neural retrieval algorithm. MobiCert-RAG now has an
opt-in prototype path with contract enforcement and a physical-run aggregation
tool, but remains a candidate v3 research direction until matched baselines and
physical-device ablations exist.

## 2. Literature check and overlap map

The following records were checked on arXiv on 2026-10-11. Their reported
results are claims from those papers, not results from this repository.

| Work | Main contribution | Overlap with this repository |
|---|---|---|
| **MobileRAG**, arXiv:2507.01079 | EcoVector partitions/partially loads index data; Selective Content Reduction reduces LM input | Overlaps mobile index efficiency and context reduction; our current hybrid search is not a replacement for EcoVector |
| **Pocket RAG**, arXiv:2602.13229 | Offline Android first-aid RAG with hybrid retrieval, selective compression, batched prompt decoding, and quantization caching | Overlaps the complete offline mobile pipeline and hybrid/compression direction |
| **SmartRAG**, arXiv:2607.14661 | On-device entity learning, provenance-preserving graph memory, and graph/lexical/dense retrieval | Directly overlaps entity/provenance-based hallucination reduction; a graph-based claim would not be new |
| **PerCache**, arXiv:2601.11553 | Predictive hierarchical cache for query and QKV reuse on mobile | Invalidates a generic “mobile RAG cache” claim |
| **CORE / Less is More**, arXiv:2606.20571 | SLM-free answer/clue extraction and residual retrieval for edge prompt compression | Directly overlaps lightweight extractive compression |
| **Lightweight Chunk Selection for Mobile RAG**, arXiv:2608.03148 | Evidence-alignment selector using query/generator/candidate features | Invalidates a generic “evidence-aware chunk selector” claim |
| **PACE**, arXiv:2608.25115 | Evidence coverage ordering and pressure-adaptive reranking budgets | Overlaps coverage-aware selection and adaptive ranking cost |
| **From Retrieved Context to Runtime Control**, arXiv:2608.19535 | Telemetry-informed adaptive compression on edge hardware | Overlaps device-state-driven compression policies |
| **AdaGATE**, arXiv:2605.05245 | Entity-centric gap tracking, targeted micro-queries, utility selection, replacement, and token-budgeted evidence repair | Directly overlaps the original MobiCert controller proposal; must be a primary baseline |
| **Question Decomposition for RAG**, ACL 2025, DOI:10.18653/v1/2025.acl-srw.32 | LLM sub-question decomposition, per-sub-question retrieval, merged-pool reranking | Establishes the demand-decomposition and complementary-evidence baseline |
| **FAIR-RAG**, arXiv:2510.22344 | Structured evidence assessment, explicit gap filling, adaptive query refinement, iterative refinement | Overlaps sufficiency/gap-driven retrieval and faithful generation |
| **A2RAG**, arXiv:2601.21162 | Sufficiency verification with targeted graph/text refinement and adaptive escalation | Overlaps escalation and provenance-backed refinement |
| **SURE-RAG**, arXiv:2605.03534 | Set-level claim/evidence sufficiency, uncertainty features, and selective answering | Overlaps answer-level verification and auditable abstention |
| **EvidentialRAG**, arXiv:2607.10491 | Conflict-preserving evidential fusion and direct/conflict/abstain routing | Overlaps conflict-aware answering; our no-auxiliary-model constraint is narrower |
| **VikingRAG**, arXiv:2609.11390 | Evidence-gap-driven multi-round retrieval and adaptive escalation with trace reuse | Overlaps gap-driven escalation and retrieval-cost reduction |
| **BELIEFRAG**, arXiv:2609.39139 | Persistent state over sufficiency, reliability, conflict, uncertainty, gaps, and acquisition cost | Overlaps explicit evidence state and action selection; cited as a later adjacent record |
| Google AI Edge RAG documentation | Official on-device RAG building blocks and deployment guidance | Establishes the platform context; it is not a research baseline |

Primary records:

- https://arxiv.org/abs/2507.01079
- https://arxiv.org/abs/2602.13229
- https://arxiv.org/abs/2607.14661
- https://arxiv.org/abs/2601.11553
- https://arxiv.org/abs/2606.20571
- https://arxiv.org/abs/2608.03148
- https://arxiv.org/abs/2608.25115
- https://arxiv.org/abs/2608.19535
- https://arxiv.org/abs/2605.05245
- https://aclanthology.org/2025.acl-srw.32/
- https://arxiv.org/abs/2510.22344
- https://arxiv.org/abs/2601.21162
- https://arxiv.org/abs/2605.03534
- https://arxiv.org/abs/2607.10491
- https://arxiv.org/abs/2609.11390
- https://arxiv.org/abs/2609.39139
- https://ai.google.dev/edge/mediapipe/solutions/genai/rag

The closest invalidation risk for the original controller is **AdaGATE**. The
closest risks for the revised protocol are **SURE-RAG** and **EvidentialRAG**.
MobiCert-RAG must therefore claim only a deterministic, mobile-first
**source-span answer contract** if the implementation actually supplies one.
It must not claim the first evidence-aware selector, gap-aware retriever,
uncertainty-aware RAG system, or abstaining RAG system.

## 3. Proposed method: MobiCert-RAG

### 3.0 Revised contribution boundary

The following modules are **not** the claimed invention: query decomposition,
gap detection, targeted retrieval, marginal coverage selection, or adaptive
stopping. MobiCert-RAG may reuse them and must compare against them.

The proposed contribution is the **answer contract** that survives the full
mobile pipeline:

```text
question -> typed demands -> source-span evidence -> integrity checks
         -> exact prompt budget -> constrained answer -> per-slot audit
```

The contract is a structured object that is produced before generation and
checked after generation. It contains one record per requested demand, the
source spans allowed to support that demand, exact values/units/polarity that
may be emitted, and the status of the demand. The contract is intentionally
smaller and more deterministic than an auxiliary verifier model.

The research question is:

> Can a typed, source-span answer contract provide useful certified partial
> answers and reduce unsupported factual claims on low-resource Android devices
> without the latency and memory cost of an auxiliary neural verifier?

That question is narrower than “can gap-aware retrieval improve RAG?” and is
the part that the experiments must be designed to answer.

### 3.1 Design goal

For a question with several requested facts, a top-ranked chunk can be highly
similar but still insufficient. For example:

```text
Question: What is the AeroSense X7 battery life and weight?

Chunk A: The AeroSense X7 is our field device.
Chunk B: Its QuantumCell battery lasts 72 hours and it weighs 2.3 kg.
```

Keeping only Chunk A is a retrieval success by similarity but an answer failure.
Keeping only a budget-truncated fragment of Chunk B can also lose the subject
link. The system should represent the question as two demands—`battery life`
and `weight`—and select evidence that covers both while retaining the entity
bridge.

### 3.2 Evidence atoms and integrity signatures

At ingestion, each structure-preserving chunk is divided into extractive
evidence units. Each unit receives a compact, deterministic signature:

```text
EvidenceUnit {
  documentId, chunkId, section, sourceOrder,
  text,
  entities: normalized entity terms,
  attributes: normalized field/property terms,
  numbers: exact numeric strings,
  units: normalized units,
  polarity: positive | negated | unknown,
  temporalMarkers: dates/years/relative-time terms,
  factKeys: (entity, attribute, time-scope) keys,
}
```

The signature is metadata; the source text remains unchanged and is what gets
passed to the model. Numeric strings, units, negations, and conflicting values
are never removed as “duplicates.” A deterministic parser is sufficient for the
first version. It can recognize table columns, colon-delimited fields, common
units, negative markers, and adjacent entity continuations without a second
language model.

### 3.3 Query demand slots

The query is decomposed into a small set of demand slots:

```text
DemandSlot {
  id,
  clause,
  entityTerms,
  attributeTerms,
  numeric/unit constraints,
  required: true | false,
  enumeration: true | false,
}
```

Examples:

| Query | Demand slots |
|---|---|
| “Who is the Chief Legal Officer and which office?” | person/role; office/location |
| “Battery life and weight of AeroSense X7” | battery duration; device weight; AeroSense X7 bridge |
| “What are the products?” | enumerated product names; product-section coverage |
| “Does the policy allow refunds?” | policy/refund; positive or negated allowance |

The parser does not need to infer a complete semantic parse. It needs a
conservative set of slots. Over-splitting causes more retrieval, while
under-splitting is caught by the multi-demand query evaluation set.

### 3.4 Certificate-gated selection

For each candidate evidence unit `u` and demand slot `d`, calculate:

```text
coverage(u,d) =
  exact entity/attribute/value match
  + lexical match
  + bounded fuzzy entity match
  + section/bridge support
```

The selector maximizes marginal demand coverage per estimated prompt token:

\[
U(u \mid S,q) =
\frac{\sum_{d\in D} w_d\,\mathbf{1}[d\text{ newly covered by }u]
      + \alpha R(u,q) + \beta B(u,S)}
     {(\widehat{tokens}(u)+\tau)^{\rho}}
     - \gamma O(u,S).
\]

Here `R` is the existing dense/lexical relevance, `B` is an entity-bridge
bonus, and `O` penalizes containment overlap. The selection is bounded and
greedy, so it remains predictable on a phone.

The selector emits a certificate rather than only a score:

```text
EvidenceCertificate {
  slotId,
  status: covered | ambiguous | missing,
  sourceIds,
  exactTerms,
  numericValues,
  polarity,
  conflictingValues,
}
```

Selection stops early when all required slots are `covered`, or expands the
candidate frontier only for `missing` slots. If the device budget is exhausted,
the answer prompt tells the model which slots are covered and which are not.
The app can therefore answer one supported part instead of inventing the
unsupported part or refusing the whole question.

### 3.7 Answer-contract enforcement

The contract must affect generation, not only logging. The first implementation
should use constrained, citation-bearing output rather than unconstrained prose:

```text
Answer each requested item with [source-id] citations.
For a `missing` item write: Not found in the uploaded documents.
For an `ambiguous` item report the competing values and cite both sources.
Do not introduce a number, unit, date, named entity, or polarity not present
in the allowed evidence for that item.
```

After generation, a deterministic checker extracts candidate protected tokens
(numbers, units, dates, named entities, negation markers, and citation IDs).
It compares them with the contract's allowed token set and records:

```text
AnswerContractAudit {
  slotId,
  citedSourceIds,
  unsupportedProtectedTokens,
  missingRequiredCitation: boolean,
  status: certified | repaired | abstained | unverifiable,
}
```

For the safety-preserving mode, an unsupported protected value is not silently
accepted. The system either replaces the slot with the fixed refusal phrase or
marks it `unverifiable` for evaluation. The raw model output remains available
for research logs; the displayed answer uses the audited form. This distinction
allows a fair measurement of both **unsupported-claim rate** and **answer
utility**, rather than hiding failures in a global refusal metric.

This layer is the main difference from AdaGATE's described pipeline: AdaGATE
assembles a compact evidence set for a black-box generator, whereas MobiCert's
research hypothesis is that a typed contract can constrain and audit the final
answer with no auxiliary LLM. The difference must be tested directly; it is not
assumed to imply novelty.

### 3.5 Conflict and negation firewall

For each `(entity, attribute, time-scope)` fact key:

1. exact duplicate units may be merged;
2. differing values, units, dates, or polarity are retained;
3. if two high-relevance values conflict, the certificate becomes `ambiguous`;
4. the prompt instructs the model to report both values with citations and not
   silently reconcile them.

This targets a failure mode that ordinary deduplication can create: removing a
second unit because it has high word overlap can erase a changed price,
version, date, or negation. It is also directly testable with synthetic
contradiction queries.

### 3.6 Budget escalation policy

MobiCert-RAG uses a three-stage search:

```text
Stage 0: compact lexical/anchor lookup
Stage 1: dense + lexical hybrid search over the bounded candidate pool
Stage 2: section-neighbor expansion only for uncertified demand slots
```

The prompt budget is allocated to the minimum certificate first. An extra
retrieval or context expansion is allowed only when its expected uncovered-slot
gain exceeds its estimated token and device cost:

\[
\frac{\Delta\text{certified-demand-weight}}
     {\Delta\text{prefill-tokens} + \eta\,\Delta\text{retrieval-cost}}
     > \theta_{device}.
\]

`theta_device` is selected from measured phone operating points and can depend
on thermal/memory state. This is a runtime policy, not a learned neural model.
The critical distinction is that device telemetry changes the cost threshold,
while evidence certificates determine whether more evidence is needed.

## 4. Why this can reduce hallucinations

MobiCert-RAG does not prove model truthfulness. It reduces identifiable causes
of hallucination:

- unsupported question clauses are marked `missing` instead of being silently
  omitted from retrieval;
- facts are passed as source excerpts rather than paraphrased summaries;
- citations map selected evidence back to document and chunk IDs;
- conflicting numbers, units, dates, and negations are preserved;
- the model is instructed to abstain per missing slot, not necessarily for the
  whole query;
- evaluation separates retrieval coverage, answer correctness, citation
  support, and refusal correctness.

The paper should say **hallucination reduction under a labeled protocol**, not
“hallucination-free.”

## 5. Implementation plan

### Phase A: instrument the current baseline

Already mostly available:

- `rag_strategy`, candidate IDs, selected IDs, context budget, prompt token
  count, token checks, and query-term coverage;
- native prompt/decode timings and device telemetry;
- label-dependent retrieval and answer metrics.

Before changing the algorithm, freeze this as the `EBM-RAG-v2` baseline and
record model hash, embedding identity, index revision, and prompt template.

### Phase B: implement the certificate layer

Add the following modules:

- `chat/utils/rag/evidenceAtoms.ts`: source-unit signatures and fact keys;
- `chat/utils/rag/questionDemands.ts`: conservative demand-slot extraction;
- `chat/utils/rag/evidenceCertificate.ts`: coverage, ambiguity, and
  per-slot status;
- `chat/utils/rag/certificateSelector.ts`: bounded marginal-gain selection and
  budget escalation;
- `chat/utils/rag/contradictionFixtures.ts`: labeled numeric/negation fixtures
  for regression tests.

Extend `VectorEntry` with compact atom metadata and persist a versioned index.
Keep the existing raw chunk text and embedding format so that a certificate can
always point to the original source span.

### Phase C: integrate the prompt and metrics

Prototype status on 2026-10-11:

- `evidenceAtoms.ts`, `questionDemands.ts`, and `evidenceCertificate.ts` are
  implemented and covered by regression tests.
- `ragPipeline.ts` accepts `policy: 'mobcert-v1'` and emits the answer contract,
  selected atoms, citation-bearing excerpts, and per-demand status.
- `qa.ts` records a contract audit and conservatively removes sentences with
  unsupported protected numeric/entity/unit values in the opt-in policy.
- `ChatScreen.tsx` accepts the benchmark trigger field
  `ragPolicy: 'mobcert-v1'`; normal interactive chat still uses EBM-RAG v2.
- The current checker audits protected tokens and citation coverage. It is not
  a full entailment model and cannot justify a general “certified correct
  answer” claim. A labeled factuality evaluation remains mandatory.

Example controlled trigger:

```json
{
  "query": "What is the battery life and weight of AeroSense X7?",
  "chatMode": "document",
  "inferenceMode": "adaptive",
  "ragPolicy": "mobcert-v1",
  "measurePromptTokens": true
}
```

The document prompt should include a small machine-readable block such as:

```text
Evidence status:
- battery life: covered by [1]
- weight: covered by [2]
- warranty period: missing

Answer covered items with citations. State that the missing item is not found.
If sources conflict, report the conflict and cite both sources.
```

Add these metrics per query:

- `demand_slot_count`;
- `certified_slot_count` and `certified_slot_weight`;
- `missing_slot_count`;
- `ambiguous_fact_key_count`;
- `contract_audit_status` and `contract_repair_count`;
- `unsupported_protected_token_count`;
- `required_citation_coverage`;
- `numeric_value_recall`;
- `polarity_preservation`;
- `citation_support_precision`;
- `false_refusal_rate`;
- `unsupported_answer_rate`;
- `prefill_tokens_per_certified_slot`;
- `extra_stage_invocations` and `certificate_overhead_ms`.

## 6. Evaluation required before an arXiv submission

### Baselines

Use the same embedding model, LLM, index, prompt template, and device for:

1. dense top-k;
2. BM25 top-k;
3. current hybrid fixed-top-k;
4. current EBM-RAG v2;
5. an AdaGATE-style gap-aware controller, reproduced as faithfully as possible
   under the same device constraints;
6. MobiCert-RAG selection without answer-contract enforcement;
7. full MobiCert-RAG with source-span contract enforcement;
8. a strong answer-verification baseline such as SURE-RAG's set-level decision
   protocol, or a documented lightweight approximation if its verifier cannot
   run on the target phone;
9. an oracle evidence selection upper bound.

Do not describe MobileRAG or Pocket RAG numbers as our baselines unless their
implementations, datasets, models, and hardware are reproduced. Cite their
results as related work and compare mechanisms carefully.

### Query categories

Build a labeled set containing at least:

- single-fact questions;
- multi-demand questions requiring two or more chunks;
- enumeration questions;
- entity-bridge questions where the name and property are separated;
- typo and Unicode variants;
- numeric/unit questions;
- negation questions;
- conflicting-version/date/price questions;
- answerable questions with distractor chunks;
- unanswerable questions requiring correct refusal.

The current Zephyr handbook fixtures are a useful seed, not a sufficient
benchmark. Every query needs gold relevant source units, demand slots, expected
answer facts, and an expected answer/refusal action.

### Quality metrics

Report separately:

- demand-slot recall and precision;
- source-unit Recall@K and MRR;
- answer token F1/ROUGE-L only as secondary metrics;
- citation support precision and unsupported-claim rate;
- numeric/unit exactness;
- negation/polarity accuracy;
- contradiction disclosure accuracy;
- false-refusal rate;
- human or expert factual labels for a final subset.

Word overlap is not entailment. The existing `metrics.ts` behavior of leaving
groundedness and correctness null without labels should be preserved.

### Efficiency metrics

On the Redmi 12 5G, collect at least five warm repetitions per query/configuration
after one excluded warm-up, plus a separate cold-start block:

- retrieval, certificate, prompt-tokenization, prefill, TTFT, decode, and total
  latency;
- prompt tokens and generated tokens;
- peak PSS/native memory where consistently measurable;
- battery temperature, Android thermal status, CPU frequencies, and battery
  percentage;
- energy if a repeatable external or hardware measurement is available.

Report median, IQR, p95, standard deviation, and paired bootstrap confidence
intervals. Randomize query order within matched blocks and keep model/index
hashes fixed. The current repository audit correctly notes that synthetic
timing artifacts cannot support physical performance claims.

### Ablations

The central ablations are:

- no demand decomposition;
- no certificate gate;
- no answer-contract enforcement;
- unconstrained prose versus citation-bearing constrained output;
- fixed top-k instead of frontier expansion;
- no numeric/unit firewall;
- no conflict preservation;
- fixed context budget versus certificate budget;
- telemetry threshold disabled;
- dense-only, lexical-only, and hybrid retrieval.

The method is promising only if it improves unsupported-claim and false-refusal
rates at a matched quality target, while reducing prompt work or keeping its
certificate overhead below the saved prefill time.

## 7. Defensible paper claims

### Claims we may make after successful experiments

- We present a fully offline, training-free evidence-certificate protocol for
  resource-constrained mobile RAG.
- The protocol allocates retrieval/context budget by uncovered answer demands,
  rather than a fixed top-k or compression ratio.
- It preserves provenance, numerical values, units, polarity, and conflicts
  during extractive evidence selection.
- On a labeled mobile document-QA suite, it improves evidence coverage and
  reduces unsupported claims or false refusals at a measured latency/energy
  operating point.
- We provide reproducible physical-device traces and an open evaluation
  harness.

### Claims we should not make

- “The first mobile RAG system.”
- “The first hybrid mobile retriever.”
- “The first adaptive compression or context-budget method.”
- “Hallucination-free answers.”
- “State of the art” without reproducing matching baselines.
- Any physical speedup from the existing simulated benchmark files.

## 8. Draft arXiv abstract (results intentionally blank)

> On-device retrieval-augmented generation enables private question answering
> over personal documents, but mobile systems face a coupled failure: a
> similarity-ranked context can be too small to answer all parts of a question,
> while unnecessary context increases prompt-prefill latency and memory use.
> We present MobiCert-RAG, a training-free evidence-certificate protocol for
> offline mobile RAG. The method decomposes a question into conservative demand
> slots, represents retrieved text as extractive evidence units with provenance
> and integrity signatures, and greedily allocates retrieval and prompt budget
> according to marginal certified coverage per token. A device-aware frontier
> expansion stage retrieves additional evidence only for uncertified demands.
> Numeric values, units, temporal qualifiers, negations, and conflicting facts
> are preserved rather than removed as lexical duplicates. The generator is
> given per-slot evidence status and is instructed to cite supported facts and
> abstain only on unsupported slots. We implement MobiCert-RAG in a fully
> offline Android application and evaluate it against dense, lexical, hybrid,
> fixed-top-k, and evidence-budgeted baselines on labeled multi-demand document
> questions. On-device experiments measure evidence quality, unsupported-claim
> rate, refusal correctness, prompt-prefill latency, memory, thermal state, and
> energy. The results will establish whether certificate-gated budgeting offers
> a better factuality/efficiency frontier than existing mobile RAG policies.

## 9. Submission checklist

Before uploading to arXiv:

1. Search arXiv, Semantic Scholar, Google Scholar, and ACM/IEEE indexes again
   immediately before submission.
2. Freeze a commit and release the code, fixture documents, query labels,
   device metadata, model hashes, and raw run records.
3. Replace every placeholder or exploratory physical result with a reproducible
   run ID and confidence interval.
4. Include negative results if MobiCert-RAG does not beat the best fixed policy.
5. State the target device and model/embedding versions precisely.
6. Separate algorithmic contributions from implementation conveniences.
7. Have a domain expert review numeric, medical, legal, or safety-critical
   evaluation claims before making them.
