# HACER-RAG: Complete Paper Text for Humanization

> **Instructions for Humanizing:**
> - You can freely rewrite, simplify, rephrase, or polish any sentences to sound more natural and human.
> - Please keep the core empirical numbers (e.g., percentages, latencies, p-values) intact.
> - Keep markers like `[Cite: ...]` and `[Table X]` so they can be linked back to LaTeX macros.
> - Once you finish humanizing, send the updated text (or sections of it) back, and it will be reinjected into the LaTeX paper.

---

## Title
HACER-RAG: Evidence-Version-Aware KV-Cache Reuse and Selective Repair for On-Device Retrieval-Augmented Generation

## Authors
Mobile Systems and Language Model Research Group  
PortingLLM Research Project  
{systems-research, mobile-llm}@portingllm.org

---

## Abstract
On-device Retrieval-Augmented Generation (RAG) offers private, low-latency question answering, but small language models (SLMs) on mobile CPUs pay a severe prompt-prefill cost whenever retrieved context is re-encoded. Restored Key-Value (KV) caches remove this cost, yet mobile runtimes treat cache files as opaque snapshots, causing silent stale-cache hallucination when evidence changes and exposing context overflows and continuation-boundary errors. We present HACER-RAG (Hierarchical Adaptive Cache, Evidence-version-aware Repair for on-device RAG), which binds every KV segment to SHA-256 chunk hashes under a canonical evidence layout, re-prefills only changed downstream chunks, falls back to scratch prefill on corruption, and enforces a JNI-level context-budget guard. 

We evaluate 4-bit Gemma 2 2B via llama.rn/llama.cpp across two physical Android tiers: mid-range (Snapdragon 4 Gen 2, UFS 2.2) and flagship (Snapdragon 8 Elite, UFS 4.0). Across 90 matched pairs and nine configurations (K in {1, 2, 4} chunks, context capacity in {512, 1024, 2048}), the mid-range device achieves prefill compute reductions of 19.34--21.21%, 38.53--40.37% and 86.06--86.67%, saving up to 47.33 s (3.65x wall speedup) with zero truncation and 100% deterministic output parity. The flagship shows identical scaling (16.35%, 36.73%, 85.23%; 1.08 percentage points from Device A at K=4) despite a 4.04x raw compute advantage. 

Balanced Two-Way ANOVAs show reductions are governed by cached tokens (p < 10^-15), not context capacity (p = 0.650--0.981) or interaction (p > 0.72). UFS 4.0 restores a 4-chunk session in 48.0--56.2 ms, saving 10.8--11.5 s (204x--229x). Multi-hop recall stays at 100%, and across five dynamic-corpus failure scenarios HACER-RAG eliminates the stale hallucinations and crashes where naive session reuse fails.

## Keywords
On-device machine learning, retrieval-augmented generation, KV-cache reuse, mobile systems, small language models, cache invalidation.

---

## Section I: Introduction
On-device RAG offers privacy, offline capability and predictable availability [Cite: Lewis et al., Wu et al.]: a local retriever (e.g., SQLite BM25 or a mobile vector index) selects evidence chunks, builds an augmented prompt and feeds it to an on-device SLM [Cite: Guu et al., Liu et al.]. Mobile hardware is tightly constrained [Cite: Alizadeh et al., Xu et al.]. Budget handsets pair heterogeneous SoCs (Snapdragon 4 Gen 2: 2x A78 + 6x A55) with 4--6 GB LPDDR4X [Cite: Reddi et al.], and even flagships (Snapdragon 8 Elite, 12 GB LPDDR5X) face thermal and battery envelopes. For 4-bit models [Cite: Dettmers et al., Frantar et al.] such as Gemma 2 2B Q4_K_M [Cite: Gemma Team], decode runs at 20--30 tok/s on budget and 60--80 tok/s on flagship cores (memory-bandwidth bound), while prefill of a ~400-token prompt takes 50--65 s on mid-range and 12--14 s on flagship CPUs.

Because users query the same corpus repeatedly, the retriever returns identical or overlapping chunks [Cite: Lu et al., Liu et al.], yet a conventional runtime re-encodes them every time, draining the battery and inducing throttling. Caching KV tensors is the natural fix. On Device A, prefilling a 395-token prompt (4 threads) takes 54.2 s (7.3 tok/s) whereas reading a 40 MB KV cache from UFS 2.2 takes under 350 ms; on Device B (UFS 4.0) prefill takes 13.1 s versus 48.0 ms to load. Prefill is thus more than 150x slower than flash restoration.

However, runtimes such as llama.rn and llama.cpp [Cite: Gerganov] expose saveSession/loadSession as file-level snapshots meant for conversational resumption: a cache is tied to a path or ID with no check of semantic validity. Applied blindly to RAG this exposes three failures:
1. **Silent stale-KV hallucination:** If a document is edited or revoked (a policy, medical record or price list), restoring the old session makes the model attend to obsolete KV tensors and emit outdated facts with high confidence [Cite: Ji et al.].
2. **Context headroom violations:** Runtimes allocate a fixed KV arena of 2 * L * H * D * n_ctx elements (L layers, H heads, head dimension D), so developers choose small n_ctx (512/1024). llama.cpp also reserves 4 internal tokens; if prefix, query and generation budget exceed the limit, leading context is silently dropped, corrupting the attention mask. Oversized arenas risk Android Low Memory Killer (LMK) termination.
3. **Continuation boundary misalignment:** Appending query text to a restored prefix without modeling tokenizer boundary merging causes index misalignment, garbled output or redundant re-evaluation.

On 4 GB phones, caches must be persisted to flash between interactions, which makes validity checking essential.

### Research Questions
- **RQ1 (scaling and context independence):** How does prefill reduction scale with cached prefix tokens on budget and flagship tiers, and does allocated n_ctx have a significant effect?
- **RQ2 (freshness):** Can lightweight cryptographic evidence versioning eliminate stale-KV hallucination under updates, revocations and reorderings without full scratch prefill?
- **RQ3 (multi-hop integrity):** Does partial prefix reuse preserve factual accuracy when answers need synthesis across several retrieved passages?
- **RQ4 (I/O trade-off):** What is the latency trade-off between reading KV caches from UFS 2.2/4.0 flash and native CPU prefill?

### Contributions
1. The HACER-RAG architecture: canonical layout, SHA-256 hashing, context-budget safety and selective repair.
2. A 90-pair scaling benchmark over nine configurations: 19.3--86.7% prefill reduction, up to 3.65x wall speedup.
3. Generalization to a flagship phone: 4.04x raw speedup, identical relative scaling.
4. Two-way ANOVA on both platforms: cached tokens, not context capacity, drive reduction (p < 10^-15).
5. Multi-chunk integrity: 44.13% mean reduction, 1.63x wall speedup, 100% multi-hop recall, failing incomplete-evidence ablations.
6. A five-scenario staleness suite where naive reuse hallucinates or crashes and HACER-RAG does not.

### Non-claims
We do not claim that saving and restoring KV caches is novel; our contribution is evidence-version-aware binding, selective repair, context-budget safety and mobile systems characterization. We claim no universal law across transformer architectures or SoCs: all results are bounded to the two phones, llama.rn v0.5.8/llama.cpp and Gemma 2 2B Instruct Q4_K_M. The five scenarios are proof-of-concept stress tests, not a proof of correctness for arbitrary document edits.

---

## Section II: Background and Related Work
**KV-cache fundamentals:** In a decoder-only transformer, token t attends only to positions tau <= t through a causal mask. The Key/Value tensors of any prefix of length P therefore do not depend on later tokens and can be reused without re-evaluation, provided model weights and positional encodings [Cite: Su et al.] are unchanged. llama.cpp [Cite: Gerganov] serializes the KV arena and the evaluated token sequence; restoring a session lets prefill of the continuation start at position P + 1.

**KV reuse in RAG:** TurboRAG [Cite: Lu et al.] precomputes document caches offline but needs positional transformations and optional fine-tuning. KVLink [Cite: Yang et al.] adds trainable linking tokens and modifies the model, which is incompatible with off-the-shelf quantized models. PerCache [Cite: Liu et al.] pre-loads user profiles but does not address cryptographic versioning, revocation or multi-chunk repair. Server engines (vLLM [Cite: Kwon et al.], SGLang) share prefixes in GPU memory and ignore cold flash I/O, thermal envelopes and Android memory eviction. KV compression such as KIVI [Cite: Liu et al.] and H2O [Cite: Zhang et al.] shrinks memory but neither removes prefill for new documents nor detects staleness.

**Differentiation:** HACER-RAG contributes:
(i) evidence-version-aware binding of each persistent KV segment to SHA-256 hashes of its chunks,
(ii) a selective repair policy that invalidates affected segments and re-prefills minimally while keeping valid prefix state,
(iii) JNI-level context-budget and boundary-token safety, and
(iv) a rigorous flash-versus-compute characterization across two device tiers.

---

## Section III: HACER-RAG Architecture
[Figure 1: Architecture Diagram - 1. Evidence Ingestion -> 2. Canonical Retrieval -> 3. Cache Decision Engine -> 4. Context-Budget Guard -> 5. Native Execution. Cases: Case 1 Exact Hit, Case 2 Selective Repair, Case 3 Scratch Fallback]

HACER-RAG has five subsystems:

**1. Evidence ingestion and versioning:** Documents are split into chunks C = {C_1, ..., C_M}, each stored with a digest H(C_i) = SHA-256(Canonicalize(C_i)), creation timestamp, document ID and token count in SQLite. Any edit, deletion or insertion changes H(C_i).

**2. Canonical evidence layout:** Retrievers rank chunks by score, so small fluctuations return permutations ([C_2, C_1] vs. [C_1, C_2]). Under causal attention a permutation changes every token and voids any saved prefix (0% reuse). HACER-RAG fixes the order deterministically, pi_canon = Sort({C_1, ..., C_k}) by persistent document and chunk IDs, so prefixes remain stable regardless of ranking noise.

**3. Adaptive cache decision engine:** For canonical chunks the engine forms prefix P = FormatPrompt(pi_canon(.)) and key K_session = SHA-256(P) and inspects local flash:
- *Case 1 (exact hit):* a valid session matches and all H(C_i) agree; only query Q is prefilled.
- *Case 2 (selective repair):* chunks C_1..C_j are unchanged, so the prefix KV is loaded and only C_{j+1}..C_k and Q are prefilled.
- *Case 3 (safe invalidation):* a revoked, missing, corrupt or malformed file invalidates the session, purges orphans and restarts from position 0 (Algorithm 1).

[Algorithm 1: HACER Cache Validation and Repair Policy]
- Input: chunks C = [C_1..C_k], query Q, registry R
- Step 1: CanonicalSort(C) -> P = FormatPrefix(C_c), K = SHA-256(P)
- Step 2: If K in R and VerifyHashes(C_c, R[K]) and LoadSession(R[K]) -> Case 1: PrefillOnline(Q); return
- Step 3: Find largest j* where prefix [1:j*] is in R. If j* > 0 and LoadSession(prefix j*) -> Case 2: PrefillOnline(remaining chunks + Q); SaveSession(K); return
- Step 4: Otherwise Case 3: Purge invalid sessions, reset context arena, PrefillOnline(P + Q); SaveSession(K).

**4. Native execution and telemetry:** The layer wraps llama.rn v0.5.8 and records kernel I/O counters from `/proc/{PID}/io` (separating flash reads from page-cache hits), Android PSS from `dumpsys meminfo` (LMK risk), and battery temperature and CPU frequencies from sysfs (thermal throttling).

**5. Blind reuse versus HACER:** Blind reuse (llama_load_session_file) treats `session.bin` as authoritative without checking the documents behind it, so reordered chunks give 0% reuse, changed or revoked documents yield silent stale answers, overflows truncate or trigger LMK kills, and boundary tokens merge uncontrolled. HACER instead sorts by persistent IDs, verifies SHA-256 chunk hashes, applies the JNI headroom test before inference, accounts for the +1 boundary token and separates flash reads from page-cache hits with `/proc/{PID}/io`.

---

## Section IV: Context-Budget Safety and Boundary Alignment
**Headroom guard:** Initializing n_ctx >= 4096 can reserve over 1.2 GB of contiguous KV memory and trigger LMKD termination on 4 GB devices. Before inference the JNI guard enforces:
[Equation 1]: `(n_ctx - Delta_res) - (N_prompt + N_predict) >= 0`
where Delta_res = 4 tokens are reserved by llama.cpp for sequence demarcation and end-of-turn markers. On violation the request is rejected at the React Native bridge instead of being silently truncated. With N_prompt = 395 and N_predict = 35 the headroom is 78, 590 and 1614 tokens for n_ctx = 512, 1024 and 2048. Across all 182 recorded executions the reported headroom matched this budget exactly, with zero engine truncations (truncated == 0).

**Boundary-token invariant:** Naive code assumes a session restored with N_loaded tokens leaves N_prompt - N_loaded tokens to evaluate. Tracing llama.rn and llama.cpp shows otherwise: because BPE merges leading whitespace or delimiters with preceding context, prefill restarts at index N_loaded - 1 so that the first new token sees full cross-boundary context. Hence for N_loaded > 0:
[Equation 2]: `N_eval = N_prompt - N_loaded + 1`
and N_eval = N_prompt = 395 for a cold start. Applying Eq. 2 gives N_eval = 312, 238 and 55 for K = 1, 2, 4 (where N_loaded = 84, 158, 341). Ignoring the +1 yields off-by-one prefill errors, corrupted positions or wrong compute attribution.

---

## Section V: Experimental Methodology
[Table I: Physical Testbeds]
- Device A (Mid-range): Snapdragon 4 Gen 2 (SM4450), 2x A78 + 6x A55, 4 GB LPDDR4X, 128 GB UFS 2.2 (~280 MB/s), Android 14.
- Device B (Flagship): Snapdragon 8 Elite (SM8750), 6x Oryon Perf + 2x Oryon Prime, 12 GB LPDDR5X (3.0x RAM), 256 GB UFS 4.0 (1,700 MB/s), Android 16.

**Software and model:** A React Native app (com.chat) calls llama.rn v0.5.8, which compiles llama.cpp with ARM NEON and OpenMP. The model is Gemma 2 2B Instruct in GGUF Q4_K_M (~1.708 GB; SHA-256 e0aee85060f168f0...). Decoding is greedy (temperature 0) with N_predict = 35 for strict determinism.

**Workload:** A synthetic enterprise corpus (Zephyr Dynamics) yields a canonical prompt of four evidence chunks plus a multi-hop question:
- Chunk 1: Company overview (founded 2011, 847 employees)
- Chunk 2: Leadership (CEO Priya Nandakumar, CLO Jorge Fonseca, CTO Marcus Lobell, CFO Saoirse Hennessey)
- Chunk 3: AeroSense X7 sensor (12,000 m altitude, 72 h battery)
- Chunk 4: SwarmKit Pro (2,048 drones, $1,200 standard and $3,800 full-capacity tiers)
The question asks the founding year, CLO, battery life and full-capacity price. The prompt tokenizes to exactly N_prompt = 395 tokens with SHA-256 digest a5f1db09b904e813...27c6d, identical in every execution.

**Protocol:** 
(i) Each baseline run is preceded by flushing the context arena with an orthogonal conversational prompt so no KV tokens linger.
(ii) Order within each pair (baseline-first or reuse-first) is randomized to avoid bias from heating or background tasks.
(iii) Per run we record native prefill time (prompt_ms) and decode time (predicted_ms) from llama.cpp timers, end-to-end wall time at the bridge, `/proc/{PID}/io` deltas, PSS, and the generated text for byte-level validation.

**Systems compared:**
(a) Full prefill: the whole prompt is evaluated in a fresh arena (N_loaded = 0).
(b) Ordinary blind session reuse: llama.rn session restore by fixed file name.
(c) HACER: canonical layout, SHA-256 verification, JNI budget enforcement, boundary alignment and selective repair.

**Metrics:** 
Native prefill latency T_prefill (s); prefill reduction Delta_prefill = (T_base - T_reuse) / T_base * 100%; wall speedup S_wall = T_base_wall / T_reuse_wall with 95% bootstrap CIs; flash cache-load overhead T_load (ms); storage read volume (classifying cold flash reads >= 85% from storage vs. warm page-cache hits >= 85% from buffer cache); multi-hop factual recall (%); and stale-fact hallucination rate (%) after document updates.

---

## Section VI: Scaling Results
**Audit of 182 records:** An independent audit of `scaling_trials.jsonl` found exactly 90 completed pairs (10 baseline + 10 reuse runs for each of C1--C9, 180 executions). Two incomplete executions were quarantined: C5_K2_CTX1024_rep1 (baseline interrupted by a carrier focus dialog) and C7_K4_CTX512_rep2 (baseline timed out from background preemption). The prompt digest was identical in all 182 records, truncated == 0 in 100% of runs, and all executions produced the byte-identical 35-token output: *"Zephyr Dynamics was founded in 2011, the CLO is Jorge Fonseca, the AeroSense X7 battery lasts 72 hours, and the full-capacity tier"*.

[Table II: Device A Scaling Matrix (90 matched pairs, Mean +- Std over 10 pairs)]
- C1 (K=1, CTX=512): Base 53.27s, Reuse 42.85s, Reduction 19.34% [15.50, 22.68], Saved 10.43s, Wall 1.19x
- C2 (K=1, CTX=1024): Base 56.03s, Reuse 44.39s, Reduction 20.48% [16.28, 23.73], Saved 11.64s, Wall 1.22x
- C3 (K=1, CTX=2048): Base 49.48s, Reuse 38.93s, Reduction 21.21% [19.10, 23.28], Saved 10.55s, Wall 1.21x
- C4 (K=2, CTX=512): Base 51.28s, Reuse 30.55s, Reduction 40.37% [38.66, 41.94], Saved 20.72s, Wall 1.51x
- C5 (K=2, CTX=1024): Base 50.73s, Reuse 30.84s, Reduction 39.16% [35.55, 42.59], Saved 19.89s, Wall 1.49x
- C6 (K=2, CTX=2048): Base 51.01s, Reuse 31.30s, Reduction 38.53% [35.52, 41.08], Saved 19.71s, Wall 1.48x
- C7 (K=4, CTX=512): Base 54.08s, Reuse 7.18s, Reduction 86.67% [86.18, 87.25], Saved 46.90s, Wall 3.60x
- C8 (K=4, CTX=1024): Base 54.89s, Reuse 7.56s, Reduction 86.20% [85.73, 86.69], Saved 47.33s, Wall 3.65x
- C9 (K=4, CTX=2048): Base 54.20s, Reuse 7.55s, Reduction 86.06% [85.25, 86.76], Saved 46.65s, Wall 3.27x

**Device A scaling:** Reduction scales monotonically with cached prefix tokens: about 19--21% at K=1 (84 tokens), 38--40% at K=2 (158) and 86--87% at K=4 (341), where prefill falls from about 54 s to 7.2--7.6 s.

[Figure 2: Prefill Reduction Chart - (a) Device A across nine configurations with 95% bootstrap CIs; (b) Mean reduction by cached-token count on both tiers]

**Cross-device scaling:** Device B delivers an average raw prefill speedup of 4.04x, cutting baseline prefill from ~54 s to 11.92--14.18 s. Despite the leap from Cortex-A78 to Oryon cores, the relative saving follows the same scaling with cached tokens: at K=4 Device A averages 86.31% and Device B 85.23% (1.08 percentage points apart); at K=1 and K=2, Device B averages 16.35% and 36.73% (Device A: 20.34% and 39.35%).

[Table III: Cross-Device Prefill and Reduction: Device A vs. Device B]
- Baseline compute speedup: 3.49x to 4.47x (mean 4.04x)
- Reuse prefill: Device A (7.18--44.39s) vs Device B (1.79--11.22s)
- Reductions: K=1 (~20% on A vs ~16% on B), K=2 (~39% on A vs ~37% on B), K=4 (~86% on A vs ~85% on B)
- Session load time on Device B: 35.6 ms to 56.2 ms

**Storage I/O (UFS 2.2 vs. 4.0):** Kernel I/O counters classify reuse runs into access tiers. On Device A, cold flash reads averaged 256.0 ms and warm page-cache hits 82--115 ms (120--280 MB/s). On Device B, sequential reads reached 1,700 MB/s, loading the 37.45 MB 4-chunk session in 48.0--56.2 ms. Loading 341 tokens thus costs ~50 ms while saving 10,863--11,460 ms of Oryon CPU time, a storage-to-compute speedup multiplier of 204x--229x. On Device A the load costs < 0.35 s against up to 47.3 s saved.

---

## Section VII: Multi-Chunk Correctness and Latency Isolation
Multi-hop questions [Cite: Yang et al.] need joint extraction from several passages, so a key concern is whether reusing precomputed leading chunks degrades recall. We built three two-chunk cases:
- Case 1: Remote-work policy ($800 one-time equipment stipend and $50/month internet allowance)
- Case 2: AeroSense X7 battery (72 h) and SwarmKit Pro capacity (2,048 drones)
- Case 3: CEO office (Stockholm) and sensor ceiling (12,000 m)
Each ran three matched baseline/reuse pairs (9 pairs, 18 executions) on Device A, recording prefill, decode, flash-load overhead, wall time and per-hop correctness.

[Table IV: Multi-Chunk Paired Trials on Device A (9 pairs)]
- Case 1 (187 base / 103 reuse tokens): Base 33.15s, Reuse 17.89s, Load 346ms, Reduction 44.30%, Wall 1.68x
- Case 2 (218 base / 119 reuse tokens): Base 45.41s, Reuse 21.04s, Load 226ms, Reduction 53.36%, Wall 1.86x
- Case 3 (156 base / 99 reuse tokens): Base 31.17s, Reuse 20.37s, Load 196ms, Reduction 34.74%, Wall 1.34x
- Overall Mean: Base 36.58s, Reuse 19.77s, Load 256ms, Reduction 44.13%, Wall 1.63x. Recall was 100% in all rows.

**Case 3 anomaly:** A preliminary single run showed baseline prefill of 19.14 s versus 20.58 s with reuse despite 58 cached tokens. Kernel I/O tracing found an asynchronous Android memory-compaction event (kswapd0) triggered by background carrier services, which reclaimed dirty pages and forced over 693 MB of flash reads during prefill, saturating the LPDDR4X bus. Under three monitored paired repetitions Case 3 behaved consistently: prefill fell from 31.17 +- 5.99 s to 20.37 +- 4.10 s (34.74% +- 2.30% reduction, about 10.8 s saved), flash load averaged 196.3 ms (< 1% of wall time) and wall speedup was 1.34x.

**Factual preservation and ablation:** All 9 pairs achieved 100% factual recall on both hops on both devices; outputs were, e.g., *"The one-time remote equipment stipend is $800 USD and the monthly internet allowance is $50 USD"* (Case 1), *"The AeroSense X7 battery lasts 72 hours, and SwarmKit Pro supports up to 2,048 drones"* (Case 2) and *"The CEO is based in the Stockholm office, and the AeroSense X7 can operate at altitudes up to 12,000 meters"* (Case 3). In an incomplete-evidence ablation providing only Chunk A, the model failed Hop 2 in 100% of runs, showing that answers depend on joint evidence rather than parametric memory.

---

## Section VIII: Dynamic Evidence and Staleness Evaluation
Personal knowledge bases change constantly. We ran five controlled failure scenarios on the physical testbed, comparing blind reuse with HACER-RAG:

[Table V: Dynamic Evidence Suite: Blind Session Reuse vs. HACER Adaptive Policy]
- **Scenario A (Content Update):** Equipment stipend updated $400 (2021) -> $800 (2024). 
  - Blind reuse: Outputs "$400 USD" (stale fact served).
  - HACER: Hash mismatch -> invalidate -> re-prefill updated chunk -> "$800 USD" (freshness restored).
- **Scenario B (Deletion / Revocation):** Childcare subsidy ($500/mo) revoked and deleted.
  - Blind reuse: Outputs "Answer: $500" (revoked fact hallucinated).
  - HACER: Document ID absent -> evict cache -> evaluate current corpus -> "None" (safely abstains).
- **Scenario C (Changed Ordering):** Retriever returned [C_2, C_1] instead of cached [C_1, C_2].
  - Blind reuse: Full prompt evaluated from scratch (0% reuse; 54s prefill).
  - HACER: Canonical sort -> exact cache hit -> correct answer (eliminates permutation misses).
- **Scenario D (Partial Update):** C_1 (leadership) unchanged; C_2 (pricing) $1,200 -> $1,600.
  - Blind reuse: Outputs "Priya, $1,200" (stale fact).
  - HACER: H(C_1) verified, C_1 KV kept, C_2 re-prefilled -> "$1,600" (prefix savings kept, downstream refreshed).
- **Scenario E (Corrupt File):** Cache binary truncated (0 bytes or bad header).
  - Blind reuse: Native segmentation fault, host app crash.
  - HACER: Size/header check before JNI -> purge -> scratch fallback -> "Stockholm" (guarantees crash resilience).

---

## Section IX: Statistical Analysis of Context Allocation
To test whether context capacity matters, we fit balanced 3x3 two-way ANOVAs with Factor A the cached prefix size K (84/158/341 tokens) and Factor B the context capacity: 10 pairs per cell on Device A (N = 90) and 5 per cell on Device B (N = 45).

[Table VI: Two-Way ANOVA on Prefill Reduction]
- Factor A (cached tokens K): Device A (F = 1,920.56, p = 5.70 x 10^-69, eta_p^2 = 0.9793); Device B (F = 107.67, p = 6.66 x 10^-16, eta_p^2 = 0.8568).
- Factor B (context n_ctx): Device A (F = 0.019, p = 0.9808 not significant, eta_p^2 = 0.0005); Device B (F = 0.44, p = 0.650 not significant, eta_p^2 = 0.0236).
- Interaction K x n_ctx: Device A (F = 0.510, p = 0.7287, not significant); Device B (F = 0.33, p = 0.857, not significant).

Three conclusions follow:
1. *Cached prefix dominates:* on Device A, K explains 97.88% of total variance (SS_A / SS_total = 69,178.69 / 70,674.93).
2. *No context-capacity effect:* p = 0.9808 on A (an alternative audit reports F = 0.81, p = 0.449) and p = 0.650 on B, so a larger arena does not perturb prefix prefill savings.
3. *No interaction,* so reuse scaling does not depend on buffer sizing.

A multiple regression `Delta_prefill = beta_0 + beta_tokens * N_cached + beta_ctx * n_ctx + epsilon` on the 90 Device A pairs gives:
- beta_0 = -1.080 +- 1.216 (p = 0.377)
- beta_tokens = +0.2567 +- 0.0040 (t = 63.43, p = 1.26 x 10^-74, 95% CI [0.2487, 0.2646])
- beta_ctx = -0.000109 +- 0.00068 (p = 0.874, CI [-0.00145, +0.00123] spanning zero)
- R^2 = 0.9788 (adj. 0.9783, F(2, 87) = 2,010, p < 10^-73).
Each cached token adds about 0.2567 percentage points, matching the theoretical 100% / 395 approx 0.2532.

**Reframed empirical claim:** Within the tested envelope (Snapdragon 4 Gen 2 and 8 Elite, Gemma 2 2B, n_ctx in {512, 1024, 2048}), prefill reduction tracks cached prefix tokens (beta = +0.2567, R^2 = 0.9788) and not allocated context capacity.

---

## Section X: Discussion and Limitations
**Storage--compute asymmetry:** Dense CPU prefill reaches only 7.29 tok/s on Snapdragon 4 Gen 2 (54.2 s for 395 tokens) and 28.4 tok/s on Snapdragon 8 Elite (11.9--14.2 s), whereas a 37--40 MB KV binary loads in 256.0 ms (UFS 2.2) or 48.0--56.2 ms (UFS 4.0). The asymmetry exceeds 150:1 on Device A and 204:1--229:1 on Device B. Unlike servers with H100-class GPUs prefilling thousands of tokens per second, mobile platforms invert the trade-off, and flash becomes a fast, energy-efficient cache for expensive computation.

**Memory pressure:** On the 4 GB device the OS and services use 1.8--2.2 GB, and com.chat reached PSS of about 2.28 GB (2,280,940 KB; Device A) and 1,986--2,176 MB (Device B) with the 1.7 GB weights and arena. Raising n_ctx to 4096 would trigger an LMK SIGKILL, so JNI budgeting bounds operation to n_ctx <= 2048 (n_ctx in {512, 1024, 2048}); KV state lives on flash and the arena is released between interactions.

**Thermal and energy:** In continuous Device A runs battery temperature rose from 29.5 deg C to 38.2 deg C and DVFS throttled clocks to 1.6 GHz. Eliminating up to 86.67% of prefill (46.7--47.3 s on A, 10.8--11.5 s on B) shortens high-power CPU residency, conserving battery and keeping decode responsive.

**Implications:**
1. Evidence binding is mandatory when local documents change; hashing adds only microseconds.
2. Canonical layout removes the permutation penalty.
3. Selective repair keeps most savings while guaranteeing freshness.
4. Flagship hardware still needs prefill optimization: cutting 13 s to 1.8 s separates unacceptable lag from fluid conversation.

**Threats to internal validity:** Commercial Android hosts run non-deterministic background daemons, as the Case 3 anomaly showed. We mitigated this with matched pairs, randomized order, context flushing, kernel I/O logging, bootstrap CIs, and interleaved trials with `dumpsys battery` logging against junction-temperature throttling. The two quarantined trials were logged with full telemetry before analysis.

**Threats to external validity:** Only two phones (mid-range and flagship) were tested; SoCs with NPUs (Hexagon, MediaTek APU) or eMMC 5.1 storage, and other memory or thermal profiles, may differ. Only Gemma 2 2B Q4_K_M (interleaved local/global attention, GQA, RoPE) was evaluated; sliding-window-only or state-space models (e.g., Mamba) may have different prefix-caching trade-offs. The five dynamic scenarios are targeted stress tests, not an exhaustive survey of document mutation patterns.

---

## Section XI: Reproducibility and Claim-to-Evidence Matrix
Table VII maps each major claim to its dataset and artifacts (`scaling_trials.jsonl`, `device_b_scaling_summary.json`, `anova_context_invariance.json`, `rigorous_multichunk_and_staleness_results.json`, `scaling_trials.summary.json`). All harnesses, analysis scripts, raw JSON/JSONL logs and figure routines are in the project repository: scaling controllers `benchmark/run_hacer_scaling_matrix.py` (mid-range) and `run_hacer_device_b_scaling.py` (flagship), `run_hacer_rigorous_evaluation.py` (multi-chunk and staleness), `audit_scaling_records.py` (independent audit), `analyze_context_invariance.py` (ANOVA and regression), `paper/scripts/generate_figures.py` (figures), and the synthesis report `benchmark/results/CROSS_DEVICE_EVALUATION_REPORT.md`.

[Table VII: Claim-to-Evidence Verification Matrix]
- Prefix compute reduction 19.3%--86.7%: 90 pairs (Dev. A), 45 pairs (Dev. B) -> A: 19.34--86.67%; B: 9.32--86.20%; 1.08 pp gap at K=4; up to 3.65x wall.
- Context-capacity independence: Two-way ANOVA, both devices -> A: p = 0.449--0.981; B: p = 0.650, eta_p^2 = 0.0236; beta_ctx = -0.000109, p = 0.874.
- Multi-hop recall preserved (100%): 3 cases x 3 reps -> 9/9 pairs 100% on both hops; 44.13% reduction; 1.63x; ablations fail 100%.
- Case 3 anomaly resolved: Repeated pairs with I/O profiling -> Compaction noise isolated; 34.74% +- 2.30%, 10.8 s saved, 196.3 ms flash.
- Stale-KV hallucination eliminated: Five dynamic scenarios -> Naive reuse stale in 3/3 mutation cases; HACER 100% fresh, abstains, recovers.
- Flash vs. compute asymmetry: `/proc/{PID}/io` kernel counters -> 256.0 ms (UFS 2.2) vs. 48.0 ms (UFS 4.0); 47.3 s / 11.5 s saved; 150x--229x.
- Context safety, zero truncation: Full logs, both devices -> Headroom >= 78 tokens; truncated == 0 in 100%; byte-identical output.

---

## Section XII: Conclusion
On-device RAG is bottlenecked by prompt prefill on low-power CPUs, and existing mobile runtimes expose KV reuse only as opaque session files that invite stale hallucination, context overflow and boundary errors. HACER-RAG binds persistent KV tensors to document versions through a canonical layout and SHA-256 chunk hashing, repairs only changed downstream chunks, and enforces a JNI-level budget guard. On two Android tiers it reduced prefill by up to 86.67% and 86.20%, with 100% output parity and zero truncation. Cached tokens, not context capacity, govern savings (p < 10^-15 vs. p = 0.650--0.981); multi-hop recall stays at 100%; and all five dynamic scenarios are handled where naive reuse fails. Evidence-version-aware KV reuse is thus a practical foundation for fast, private and fresh on-device conversational AI.
