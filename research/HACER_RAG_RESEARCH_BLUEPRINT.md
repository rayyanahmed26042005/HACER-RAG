# HACER-RAG: A Quality-Constrained Hardware-Adaptive Runtime for KV-Cache Reuse and Boundary-Aware Recomputation in Mobile RAG

**Status:** Research blueprint; no results claimed
**Target venues:** MobiSys, ASPLOS, MLSys, EMNLP
**Scope:** Heterogeneous Android/iOS-class mobile devices running off-the-shelf, quantized decoder-only language models
**Date:** 2026-10-17
**MobCert status:** Frozen comparator/protocol; no MobCert implementation changes are part of this paper

> **Evidence boundary.** Every quantitative result in this document is an experimental target, acceptance criterion, or `[X.X]` placeholder. The current Redmi 12 5G measurements in this repository are engineering artifacts and are not evidence for the claims proposed here. The paper must not report a speedup, accuracy preservation, energy reduction, or generalization result until the protocol below produces it on the specified devices and workloads.

## 1. Executive summary

This paper proposes **HACER-RAG** (**H**ardware-**A**daptive **C**ache, **E**rror-bounded **R**ecomputation for RAG), a mobile inference runtime that chooses, for each query and retrieved chunk, how much KV state to reuse, how precisely to store it, how much Flash data to load, and which cross-chunk dependencies to recompute.

The central claim is deliberately narrower than “precomputed KV is always valid”:

> A mobile RAG runtime can improve the latency–memory–energy Pareto frontier by combining (i) explicit cache-validity modes, (ii) split-precision anchor/bulk storage, (iii) boundary-aware selective recomputation, and (iv) a runtime scheduler calibrated to measured device state, while exposing the quality cost of approximate reuse instead of hiding it.

The paper treats independently encoded chunk KV caches as **approximate proposals**, not exact representations of the final concatenated sequence. Exact prefix reuse, approximate chunk reuse, and full recomputation are distinct runtime modes. The scheduler may reject reuse when the validity key, position transform, attention-mask semantics, or quality guard is not satisfied.

The proposed novelty is therefore the **joint, quality-constrained policy** rather than any single component in isolation:

1. A split-precision cache whose high-precision anchors serve both as robust evidence and as a low-cost routing index.
2. Two portable cross-chunk repair designs that make the residual-stream boundary explicit rather than assuming that token sparsity alone makes a standard decoder sparse.
3. A hardware- and runtime-parameterized controller that selects bit width, Flash pages, recomputation design, and recomputation ratio under latency, memory, energy, and thermal constraints.
4. A falsifiable evaluation of the quality–cost frontier from `0%` reuse repair to full prefill across cold/warm Flash state, device tiers, thermal states, and multi-hop reasoning tasks.

## 2. Abstract

Retrieval-augmented generation (RAG) on mobile devices repeatedly re-encodes retrieved document chunks, making prompt prefill a dominant cost when compute is limited, while persistent KV storage introduces Flash bandwidth, memory pressure, quantization error, and cross-chunk correctness problems. Existing chunk-level KV reuse methods reduce online computation but either rely on model changes, treat independently encoded caches as directly composable, or use fixed recomputation policies that are not calibrated to mobile hardware. We present HACER-RAG, a portable runtime design for quality-constrained KV reuse across heterogeneous mobile devices. HACER-RAG stores each chunk as a split-precision cache: a high-precision anchor subspace supports query-relevance routing and selective repair, while a low-bit bulk payload reduces Flash traffic and memory footprint. To address the residual-stream boundary problem, we compare boundary activation caching, which bypasses shallow-layer recomputation for selected anchors, with all-layer ultra-sparse recomputation, which avoids intermediate activation storage. A hardware-parameterized scheduler selects cache precision, Flash pages, and recomputation ratio using measured Flash bandwidth, dequantization rate, prefill throughput, available memory, and thermal state. We define explicit cache validity keys and evaluate approximate reuse as a quality-cost tradeoff rather than exact equivalence. Across budget, mid-range, and flagship mobile profiles, and on single-hop and multi-hop RAG benchmarks, the study will measure whether HACER-RAG improves the latency–energy–memory Pareto frontier without exceeding a preregistered quality-loss bound. The final paper will replace all outcome placeholders with paired repeated-run measurements, confidence intervals, and held-out-device results: `[X.X]%` change in p95 TTFT, `[X.X]%` change in energy/query, `[X.X]%` peak-PSS change, and `[X.X]` quality loss at the selected operating point.

## 3. Contributions and non-claims

### 3.1 Intended contributions

The completed paper should make the following contributions if, and only if, the experiments support them:

1. **A validity-aware formulation.** We formalize when a cached KV block is exact, when it is an approximate proposal, and what metadata is required to prevent unsafe reuse across prompt templates, positions, model revisions, and attention masks.
2. **A split-precision anchor/bulk cache.** We define a query-independent anchor score that combines attention persistence, activation salience, boundary sensitivity, and quantization vulnerability. Anchors are retained at INT8/FP16 and exposed as an in-RAM routing index; bulk tokens use a selectable 2-bit/4-bit representation.
3. **Boundary-aware selective recomputation.** We implement and compare two runtime designs: boundary activation caching and all-layer ultra-sparse recomputation. The paper explicitly accounts for residual-stream dependencies and does not assume that skipping a token at one layer is free at the next layer.
4. **A hardware-adaptive policy.** We present a device-agnostic cost model and a conservative scheduler that chooses `(bit width, cache pages, recomputation design, recomputation ratio, eviction action)` from measured hardware and runtime state.
5. **A reproducible evaluation protocol.** We provide cold/warm Flash controls, sustained thermal blocks, memory-pressure conditions, exact model/index hashes, paired query order, statistical tests, and an accuracy–latency–energy Pareto analysis across heterogeneous devices.

### 3.2 Explicit non-claims

The blueprint does **not** claim that:

- arbitrary independently encoded KV caches are exact under concatenation;
- the method is model-agnostic before testing multiple architectures and positional schemes;
- a fixed `20%` recomputation policy is optimal;
- anchors selected by Key-vector norm are semantically sufficient;
- energy or thermal gains exist before direct measurement;
- a single phone represents a heterogeneous mobile population;
- current MobCert results establish the proposed cache method;
- any `[X.X]` value below is measured.

## 4. Research questions and falsifiable hypotheses

### RQ1 — Can anchors reduce cache traffic without hiding semantic loss?

**H1 (split precision versus uniform quantization).** At a fixed cache-storage budget and a fixed answer-quality tolerance `ε_Q`, a split-precision cache with an anchor routing index will reduce online bulk bytes loaded and/or dequantized compared with uniform KIVI-style 2-bit/4-bit storage, without increasing the tail quality loss beyond `ε_Q`.

**Falsification rule:** H1 is rejected if, on the preregistered primary workload, split precision does not reduce measured Flash/dequantization work at the same quality constraint, or if the reduction disappears after charging anchor lookup, metadata, page alignment, and false-negative routing costs.

### RQ2 — Which repair design handles the residual-stream boundary?

**H2 (selective cross-chunk repair).** Given a fixed quality-loss bound, a boundary-aware selective recomputation policy can achieve lower TTFT and energy than full prefill while outperforming naive zero-repair chunk concatenation. Design A and Design B will be compared at equal measured recomputed-token fraction, equal quality tolerance, and equal cache budget.

**Falsification rule:** H2 is rejected if neither design produces a statistically significant Pareto improvement over full prefill and naive reuse, or if the apparent gain comes only from omitting failed/abstained queries.

### RQ3 — Does hardware adaptation matter beyond a fixed heuristic?

**H3 (hardware-parameterized scheduling).** A scheduler using measured Flash bandwidth, SIMD dequantization throughput, prefill throughput, free memory, battery state, and thermal state will reduce p95 SLO violations or energy/query relative to the best fixed policy on held-out device/state combinations.

**Falsification rule:** H3 is rejected if a fixed policy selected using the training devices matches or exceeds the adaptive scheduler on held-out devices and thermal/memory states, after including scheduler overhead and calibration cost.

### RQ4 — How much multi-hop quality is recovered as repair increases?

**H4 (accuracy–cost frontier).** Multi-hop QA quality will vary measurably as the repair ratio increases from `r=0%` to `r=5%`, `r=20%`, and `r=100%` full prefill. The paper will quantify the frontier rather than assume that a 20% point is sufficient.

**Falsification rule:** H4 is rejected if the quality curve is statistically indistinguishable across repair ratios, or if selective repair cannot recover quality at lower measured cost than full prefill.

## 5. Prior work and precise differentiation

### 5.1 Chunk KV reuse

TurboRAG precomputes document KV caches and retrieves them for online use, including analysis of masks and positional embeddings and a fine-tuning path for preserving quality ([Lu et al., 2024](https://arxiv.org/abs/2410.07590)). KVLink independently precomputes document KV caches, applies position adjustment, and introduces trainable special tokens to restore cross-document interaction ([Yang et al., 2025](https://arxiv.org/abs/2502.16002)). HACER-RAG differs in three ways: it targets off-the-shelf quantized mobile SLMs, treats independently encoded caches as approximate unless validity conditions hold, and makes the repair/precision decision a hardware-state-constrained runtime action rather than a fixed deployment assumption.

### 5.2 Positional handling and deferred rotation

LazyAttention is included as a positional-encoding baseline in the planned implementation. Its deferred RoPE handling can remove position-dependent tensor movement, but positional alignment alone does not recreate semantic interactions that were absent when chunks were encoded independently. The exact publication metadata and implementation revision must be frozen in the final bibliography before submission; the placeholder label “LazyAttention” must not be used as a substitute for a verified citation.

### 5.3 Selective recomputation

ProphetKV is treated as the fixed selective-recomputation baseline requested by the study design. HACER-RAG differs by making the repair set and ratio hardware/state dependent, by explicitly comparing boundary activation caching with all-layer sparse repair, and by measuring the cost of the residual-stream boundary rather than assuming that a static approximately-20% full-layer policy transfers to mobile CPUs. The final manuscript must verify the exact ProphetKV bibliographic record and implementation conditions before quoting any result.

### 5.4 Mobile hierarchical caching

PerCache studies predictive hierarchical caching for personalized mobile RAG and adapts configuration to dynamic system loads ([Liu et al., 2025/2026](https://arxiv.org/abs/2601.11553)). HACER-RAG should not claim that PerCache is non-adaptive. Its differentiator is narrower: it adapts **KV representation precision, Flash-page loading, and cross-chunk repair** using an explicit validity/error model, and it evaluates cold-start personal queries where future-query prediction is unavailable.

### 5.5 KV quantization

KIVI studies tuning-free asymmetric 2-bit KV quantization and reports different quantization granularities for Keys and Values ([Liu et al., 2024](https://arxiv.org/abs/2402.02750)). HACER-RAG uses KIVI-style uniform quantization as a baseline and asks a different question: whether a small high-precision subspace can serve both as a routing index and as a correctness-sensitive repair set when quantization and independently encoded chunk boundaries interact.

### 5.6 Position in the literature

The defensible novelty statement is:

> Prior work supplies important pieces—chunk KV reuse, position handling, selective recomputation, mobile hierarchical caching, and KV quantization. HACER-RAG proposes and tests their joint decision problem under an explicit mobile hardware state and a measurable quality constraint.

This is a hypothesis about a combined system contribution, not a claim of priority over every individual component.

## 6. Formal problem statement

Let a decoder-only model have `L` transformer layers, `H` KV heads, head width `d`, and a tokenizer `τ`. A RAG request is

\[
q=(x_{\mathrm{sys}},x_{\mathrm{usr}},\mathcal{D}_q,\pi),
\]

where `x_sys` is the fixed system/template prefix, `x_usr` is the user query, `\mathcal{D}_q=(D_1,\ldots,D_m)` is the ordered retrieved chunk sequence, and `π` is the prompt/attention-mask policy. Let `N=\sum_i n_i` be the number of retrieved document tokens and `P` the query/prefix token count.

For every chunk `D_i`, an offline encoder produces a local cache proposal

\[
\widetilde{C}_i^\ell=(\widetilde{K}_i^\ell,\widetilde{V}_i^\ell),
\quad \ell\in\{0,\ldots,L-1\},
\]

under a declared local context, relative position origin, attention mask, model hash, tokenizer hash, and quantization configuration. The exact full-prefill cache for the assembled request is

\[
C_{\mathrm{full}}^\ell(q)=\operatorname{KV}_\ell(x_{\mathrm{sys}}\oplus x_{\mathrm{usr}}\oplus D_1\oplus\cdots\oplus D_m;\pi).
\]

In general, `\widetilde{C}_i^\ell` is not equal to the corresponding slice of `C_full^\ell(q)` because hidden states depend on preceding tokens and attention interactions. HACER-RAG therefore selects an action

\[
a_q=\big(b_q,\,\mathcal{P}_q,\,d_q,\,r_q,\,L_{\mathrm{local},q},\,e_q\big),
\]

where:

- `b_q\in\{2,4,8,16\}` is the bulk precision;
- `\mathcal{P}_q` is the set of Flash pages loaded or dequantized;
- `d_q\in\{\mathrm{exact},\mathrm{A},\mathrm{B},\mathrm{full}\}` is the reuse/repair design;
- `r_q` is the selected recomputed-token ratio;
- `L_local,q` is the shallow-layer boundary for Design A;
- `e_q` is an eviction or fallback action.

The runtime optimizes a constrained objective, not a single latency number:

\[
\begin{aligned}
\min_{a_q}\quad &
\widehat{T}(q,a_q)+\lambda_E\widehat{E}(q,a_q)+\lambda_M\widehat{M}(q,a_q)+\lambda_S\widehat{S}(q,a_q)\\
\text{s.t.}\quad &
\widehat{\Delta Q}(q,a_q)\le \epsilon_Q,\\
&\widehat{M}_{\mathrm{peak}}(q,a_q)\le M_{\mathrm{free}}-M_{\mathrm{guard}},\\
&\widehat{T}_{\mathrm{p95}}(q,a_q)\le T_{\mathrm{SLO}},\\
&T_{\mathrm{thermal}}(q,a_q)\le T_{\mathrm{max}}.
\end{aligned}
\]

Here `\Delta Q` is a measured or calibrated quality-loss proxy, `S` is an SLO violation indicator, and `M_guard` is a safety reserve for the operating system and the application. If no action satisfies the constraints, the runtime falls back to full prefill or refuses cache reuse; it must not silently violate the declared quality bound.

## 7. H1: split-precision anchor/bulk cache

### 7.1 Cache partition

For each layer `\ell` and chunk token `t`, define a token record

\[
z_{i,t}^\ell=[k_{i,t}^\ell;v_{i,t}^\ell],\qquad t\in[1,n_i].
\]

The chunk is partitioned into an anchor set `A_i^\ell` and bulk set `B_i^\ell`, with `A_i^\ell\cap B_i^\ell=\varnothing` and `A_i^\ell\cup B_i^\ell=[1,n_i]`. The persisted representation is

\[
\mathcal{S}_i^\ell=
\underbrace{\{z_{i,t}^\ell\text{ at }p_A\}_{t\in A_i^\ell}}_{\text{anchor subspace}}
\;\cup\;
\underbrace{\{Q_{b}(z_{i,t}^\ell),s_{i,t}^\ell\}_{t\in B_i^\ell}}_{\text{bulk payload}},
\]

where `p_A\in\{8,16\}` is the anchor precision, `b\in\{2,4\}` is the bulk precision, and `s` contains per-group scales/zero points and checksums. KIVI-compatible per-channel Key and per-token Value grouping is one baseline; a uniform group-wise quantizer is another baseline.

### 7.2 Anchor score

Anchors must not be selected by Key norm alone. The offline anchor score for token `t` is

\[
\begin{aligned}
s_{i,t}={}&w_0+w_1\,\operatorname{Sink}_{i,t}+w_2\,\log(1+\|k_{i,t}\|_2)\\
&+w_3\,\log(1+\|v_{i,t}\|_2)+w_4\,\operatorname{AttnMass}_{i,t}
 +w_5\,\operatorname{Boundary}_{i,t}\\
&+w_6\,\operatorname{QuantRisk}_{i,t}(b)
 +w_7\,\operatorname{LexSalience}_{i,t}
 +w_8\,\operatorname{EntitySalience}_{i,t}.
\end{aligned}
\]

The features are defined without using the evaluation answer:

- `Sink` marks persistent attention-sink positions;
- `AttnMass` is cumulative intra-chunk attention over a calibration corpus;
- `Boundary` increases near chunk starts, ends, headings, table headers, and entity transitions;
- `QuantRisk` is a held-out calibration estimate of the logit/KV change caused by bulk quantization;
- `LexSalience` and `EntitySalience` use document-local statistics, not the future query.

The primary anchor selector is top-`k_A` by `s`. Ablations compare: (a) sinks plus Key/Value norm outliers, (b) cumulative intra-chunk attention, (c) the full score above, and (d) uniform random/uniform quantization. A query-conditioned variant may re-rank only the persisted anchors using a compact query embedding; it may not inspect the bulk payload before the routing decision.

### 7.3 Anchor routing before bulk load

The anchor index stores a compact per-chunk prototype `g_i` and anchor vectors. For a query representation `u_q`, the router computes

\[
\rho_i(q)=\alpha\,\operatorname{cos}(u_q,g_i)+
\frac{1}{|A_i|}\sum_{t\in A_i}\operatorname{TopKSim}(u_q,k_{i,t})+
\beta\,\operatorname{LexOverlap}(q,D_i).
\]

Only chunks with `\rho_i(q)\ge\theta_\rho` or in the top `K_route` are eligible for bulk page loads. The router reports false negatives against full retrieval and reports the bytes saved after page rounding. The routing index is not allowed to change the fixed retrieval ground truth; it is evaluated as a cache-loading optimization after retrieval.

### 7.4 H1 measurements

For each quantization/anchor policy, report:

1. bytes stored per token and per chunk;
2. anchor-index RAM;
3. Flash pages touched, read bytes, and read amplification;
4. dequantized tokens and dequantization time;
5. router latency and false-negative rate;
6. next-token KL divergence and logit agreement against full prefill;
7. answer EM/F1, citation/evidence support, hallucination, and multi-hop exactness;
8. quality at matched cache bytes and matched quality-loss bound.

## 8. H2: selective cross-chunk recomputation

### 8.1 RoPE phase alignment

Let `p^local_{i,t}` be the position used during chunk-local encoding and `p^global_{i,t}` the position after assembly. For standard RoPE, define

\[
\Delta p_{i,t}=p^global_{i,t}-p^local_{i,t},\qquad
K'_{i,t}^\ell=R_\ell(\Delta p_{i,t})\widetilde{K}_{i,t}^\ell.
\]

Values are not RoPE-rotated. The runtime must use the model's actual RoPE base, scaling, per-head frequency layout, and position-interpolation rule. A cache record whose RoPE configuration does not match the active model is invalid. For ALiBi, learned absolute embeddings, sliding-window masks, or other position mechanisms, the phase transform is replaced by an architecture-specific validity adapter; the method must fall back to full recomputation if no adapter is implemented.

RoPE alignment removes a position-origin mismatch; it does **not** restore cross-chunk semantic attention. That residual error is the target of selective recomputation.

### 8.2 Boundary and influence scores

Let `S_q` be the set of tokens recomputed online. We score a candidate token with

\[
u_{i,t}(q)=
\eta_1\rho_i(q)+\eta_2\operatorname{Boundary}_{i,t}
 +\eta_3\operatorname{Influence}_{i,t}(q)
 +\eta_4\operatorname{Uncertainty}_{i,t}(b)
 +\eta_5\operatorname{CrossHop}_{i,t}(q).
\]

`Influence` may be estimated from a small probe batch, attention rollout, or gradient-free logit sensitivity. `Uncertainty` is calibrated against the difference between approximate and full logits on held-out queries. `CrossHop` marks entities, relation-bearing spans, table headers, and bridge terms that connect multiple retrieved chunks. The selection is constrained by

\[
|S_q|\le r_q N,\qquad r_q\in\{0,0.01,0.02,0.05,0.10,0.20,1.0\}.
\]

The exact set and all selection overhead are logged.

### 8.3 Design A — boundary activation caching

Choose a shallow boundary `L_local<L`. Offline, persist quantized intermediate activations for anchor tokens:

\[
A^{L_{local}-1}_{i,t}=h^{L_{local}-1}_{i,t},\qquad t\in A_i.
\]

At query time:

1. load and dequantize the selected anchor activations;
2. apply the global position/mask adapter where valid;
3. recompute the selected boundary and bridge tokens through layers `L_local,\ldots,L-1`;
4. use cached proposal K/V for non-selected tokens, subject to the active attention mask;
5. run the query/prefix prefill and decode.

Design A trades additional Flash bytes and activation quantization error for bypassed shallow-layer work. Its boundary activation format records the layer, normalization state, tensor layout, quantizer, and model hash. If a selected token depends on an unavailable preceding activation, the runtime expands the recomputation frontier or falls back to Design B/full prefill.

### 8.4 Design B — all-layer ultra-sparse recomputation

Design B stores no intermediate hidden states. It recomputes only `S_q` through all layers `0,\ldots,L-1` and uses cached K/V proposals for the remaining tokens. The implementation must make the dependency approximation explicit: a recomputed token may attend to cached keys/values, but a cached token is not silently treated as if it had incorporated the recomputed token unless the attention schedule actually does so.

This design is expected to use `r_q\le5%` for mobile CPU profiles, but that is a search range, not a claimed optimum. Its purpose is to test whether a smaller all-layer repair can beat a larger shallow-layer bypass once activation I/O is charged.

### 8.5 Residual-stream boundary invariant

The runtime must expose a per-layer dependency trace:

\[
\mathcal{G}_\ell(S_q)=\{(t,j):t\in S_q,\;j\le t,\;\text{attention edge }t\leftarrow j\text{ is evaluated at layer }\ell\}.
\]

An implementation is not considered a valid selective-recompute result unless it records:

- which tokens were recomputed at each layer;
- which cached K/V blocks were read by those tokens;
- whether a boundary activation or token hidden state was loaded;
- whether the attention mask was exact, relaxed, or approximate;
- the fallback path when dependency requirements were unmet.

### 8.6 H2 measurements

The primary comparison is a frontier with `r=0%` (RoPE-aligned reuse), `r=1%`, `r=2%`, `r=5%`, `r=10%`, `r=20%`, and `r=100%` full prefill. For each point, report quality, TTFT, p95 TTFT, energy, peak PSS, Flash bytes, recomputed FLOPs/tokens, and cache-management overhead. Results must be shown separately for single-hop and multi-hop queries.

## 9. H3: hardware-parameterized cost model and scheduler

### 9.1 Device state

The scheduler consumes a measured state vector rather than a device-name lookup:

\[
s_d=\big(B_{flash},R_{dequant},C_{prefill}^{(\ell)},
M_{free},P_{idle},P_{load},T_{batt},T_{thermal},
E_{battery},\chi_{cache},\chi_{os}\big).
\]

The terms are:

- `B_flash`: sustained sequential and random Flash read bandwidth at the relevant page sizes;
- `R_dequant`: SIMD dequantization throughput by bit width and tensor layout;
- `C_prefill^(\ell)`: calibrated prefill throughput by layer group and batch shape;
- `M_free`: memory available to the runtime after a measured OS/application reserve;
- `P_idle,P_load`: measured or calibrated power states;
- `T_batt,T_thermal`: battery temperature and Android/iOS thermal signal;
- `E_battery`: energy state or battery-current estimate;
- `χ_cache`: page-cache warmth and cache residency;
- `χ_os`: memory pressure/background-load state.

The calibration procedure must measure each parameter on every device/profile with confidence intervals. The scheduler may use a small online correction, but it must not use future answer quality or post-hoc timing.

### 9.2 Latency model

For action `a=(b,\mathcal{P},d,r,L_local,e)`, the predicted TTFT is

\[
\begin{aligned}
\widehat{T}_{TTFT}(q,a|s_d)={}&T_{embed}+T_{retrieve}+T_{route}\\
&+\underbrace{\frac{\operatorname{Bytes}_{IO}(b,\mathcal{P})}{B_{flash}(\chi_{cache})}}_{T_{Flash\_IO}}\\
&+\underbrace{\frac{\operatorname{Bytes}_{dequant}(b,\mathcal{P})}{R_{dequant}(b)}}_{T_{dequant}}\\
&+T_{RoPE}(\mathcal{P})+T_{activation\_load}(d)\\
&+\underbrace{\sum_{\ell\in\mathcal{L}(d)}
\frac{F_\ell(q,r,L_local)}{C_{prefill}^{(\ell)}(s_d)}}_{T_{recompute}}\\
&+T_{query\_prefill}(q,a)+T_{first\_decode}(q,a)\\
&+T_{scheduler}+T_{fallback}.
\end{aligned}
\]

`Bytes_IO` includes page rounding, headers, checksums, mmap faults, and `pread` completion. `T_activation_load` is zero for Design B but not for Design A. `T_fallback` is nonzero only when a guard rejects the proposed action. The model is deliberately additive at first; interaction residuals are fitted from calibration data and reported separately rather than silently folded into a “speedup” coefficient.

### 9.3 Memory and energy model

Peak memory is modeled as

\[
\widehat{M}_{peak}=M_{weights}+M_{query\_KV}+M_{anchors}^{RAM}+
M_{pages}^{resident}+M_{workspace}(d,r)+M_{OS\_guard}.
\]

Energy is estimated from measured power samples:

\[
\widehat{E}(q,a)=\int_0^{\widehat{T}_{TTFT}+T_{decode}}P(s_d(t),a)\,dt.
\]

If only battery charge is available, the paper must report the measurement resolution and uncertainty and must not present it as a precise per-query joule value. Preferred instrumentation is an external power monitor or a device-supported current/voltage trace with calibration.

### 9.4 Quality-loss model

The scheduler uses a conservative upper confidence estimate:

\[
\widehat{\Delta Q}(q,a)=
\gamma_0+\gamma_1\operatorname{BulkBytesSkipped}+
\gamma_2\operatorname{BoundaryMass}(q)+
\gamma_3\operatorname{QuantRisk}(b)+
\gamma_4\operatorname{Uncertainty}(q,a).
\]

The model is calibrated only on a training split. The held-out test split measures actual quality. If the predictor is miscalibrated, the runtime increases a guard margin `g_Q` or falls back to a safer action. The paper reports calibration error, false-safe rate, and false-unsafe rate; average prediction error alone is insufficient.

### 9.5 Runtime decision algorithm

At each request:

1. Validate the cache key and architecture/position metadata.
2. Read device state and query/retrieval features.
3. Use anchors to estimate chunk relevance and candidate bulk pages.
4. Enumerate feasible actions across bit widths, Design A/B, `r`, and page budgets.
5. Reject actions violating memory, thermal, quality, or SLO guards.
6. Choose the minimum predicted objective.
7. Execute with trace logging.
8. Update only the permitted online calibration state from observed timing/thermal data; do not train on test answers.

Pseudo-code:

```text
state <- measure_device_state()
key_ok <- validate(cache_key, model, tokenizer, template, rope, mask)
if not key_ok:
    return full_prefill()

route <- score_with_anchors(query, anchor_index)
actions <- enumerate({b, pages, design, r, L_local, fallback})
feasible <- []
for a in actions:
    cost <- predict_cost(a, state, route)
    risk <- predict_quality_loss(a, state, route)
    if cost.memory <= state.free_memory - guard_memory and
       cost.thermal <= thermal_limit and
       cost.p95 <= latency_SLO and
       risk.upper_bound <= quality_epsilon:
        feasible.append((objective(cost, risk), a))

if feasible is empty:
    return full_prefill()
return execute(min(feasible), trace=true)
```

The paper must include the scheduler's action-selection time and any calibration overhead in all end-to-end numbers.

## 10. Portable mobile runtime architecture

### 10.1 Layered components

The implementation is an extension around a `ggml`/`llama.cpp`-style graph and should preserve a portable CPU path:

1. **Retriever:** fixed dense/lexical retrieval and fixed chunk order for the primary experiment.
2. **Cache validator:** checks model/tokenizer/template/RoPE/mask/chunk hashes and cache schema version.
3. **Anchor router:** reads only the small in-RAM anchor index and chooses candidate chunks/pages.
4. **Page manager:** issues aligned `mmap` or `pread` requests, tracks residency, checksums, and eviction.
5. **Dequantizer:** vectorized 2/4/8-bit unpacking into the tensor layout expected by the attention kernel.
6. **KV graph adapter:** provides exact, approximate, Design A, Design B, and full-prefill modes through a common graph API.
7. **Scheduler:** applies the cost/risk constraints and fallback policy.
8. **Telemetry:** emits per-layer tokens, bytes, page faults, kernel time, PSS, power, temperature, and thermal state.

The core paper artifact should be implemented in C++/C with platform adapters for Android and iOS. Java/Kotlin/Swift code is limited to file descriptors, telemetry, and lifecycle integration.

### 10.2 Graph modifications

The graph needs explicit node types instead of overloading an ordinary dense prefill:

- `LLAMA_OP_KV_LOAD_ANCHOR`: load high-precision anchor K/V;
- `LLAMA_OP_KV_LOAD_BULK`: load and dequantize selected pages;
- `LLAMA_OP_KV_ROPE_SHIFT`: apply a validated position transform;
- `LLAMA_OP_KV_MERGE_APPROX`: merge cached proposals with the active attention view;
- `LLAMA_OP_BOUNDARY_ACTIVATION_LOAD`: load Design A activation blocks;
- `LLAMA_OP_SPARSE_RECOMPUTE`: execute selected token rows through a layer range;
- `LLAMA_OP_KV_VALIDATION`: check hashes, schema, positions, mask, and quantizer;
- `LLAMA_OP_FULL_PREFILL_FALLBACK`: escape hatch for invalid or unsafe reuse.

Each graph execution records an `approximation_level` and a `validity_status` (`exact`, `rope_aligned_approximate`, `boundary_repaired`, `full_prefill`, `fallback`). This prevents an approximate result from being reported as exact simply because the graph returned a token.

### 10.3 Binary Flash container

The proposed container is a versioned, append-only file with a fixed header and independently checksummed pages:

```text
HACERKV\0
schema_version: u32
model_sha256: 32 bytes
tokenizer_sha256: 32 bytes
template_sha256: 32 bytes
rope_id, rope_base, rope_scale: fixed fields
mask_policy_id: fixed field
chunk_id, chunk_sha256, token_count
layer_count, kv_heads, head_dim
anchor_precision, bulk_precision
page_size, alignment, index_offset, page_count
anchor_index_offset, activation_index_offset
per-layer tensor descriptors
page directory: {offset, length, layer, chunk, token_range, crc32c}
payload pages: anchor blocks, bulk blocks, optional activations
```

Required properties:

- data offsets are aligned to the measured Flash/page granularity, initially `[4 KiB/16 KiB]` candidates;
- page directory entries permit partial `pread` without reading neighboring chunks;
- every page is independently checksummed;
- tensor shape, endian, quantizer, and SIMD layout are explicit;
- `mmap` is used for sequential warm access only when residency is observable; `pread` is used for bounded random page reads;
- file locking and atomic manifest replacement prevent partially written cache records;
- a cache is invalidated when any validity-key component changes.

The experiment must compare `mmap`, aligned `pread`, and an OS-buffered path, including page-fault and read-amplification costs.

### 10.4 SIMD dequantization

The dequantization pipeline is:

1. issue aligned page reads;
2. validate CRC and tensor descriptor;
3. unpack 2/4-bit groups using NEON on ARM64 and a scalar reference path;
4. apply per-group scales/zero-points;
5. fuse dequantization with RoPE rotation where the architecture permits;
6. write into the attention kernel's contiguous or tiled layout;
7. release pages under the scheduler's memory policy.

The portable implementation must have a deterministic scalar fallback and must record whether the SIMD path was used. The primary results must not depend on an unreported vendor-specific accelerator.

## 11. Experimental protocol

### 11.1 Hardware matrix

The primary study uses at least two physical devices per tier and a held-out device per tier when available. If procurement limits the study, CPU-throttled profiles may supplement but may not be relabeled as independent hardware. The final paper must list exact SoC, CPU clusters, accelerator availability, RAM, Flash type, OS build, kernel, thermal policy, and power instrumentation.

| Tier | Intended operating regime | Required measured variables | Minimum study cells |
|---|---|---|---:|
| Budget | CPU-bound, low RAM, slower Flash | `B_flash`, `R_dequant`, prefill throughput, free RAM, sustained thermal state | `[N_budget]` devices |
| Mid-range | mixed compute/memory bound, thermal transition | memory bandwidth, Flash, p50/p95 prefill, thermal transition time | `[N_mid]` devices |
| Flagship | high compute, OS pressure and energy dominated | CPU/GPU/NPU path, energy/query, background pressure, cache warmth | `[N_flagship]` devices |

Each physical device contributes a cool-start block, a warm-Flash block, a memory-pressure block, a thermally sustained block, and a battery/charging block if safe and supported.

### 11.2 Models and inference settings

Use at least two off-the-shelf decoder models with different attention/position configurations, for example:

- Model A: `[model name, parameter count, GGUF/GGML revision, quantization, SHA-256]`;
- Model B: `[model name, parameter count, attention type, RoPE/position scheme, quantization, SHA-256]`.

The exact models are frozen before collection. Record tokenizer, chat template, context length, threads, accelerator layers, batch size, micro-batch size, sampling seed, stop rules, and output-token cap. The primary quality comparison uses deterministic decoding; a separate robustness block may use fixed-seed stochastic decoding.

The retrieval pipeline is held constant across all cache policies: same document hashes, chunk IDs, embedding model, index hash, top-`K` retrieval results, order, prompt template, and citation policy. Cache reuse must not change retrieval quality by changing the retriever. The experiment first records a retrieval trace, then replays it through every runtime policy.

### 11.3 Workloads

The primary workloads are:

1. **Single-hop QA:** a controlled subset of Natural Questions or equivalent, with one relevant chunk and distractors.
2. **HotpotQA:** two-hop and comparison questions with annotated supporting facts.
3. **2WikiMultiHopQA:** multi-document reasoning with explicit bridge structure.
4. **MuSiQue:** compositional multi-hop questions with varying hop counts.
5. **Personal/cold-start corpus:** locally generated or permissioned documents with query distribution shifts and no future-query predictor.

For every benchmark, publish the exact sampled IDs, retrieval corpus, chunking script, prompt, and license-compatible preprocessing. Separate one-hop versus two-/three-/four-hop queries, short versus long chunks, repeated versus novel chunks, high versus low lexical overlap, stable versus reordered chunks, and answerable versus insufficient-evidence queries.

### 11.4 Cache and Flash conditions

The experiment has three cache states:

- **Cold Flash:** after a documented cache-eviction procedure. On Android, `drop_caches` may require root and is not assumed available; use a controlled reboot, file-size eviction workload, or a documented OS-supported alternative and report which one was used.
- **Warm Flash:** repeat access after a fixed warm-up sequence, verifying page residency through page-fault/read telemetry rather than assuming it.
- **Mixed:** a randomized sequence with a measured fraction `[p_cold]` of cold or evicted pages.

The same query order and cache-state schedule are used for all policies within a paired block. The schedule is counterbalanced so a policy does not systematically receive the coolest or warmest device state.

### 11.5 Thermal and memory controls

Run each policy in cool idle state, thermally sustained state after `[T_sustain]` minutes of a fixed workload, background-load state with a reproducible CPU/memory stressor, low-free-memory state with an explicitly documented memory reservation, and charging/battery-powered state where instrumentation permits.

Record pre/post battery temperature, thermal status, CPU frequencies, PSS/RSS, page faults, memory pressure, and OS background activity. Do not infer thermal causality from temperature alone; use the OS thermal signal and a frequency/throughput trace.

### 11.6 Baselines and ablations

| ID | Baseline/policy | Purpose |
|---|---|---|
| B0 | Full prefill, standard KV | Quality and compute reference |
| B1 | Exact prefix/prompt cache only | Separates safe reuse from approximate chunk reuse |
| B2 | Naive independently encoded chunk concatenation, RoPE aligned, `r=0%` | TurboRAG-style reuse without repair |
| B3 | Fixed uniform 2-bit and 4-bit KV | KIVI-style storage/quantization comparison |
| B4 | Fixed 20% all-layer recomputation | ProphetKV-style fixed repair comparison |
| B5 | Similarity-threshold page loader | Non-hardware-adaptive routing heuristic |
| B6 | Predictive/hierarchical cache where reproducible | PerCache-family comparison |
| B7 | HACER-RAG with uniform anchors | Tests whether the anchor score matters |
| B8 | HACER-RAG, Design A only | Tests boundary activation caching |
| B9 | HACER-RAG, Design B only | Tests all-layer ultra-sparse repair |
| B10 | HACER-RAG full scheduler | Joint contribution |
| B11 | Offline oracle | Measures policy headroom using post-hoc measured cost/quality |

Required ablations include removing anchor routing, replacing the anchor score with Key/Value norm outliers, cumulative attention only, uniform quantization at equal bytes, boundary tokens, query probes, adaptive `r`, thermal input, memory-pressure input, and comparing `mmap` with aligned `pread`.

### 11.7 Metrics

**Systems:** TTFT, prefill, first-decode, decode throughput; p50/p95/p99 latency; SLO violation rate; retrieval/routing/Flash/dequantization/RoPE/recompute/scheduler time; bytes/pages/page faults/read amplification; hit/fallback rates; recomputed tokens/FLOPs by layer; peak PSS/RSS; energy/query; battery temperature, thermal state, frequencies, and sustained throughput.

**Cache approximation:** next-token KL divergence; logit cosine similarity and top-1 agreement; perplexity change; answer disagreement; citation/evidence disagreement; worst-decile and 95th-percentile quality loss; quality as a function of cache age, chunk reorder, bit width, and repair ratio.

**RAG quality:** EM/F1 or exact correctness; supporting-fact recall/precision; citation entailment; labeled faithfulness/groundedness; hallucination and unsupported-claim rate; refusal correctness; multi-hop bridge and hop-completion accuracy.

Metrics must distinguish `not_applicable`, `not_measured`, `skipped`, and `failed`. A skipped inference is not a fully grounded answer.

## 12. Statistical protocol and falsification tests

### 12.1 Repetitions and pairing

For every `(device, model, dataset slice, cache state, thermal state, policy)` cell:

- use at least `[R_warm]` warm repetitions and `[R_cold]` cold repetitions;
- randomize query order within blocks;
- pair policies on the same query and hardware state;
- discard only predeclared setup failures, preserving a failure ledger;
- report median, IQR, p95, mean, standard deviation, and sample count.

Use paired bootstrap confidence intervals over queries and devices. Use a mixed-effects model with policy as a fixed effect and device/query as random effects when sample size permits. Correct for multiple comparisons across primary hypotheses. The final paper must include effect sizes and confidence intervals, not only p-values.

### 12.2 Primary acceptance table

The paper should preregister the following decision table before final collection:

| Hypothesis | Primary comparison | Required evidence | Outcome placeholder |
|---|---|---|---|
| H1 | Split precision vs uniform quantization at equal bytes/quality | Lower bytes/dequant time with quality loss `≤ε_Q` and CI excluding zero | `[X.X]%` byte reduction; `[X.X]` quality delta |
| H2 | Design A/B vs B2/B0 at equal quality target | Lower p95 TTFT or energy, with dependency trace and no hidden fallback | `[X.X]%` TTFT; `[X.X]%` energy |
| H3 | Adaptive scheduler vs best fixed policy on held-out states/devices | Lower SLO violation or objective with calibration overhead charged | `[X.X]` violation change |
| H4 | Quality/cost curve over `r` | Measured multi-hop frontier and statistical uncertainty | `[X.X]` recovered quality at `[X.X]%` cost |

### 12.3 What counts as a negative result

The paper is scientifically successful if it finds that uniform quantization is better on some tier, full prefill is preferable under a cold-cache or high-bandwidth regime, Design A loses because activation I/O dominates, Design B loses because all-layer sparse kernels underutilize SIMD, adaptive scheduling does not generalize to unseen devices, or multi-hop quality cannot be recovered below a large repair ratio. These outcomes falsify parts of the design and should be reported rather than hidden behind a tuned operating point.

## 13. Reproducibility and artifact requirements

Every result row must include:

```text
run_id
device_manifest_sha256
os_build_and_kernel
model_sha256
tokenizer_sha256
embedding_sha256
document_corpus_sha256
chunker_revision
retrieval_index_sha256
prompt_template_sha256
cache_container_schema
cache_manifest_sha256
policy_and_ablation_id
cache_state
thermal_state
memory_pressure_state
query_id
seed
raw_trace_path
quality_label_version
```

The artifact must provide a scalar reference implementation for quantization/cache merge; deterministic tests for RoPE shifts, page alignment, checksums, invalidation, and fallback; a trace-replay tool for scheduler decisions; raw per-layer traces; calibration/collection/evaluation/bootstrap scripts; and a manifest distinguishing physical devices, CPU-throttled profiles, emulators, and simulation.

The existing repository's MobCert code and reports are frozen and may be included only as a read-only comparator/protocol if a future experiment explicitly requires it. No HACER-RAG claim may depend on changing MobCert behavior.

## 14. Failure modes and safeguards

| Failure mode | Detection | Required response |
|---|---|---|
| Model/template mismatch | validity-key hash check | full prefill |
| RoPE or mask mismatch | cache metadata validation | invalidate cache |
| Chunk order changed | order hash | re-align or full prefill |
| Anchor false negative | full-retrieval replay | count and report; raise `K_route` |
| Bulk page checksum failure | CRC32C/hash | discard page and fallback |
| OOM/memory pressure | allocation/PSS guard | evict, lower precision, or full prefill |
| Thermal state changes | telemetry threshold | reschedule/fallback and report |
| Quality predictor miscalibration | held-out audit | increase safety margin |
| Sparse graph dependency violation | per-layer trace | mark approximate or reject result |
| Stale/personal cache leakage | user/document namespace | isolate and invalidate |
| Scheduler overhead dominates | trace timing | report; choose static/full prefill if cheaper |

## 15. Paper organization

1. **Introduction:** mobile RAG tension; why chunk KV reuse is not automatically valid; research question and contributions.
2. **Background and motivation:** mobile hardware, RAG prefill, KV cache, quantization, RoPE, residual-stream dependencies.
3. **Problem formulation:** exact versus approximate cache validity and quality-constrained device-state optimization.
4. **HACER-RAG design:** anchor/bulk cache, routing, RoPE alignment, Design A/B, scheduler.
5. **Implementation:** `ggml` graph, container, mmap/pread, NEON/scalar kernels, telemetry.
6. **Experimental methodology:** device matrix, model/corpus hashes, workloads, baselines, statistical plan.
7. **Results:** H1–H4, Pareto curves, sustained thermal blocks, held-out-device generalization, failure cases.
8. **Ablations and sensitivity:** anchor policies, page sizes, bit widths, repair ratios, scheduler inputs.
9. **Limitations and threats to validity:** architecture scope, device access, power instrumentation, retrieval dependence, approximate correctness.
10. **Related work:** TurboRAG, KVLink, LazyAttention, ProphetKV, PerCache, KIVI, mobile inference/offload systems.
11. **Conclusion:** what measurements support and what remains open.

## 16. Planned result tables and figures

### Table 1 — Device and model manifest

| Tier | Device/SoC | RAM | Flash | OS/kernel | Model | Quantization | Power instrument |
|---|---|---:|---|---|---|---|---|
| Budget | `[device]` | `[X] GB` | `[type]` | `[build]` | `[model]` | `[quant]` | `[instrument]` |
| Mid-range | `[device]` | `[X] GB` | `[type]` | `[build]` | `[model]` | `[quant]` | `[instrument]` |
| Flagship | `[device]` | `[X] GB` | `[type]` | `[build]` | `[model]` | `[quant]` | `[instrument]` |

### Table 2 — Main performance/quality result

| Policy | Cache state | TTFT p50 | TTFT p95 | Energy/query | Peak PSS | Quality | Multi-hop bridge | SLO violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Full prefill | `[cold/warm]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` |
| Naive reuse | `[cold/warm]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` |
| Fixed repair | `[cold/warm]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` |
| HACER-RAG | `[cold/warm]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` | `[X.X]` |

Every number is reported with `[95% CI]`, sample count, and exact policy configuration.

### Figure plan

1. **Runtime decision diagram:** retrieval trace → validity check → anchor routing → page selection → scheduler action → Design A/B/full prefill → decode, with fallback and telemetry.
2. **Accuracy–latency–energy Pareto frontier:** every `r` and bit width, separated by hop count; annotate adaptive and oracle points.
3. **Hardware-state adaptation:** scheduler choices as Flash bandwidth, free RAM, temperature, and prefill throughput vary, including held-out traces.
4. **Residual-stream boundary ablation:** naive reuse, Design A, Design B, and full prefill by layer, with activation bytes and logit divergence.

## 17. Implementation roadmap

### Phase 0 — Freeze and inventory

- Freeze MobCert files, behavior, and historical reports.
- Freeze the model, tokenizer, prompt, corpus, chunk IDs, and retrieval replay.
- Record the current repository revision and preserve old results as audit artifacts.

### Phase 1 — Exact reuse and trace infrastructure

- Implement cache validity keys and full-prefill reference traces.
- Add scalar RoPE-shift tests and exact-prefix reuse.
- Add per-layer, per-token, per-page trace schema.
- Build the binary container reader/writer with CRC and atomic manifests.

### Phase 2 — Quantized anchor/bulk cache

- Implement uniform 2/4-bit baselines.
- Implement anchor scoring and in-RAM routing.
- Validate bytes, pages, false negatives, and logit divergence offline.

### Phase 3 — Design A/B

- Implement boundary activation cache.
- Implement all-layer sparse recomputation.
- Add dependency-trace validation and conservative fallback.

### Phase 4 — Scheduler and calibration

- Build device calibration harness.
- Fit cost and quality-loss predictors on training devices only.
- Implement guard bands, held-out-device evaluation, and policy replay.

### Phase 5 — Collection and paper

- Run randomized paired blocks across device tiers and states.
- Lock raw artifacts before analysis.
- Generate confidence intervals and Pareto plots.
- Replace `[X.X]` placeholders only with trace-backed results.

## 18. Current status and next concrete action

The current workspace is **not yet a native implementation of HACER-RAG**. It contains an offline RAG engineering baseline, physical telemetry/evaluation infrastructure, frozen MobCert work, and a new host-side reference for the HACERKV manifest and scheduler contracts. The host reference is not a mobile measurement and does not export/import llama.cpp KV tensors. The next native research implementation should begin with Phase 1 rather than modifying MobCert or using the existing nine-query Zephyr evaluation as the main paper benchmark.

The first falsifiable prototype milestone is:

> On one selected model and one selected device, reproduce full-prefill logits, exact-prefix reuse, and RoPE-aligned independent-chunk reuse from the same cache manifest; then quantify next-token KL divergence and page/compute costs before implementing selective repair.

No paper-quality claim should be made until this prototype demonstrates that the runtime can distinguish exact reuse from approximate reuse and can fall back safely.

### 18.1 Current ADB and MobCert boundary

At the latest collection attempt, `adb devices -l` returned no attached device.
The Redmi/MobCert artifacts therefore remain frozen read-only comparators. The
new HACER-RAG work does not depend on MobCert behavior, does not change its
source, and must not use a disconnected or stale ADB result as evidence. A
future phone run is valid only when the request-correlated runner records a
live device identity, APK/model/index hashes, and a completed terminal result.

## References to verify before submission

- Songshuo Lu et al., “TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text,” arXiv:2410.07590, 2024. <https://arxiv.org/abs/2410.07590>
- Jingbo Yang et al., “KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse,” arXiv:2502.16002, 2025. <https://arxiv.org/abs/2502.16002>
- Zirui Liu et al., “KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache,” ICML 2024, arXiv:2402.02750. <https://arxiv.org/abs/2402.02750>
- Kaiwei Liu et al., “PerCache: Predictive Hierarchical Cache for RAG Applications on Mobile Devices,” arXiv:2601.11553, 2025/2026. <https://arxiv.org/abs/2601.11553>
- LazyAttention and ProphetKV: exact title, venue, version, and URL must be verified from the authoritative proceedings/arXiv record before the bibliography is finalized.
