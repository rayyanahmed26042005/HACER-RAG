# HACER-RAG Physical Scaling Matrix Study (180 Executions on Xiaomi Redmi 12 5G)

**Target Device:** Xiaomi Redmi 12 5G (`36fef3202145`), Snapdragon 4 Gen 2 (2x Cortex-A78 @ 2.2 GHz + 6x Cortex-A55 @ 2.0 GHz), 8 GB RAM, Android 14 / HyperOS  
**Model & Engine:** Gemma 2 2B Instruct (`gemma-2-2b-it-Q4_K_M.gguf`), `llama.rn` v0.5.8 (llama.cpp)  
**Workload & Budget:** Byte-for-byte canonical RAG prompt (1,567 chars, 395 tokens), generation budget $n_{\text{predict}} = 35$, greedy decoding ($T = 0$, top-p $= 1$).  
**Experimental Design:** $3 \times 3$ full matrix ($K \in \{1, 2, 4\}$ chunks $\times n_{\text{ctx}} \in \{512, 1024, 2048\}$), 10 paired repetitions per cell = 90 matched pairs (180 executions total), randomized within-pair execution order, inter-trial context flushing, and low-level `/proc/{PID}/io` block telemetry.

---

## 1. Executive Summary & Core Research Findings

1. **Monotonic Latency–Cache-Size Scaling Confirmed:**  
   The physical experiments on the Snapdragon 4 Gen 2 demonstrate that prefill latency reduction tracks prefix cache size with near-mathematical linearity, regardless of allocated context capacity:
   - **$K=1$ Cached Chunk (84 tokens / 21.2% prefix):** **~20.3%** mean prefill reduction (10.3s saved), **1.20x** end-to-end wall-clock speedup.
   - **$K=2$ Cached Chunks (158 tokens / 39.9% prefix):** **~39.4%** mean prefill reduction (20.1s saved), **1.49x** end-to-end wall-clock speedup.
   - **$K=4$ Cached Chunks (341 tokens / 86.1% prefix):** **~86.3%** mean prefill reduction (47.0s saved), **3.51x** end-to-end wall-clock speedup.
2. **Context-Capacity Invariance:**  
   Expanding context capacity from $n_{\text{ctx}} = 512$ to $1024$ and $2048$ does not degrade cache reuse efficiency: at $K=4$, prefill reductions are **86.7%** ($n_{\text{ctx}}=512$), **86.2%** ($n_{\text{ctx}}=1024$), and **86.1%** ($n_{\text{ctx}}=2048$), with overlapping 95% bootstrap confidence intervals.
3. **100% Invariant Ground Truth:**  
   - **Prompt Invariance:** The canonical prompt SHA-256 (`a5f1db09b904e81393eb06119c537db0e337ed12cfcf78b927e6e9e030727c6d`) is identical across all 182 records.
   - **Answer Invariance:** Every single run generated the exact byte-identical fact-accurate answer text:
     > *" Zephyr Dynamics was founded in 2011, the CLO is Jorge Fonseca, the AeroSense X7 battery lasts 72 hours, and the full-capacity tier"*
   - **Zero Truncation:** `truncated = false` on 182/182 executions. All context budgets satisfied the enforced inequality $(395 + 35) \le n_{\text{ctx}} - 4$.
4. **Complete Dataset Accounting:**  
   A total of **182 execution records** were logged. Exactly 90 matched pairs (10 pairs per configuration) completed full execution. Two incomplete single runs (one from C5 and one from C7, caused by transient carrier Class 0 popups) were preserved in the raw JSONL log and classified under `incomplete_pairs` rather than discarded.

---

## 2. Full $3 \times 3$ Scaling Matrix Statistical Results

All metrics reflect 10 paired repetitions per configuration ($N=10$). Bootstrap 95% confidence intervals are computed using 2,000 resamples.

| Config ID | $K$ | $n_{\text{ctx}}$ | Matched Pairs | Baseline Prefill (Mean $\pm$ Std) | Reuse Prefill (Mean $\pm$ Std) | Prefill Paired $\Delta$ [95% Bootstrap CI] | Prefill Reduction % [95% Bootstrap CI] | Wall-Clock Baseline (Mean) | Wall-Clock Reuse (Mean) | Wall Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C1_K1_CTX512`** | 1 | 512 | 10 | 53.27s $\pm$ 2.73s | 42.85s $\pm$ 2.00s | 10.43s [8.16s, 12.45s] | **19.3%** [15.5%, 22.7%] | 63.98s | 53.71s | **1.19x** |
| **`C2_K1_CTX1024`** | 1 | 1024 | 10 | 56.03s $\pm$ 5.94s | 44.39s $\pm$ 4.16s | 11.64s [9.08s, 14.04s] | **20.5%** [16.3%, 23.7%] | 67.03s | 55.15s | **1.22x** |
| **`C3_K1_CTX2048`** | 1 | 2048 | 10 | 49.48s $\pm$ 2.97s | 38.93s $\pm$ 1.98s | 10.55s [9.31s, 11.93s] | **21.2%** [19.1%, 23.3%] | 59.62s | 49.31s | **1.21x** |
| **`C4_K2_CTX512`** | 2 | 512 | 10 | 51.28s $\pm$ 3.12s | 30.55s $\pm$ 1.97s | 20.72s [19.36s, 22.00s] | **40.4%** [38.7%, 41.9%] | 61.58s | 40.77s | **1.51x** |
| **`C5_K2_CTX1024`** | 2 | 1024 | 10 | 50.73s $\pm$ 1.58s | 30.84s $\pm$ 2.79s | 19.89s [18.04s, 21.80s] | **39.2%** [35.5%, 42.6%] | 61.16s | 41.00s | **1.49x** |
| **`C6_K2_CTX2048`** | 2 | 2048 | 10 | 51.01s $\pm$ 2.08s | 31.30s $\pm$ 1.86s | 19.71s [17.91s, 21.25s] | **38.5%** [35.5%, 41.1%] | 61.68s | 41.64s | **1.48x** |
| **`C7_K4_CTX512`** | 4 | 512 | 10 | 54.08s $\pm$ 4.21s | 7.18s $\pm$ 0.40s | 46.90s [44.58s, 49.54s] | **86.7%** [86.2%, 87.2%] | 64.57s | 17.93s | **3.60x** |
| **`C8_K4_CTX1024`** | 4 | 1024 | 10 | 54.89s $\pm$ 2.46s | 7.56s $\pm$ 0.28s | 47.33s [45.87s, 48.96s] | **86.2%** [85.7%, 86.7%] | 65.54s | 17.96s | **3.65x** |
| **`C9_K4_CTX2048`** | 4 | 2048 | 10 | 54.20s $\pm$ 1.64s | 7.55s $\pm$ 0.61s | 46.65s [45.62s, 47.90s] | **86.1%** [85.2%, 86.8%] | 65.16s | 19.92s | **3.27x** |

---

## 3. Detailed Latency Distribution & Variability

| Config ID | Prefill Baseline Median | Prefill Reuse Median | Baseline Wall Median | Reuse Wall Median | Wall Paired $\Delta$ [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`C1_K1_CTX512`** | 53.04s | 42.51s | 64.32s | 52.86s | 10.27s [7.69s, 12.52s] |
| **`C2_K1_CTX1024`** | 53.62s | 42.61s | 64.52s | 52.61s | 11.88s [8.83s, 14.60s] |
| **`C3_K1_CTX2048`** | 47.94s | 38.81s | 58.95s | 49.26s | 10.31s [8.65s, 12.12s] |
| **`C4_K2_CTX512`** | 52.05s | 31.70s | 62.92s | 42.07s | 20.81s [19.43s, 21.97s] |
| **`C5_K2_CTX1024`** | 50.97s | 30.61s | 61.26s | 40.51s | 20.17s [17.81s, 22.54s] |
| **`C6_K2_CTX2048`** | 51.59s | 31.03s | 62.55s | 41.80s | 20.05s [18.17s, 21.78s] |
| **`C7_K4_CTX512`** | 53.56s | 7.22s | 63.58s | 18.09s | 46.64s [44.07s, 49.78s] |
| **`C8_K4_CTX1024`** | 54.89s | 7.52s | 65.06s | 17.63s | 47.59s [46.12s, 49.21s] |
| **`C9_K4_CTX2048`** | 53.74s | 7.16s | 64.35s | 18.17s | 45.24s [42.23s, 48.15s] |

---

## 4. Hardware Telemetry & Storage I/O Analysis

- **Storage Condition Classification:** Across the 90 reuse executions:
  - **`cold_verified`:** 39 executions (physical storage read bytes $\ge 85\%$ of session size, directly measured from `/proc/{PID}/io` `read_bytes`).
  - **`warm_candidate`:** 51 executions (process read requests $\ge 85\%$ of session size via `rchar`, with block-level I/O partially absorbed by Android OS page cache).
- **Session File Footprint on UFS Flash:**
  - $K=1$ session file: **10,078,198 bytes** (~10.08 MB)
  - $K=2$ session file: **17,960,086 bytes** (~17.96 MB)
  - $K=4$ session file: **37,451,782 bytes** (~37.45 MB)
- **Thermal Behavior:** Device battery temperature stayed within **38.0°C to 42.0°C** across all 182 executions, managed by randomized within-pair order and inter-trial context flushing.
- **Memory Footprint (PSS):** Proportional Set Size remained stable at **~2,015 MB** ($n_{\text{ctx}}=512$), **~2,105 MB** ($n_{\text{ctx}}=1024$), and **~2,217 MB** ($n_{\text{ctx}}=2048$).

---

## 5. Token Accounting & Context Safety Telemetry

Across all 182 execution records:
- `prompt_tokens_total`: **395** native tokens (constant across all configurations).
- `tokens_loaded_from_session`: **0** (Baseline), **84** ($K=1$), **158** ($K=2$), **341** ($K=4$).
- `tokens_evaluated_this_run`: **395** (Baseline), **312** ($K=1$), **238** ($K=2$), **55** ($K=4$).
- `resident_kv_tokens_post_decode`: **430** tokens ($396 \text{ prompt tokens with BOS} + 34 \text{ predicted tokens}$).
- `safety_headroom_enforced`: **78** ($n_{\text{ctx}}=512$), **590** ($n_{\text{ctx}}=1024$), **1614** ($n_{\text{ctx}}=2048$).
- `raw_context_remaining`: **82** ($n_{\text{ctx}}=512$), **594** ($n_{\text{ctx}}=1024$), **1618** ($n_{\text{ctx}}=2048$).

---

## 6. Analytical Conclusion & Implications for HACER-RAG

The empirical evidence from 180 controlled physical executions on the Snapdragon 4 Gen 2 firmly establishes:
1. **Prefill Compute Dominance:** On resource-constrained mobile CPUs (Cortex-A78/A55 without NPU acceleration), prefill latency represents over **80%** of total response time for moderate context lengths (395 tokens take ~53 seconds to prefill vs. ~8.5 seconds to decode 34 tokens).
2. **Exact-Prefix Cache Efficiency:** By preserving and restoring exact prompt prefixes, HACER-RAG cuts prefill latency by **86.3%** when 4 chunks are cached, reducing end-to-end response time from **~65 seconds down to ~18 seconds** (a **3.5x speedup**) with 100% answer identity and zero truncation.
3. **Context Capacity Scalability:** Context capacity ($n_{\text{ctx}}$) can be dynamically expanded to 2048 tokens without penalizing cache reuse efficiency or inducing out-of-memory errors on an 8 GB Android 14 device.
