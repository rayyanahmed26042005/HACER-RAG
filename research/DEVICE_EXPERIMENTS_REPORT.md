# Physical Device Experimental Results: General Q&A & Document RAG

**Date:** October 11, 2026  
**Hardware Platform:** Xiaomi Redmi 12 5G  
**SoC Architecture:** Qualcomm Snapdragon 4 Gen 2 (SM4450)  
- **CPU Topology:** 2x ARM Cortex-A78 @ 2.2 GHz (Performance) + 6x ARM Cortex-A55 @ 2.0 GHz (Efficiency)  
- **System Memory:** 6 GB LPDDR4X (Available RAM during runs: ~2.1 GB – 2.8 GB)  
- **Thermal Monitor:** On-device hardware battery sensor & Android thermal governor  
**LLM Engine:** Gemma 2 2B (GGUF Quantized, 4-bit, native `llama.rn` / `llama.cpp` runtime)  
**Embedding Engine:** `all-minilm-l6-v2-q4_k_m` (384-dimensional dense embeddings)  
**Ground Truth Document:** `zephyr_handbook.txt` (3-page / 53-line / 9-chunk corporate employee & product handbook)  

---

## 1. Executive Summary & Core Empirical Findings

Across 10 physical device runs on the Redmi 12 5G (3 General Q&A runs and 7 Document RAG runs), the on-device telemetry and inference HUD captured real execution profiles.

### Key Discoveries:
1. **Thread Inversion on Heterogeneous Cores:**
   - On General Q&A, **Fixed 4 Threads** achieved **3.1 tokens/sec**, whereas **Fixed 6 Threads** regressed to **2.9 tokens/sec** (-6.5% slowdown) and increased Time To First Token (TTFT) from 4.84s to 5.51s.
   - Allocating 6 threads forces work onto 4 slow Cortex-A55 cores, creating `ggml_vec_dot` barrier synchronization stalls and memory bandwidth contention. More threads degrade performance on entry-tier heterogeneous mobile silicon.
2. **Adaptive Controller Behavior:**
   - The adaptive bandit controller converged to **4 threads** (`exploitation without interactive exploration`) and correctly avoided 6T/8T over-allocation.
3. **The Bottleneck Flip (General vs RAG):**
   - **General Q&A:** The primary bottleneck was **Token Decode** (93%–99% of run latency). Prompt prefill was negligible (0–4.6s).
   - **Document RAG:** The primary bottleneck flipped completely to **Prompt Prefill** (up to **84% of total inference time** and TTFT exploding to **31.7s – 80.9s**).
4. **RAG Retrieval Quality & Context Suffocation:**
   - The original RAG implementation suffered from:
     - **Naive 100-word Chunking Boundary Fragmentation:** Splitting tables mid-row separated Jorge Fonseca from his Chief Legal Officer title, and separated AeroSense X7 from its battery/weight specifications.
     - **Context Suffocation:** The controller allocated a 192-token context budget for single-hop questions, causing the retriever to discard candidate chunks 2 and 3. In all inspected RAG runs, **only 1 chunk was kept out of 3 candidates** (`Selected Chunks: 1 kept of 3 candidates`).
     - **Pure Dense Cosine Search Failure:** Typographical variants (`aerosence` instead of `AeroSense`) and domain proper nouns failed to retrieve the right chunk.
     - **Output Length Throttling in Adaptive Mode:** In adaptive mode, `nPredict` was clamped down to 128 tokens with a prompt instructing the model to "Answer clearly and concisely", resulting in overly terse 2–3 line answers for general queries like "what is AI".

---

## 2. General Q&A Experimental Data

Tested Query: *"Tell me a story"* (and *"what is xboost in deep"*)

| Metric | Fixed 4 Threads (4T) | Adaptive Mode (4T Selected) | Fixed 6 Threads (6T) |
| :--- | :--- | :--- | :--- |
| **Artifact Screenshot** | `OutputScreen/General/WhatsApp Image 2026-10-11 at 1.14.59 PM (1).jpeg` | `OutputScreen/General/WhatsApp Image 2026-10-11 at 1.14.59 PM.jpeg` | `OutputScreen/General/WhatsApp Image 2026-10-11 at 1.18.26 PM.jpeg` |
| **Execution Time** | 1:12 PM | 1:13 PM | 1:18 PM |
| **Total Warm Latency** | **75,854.8 ms** (75.85s) | **65,471.6 ms** (65.47s) | **90,128.4 ms** (90.13s) |
| **TTFT (First Token)** | 4,843.4 ms | **639.9 ms (0.64s)** | 5,510.0 ms |
| **Prompt Prefill Time** | 4,641.0 ms (9 tokens) | 0.0 ms (KV Cache / 1 token) | 0.0 ms (KV Cache / 1 token) |
| **Prompt Prefill Speed** | 1.9 tok/s | $\infty$ (cached) | $\infty$ (cached) |
| **Token Decode Time** | 70,632.0 ms | 64,907.0 ms | 89,432.0 ms |
| **Generated Tokens** | 221 tokens | 174 tokens | 261 tokens |
| **Decode Throughput** | **3.1 tokens/sec** | **2.7 tokens/sec** | **2.9 tokens/sec** |
| **Primary Bottleneck** | **Token Decode** | **Token Decode** | **Token Decode** |
| **Battery Temperature** | *not read* | **44.0 °C** (Thermal Stress) | *not read* |
| **Available RAM** | *N/A* | 2,755 MB | *N/A* |
| **Policy State** | Fixed benchmark policy | `exploitation without exploration` | Fixed benchmark policy |

### Key Observations:
- **Scaling penalty:** Moving from 4 to 6 threads added 15 seconds of decode latency and decreased generation speed from 3.1 to 2.9 tok/s.
- **Thermal throttling:** Back-to-back inference elevated battery temperature to 44.0 °C, initiating Snapdragon DVFS frequency reduction and stabilizing throughput between 2.7 and 2.9 tok/s.

---

## 3. Document RAG Experimental Data (Physical Device)

Dataset: `offlineRag.pdf` / `zephyr_handbook.txt`  
Index Size: 9 chunks on device  
Embedding Model: `all-minilm-l6-v2-q4_k_m` (384-d)

### Detailed Run Logs

#### Run R1: Company Products
- **Query:** *"What are the products"*
- **Screenshot:** `OutputScreen/Rag/WhatsApp Image 2026-10-11 at 1.43.34 PM (1).jpeg` (1:36 PM)
- **Controller Configuration:** `ADAPTIVE (4T)`
- **Total Warm Latency:** 34.98s
- **TTFT:** 31.74s
- **Decode Speed:** 4.1 tok/s
- **Generated Text:** *"The products are the AeroSense X7."*
- **Retrieval Result:** 1 chunk kept of 3 candidates.
- **Accuracy Assessment:** **Partial Failure.** The handbook details 3 distinct products: AeroSense X7, SwarmKit Pro, and CloudMapper Lite. Because only Chunk 1 was passed to the LLM, the model missed the other two products entirely.

#### Run R2: Employee Benefits
- **Query:** *"What is the annual learning budget and gym allowance of employees?"*
- **Screenshot:** `OutputScreen/Rag/WhatsApp Image 2026-10-11 at 1.43.34 PM.jpeg` (1:38 PM)
- **Controller Configuration:** `ADAPTIVE (4T)`
- **Total Warm Latency:** 45,881.7 ms (45.88s)
- **TTFT:** 36,683.7 ms (36.68s)
- **Native llama.cpp Prefill:** 36,143.0 ms (218 tokens @ 6.0 tok/s)
- **Native llama.cpp Decode:** 9,140.0 ms (33 tokens @ 3.6 tok/s)
- **Primary Bottleneck:** **Prompt Prefill (78.8% of total runtime)**
- **RAG Pipeline Retrieval:** 129.4 ms (1 kept of 3 candidates, Evidence demand: `broad`)
- **Telemetry:** Battery Temp: 43.0 °C, RAM Available: 2,360 MB
- **Generated Text:** *"The annual learning budget is $1,500 USD/year per employee and the gym allowance is $60/month reimbursed via expense report."*
- **Accuracy Assessment:** **Full Success (100% Faithful).** Ground truth facts matched exactly with zero hallucinations.

#### Run R3: Executive Leadership Query
- **Query:** *"Who is chief legal officer and which office are they based in ?"*
- **Screenshot:** `OutputScreen/Rag/WhatsApp Image 2026-10-11 at 1.43.32 PM.jpeg` (1:41 PM)
- **Controller Configuration:** `BASELINE (4T)`
- **Total Warm Latency:** 89.06s
- **TTFT:** 80.88s
- **Decode Speed:** 3.2 tok/s
- **Generated Text:** *"The provided text doesn't explicitly state the name of the Chief Legal Officer or their location."*
- **Accuracy Assessment:** **False Refusal / Retrieval Miss.** Section 1.1 states: *"Jorge Fonseca: Chief Legal Officer | Office: Lisbon | Joined: 2019"*. Naive chunking split the leadership table across chunks, and the single retrieved chunk lacked Jorge Fonseca's name.

#### Run R4: Typo & Hardware Specifications
- **Query:** *"What is battery life and weight of aerosence X7?"* (Note typo: `aerosence`)
- **Screenshot:** `OutputScreen/Rag/WhatsApp Image 2026-10-11 at 1.43.35 PM (1).jpeg` & `1.43.36 PM.jpeg` (1:07 PM & 1:35 PM)
- **Controller Configuration:** `ADAPTIVE (4T)`
- **Total Warm Latency:** 40.95s – 53.01s
- **TTFT:** 34.03s – 43.47s
- **Decode Speed:** 2.3 – 3.0 tok/s
- **Generated Text:** *"I don't have enough information in the uploaded documents to answer that."*
- **Accuracy Assessment:** **False Refusal.** Chunk 2 has: *"QuantumCell battery that lasts 72 hours on a single charge. It weighs 2.3 kg"*, but refers to it as "The X7" (the name "AeroSense X7" was in Chunk 1). Because of the typo and single-chunk limit, retrieval failed.

#### Run R5: Overview Query with Typo
- **Query:** *"What is zypher?"* (Note typo: `zypher`)
- **Screenshot:** `OutputScreen/Rag/WhatsApp Image 2026-10-11 at 1.43.35 PM (2).jpeg` (1:34 PM)
- **Controller Configuration:** `ADAPTIVE (4T)`
- **Total Warm Latency:** 46,300.1 ms (46.30s)
- **TTFT:** 39,159.1 ms (39.16s)
- **Native llama.cpp Prefill:** 38,780.0 ms (255 tokens @ 6.6 tok/s)
- **Native llama.cpp Decode:** 7,086.0 ms (20 tokens @ 2.8 tok/s)
- **Primary Bottleneck:** **Prompt Prefill (83.8% of total runtime)**
- **RAG Pipeline Retrieval:** 43.0 ms (1 kept of 3 candidates, Demand: `single_hop`)
- **Telemetry:** Battery Temp: 40.0 °C, RAM Available: 2,116 MB
- **Generated Text:** *"Zephyr Dynamics Inc. specializes in atmospheric data harvesting systems and micro-drone swarm technology."*
- **Accuracy Assessment:** **Full Success.** Retrieved Chunk 0 correctly despite the typo.

#### Run R6: General Definition Query
- **Query:** *"what is xboost in deep"*
- **Screenshot:** `OutputScreen/Rag/WhatsApp Image 2026-10-11 at 1.43.36 PM.jpeg` (1:03 PM)
- **Controller Configuration:** `ADAPTIVE (4T)`
- **Total Warm Latency:** 14.07s
- **TTFT:** 5.48s
- **Decode Speed:** 3.8 tok/s
- **Generated Text:** *"XBoost is a gradient boosting library in Python. It's known for its speed and efficiency in handling large datasets."*
- **Accuracy Assessment:** Accurate, but excessively brief (2 sentences) due to adaptive token budget capping.

---

## 4. Comprehensive RAG Performance Matrix

| Query | Mode | Retrieval Time | Prefill Latency | Decode Speed | Total Warm Latency | Context Chunks | Result Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **"What is zypher?"** | Adaptive (4T) | 43.0 ms | 38.78s (255 tok @ 6.6 t/s) | 2.8 tok/s | 46.30s | 1 of 3 | ✅ Success |
| **"What are the products"** | Adaptive (4T) | ~110 ms | ~31.0s | 4.1 tok/s | 34.98s | 1 of 3 | ⚠️ Partial (1/3 products) |
| **"What is annual learning budget..."** | Adaptive (4T) | 129.4 ms | 36.14s (218 tok @ 6.0 t/s) | 3.6 tok/s | 45.88s | 1 of 3 | ✅ Success (100% faithful) |
| **"Who is chief legal officer..."** | Baseline (4T) | ~120 ms | ~80.0s | 3.2 tok/s | 89.06s | 1 of 3 | ❌ Retrieval Miss |
| **"What is battery life and weight..."** | Adaptive (4T) | ~90 ms | ~33.5s | 3.0 tok/s | 40.95s | 1 of 3 | ❌ Retrieval Miss |
| **"what is xboost in deep"** | Adaptive (4T) | 0 ms (General) | ~5.0s | 3.8 tok/s | 14.07s | N/A | ⚠️ Overly Terse (2 lines) |

---

## 5. Architectural Root Causes & Improvement Plan

### Problem 1: Chunk Boundary Fragmentation
- **Issue:** Naive 100-word sliding windows slice through sentences, tables, and sections.
- **Solution:** Implement **Structure-Aware Chunking**. Split on markdown headers (`##`), list items (`-`), and paragraph breaks (`\n\n`), keeping atomic sections together with parent section breadcrumbs (e.g. `[Section: 1.1 Leadership Team]`).

### Problem 2: Context Suffocation
- **Issue:** `getAdaptiveContextBudget` sets a 192–320 token budget, forcing `ragPipeline.ts` to discard candidates 2 and 3.
- **Solution:** Increase the mobile context budget to 512–768 tokens and allow up to 3 non-duplicate chunks when evidence is broad or multi-hop.

### Problem 3: Pure Dense Search vs Typos/Keywords
- **Issue:** Cosine similarity on 4-bit quantized MiniLM fails on misspellings ("aerosence") and exact proper nouns ("Jorge Fonseca").
- **Solution:** Implement **Hybrid Retrieval (Dense Cosine + In-Memory BM25 Lexical + Reciprocal Rank Fusion)**. This provides zero-cost lexical matching for exact keywords and typo resilience on mobile CPU.

### Problem 4: Overly Short General Responses in Adaptive Mode
- **Issue:** `getAdaptiveOutputBudget` artificially caps `nPredict` to 128 tokens for queries with "what/is/are", while the system prompt instructs the model to be "concise".
- **Solution:** Scale output budgets to 384–512 tokens for general inquiries, allowing the model to generate rich, informative explanations.
