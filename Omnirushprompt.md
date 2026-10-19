# Research Mission: Efficient On-Device LLM Inference for Resource-Constrained Android Devices

## 1. Your Role

Act as a combination of:

- A mobile LLM inference researcher.
- An academic research assistant familiar with IEEE publication standards.
- A systems performance engineer specializing in Android, llama.cpp, GGUF models, CPU/GPU/NPU execution, memory management, and thermal constraints.
- A critical research reviewer whose job is to reject weak novelty claims, unsubstantiated performance claims, and ideas that have already been adequately explored.

Your objective is not merely to optimize my application. It is to identify, validate, implement, and experimentally evaluate a technically meaningful contribution to efficient local LLM inference on resource-constrained mobile devices, with the potential to support a strong IEEE research paper.

**My primary research goal is efficient, private, fully on-device LLM inference on commodity Android smartphones.** A novel contribution is desirable but not mandatory if a carefully evaluated, reproducible systems contribution provides meaningful value. However, novelty must be investigated before selecting the final approach.

Do not assume that my proposed ideas are novel. Do not claim novelty simply because I implement a combination of existing techniques.

## 2. Existing System and Experimental Environment

The current application is a fully offline mobile RAG chatbot implemented with:

- React Native 0.78.
- llama.rn using llama.cpp for native LLM inference.
- GGUF-format quantized models.
- A locally stored document index and retrieved context.
- A local embedding model for semantic retrieval.
- Android Studio, ADB, Logcat, and physical-device profiling.

### Target physical device

- Device: Xiaomi Redmi 12 5G.
- Model identifier: 23076RN4BI.
- SoC: Qualcomm Snapdragon 4 Gen 2 (SM4450).
- GPU: Adreno 613.
- CPU: 8 cores, comprising 2 Cortex-A78 performance cores and 6 Cortex-A55 efficiency cores.
- OS: Android 15 / HyperOS.

### Current inference configuration

The baseline configuration previously used:

- Main model: Gemma 2 2B Instruct, Q4_K_M GGUF.
- Embedding model: all-MiniLM-L6-v2, quantized GGUF, 384-dimensional embeddings.
- CPU-only inference: n_gpu_layers = 0.
- n_threads = 4.
- n_ctx = 512.
- Batch and microbatch sizes previously configured at 512.
- The initialized LlamaContext is cached and reused across queries rather than reloading the main model for every query.

Verify these details against the actual repository, model metadata, configuration, and latest benchmark files before treating them as definitive. The current settings may not be optimal, and the model's exact parameter count, file size, build configuration, and quantization type must be reported accurately.

## 3. Existing Results and Known Bottlenecks

These are our preliminary observations. Treat them as baseline evidence to validate, not unquestionable facts.

### A. Physical-device benchmark

A representative RAG query, "Tell me about zypher", was executed against a local PDF named offlineRag.pdf containing nine indexed chunks.

The observed total latency was approximately 79.2 seconds.

| Component | Preliminary measurement | Interpretation |
|---|---:|---|
| Embedding and retrieval | Approximately 33–84 ms across observed runs; one representative run around 52 ms | Very small share of end-to-end latency |
| Prompt processing | Approximately 48,500 ms | Major bottleneck |
| Prompt tokens | Approximately 485 | The actual input-token count must be confirmed from the benchmark artifact |
| Prompt processing throughput | Approximately 10 tokens/second | Extremely slow for this workload |
| Token generation | Approximately 28,150 ms | Major bottleneck |
| Generated tokens | 33 | Short output, yet expensive to generate |
| Decode throughput | Approximately 1.17 tokens/second | Severe decode-performance limitation |
| UI typing animation | Approximately 2.5 seconds | Secondary application-level overhead |
| Total latency | Approximately 79.2 seconds | Unacceptable for many interactive use cases |

The embedding model, vector retrieval, and context construction do not currently appear to be the main performance problems.

**Important:** Do not conclude that attention alone causes the 48.5-second prompt-processing time. Native prompt processing can include projections, attention, MLP/FFN operations, normalization, quantized matrix multiplication, memory movement, and runtime scheduling. The exact breakdown is not yet known.

Similarly, TTFT and native prompt-processing time are different measurements. TTFT includes time from request submission until the first token becomes available, while prompt-processing time is a runtime-reported metric. Instrument and report them separately.

### B. Device resource observations

Previous profiling observed:

- Device temperature of approximately 39–42.5 °C.
- Performance-core frequencies around 1.344 GHz and efficiency-core frequencies around 921 MHz during a run.
- Potential thermal or power-related frequency limitations.

These readings do not prove thermal throttling. Establish whether the phone is thermally throttled by recording available Android thermal-status information, CPU frequencies, repeated-run performance, battery state, and temperature trends.

Previous memory readings were not sufficiently reconciled. Do not combine PSS, native heap allocation, mapped model-file size, and swap into one memory metric. Measure them separately and explain their meanings.

### C. Emulator results

A preliminary warm-query benchmark suite contained 108 repetitions across low-, medium-, and high-resource Android Virtual Device configurations.

Reported preliminary results:

- Mean latency: 7.14 seconds.
- Median latency: 6.12 seconds.
- Minimum: 3.07 seconds.
- Maximum: 12.22 seconds.
- Standard deviation: 3.82 seconds.
- Mean peak PSS: approximately 1,124.83 MB.
- Mean generation throughput: approximately 13.72 tokens/second.

These are emulator measurements, not equivalent measurements of the corresponding physical Pixel devices. Host CPU architecture, virtualization, resource allocation, and backend behavior may influence results.

The emulator and physical-phone results are not yet a controlled apples-to-apples comparison. Do not claim an 11-times hardware performance difference as a rigorous finding until model builds, prompts, token counts, output limits, warm/cold state, and benchmark procedures are matched.

### D. Existing instrumentation and artifacts

The repository may already contain:

- `chat/tasks/qa.ts`, which records native inference timing data.
- `benchmark/run_device_inference.py`.
- `benchmark/results/physical_mobile_split_benchmark.json`.
- `benchmark/benchmark.py`.
- `run_benchmark.py`.
- `analyze_bottlenecks.py`.
- Benchmark datasets, query sets, and AVD profiles.

Inspect the actual repository and use the existing instrumentation before rebuilding the benchmark infrastructure. Locate the authoritative results and determine whether reported values are reproducible.

## 4. First Requirement: Conduct a Serious Literature Review

Before proposing or implementing a research contribution, conduct a targeted literature review using current, credible academic and technical sources.

Prioritize peer-reviewed papers and recent preprints from IEEE Xplore, ACM Digital Library, USENIX, MLSys, ASPLOS, ISCA, MobiSys, MobiCom, EuroSys, and relevant arXiv papers. Inspect relevant official runtime documentation and source code as well.

Investigate research on:

1. Efficient LLM inference on smartphones and edge/mobile devices.
2. On-device local chatbots, mobile RAG, and private offline assistants.
3. Transformer prompt prefill and autoregressive decode optimization.
4. Mobile CPU scheduling, big.LITTLE scheduling, thread selection, and CPU affinity.
5. Mobile GPU inference using Vulkan, OpenCL, and supported vendor-specific backends.
6. Qualcomm AI Engine Direct, QNN, Hexagon NPU, and Adreno GPU inference.
7. Memory-constrained inference, KV-cache optimization, quantization, weight streaming, and model loading.
8. Thermal-aware and energy-aware inference scheduling.
9. Adaptive inference, dynamic model selection, early exiting, speculative decoding, token budgets, and quality-aware compute allocation.
10. Retrieval-context selection and adaptive RAG for reducing prompt-processing cost.
11. State-space models such as Mamba and recurrent or parallel alternatives such as RetNet, where relevant to mobile inference.
12. Dynamic orchestration of local LLMs and lightweight on-device tools or agents.

Search for both directly related papers and work that could invalidate a proposed novelty claim. Include recent work through the actual date of the review.

For every important paper, record:

- Full title, authors, publication venue, and year.
- DOI or stable URL where available.
- Problem addressed and core technique.
- Device, hardware, model architecture, quantization, and runtime used.
- Baselines and evaluation metrics.
- Measured improvements and limitations.
- Whether the method can realistically run on my target Android device.
- How it relates to the proposed contribution.

Build a comparison matrix of the closest existing methods. Identify overlapping techniques, implementation similarities, and differences.

**Do not invent citations, numerical results, publication venues, or novelty claims.** Read the paper or its accessible full text where feasible rather than relying solely on its title or abstract.

## 5. Second Requirement: Establish a Trustworthy Baseline

Before changing the inference architecture, reproduce and validate the current baseline.

Record the following for every benchmark run:

- Model name, parameter count, file size, quantization, and cryptographic hash.
- llama.cpp/llama.rn versions, commit or release, build flags, and backend.
- Android version, device model, available RAM, battery state, and thermal state.
- Thread count and any available CPU-affinity settings.
- Context length, batch size, microbatch size, and GPU-layer configuration.
- Cold-start/model-load time, measured separately from warm-query time.
- Input prompt tokens, retrieved chunk count, retrieved tokens, and output tokens.
- Embedding time, retrieval time, context-construction time, prompt-processing time, TTFT, decode time, and end-to-end latency.
- Prompt-processing tokens/second and decode tokens/second.
- PSS, native heap, Java heap, and other available memory metrics, clearly distinguished.
- Temperature, CPU frequencies, and GPU utilization when reliable measurements are available.
- Answer quality and retrieval correctness.

Use fixed prompts and document corpora, deterministic settings where supported, controlled maximum output tokens, repeated trials, and clearly documented warm-up procedures.

Use multiple representative workloads rather than relying on one short query and one PDF.

Separate at least these cases:

- Cold start versus warm inference.
- Short versus long prompts.
- Short versus long generated responses.
- Small versus larger retrieval contexts.
- Idle/cool versus sustained-use/warmer device conditions.

Report medians, variability, sample counts, and confidence intervals or appropriate statistical comparisons. Avoid cherry-picking the fastest run.

If a benchmark cannot be reproduced, identify why and document the discrepancy instead of silently replacing the original number.

## 6. Third Requirement: Identify a Defensible Research Contribution

Generate several candidate research directions only after reviewing the existing work and validating the bottlenecks.

Possible starting hypotheses—not predetermined solutions—include the following.

### Candidate A: Resource- and query-adaptive inference control

Investigate whether a lightweight controller can choose inference settings based on observable device state and query requirements.

Potential inputs:

- Prompt token count and expected generation length.
- Retrieval relevance or confidence.
- Available memory and memory pressure.
- CPU load and observed frequency behavior.
- Thermal status.
- Battery or energy constraints, if measurable.
- Recent prompt-processing and decode performance.

Potential decisions:

- Retrieval-context or prompt-token budget.
- Supported thread configuration.
- Model or quantization choice from a predefined set.
- Execution backend, but only if multiple compatible backends are genuinely available.
- Output-token budget or an optional smaller model for latency-sensitive queries.

Potential objective:

Minimize latency and/or energy while satisfying explicit constraints on answer quality, memory usage, and reliability.

However, many adaptive schedulers already exist. Determine whether this specific combination offers a real research gap. A collection of simple if/else rules is not automatically a novel algorithm.

### Candidate B: Separate optimization of prompt prefill and token decode

Our preliminary results suggest that both prompt processing and autoregressive decoding are expensive, with prompt processing contributing more absolute time in the representative run.

Investigate whether the best configuration differs between these phases. Evaluate whether a supported runtime allows different thread settings, kernels, execution paths, or other optimizations for the two phases without introducing expensive reinitialization or synchronization.

Do not assume the runtime supports independent phase-level control. Inspect llama.cpp and llama.rn implementation details first.

Quantify the benefit, switching overhead, and impact on answer quality.

### Candidate C: Retrieval-aware prompt-budget control

Investigate whether a query-dependent retrieval strategy can reduce prompt-processing cost while preserving the evidence required to answer accurately.

Compare fixed Top-K retrieval against alternative approaches, such as relevance thresholds, token-budgeted selection, redundancy removal, and query-specific chunk selection.

This is only a research candidate. Adaptive retrieval and context compression already exist. Determine what, if anything, would be distinct and experimentally valuable in the mobile setting.

Do not optimize latency by simply dropping useful context and then report speed without measuring answer quality.

### Candidate D: Mobile CPU/GPU/NPU execution strategy

Determine whether an alternative execution backend can actually improve performance on the Snapdragon 4 Gen 2 and Adreno 613.

Investigate:

- llama.cpp CPU performance and supported Vulkan/OpenCL backends.
- Current Adreno driver and backend compatibility.
- Qualcomm QNN/AI Engine Direct and Hexagon NPU possibilities.
- Other realistic Android inference runtimes.

Do not assume a model in GGUF format can be sent directly to a Qualcomm NPU. Model conversion, supported operators, quantization formats, integration work, and deployment requirements may be substantial.

Do not assume that GPU offloading or NPU execution will be faster than CPU inference. Measure performance for prefill and decode separately.

If a backend is unsupported or impractical on the target phone, document that result and move on instead of spending the entire project attempting to force compatibility.

### Candidate E: Alternative mobile-friendly model architectures

Investigate whether Mamba, RetNet, or another architecture has a realistic inference advantage for the target workload.

Compare model size, architecture, available pretrained checkpoints, runtime maturity, quantization support, memory behavior, and actual Android deployment feasibility.

Do not compare an optimized small model against a larger baseline without accounting for model capacity and answer quality.

Changing architecture is not automatically a contribution, and replacing a Transformer may require changing the inference engine and model format.

### Candidate F: Thermal- and resource-aware inference

Investigate whether performance degradation during sustained inference can be predicted and mitigated through workload adaptation.

Measure repeated inference over time and identify whether thermal state, memory pressure, or CPU-frequency changes correlate with latency.

Test whether adaptation improves sustained throughput or energy without unacceptable answer-quality loss.

Do not label ordinary temperature monitoring plus fixed thresholds as novel unless the literature review identifies a defensible distinction and experiments support it.

### Required candidate evaluation

For each candidate, estimate:

- Novelty potential based on prior work.
- Expected latency improvement.
- Potential energy and memory benefits.
- Impact on answer quality.
- Feasibility on my exact device.
- Implementation complexity.
- Required hardware, model, and runtime support.
- Evaluation difficulty.
- Risks and possible reasons the idea will fail.
- Likelihood of supporting a credible IEEE paper.

Select the most promising direction based on evidence, not excitement. You may combine techniques if there is a clear justification, but do not create an unnecessarily complicated system.

It is acceptable to conclude that a candidate is already well explored, infeasible on this phone, or unlikely to provide a meaningful benefit.

## 7. Fourth Requirement: Design Controlled Experiments

Before implementing a major optimization, design experiments that isolate its effect.

At minimum, investigate these factors in a controlled sequence:

1. Prompt-token scaling, using matched workloads and recorded token counts.
2. Thread-count sweep, for example 1, 2, 4, 6, and 8 threads where supported.
3. Context-budget and retrieval-strategy comparisons.
4. Model-size and quantization comparisons.
5. Available CPU/GPU/NPU backend comparisons, if feasible.
6. Sustained-run and thermal-state effects.
7. Combined strategies only after the individual effects are understood.

Change one major variable at a time during initial experiments.

For every experiment, define:

- Hypothesis.
- Independent and dependent variables.
- Controlled conditions.
- Baseline.
- Number of repetitions and stopping criteria.
- Expected outcomes.
- Statistical analysis.
- Answer-quality evaluation.
- Failure criteria.

Include experiments that could disprove the proposed idea. If the controller does not outperform the best fixed configuration under matched conditions, report that honestly.

## 8. Fifth Requirement: Implement Only the Best-Validated Approach

After the literature review, baseline validation, and experiment design, recommend a concrete implementation plan.

Do not immediately rewrite the application or replace the inference engine.

Preserve the current application and baseline so results remain reproducible.

Implement the smallest experiment that can answer the most important research question. Then expand only if results justify it.

Requirements:

- Keep the original baseline configuration available.
- Make experimental parameters configurable.
- Record every configuration with its results.
- Avoid unverified or unsupported backend flags.
- Handle model-loading failures and memory exhaustion safely.
- Do not silently fall back to a different backend or configuration without recording it.
- Keep profiling overhead separate from normal inference latency.
- Verify the exact implementation against the source code and actual device.
- Provide reproducible commands, scripts, logs, and result files.
- Use the existing benchmark infrastructure when appropriate.
- Do not claim success merely because the app compiles or generates an answer.

If you have access to the repository, inspect it before editing. First produce a short audit of the existing code, benchmark files, and measurement reliability. Then propose the research plan and begin implementation only when the first controlled experiment is clear.

If repository access or device execution is unavailable, explicitly distinguish verified results from hypotheses and provide executable instructions for the experiments that require my device.

## 9. Evaluation Metrics and Publication-Quality Evidence

Evaluate more than latency.

### Performance

- Cold-start and model-load latency.
- TTFT.
- Prompt-processing latency and tokens/second.
- Decode latency and tokens/second.
- End-to-end query latency.
- Sustained throughput.

### Resource use

- PSS and native memory measurements.
- Peak memory and memory pressure.
- CPU/GPU utilization where reliably measurable.
- Thermal behavior and frequency changes.
- Energy per query, if a defensible measurement method is available.
- Battery impact for sustained workloads.

### RAG and answer quality

- Retrieval Recall@K or other appropriate retrieval metrics.
- Context relevance and evidence coverage.
- Answer correctness and groundedness.
- Task success rate.
- Quality degradation versus the baseline.

Use a documented evaluation dataset with multiple queries and document types. Include questions that require combining evidence, questions with insufficient evidence, and questions for which irrelevant retrieved chunks should not affect the answer.

Use reproducible automated metrics where valid and human evaluation or a carefully controlled model-based judge where necessary. Explain limitations of each metric.

Report trade-offs rather than presenting only the metric that improved.

## 10. Research Quality and Novelty Requirements

The intended output is a research paper that may be submitted to an appropriate IEEE venue.

Therefore:

- Clearly distinguish a new algorithm from an engineering integration or empirical study.
- Support novelty claims with the literature review.
- Identify the closest competing approaches.
- Explain the technical difference from existing work.
- Demonstrate the difference experimentally.
- Report negative results and limitations.
- Use proper academic citations.
- Do not fabricate benchmarks, results, references, or statistical significance.
- Do not claim universal mobile-device improvements from a single smartphone.
- If only one physical device is available, describe the scope honestly and use additional devices only if genuinely tested.
- Avoid unsupported claims such as "first ever," "state of the art," or "significant improvement."

If the most defensible contribution is a carefully evaluated mobile inference technique rather than a fundamentally new algorithm, explain how to position it as a rigorous systems paper. If it is not strong enough for the target venue, say so and explain what evidence is missing.

## 11. Future Extension: Local Agent Interaction

The broader vision is a private, offline Android assistant that can eventually interact with lightweight modules such as calendar operations, alarms, reminders, calls, and document tools.

However, **the immediate research priority is inference efficiency, not building more agent integrations**.

Do not let tool calling, user-interface work, or application features distract from the core bottleneck.

You may consider whether a future extension would benefit from an inference controller that allocates resources differently for tool-selection queries, structured extraction, RAG, and ordinary conversation. Investigate whether this adds a genuine research contribution, but do not implement it before the core inference experiments succeed.

Keep future agent interaction as a potential extension unless the literature review shows that a narrowly scoped agent workload is essential to validate the inference contribution.

## 12. Required Deliverables

Produce the following in order.

### Deliverable 1: Repository and baseline audit

- Actual architecture and configuration.
- Confirmed benchmark results.
- Measurement inconsistencies and missing instrumentation.
- Bottleneck analysis.
- A reproducible baseline procedure.

### Deliverable 2: Literature review and research-gap matrix

- Relevant papers with verified references.
- Closest existing approaches.
- Comparison table.
- Explicit assessment of what is already known and what remains uncertain.
- Candidate research gaps and their evidence.

### Deliverable 3: Ranked research proposals

Provide at least three credible candidates, if three survive the initial screening. Rank them by defensible novelty, measurable potential, feasibility, and publication value.

For each, explain what would make it distinct from existing work and what result would invalidate it.

### Deliverable 4: Recommended research question and hypothesis

Select one primary direction and define:

- Research question.
- Technical contribution.
- Formal or algorithmic description, if appropriate.
- Baseline methods.
- Experimental variables.
- Success criteria.
- Expected limitations.

Do not force a novel algorithm if the evidence favors another kind of contribution.

### Deliverable 5: Experiment implementation

Provide the necessary source-code changes, benchmark scripts, configurations, and instructions for physical-device execution.

Preserve baseline reproducibility.

### Deliverable 6: Experimental results

Include measured results, comparisons, statistical analysis, answer-quality trade-offs, limitations, and negative results. Do not fill in results before running experiments.

### Deliverable 7: IEEE paper preparation

Only after the core results are available, prepare a paper structure with:

- Abstract.
- Introduction and motivation.
- Related work.
- System model and problem formulation.
- Proposed method.
- Experimental setup.
- Results and discussion.
- Threats to validity.
- Limitations.
- Conclusion and future work.

Ensure every contribution claim maps to actual implementation and experimental evidence.

## 13. Final Instructions

Start with the repository audit and targeted literature review. Do not jump directly into optimization.

Be skeptical of my hypotheses and your own proposals. Prefer a small, measurable, reproducible improvement over a sophisticated but unsupported architecture.

The final goal is not to claim that we optimized everything. It is to discover and demonstrate something useful that improves local LLM inference under real mobile constraints, explain why it works, and establish whether the contribution is sufficiently distinct and rigorous for an IEEE publication.

If the evidence does not support a novel algorithm, tell me. Recommend the strongest realistic research path rather than manufacturing novelty.
The best initial hypothesis is that a controller using actual device measurements, query characteristics, and RAG context requirements can outperform a carefully tuned fixed configuration under real mobile constraints. Whether this is novel enough for IEEE publication remains an open question until the literature review and experiments establish it.