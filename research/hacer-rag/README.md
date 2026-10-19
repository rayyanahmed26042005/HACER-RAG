# HACER-RAG Phase 1 host reference

This standalone package implements the first falsifiable prototype milestone
from [`research/HACER_RAG_RESEARCH_BLUEPRINT.md`](../HACER_RAG_RESEARCH_BLUEPRINT.md):

- deterministic cache validity keys;
- exact-prefix reuse metadata and mismatch reasons;
- scalar RoPE key rotation and metadata validation;
- HACERKV-aligned cache manifest validation and Flash page accounting;
- a host-side hardware-parameterized action-cost scheduler with conservative fallback;
- versioned, atomic reference traces;
- host-only microbenchmarks for the prototype overhead.

It is intentionally isolated from the React Native application, default RAG
path, Android telemetry, and frozen MobCert implementation. It does **not**
export or import llama.cpp KV tensors yet. `llama.rn` exposes session
save/load and cached-token counts, but not the per-layer KV graph required for
HACER's approximate chunk-cache experiments. That native adapter is a later
phase and must use a separate research harness.

The cache manifest and scheduler modules are scalar host references. They define
the schema, alignment, validity, read-amplification, cost, and fallback contracts
that a later native adapter must satisfy; they are not measurements of Android
Flash, SIMD throughput, or model quality.

## Run

From the repository root:

```bash
node --test research/hacer-rag/tests/*.test.mjs
node research/hacer-rag/bench/phase1_benchmark.mjs
```

Or from this directory:

```bash
npm test
npm run benchmark
```

The benchmark reports host reference overhead only. It must not be copied into
the paper's mobile latency, energy, or answer-quality tables.

## Validity model

The key includes model, tokenizer, prompt template, architecture, position
scheme, attention mask, quantization, tensor layout, chunk content, assembly
order, local position origin, and namespace metadata. A change in any field
produces a different SHA-256 digest and a structured mismatch path.

Independent chunk caches are never labeled `exact` by this package. Exact
prefix reuse requires a matching validity digest, template hash, and token-ID
hash for the cached prefix. RoPE rotation changes Keys only; Values are not
rotated.

## Next native milestone

After these tests pass, add a separate native/session adapter that records:

1. full-prefill reference logits;
2. exact-prefix session reuse;
3. `tokens_cached` and session-load failures;
4. prompt/token hashes and fallback reasons;
5. per-run traces using this schema.

Only after that adapter is validated should the project attempt independently
encoded chunk KV import or selective cross-chunk recomputation.

## New research modules

- `src/cacheContainer.mjs`: deterministic HACERKV metadata, aligned page
  validation, hash-based invalidation, and read-amplification accounting.
- `src/scheduler.mjs`: device-state/action cost estimates, feasible-action
  enumeration, quality/memory/latency guards, and full-prefill fallback.

These modules are intentionally testable without ADB or a phone. A real mobile
claim still requires native tensor export/import, calibrated Flash/dequantization
rates, per-layer traces, and paired physical runs.
