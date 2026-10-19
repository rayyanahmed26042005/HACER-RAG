/**
 * repairSim.mjs
 *
 * Implements the theoretical and experimental simulation of Hypothesis H2:
 * Comparing Design A (Boundary Activation Caching) vs. Design B (All-Layer Ultra-Sparse Recomputation).
 */

export const DEFAULT_MODEL_PARAMS = Object.freeze({
  layers: 18,
  hiddenDim: 2048,
  kvHeads: 4,
  headDim: 64,
  boundaryLayer: 6, // L_local: shallow layers 0...5 bypassed in Design A
});

/**
 * Simulate FLOPs, Flash I/O, and latency for Design A vs. Design B.
 *
 * @param {Object} params
 * @param {number} params.contextTokens - Total tokens in retrieved chunks (e.g. 1500)
 * @param {number} params.anchorRatio - Fraction alpha of anchor tokens (default 0.10)
 * @param {number} [params.repairRatio] - Recomputation token ratio r (default 0.05)
 * @param {Object} [params.model] - Model architectural parameters
 * @param {Object} [params.device] - Device hardware rates (bandwidth, throughput)
 * @returns {Object} Comparative metrics across all repair designs
 */
export function simulateRepairMechanisms(params) {
  const contextTokens = params.contextTokens || 1500;
  const anchorRatio = params.anchorRatio || 0.10;
  const repairRatio = params.repairRatio || 0.05;
  const model = { ...DEFAULT_MODEL_PARAMS, ...(params.model || {}) };
  const L = model.layers;
  const L_local = model.boundaryLayer;
  const D = model.hiddenDim;

  // Device hardware profile
  const device = {
    flashReadBps: params.device?.flashReadBps || 200 * 1024 * 1024, // 200 MB/s (UFS 2.2)
    prefillFlopsPerSec: params.device?.prefillFlopsPerSec || 35e9,  // 35 GFLOPs (budget CPU)
    dequantBps: params.device?.dequantBps || 1.2e9,                 // 1.2 GB/s SIMD NEON
  };

  // FLOPs per token per layer: ~ 4 * D^2 (attention projections + FFN)
  const flopsPerTokenLayer = 4 * D * D;
  const denseFullPrefillFlops = contextTokens * L * flopsPerTokenLayer;

  // Baseline 0: Full Prefill (100% compute)
  const fullPrefill = {
    mode: 'FULL_PREFILL',
    recomputedTokens: contextTokens,
    recomputedFraction: 1.0,
    layersExecuted: L,
    totalFlops: denseFullPrefillFlops,
    activationFlashBytes: 0,
    computeLatencyMs: (denseFullPrefillFlops / device.prefillFlopsPerSec) * 1000,
    activationIoLatencyMs: 0,
    totalTimeMs: (denseFullPrefillFlops / device.prefillFlopsPerSec) * 1000,
    multiHopRecoveryEstimate: 1.0, // Ground truth 100%
  };

  // Baseline 1: TurboRAG / Naive Reuse (r = 0%)
  const naiveReuse = {
    mode: 'NAIVE_REUSE',
    recomputedTokens: 0,
    recomputedFraction: 0,
    layersExecuted: 0,
    totalFlops: 0,
    activationFlashBytes: 0,
    computeLatencyMs: 0,
    activationIoLatencyMs: 0,
    totalTimeMs: 0,
    multiHopRecoveryEstimate: 0.45, // Degraded multi-hop accuracy
  };

  // Design A: Boundary Activation Caching
  // Anchors only (alpha = 10%) recomputed from layer L_local to L-1
  const anchorTokens = Math.ceil(contextTokens * anchorRatio);
  const designAFlops = anchorTokens * (L - L_local) * flopsPerTokenLayer;
  // Caching INT8 intermediate activations at layer L_local-1 for anchor tokens
  const designAActivationBytes = anchorTokens * D * 1; // 1 byte per activation (INT8)
  const designAComputeMs = (designAFlops / device.prefillFlopsPerSec) * 1000;
  const designAIoMs = (designAActivationBytes / device.flashReadBps) * 1000;

  const designA = {
    mode: 'DESIGN_A_BOUNDARY_CACHE',
    recomputedTokens: anchorTokens,
    recomputedFraction: anchorTokens / contextTokens,
    layersExecuted: L - L_local,
    totalFlops: designAFlops,
    activationFlashBytes: designAActivationBytes,
    computeLatencyMs: designAComputeMs,
    activationIoLatencyMs: designAIoMs,
    totalTimeMs: designAComputeMs + designAIoMs,
    multiHopRecoveryEstimate: 0.88, // Recovers cross-chunk attention via deep layers
    flopReductionVsFull: (denseFullPrefillFlops - designAFlops) / denseFullPrefillFlops,
  };

  // Design B: All-Layer Ultra-Sparse Recomputation
  // Recomputes top r% (e.g. 5%) tokens across all L layers
  const sparseTokens = Math.ceil(contextTokens * repairRatio);
  const designBFlops = sparseTokens * L * flopsPerTokenLayer;
  const designBComputeMs = (designBFlops / device.prefillFlopsPerSec) * 1000;

  const designB = {
    mode: 'DESIGN_B_ALL_LAYER_SPARSE',
    recomputedTokens: sparseTokens,
    recomputedFraction: sparseTokens / contextTokens,
    layersExecuted: L,
    totalFlops: designBFlops,
    activationFlashBytes: 0, // Zero extra Flash I/O!
    computeLatencyMs: designBComputeMs,
    activationIoLatencyMs: 0,
    totalTimeMs: designBComputeMs,
    multiHopRecoveryEstimate: 0.85, // Recovers across all layers for sparse set
    flopReductionVsFull: (denseFullPrefillFlops - designBFlops) / denseFullPrefillFlops,
  };

  // Determine Crossover Bandwidth where Design A matches Design B
  // Design A wins when: Time_A < Time_B
  // Compute_A + Act_IO_A < Compute_B
  // Act_IO_A < Compute_B - Compute_A
  const deltaComputeMs = designBComputeMs - designAComputeMs;
  let crossoverBandwidthBps = null;
  if (deltaComputeMs > 0) {
    crossoverBandwidthBps = (designAActivationBytes / (deltaComputeMs / 1000));
  }

  return {
    contextTokens,
    model: { layers: L, boundaryLayer: L_local, hiddenDim: D },
    device,
    comparisons: {
      fullPrefill,
      naiveReuse,
      designA,
      designB,
    },
    crossoverBandwidthBps,
    crossoverBandwidthMBps: crossoverBandwidthBps ? crossoverBandwidthBps / (1024 * 1024) : null,
    recommendation: designA.totalTimeMs < designB.totalTimeMs ? 'DESIGN_A' : 'DESIGN_B',
  };
}
