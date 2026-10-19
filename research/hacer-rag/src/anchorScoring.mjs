/**
 * anchorScoring.mjs
 *
 * Implements Hypothesis H1 (Split-Precision Anchor Subspace Partitioning)
 * from HACER_RAG_RESEARCH_BLUEPRINT.md:
 *
 * S_anchor(t) = w1 * Phi_sink(t) + w2 * alpha_intra(t) + w3 * kappa_norm(t) + w4 * sigma_quant(t)
 */

export const DEFAULT_ANCHOR_WEIGHTS = Object.freeze({
  w_sink: 0.25,
  w_intra: 0.35,
  w_norm: 0.20,
  w_quant: 0.20,
});

export const DEFAULT_ANCHOR_RATIO = 0.10; // Top 10% tokens assigned to high precision

/**
 * Compute the attention sink factor for token position t (0-indexed).
 * Early tokens absorb disproportionate attention mass.
 */
export function computeSinkFactor(t, lambda = 0.5) {
  if (t < 0) return 0;
  return Math.exp(-lambda * t);
}

/**
 * Compute composite anchor saliency score for each token in a precomputed chunk.
 *
 * @param {Object} chunkData
 * @param {number} chunkData.tokenCount - Total tokens in chunk N_k
 * @param {Array<Array<number>>} [chunkData.intraAttention] - Optional N_k x N_k intra-chunk attention matrix
 * @param {Array<number>} [chunkData.keyNorms] - L2 norm of key vector per token
 * @param {Array<number>} [chunkData.valueNorms] - L2 norm of value vector per token
 * @param {Array<number>} [chunkData.quantErrors] - Measured quantization L2 error per token
 * @param {Object} [options]
 * @param {Object} [options.weights] - Component weights summing to 1.0
 * @param {number} [options.anchorRatio] - Fraction alpha of tokens selected as anchors (default 0.10)
 * @returns {Object} Partitioned token indices and detailed scores
 */
export function scoreAndPartitionChunk(chunkData, options = {}) {
  const tokenCount = chunkData?.tokenCount;
  if (!Number.isInteger(tokenCount) || tokenCount <= 0) {
    throw new TypeError('chunkData.tokenCount must be a positive integer');
  }

  const weights = { ...DEFAULT_ANCHOR_WEIGHTS, ...(options.weights || {}) };
  const anchorRatio = Math.max(0.01, Math.min(1.0, options.anchorRatio ?? DEFAULT_ANCHOR_RATIO));

  // 1. Max norms for scaling
  const maxKeyNorm = chunkData.keyNorms ? Math.max(...chunkData.keyNorms, 1e-6) : 1.0;
  const maxValueNorm = chunkData.valueNorms ? Math.max(...chunkData.valueNorms, 1e-6) : 1.0;
  const maxQuantError = chunkData.quantErrors ? Math.max(...chunkData.quantErrors, 1e-6) : 1.0;

  const scores = [];

  for (let t = 0; t < tokenCount; t += 1) {
    // Sink factor
    const phiSink = computeSinkFactor(t);

    // Cumulative intra-chunk attention mass: sum_{j=t+1}^{N_k} A_{j, t}
    let alphaIntra = 0;
    if (chunkData.intraAttention && Array.isArray(chunkData.intraAttention)) {
      let sumAttn = 0;
      let countAttn = 0;
      for (let j = t + 1; j < tokenCount; j += 1) {
        if (chunkData.intraAttention[j] && typeof chunkData.intraAttention[j][t] === 'number') {
          sumAttn += chunkData.intraAttention[j][t];
          countAttn += 1;
        }
      }
      alphaIntra = countAttn > 0 ? sumAttn / Math.max(1, tokenCount - t - 1) : 0;
    } else {
      // Default geometric decay approximation if raw matrix is omitted
      alphaIntra = 1.0 / Math.sqrt(t + 1);
    }

    // Key-Value norm salience
    const keyNormRel = (chunkData.keyNorms?.[t] ?? 1.0) / maxKeyNorm;
    const valNormRel = (chunkData.valueNorms?.[t] ?? 1.0) / maxValueNorm;
    const kappaNorm = (keyNormRel + valNormRel) / 2.0;

    // Quantization sensitivity
    const sigmaQuant = (chunkData.quantErrors?.[t] ?? 0.5) / maxQuantError;

    // Composite Score
    const compositeScore =
      weights.w_sink * phiSink +
      weights.w_intra * alphaIntra +
      weights.w_norm * kappaNorm +
      weights.w_quant * sigmaQuant;

    scores.push({
      tokenIndex: t,
      compositeScore,
      components: {
        phiSink,
        alphaIntra,
        kappaNorm,
        sigmaQuant,
      },
    });
  }

  // Sort descending by composite score
  const ranked = [...scores].sort((a, b) => b.compositeScore - a.compositeScore);

  const anchorCount = Math.max(1, Math.floor(tokenCount * anchorRatio));
  const anchorIndices = new Set(ranked.slice(0, anchorCount).map(s => s.tokenIndex));
  const bulkIndices = new Set();

  for (let t = 0; t < tokenCount; t += 1) {
    if (!anchorIndices.has(t)) {
      bulkIndices.add(t);
    }
  }

  return {
    tokenCount,
    anchorCount,
    bulkCount: bulkIndices.size,
    anchorRatio: anchorCount / tokenCount,
    anchorIndices: Array.from(anchorIndices).sort((a, b) => a - b),
    bulkIndices: Array.from(bulkIndices).sort((a, b) => a - b),
    scores,
  };
}
