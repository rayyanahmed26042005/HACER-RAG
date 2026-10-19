/**
 * anchorRouting.mjs
 *
 * Implements the In-RAM Anchor Query Routing Index from HACER_RAG_RESEARCH_BLUEPRINT.md.
 * Evaluates query relevance using only resident anchor keys prior to issuing Flash I/O
 * for bulk chunk payloads.
 */

/**
 * Compute dot product of two equal-length arrays.
 */
function dotProduct(vecA, vecB) {
  let sum = 0;
  const len = Math.min(vecA.length, vecB.length);
  for (let i = 0; i < len; i += 1) {
    sum += vecA[i] * vecB[i];
  }
  return sum;
}

/**
 * Compute vector L2 norm.
 */
function l2Norm(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i += 1) {
    sum += vec[i] * vec[i];
  }
  return Math.sqrt(sum) || 1e-6;
}

/**
 * In-RAM Anchor Routing Engine
 */
export class AnchorRoutingIndex {
  constructor(options = {}) {
    this.headDim = options.headDim || 64;
    this.routingThreshold = options.routingThreshold ?? 0.35;
    this.backgroundThreshold = options.backgroundThreshold ?? 0.15;
    // Map of chunkId -> { anchorKeys: Array<Array<number>>, anchorTokenIndices: Array<number> }
    this.chunks = new Map();
  }

  /**
   * Register a chunk's resident anchor keys in RAM.
   *
   * @param {string} chunkId
   * @param {Array<Array<number>>} anchorKeys - Matrix of shape [n_anchors, head_dim]
   * @param {Array<number>} anchorTokenIndices - Original chunk token positions
   */
  registerChunk(chunkId, anchorKeys, anchorTokenIndices) {
    if (!chunkId || !Array.isArray(anchorKeys)) {
      throw new TypeError('chunkId and anchorKeys array required');
    }
    this.chunks.set(chunkId, {
      chunkId,
      anchorKeys,
      anchorTokenIndices: anchorTokenIndices || anchorKeys.map((_, i) => i),
      anchorCount: anchorKeys.length,
    });
  }

  /**
   * Evaluate a query vector against all resident anchor chunks in RAM.
   *
   * @param {Array<number>} queryVector - Query embedding or projected query key vector
   * @returns {Array<Object>} Sorted routing decisions per chunk
   */
  routeQuery(queryVector) {
    if (!Array.isArray(queryVector) || queryVector.length === 0) {
      throw new TypeError('queryVector must be a non-empty array');
    }

    const qNorm = l2Norm(queryVector);
    const results = [];

    for (const [chunkId, chunk] of this.chunks.entries()) {
      let maxScore = -Infinity;
      let sumScore = 0;

      for (let i = 0; i < chunk.anchorKeys.length; i += 1) {
        const aKey = chunk.anchorKeys[i];
        const dot = dotProduct(queryVector, aKey);
        const norm = l2Norm(aKey);
        const cosSim = dot / (qNorm * norm);

        if (cosSim > maxScore) maxScore = cosSim;
        sumScore += cosSim;
      }

      const meanScore = chunk.anchorKeys.length > 0 ? sumScore / chunk.anchorKeys.length : 0;
      // Combined routing metric: 70% peak anchor similarity + 30% aggregate anchor coverage
      const routingScore = 0.70 * Math.max(0, maxScore) + 0.30 * Math.max(0, meanScore);

      let decision = 'BYPASS'; // Zero Flash I/O
      let recommendedBitWidth = 0;

      if (routingScore >= this.routingThreshold) {
        decision = 'STREAM_HIGH'; // Load 4-bit bulk payload
        recommendedBitWidth = 4;
      } else if (routingScore >= this.backgroundThreshold) {
        decision = 'STREAM_LOW'; // Load compressed 2-bit bulk payload
        recommendedBitWidth = 2;
      }

      results.push({
        chunkId,
        routingScore,
        maxScore,
        meanScore,
        decision,
        recommendedBitWidth,
      });
    }

    // Sort descending by routing score
    results.sort((a, b) => b.routingScore - a.routingScore);
    return results;
  }

  /**
   * Calculate Flash I/O traffic savings given routing decisions.
   *
   * @param {Array<Object>} routingDecisions
   * @param {number} tokensPerChunk
   * @param {number} kvBytesPerToken - Baseline uncompressed FP16 bytes per token
   */
  calculateTrafficReduction(routingDecisions, tokensPerChunk = 256, kvBytesPerToken = 128) {
    const uncompressedBaselineBytes = routingDecisions.length * tokensPerChunk * kvBytesPerToken;
    let actualFlashBytes = 0;

    for (const r of routingDecisions) {
      if (r.decision === 'STREAM_HIGH') {
        // Bulk tokens loaded at 4-bit (4/16 = 0.25 of baseline)
        actualFlashBytes += tokensPerChunk * kvBytesPerToken * (4 / 16);
      } else if (r.decision === 'STREAM_LOW') {
        // Bulk tokens loaded at 2-bit (2/16 = 0.125 of baseline)
        actualFlashBytes += tokensPerChunk * kvBytesPerToken * (2 / 16);
      }
      // BYPASS chunks load 0 bulk bytes from Flash!
    }

    const savedBytes = uncompressedBaselineBytes - actualFlashBytes;
    const reductionRatio = uncompressedBaselineBytes > 0 ? savedBytes / uncompressedBaselineBytes : 0;

    return {
      uncompressedBaselineBytes,
      actualFlashBytes,
      savedBytes,
      reductionRatio,
      bypassedChunks: routingDecisions.filter(r => r.decision === 'BYPASS').length,
      highPriorityChunks: routingDecisions.filter(r => r.decision === 'STREAM_HIGH').length,
      lowPriorityChunks: routingDecisions.filter(r => r.decision === 'STREAM_LOW').length,
    };
  }
}
