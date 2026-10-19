import test from 'node:test';
import assert from 'node:assert/strict';
import { scoreAndPartitionChunk, computeSinkFactor } from '../src/anchorScoring.mjs';
import { AnchorRoutingIndex } from '../src/anchorRouting.mjs';
import { simulateRepairMechanisms } from '../src/repairSim.mjs';
import { chooseAction, estimateActionCost } from '../src/scheduler.mjs';

test('computeSinkFactor generates monotonic exponential decay', () => {
  const f0 = computeSinkFactor(0);
  const f1 = computeSinkFactor(1);
  const f5 = computeSinkFactor(5);

  assert.equal(f0, 1.0);
  assert.ok(f1 < f0);
  assert.ok(f5 < f1);
  assert.equal(computeSinkFactor(-1), 0);
});

test('scoreAndPartitionChunk correctly selects anchors and preserves partitions', () => {
  const tokenCount = 100;
  // Synthetic key norms: token 0 (sink), token 42 (outlier), rest normal
  const keyNorms = Array.from({ length: tokenCount }, (_, i) => (i === 42 ? 10.0 : 1.0));
  const valueNorms = Array.from({ length: tokenCount }, () => 1.0);
  const quantErrors = Array.from({ length: tokenCount }, (_, i) => (i === 42 ? 2.0 : 0.1));

  const result = scoreAndPartitionChunk(
    { tokenCount, keyNorms, valueNorms, quantErrors },
    { anchorRatio: 0.10 } // 10% anchors = 10 tokens
  );

  assert.equal(result.tokenCount, 100);
  assert.equal(result.anchorCount, 10);
  assert.equal(result.bulkCount, 90);
  assert.equal(result.anchorIndices.length, 10);
  assert.equal(result.bulkIndices.length, 90);

  // Both token 0 (sink) and token 42 (norm outlier) must be selected as anchors
  assert.ok(result.anchorIndices.includes(0), 'Token 0 (sink) should be an anchor');
  assert.ok(result.anchorIndices.includes(42), 'Token 42 (norm outlier) should be an anchor');

  // Verify no overlap between anchors and bulk tokens
  for (const idx of result.anchorIndices) {
    assert.ok(!result.bulkIndices.includes(idx), `Token ${idx} should not appear in both sets`);
  }
});

test('AnchorRoutingIndex correctly routes query and calculates traffic reduction', () => {
  const router = new AnchorRoutingIndex({ headDim: 4, routingThreshold: 0.40, backgroundThreshold: 0.15 });

  // Chunk 1: Keys aligned with [1, 0, 0, 0] (Relevant)
  router.registerChunk(
    'chunk-doc1-sec2',
    [
      [1.0, 0.05, 0.0, 0.0],
      [0.9, 0.1, 0.0, 0.0],
    ],
    [0, 5]
  );

  // Chunk 2: Keys orthogonal [-0.1, 0.0, 1.0, 0.0] (Irrelevant)
  router.registerChunk(
    'chunk-unrelated',
    [
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.9, 0.1],
    ],
    [0, 10]
  );

  // Query vector strongly pointing towards dimension 0
  const query = [1.0, 0.0, 0.0, 0.0];
  const decisions = router.routeQuery(query);

  assert.equal(decisions.length, 2);
  assert.equal(decisions[0].chunkId, 'chunk-doc1-sec2');
  assert.equal(decisions[0].decision, 'STREAM_HIGH');
  assert.equal(decisions[0].recommendedBitWidth, 4);

  assert.equal(decisions[1].chunkId, 'chunk-unrelated');
  assert.equal(decisions[1].decision, 'BYPASS'); // Skipped with 0 Flash I/O!
  assert.equal(decisions[1].recommendedBitWidth, 0);

  const traffic = router.calculateTrafficReduction(decisions, 256, 128);
  assert.ok(traffic.reductionRatio > 0.70, 'Routing should save >70% Flash traffic by bypassing unrelated chunks');
  assert.equal(traffic.bypassedChunks, 1);
  assert.equal(traffic.highPriorityChunks, 1);
});

test('simulateRepairMechanisms evaluates Design A vs Design B and crossover point', () => {
  const sim = simulateRepairMechanisms({
    contextTokens: 1500,
    anchorRatio: 0.05,  // 5% anchors: 75 tokens * 12 layers = 900 token-layers
    repairRatio: 0.10,  // 10% sparse repair: 150 tokens * 18 layers = 2700 token-layers
    model: { layers: 18, boundaryLayer: 6, hiddenDim: 2048 },
    device: {
      flashReadBps: 200 * 1024 * 1024, // 200 MB/s (UFS 2.2 budget)
      prefillFlopsPerSec: 35e9,        // 35 GFLOPs budget CPU
    },
  });

  const full = sim.comparisons.fullPrefill;
  const designA = sim.comparisons.designA;
  const designB = sim.comparisons.designB;

  // Design A bypasses 6 layers for anchors: executes 12 layers for 75 tokens
  assert.ok(designA.totalFlops < full.totalFlops);
  assert.ok(designA.flopReductionVsFull > 0.90, 'Design A should reduce prefill FLOPs by >90%');
  assert.ok(designA.activationFlashBytes > 0, 'Design A requires caching activations');

  // Design B executes all 18 layers for 150 tokens
  assert.ok(designB.totalFlops < full.totalFlops);
  assert.equal(designB.activationFlashBytes, 0, 'Design B has zero activation Flash I/O');

  // Both should be vastly faster than full prefill
  assert.ok(designA.totalTimeMs < full.totalTimeMs);
  assert.ok(designB.totalTimeMs < full.totalTimeMs);
  assert.ok(Number.isFinite(sim.crossoverBandwidthMBps), 'Crossover bandwidth should be finite when Design A compute < Design B compute');
});

test('Hardware cost scheduler chooses feasible action and falls back safely under constraints', () => {
  const budgetState = {
    flashReadBytesPerSecond: 200 * 1024 * 1024,
    dequantBytesPerSecond: 1.2e9,
    prefillFlopsPerSecond: 35e9,
    pageSize: 4096,
    freeMemoryBytes: 500 * 1024 * 1024, // 500 MB free RAM
  };

  const workload = {
    tokens: 1500,
    layers: 18,
    kvBytesPerToken: 128,
    flopsPerTokenLayer: 16384,
    activationBytes: 307200,
    baseRisk: 0.04,
  };

  // Normal constraints: scheduler should pick a fast low-bit sparse action
  const normalChoice = chooseAction({
    state: budgetState,
    workload,
    constraints: { maxLatencyMs: 5000, maxQualityLoss: 0.05 },
  });

  assert.equal(normalChoice.fallback, false);
  assert.ok(normalChoice.selected.repairRatio <= 0.20);
  assert.ok(normalChoice.cost.predictedMs < 5000);

  // Impossible constraint (e.g. quality loss < 0.001 when min risk is 0.005): scheduler must trigger fallback
  const impossibleChoice = chooseAction({
    state: budgetState,
    workload,
    constraints: { maxLatencyMs: 0.0001, maxQualityLoss: 0.001 },
  });

  assert.equal(impossibleChoice.fallback, true);
  assert.equal(impossibleChoice.selected.design, 'full');
});
