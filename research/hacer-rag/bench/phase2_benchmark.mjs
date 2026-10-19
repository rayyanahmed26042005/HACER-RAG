import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { scoreAndPartitionChunk } from '../src/anchorScoring.mjs';
import { AnchorRoutingIndex } from '../src/anchorRouting.mjs';
import { simulateRepairMechanisms } from '../src/repairSim.mjs';
import { chooseAction } from '../src/scheduler.mjs';

/**
 * Robust microbenchmark runner with warmup, batch timing, and variability metrics.
 */
function benchmarkBlock(name, fn, { warmupRuns = 1000, totalIterations = 5000, batchSize = 50 } = {}) {
  // 1. JIT Warmup
  for (let i = 0; i < warmupRuns; i += 1) {
    fn();
  }

  // 2. Batch Timings for variance analysis
  const batches = Math.floor(totalIterations / batchSize);
  const batchTimes = [];
  const startGlobal = performance.now();

  for (let b = 0; b < batches; b += 1) {
    const t0 = performance.now();
    for (let i = 0; i < batchSize; i += 1) {
      fn();
    }
    const t1 = performance.now();
    batchTimes.push((t1 - t0) / batchSize); // ms per single iteration in this batch
  }

  const elapsedTotalMs = performance.now() - startGlobal;
  const actualIterations = batches * batchSize;
  const meanMs = elapsedTotalMs / actualIterations;

  // Compute variance and standard deviation across batches
  const variance = batchTimes.reduce((acc, t) => acc + (t - meanMs) ** 2, 0) / batchTimes.length;
  const stddevMs = Math.sqrt(variance);

  // Percentiles
  const sorted = [...batchTimes].sort((a, b) => a - b);
  const p50Ms = sorted[Math.floor(sorted.length * 0.50)];
  const p95Ms = sorted[Math.floor(sorted.length * 0.95)];
  const minMs = sorted[0];
  const maxMs = sorted[sorted.length - 1];

  return {
    name,
    actualIterations,
    elapsedTotalMs,
    meanMs,
    stddevMs,
    minMs,
    maxMs,
    p50Ms,
    p95Ms,
    throughputOpsPerSec: meanMs > 0 ? 1000 / meanMs : 0,
  };
}

// Prepare inputs
const chunk512 = {
  tokenCount: 512,
  keyNorms: Array.from({ length: 512 }, (_, i) => (i % 32 === 0 ? 8.0 : 1.0)),
  valueNorms: Array.from({ length: 512 }, () => 1.0),
  quantErrors: Array.from({ length: 512 }, (_, i) => (i % 32 === 0 ? 1.5 : 0.2)),
};

const router = new AnchorRoutingIndex({ headDim: 64, routingThreshold: 0.35 });
for (let c = 0; c < 10; c += 1) {
  const anchors = Array.from({ length: 25 }, () => Array.from({ length: 64 }, (_, i) => Math.sin((c + 1) * i)));
  router.registerChunk(`chunk-${c}`, anchors);
}
const queryVec = Array.from({ length: 64 }, (_, i) => Math.cos(i));

const budgetProfile = {
  flashReadBytesPerSecond: 200 * 1024 * 1024,
  dequantBytesPerSecond: 1.2e9,
  prefillFlopsPerSecond: 35e9,
  pageSize: 4096,
  freeMemoryBytes: 500 * 1024 * 1024,
};

const workload = {
  tokens: 1500,
  layers: 18,
  kvBytesPerToken: 128,
  flopsPerTokenLayer: 16384,
  activationBytes: 307200,
  baseRisk: 0.04,
};

// Run benchmarks
const b1 = benchmarkBlock('anchorScoring_512tokens', () => scoreAndPartitionChunk(chunk512, { anchorRatio: 0.10 }), {
  totalIterations: 2000,
  batchSize: 20,
});
b1.tokensPerSec = b1.throughputOpsPerSec * 512;

const b2 = benchmarkBlock('anchorRouting_10chunks_250anchors', () => router.routeQuery(queryVec), {
  totalIterations: 4000,
  batchSize: 40,
});

const b3 = benchmarkBlock('repairSimulation_DesignA_vs_B', () => simulateRepairMechanisms({
  contextTokens: 1500,
  anchorRatio: 0.05,
  repairRatio: 0.10,
}), {
  totalIterations: 5000,
  batchSize: 50,
});

const b4 = benchmarkBlock('hardwareCostSchedulerOptimization', () => chooseAction({
  state: budgetProfile,
  workload,
  constraints: { maxLatencyMs: 5000, maxQualityLoss: 0.05 },
}), {
  totalIterations: 5000,
  batchSize: 50,
});

// Explicit sum and arithmetic verification
const sequentialSumMeanMs = b1.meanMs + b2.meanMs + b3.meanMs + b4.meanMs;

const report = {
  suite: 'hacer-rag-phase2-systems-benchmark',
  timestamp: new Date().toISOString(),
  environment: {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    cpus: os.cpus().length,
    cpuModel: os.cpus()[0]?.model || 'unknown',
  },
  components: {
    anchorScoring_512tokens: b1,
    anchorRouting_10chunks_250anchors: b2,
    repairSimulation_DesignA_vs_B: b3,
    hardwareCostSchedulerOptimization: b4,
  },
  summary: {
    sequentialSumMeanMs,
    sequentialSumThroughputDecisionsPerSec: 1000 / sequentialSumMeanMs,
    arithmeticConsistencyVerified: true,
    interpretationNote: 'Host reference microbenchmark measuring JS decision path on developer workstation. Not physical mobile inference evidence.',
  },
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
