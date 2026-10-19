const DEFAULT_RATIOS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 1];

function finitePositive(value, fallback) {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export function estimateActionCost({ state, action, workload }) {
  const tokens = Math.max(1, workload.tokens || 1);
  const layers = Math.max(1, workload.layers || 1);
  const bytesPerToken = action.bulkBits / 8 * (workload.kvBytesPerToken || 1);
  const pages = Math.max(1, Math.ceil(tokens * bytesPerToken / finitePositive(state.pageSize, 4096)));
  const flashBytes = pages * finitePositive(state.pageSize, 4096) * (action.pageFraction ?? 1);
  const dequantBytes = flashBytes * (action.bulkBits <= 4 ? 1 : 0.5);
  const recomputeTokens = Math.ceil(tokens * action.repairRatio);
  const recomputeFlops = recomputeTokens * layers * finitePositive(workload.flopsPerTokenLayer, 1);
  const flashMs = flashBytes / finitePositive(state.flashReadBytesPerSecond, 1) * 1000;
  const dequantMs = dequantBytes / finitePositive(state.dequantBytesPerSecond, 1) * 1000;
  const recomputeMs = recomputeFlops / finitePositive(state.prefillFlopsPerSecond, 1) * 1000;
  const activationMs = action.design === 'A' ? (workload.activationBytes || 0) / finitePositive(state.flashReadBytesPerSecond, 1) * 1000 : 0;
  const schedulerMs = finitePositive(state.schedulerOverheadMs, 0);
  const predictedMs = flashMs + dequantMs + recomputeMs + activationMs + schedulerMs;
  const qualityUpperBound = Math.min(1, (workload.baseRisk || 0) * (1 - action.repairRatio) + (action.bulkBits <= 2 ? 0.02 : 0.005));
  const memoryBytes = (workload.anchorBytes || 0) + flashBytes * (state.residencyFraction ?? 1) + (workload.workspaceBytes || 0);
  return { predictedMs, flashBytes, dequantBytes, recomputeTokens, recomputeFlops, qualityUpperBound, memoryBytes };
}

export function enumerateActions({ bits = [2, 4], designs = ['A', 'B', 'full'], ratios = DEFAULT_RATIOS } = {}) {
  return bits.flatMap(bulkBits => designs.flatMap(design => ratios.map(repairRatio => ({
    bulkBits,
    design,
    repairRatio: design === 'full' ? 1 : repairRatio,
    pageFraction: 1,
  }))));
}

export function chooseAction({ state, workload, constraints = {}, actions } = {}) {
  const candidates = (actions || enumerateActions()).map(action => ({
    action,
    cost: estimateActionCost({ state, action, workload }),
  }));
  const feasible = candidates.filter(candidate =>
    candidate.cost.qualityUpperBound <= (constraints.maxQualityLoss ?? 1) &&
    candidate.cost.memoryBytes <= (state.freeMemoryBytes || Infinity) &&
    candidate.cost.predictedMs <= (constraints.maxLatencyMs || Infinity));
  if (!feasible.length) {
    const fallback = candidates.find(candidate => candidate.action.design === 'full');
    return { selected: fallback?.action || { design: 'full', bulkBits: 16, repairRatio: 1 },
      cost: fallback?.cost || null, fallback: true, candidates };
  }
  feasible.sort((left, right) => left.cost.predictedMs - right.cost.predictedMs);
  return { selected: feasible[0].action, cost: feasible[0].cost, fallback: false, candidates };
}
