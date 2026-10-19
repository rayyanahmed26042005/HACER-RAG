/**
 * Per-request inference policy used by the controlled mobile experiments.
 *
 * The historical fixed configuration remains available as an explicit mode.
 * The shipped chat path uses the adaptive policy, which learns from native
 * timings observed in this process. Keeping the policy here, instead of
 * hiding it in the UI, makes the selected policy part of every run record.
 */

import type { DeviceState } from './deviceTelemetry';
import type { RagLabels } from './rag/metrics';

export type InferenceMode = 'baseline' | 'fixed' | 'adaptive';
export type RagPolicy = 'ebmrag-v2' | 'mobcert-v1';

export interface InferenceRequestOptions {
  mode?: InferenceMode;
  nThreads?: number;
  contextTokenBudget?: number;
  maxOutputTokens?: number;
  candidateThreads?: number[];
  deviceState?: DeviceState;
  /** Allow deliberate candidate exploration during a controlled run. */
  allowExploration?: boolean;
  /** Include the extra native tokenize call in a profiling run. */
  measurePromptTokens?: boolean;
  /** Explicitly toggle RAG overlap deduplication. */
  deduplicate?: boolean;
  candidateK?: number;
  maxSelectedChunks?: number;
  ragLabels?: RagLabels;
  /** Research-only opt-in; the production default remains EBM-RAG v2. */
  ragPolicy?: RagPolicy;
  /** Allow a complete MobCert numeric/unit certificate to answer without Gemma. */
  allowDirectEvidenceAnswer?: boolean;
  /** Benchmark correlation metadata; ignored by interactive callers. */
  benchmarkRequestId?: string;
  benchmarkQueryId?: string;
  benchmarkCorpusId?: string;
}

export interface InferencePlan {
  mode: InferenceMode;
  nThreads: number;
  nPredict: number;
  contextTokenBudget?: number;
  evidenceDemand: 'single_hop' | 'multi_hop' | 'broad';
  workloadBucket: string;
  selectionReason: string;
  deduplicate?: boolean;
  candidateK?: number;
  maxSelectedChunks?: number;
}

export interface InferenceObservation {
  plan: InferencePlan;
  totalInferenceMs: number;
  prefillMs: number;
  decodeMs: number;
}

export const BASELINE_INFERENCE_CONFIG = Object.freeze({
  n_ctx: 512,
  n_threads: 4,
  n_gpu_layers: 0,
});

// The application uses the optimized policy by default. Set this to
// 'baseline' when collecting a strict historical-baseline run; the controller
// remains available through the benchmark trigger regardless of this value.
export const DEFAULT_CHAT_INFERENCE_MODE: InferenceMode = 'adaptive';

const DEFAULT_OUTPUT_TOKENS = 512;
const DEFAULT_CANDIDATE_THREADS = [2, 4, 6, 8];
const EMA_ALPHA = 0.35;

type ThreadStats = {
  count: number;
  emaTotalMs: number;
  emaPrefillMs: number;
  emaDecodeMs: number;
};

const observations = new Map<string, Map<number, ThreadStats>>();

function getWorkloadBucket(promptTokens: number, chatMode: 'general' | 'document'): string {
  const size = promptTokens <= 192 ? 'short' : promptTokens <= 384 ? 'medium' : 'long';
  return `${chatMode}:${size}`;
}

export function classifyEvidenceDemand(
  query: string,
  chatMode: 'general' | 'document',
): 'single_hop' | 'multi_hop' | 'broad' {
  if (chatMode === 'general') return 'broad';
  const normalized = query.trim().toLowerCase();
  const words = normalized.split(/\s+/).filter(Boolean);
  if (isBroadEntityQuery(query)) return 'broad';
  if (
    words.length >= 18 ||
    /\b(compare|contrast|difference|both|each|multiple|combine|relationship|and|all|list)\b/.test(normalized)
  ) {
    return 'multi_hop';
  }
  if (
    /\b(why|how does|how do|explain|steps|process|main|causes?|products|offices|services|types)\b/.test(normalized) ||
    words.length >= 9
  ) {
    return 'broad';
  }
  return 'single_hop';
}

function getAdaptiveContextBudget(demand: 'single_hop' | 'multi_hop' | 'broad'): number {
  // The Gemma context is 512 tokens. Reserve room for the instruction,
  // question, and chat template; EBM-RAG compresses evidence when needed.
  if (demand === 'single_hop') return 300;
  if (demand === 'multi_hop') return 360;
  return 340;
}

function isBroadEntityQuery(query: string): boolean {
  if (/\b(?:battery|weight|office|offices|price|cost|capacity|role|salary|budget|allowance|version|status|date|year|years|hours|products|services|types)\b/i.test(query)) return false;
  return /^(?:what|who)\s+(?:is|are)\s+[^?!.]+[?!.]?$/i.test(query.trim()) ||
    /^(?:tell\s+me\s+about|describe)\s+[^?!.]+[?!.]?$/i.test(query.trim());
}

export function getGeneralOutputLimit(query: string): number {
  const normalized = query.trim().toLowerCase();
  if (/^(?:what|who|where|when)\s+is\b/.test(normalized)) return 64;
  if (/\b(?:explain|define|why|how)\b/.test(normalized)) return 128;
  if (/\b(?:story|essay|detail|elaborate|code|write|comprehensive)\b/.test(normalized)) return 256;
  return 128;
}

function getAdaptiveOutputBudget(
  query: string,
  demand: 'single_hop' | 'multi_hop' | 'broad',
  chatMode: 'general' | 'document' = 'general',
): number {
  // General mode is decode-bound on CPU-only mobile execution. Keep ordinary
  // answers concise; explicit long-form requests receive a larger runway.
  if (chatMode === 'general') {
    return getGeneralOutputLimit(query);
  }

  // In Document RAG mode, scale budget based on evidence demand:
  // Factual lookups -> 192 tokens (concise yet complete without getting cut off)
  // Multi-hop / broad analysis -> 384 to 448 tokens
  if (isBroadEntityQuery(query)) return 320;
  if (demand === 'single_hop') return 192;
  if (demand === 'multi_hop') return 448;
  return 320;
}

function getDefaultCandidateK(
  demand: 'single_hop' | 'multi_hop' | 'broad',
  options: InferenceRequestOptions,
): number {
  // MobCert needs a wider bounded frontier for broad entity requests. A top-3
  // frontier can contain only a title/footer mention and make a supported
  // entity look empty, even though the rest of the document is indexed.
  if (options.ragPolicy === 'mobcert-v1' && demand === 'broad') return 8;
  return 3;
}

function getSafeThreadCandidates(options: InferenceRequestOptions): number[] {
  const configured = normalizeCandidates(options.candidateThreads);
  const availableProcessors = options.deviceState?.availableProcessors || 8;
  const thermalStatus = options.deviceState?.thermalStatus ?? 0;
  const lowMemory = options.deviceState?.lowMemory ?? false;
  const thermalLimited = thermalStatus >= 3;
  const maxThreads = Math.max(1, Math.min(availableProcessors, thermalLimited || lowMemory ? 4 : 8));
  const safeCandidates = configured.filter(threadCount => threadCount <= maxThreads);
  return safeCandidates.length > 0 ? safeCandidates : [Math.min(BASELINE_INFERENCE_CONFIG.n_threads, maxThreads)];
}

function getStats(bucket: string): Map<number, ThreadStats> {
  let stats = observations.get(bucket);
  if (!stats) {
    stats = new Map<number, ThreadStats>();
    observations.set(bucket, stats);
  }
  return stats;
}

function normalizeCandidates(candidates?: number[]): number[] {
  const values = (candidates || DEFAULT_CANDIDATE_THREADS)
    .filter(Number.isFinite)
    .map(value => Math.max(1, Math.round(value)));
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

/**
 * Select the configuration for one completion.
 *
 * Adaptive mode explores each candidate once for a workload bucket, then
 * exploits the lowest observed native inference latency. This is deliberately
 * a small, interpretable controller; it is not presented as a new scheduling
 * algorithm. Device experiments must establish whether this policy helps.
 */
export function selectInferencePlan(
  promptTokens: number,
  chatMode: 'general' | 'document',
  options: InferenceRequestOptions = {},
  query = '',
): InferencePlan {
  const mode = options.mode || 'baseline';
  const evidenceDemand = classifyEvidenceDemand(query, chatMode);
  const workloadBucket = `${getWorkloadBucket(promptTokens, chatMode)}:${evidenceDemand}`;
  const nPredict = options.maxOutputTokens && options.maxOutputTokens > 0
    ? Math.round(options.maxOutputTokens)
    : mode === 'adaptive'
      ? getAdaptiveOutputBudget(query, evidenceDemand, chatMode)
      : DEFAULT_OUTPUT_TOKENS;

  if (mode === 'baseline') {
    return {
      mode,
      nThreads: BASELINE_INFERENCE_CONFIG.n_threads,
      nPredict,
      contextTokenBudget: options.contextTokenBudget,
      evidenceDemand,
      workloadBucket,
      selectionReason: 'historical fixed baseline',
      deduplicate: options.deduplicate ?? false,
      candidateK: options.candidateK ?? getDefaultCandidateK(evidenceDemand, options),
      maxSelectedChunks: options.maxSelectedChunks ?? 3,
    };
  }

  if (mode === 'fixed') {
    const nThreads = options.nThreads && options.nThreads > 0
      ? Math.round(options.nThreads)
      : BASELINE_INFERENCE_CONFIG.n_threads;
    return {
      mode,
      nThreads,
      nPredict,
      contextTokenBudget: options.contextTokenBudget,
      evidenceDemand,
      workloadBucket,
      selectionReason: options.nThreads ? 'benchmark-requested fixed thread count' : 'fixed baseline fallback',
      deduplicate: options.deduplicate ?? false,
      candidateK: options.candidateK ?? getDefaultCandidateK(evidenceDemand, options),
      maxSelectedChunks: options.maxSelectedChunks ?? 3,
    };
  }

  const candidates = getSafeThreadCandidates(options);
  const adaptiveContextBudget = options.contextTokenBudget && options.contextTokenBudget > 0
    ? Math.round(options.contextTokenBudget)
    : isBroadEntityQuery(query) ? 340 : getAdaptiveContextBudget(evidenceDemand);
  const candidateK = options.candidateK ?? (evidenceDemand === 'multi_hop' ? 10 : 8);
  const maxSelectedChunks = options.maxSelectedChunks ?? (evidenceDemand === 'single_hop' ? 3 : 6);
  const stats = getStats(workloadBucket);
  const unexplored = candidates.filter(threadCount => !stats.has(threadCount));

  let nThreads: number;
  let selectionReason: string;
  if (options.nThreads && candidates.includes(Math.round(options.nThreads))) {
    nThreads = Math.round(options.nThreads);
    selectionReason = 'benchmark-requested adaptive candidate';
  } else if (options.allowExploration && unexplored.length > 0) {
    // Start at the existing baseline when it is a candidate, then visit the
    // remaining candidates in ascending order. This makes the first run less
    // surprising while still guaranteeing exploration.
    nThreads = unexplored.includes(BASELINE_INFERENCE_CONFIG.n_threads)
      ? BASELINE_INFERENCE_CONFIG.n_threads
      : unexplored[0];
    selectionReason = 'exploration of an unmeasured thread candidate';
  } else if (stats.size > 0) {
    nThreads = candidates.reduce((best, candidate) => {
      const bestStats = stats.get(best)!;
      const candidateStats = stats.get(candidate)!;
      if (!bestStats) return candidate;
      if (!candidateStats) return best;
      return candidateStats.emaTotalMs < bestStats.emaTotalMs ? candidate : best;
    }, candidates[0]);
    selectionReason = options.allowExploration
      ? 'exploitation of lowest observed native inference latency'
      : 'exploitation without interactive exploration';
  } else {
    nThreads = candidates.includes(BASELINE_INFERENCE_CONFIG.n_threads)
      ? BASELINE_INFERENCE_CONFIG.n_threads
      : candidates[0];
    selectionReason = 'safe baseline until controlled observations exist';
  }

  return {
    mode,
    nThreads,
    nPredict,
    contextTokenBudget: adaptiveContextBudget,
    evidenceDemand,
    workloadBucket,
    selectionReason,
    deduplicate: options.deduplicate ?? true,
    candidateK,
    maxSelectedChunks,
  };
}

export function recordInferenceObservation(observation: InferenceObservation): void {
  if (observation.plan.mode !== 'adaptive' || !Number.isFinite(observation.totalInferenceMs)) {
    return;
  }

  const stats = getStats(observation.plan.workloadBucket);
  const current = stats.get(observation.plan.nThreads);
  if (!current) {
    stats.set(observation.plan.nThreads, {
      count: 1,
      emaTotalMs: observation.totalInferenceMs,
      emaPrefillMs: observation.prefillMs,
      emaDecodeMs: observation.decodeMs,
    });
    return;
  }

  current.count += 1;
  current.emaTotalMs = EMA_ALPHA * observation.totalInferenceMs + (1 - EMA_ALPHA) * current.emaTotalMs;
  current.emaPrefillMs = EMA_ALPHA * observation.prefillMs + (1 - EMA_ALPHA) * current.emaPrefillMs;
  current.emaDecodeMs = EMA_ALPHA * observation.decodeMs + (1 - EMA_ALPHA) * current.emaDecodeMs;
}

export function resetInferenceController(): void {
  observations.clear();
}

export function getInferenceControllerSnapshot(): Record<string, Record<string, ThreadStats>> {
  const snapshot: Record<string, Record<string, ThreadStats>> = {};
  observations.forEach((threadStats, bucket) => {
    snapshot[bucket] = {};
    threadStats.forEach((stats, threadCount) => {
      snapshot[bucket][String(threadCount)] = { ...stats };
    });
  });
  return snapshot;
}
