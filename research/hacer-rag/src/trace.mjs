import { appendFile, mkdir, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

export const TRACE_SCHEMA_VERSION = 1;

const ALLOWED_STATUS = new Set(['exact', 'rope_aligned_approximate', 'boundary_repaired', 'full_prefill', 'fallback']);

export function createTrace({
  runId,
  deviceManifestSha256 = null,
  modelSha256,
  tokenizerSha256,
  templateSha256,
  cacheManifestSha256 = null,
  queryId,
  retrievalReplayId = null,
  validityDigest = null,
  validityStatus,
  reuseMode,
  cacheState = 'not_measured',
  thermalState = 'not_measured',
  memoryPressureState = 'not_measured',
  promptTokenCount,
  prefix = null,
  rope = null,
  fallbackReason = null,
  timing = {},
  counters = {},
}) {
  if (!runId || !queryId || !modelSha256 || !tokenizerSha256 || !templateSha256) {
    throw new TypeError('runId, queryId, modelSha256, tokenizerSha256, and templateSha256 are required');
  }
  if (!ALLOWED_STATUS.has(validityStatus)) throw new TypeError(`invalid validityStatus:${validityStatus}`);
  return {
    traceSchemaVersion: TRACE_SCHEMA_VERSION,
    runId,
    deviceManifestSha256,
    modelSha256,
    tokenizerSha256,
    templateSha256,
    cacheManifestSha256,
    queryId,
    retrievalReplayId,
    validityDigest,
    validityStatus,
    reuseMode: reuseMode || 'unknown',
    cacheState,
    thermalState,
    memoryPressureState,
    promptTokenCount: Number.isInteger(promptTokenCount) ? promptTokenCount : null,
    prefix,
    rope,
    fallbackReason,
    timing: normalizeNumbers(timing),
    counters: normalizeNumbers(counters),
    createdAt: new Date().toISOString(),
  };
}

export function validateTrace(trace) {
  const errors = [];
  if (trace?.traceSchemaVersion !== TRACE_SCHEMA_VERSION) errors.push('trace_schema_version');
  for (const field of ['runId', 'queryId', 'modelSha256', 'tokenizerSha256', 'templateSha256']) {
    if (!trace?.[field]) errors.push(field);
  }
  if (!ALLOWED_STATUS.has(trace?.validityStatus)) errors.push('validityStatus');
  if (trace?.promptTokenCount !== null && trace?.promptTokenCount !== undefined &&
      (!Number.isInteger(trace.promptTokenCount) || trace.promptTokenCount < 0)) errors.push('promptTokenCount');
  return { valid: errors.length === 0, errors };
}

export async function writeTraceAtomic(path, trace) {
  const result = validateTrace(trace);
  if (!result.valid) throw new Error(`invalid_trace:${result.errors.join(',')}`);
  await mkdir(dirname(path), { recursive: true });
  const tempPath = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(tempPath, `${JSON.stringify(trace, null, 2)}\n`, 'utf8');
  await rename(tempPath, path);
  return path;
}

export async function appendTraceJsonl(path, trace) {
  const result = validateTrace(trace);
  if (!result.valid) throw new Error(`invalid_trace:${result.errors.join(',')}`);
  await mkdir(dirname(path), { recursive: true });
  await appendFile(path, `${JSON.stringify(trace)}\n`, 'utf8');
  return path;
}

function normalizeNumbers(value) {
  if (!value || typeof value !== 'object') return {};
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, Number.isFinite(item) ? item : item ?? null]));
}
