import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  buildValidityRecord,
  compareValidityRecords,
  createValidityKey,
  explainValidityMismatch,
} from '../src/validityKey.mjs';
import {
  createPrefixReuseMetadata,
  markPrefixOutcome,
  tokenIdsHash,
  validateExactPrefixReuse,
} from '../src/prefixReuse.mjs';
import {
  rotateKeyVector,
  shiftKeyBatch,
  validateRoPEMetadata,
  validateRopeConfig,
} from '../src/ropeScalar.mjs';
import {
  appendTraceJsonl,
  createTrace,
  validateTrace,
  writeTraceAtomic,
} from '../src/trace.mjs';
import { alignUp, createCacheManifest, estimateReadAmplification, validateCacheManifest } from '../src/cacheContainer.mjs';
import { chooseAction, estimateActionCost } from '../src/scheduler.mjs';

function metadata(overrides = {}) {
  return {
    modelSha256: 'model-a',
    tokenizerSha256: 'tokenizer-a',
    templateSha256: 'template-a',
    architecture: { family: 'decoder-only', layers: 4, kvHeads: 2, headDim: 8 },
    position: { scheme: 'rope', base: 10000, scale: 1, frequencyLayout: 'interleaved' },
    attentionMask: { policy: 'causal', window: null },
    quantization: { key: 'fp16', value: 'fp16', bulk: 'q4_0' },
    tensorLayout: { endian: 'little', order: 'layer-head-token-dim' },
    chunk: { id: 'doc-a:0', contentSha256: 'chunk-a', tokenIdsSha256: tokenIdsHash([1, 2, 3]) },
    assembly: { orderSha256: 'order-a', localPositionOrigin: 0 },
    namespace: { user: 'test', profile: 'default' },
    ...overrides,
  };
}

test('validity keys are deterministic under object insertion order', () => {
  const first = createValidityKey(metadata());
  const second = createValidityKey({ ...metadata(), architecture: { kvHeads: 2, headDim: 8, layers: 4, family: 'decoder-only' } });
  assert.equal(first.digest, second.digest);
  assert.equal(first.algorithm, 'sha256-canonical-json-v1');
});

test('validity keys explain model, template, order, and quantizer changes', () => {
  const expected = createValidityKey(metadata());
  const actual = createValidityKey(metadata({
    modelSha256: 'model-b',
    templateSha256: 'template-b',
    quantization: { key: 'q8', value: 'q8', bulk: 'q2' },
    assembly: { orderSha256: 'order-b', localPositionOrigin: 0 },
  }));
  const paths = explainValidityMismatch(expected, actual).map(item => item.path);
  assert.ok(paths.includes('$.modelSha256'));
  assert.ok(paths.includes('$.templateSha256'));
  assert.ok(paths.includes('$.quantization.bulk'));
  assert.ok(paths.includes('$.assembly.orderSha256'));
  assert.equal(compareValidityRecords(expected.record, actual.record).length, paths.length);
});

test('validity metadata requires all cache-safety domains', () => {
  const incomplete = metadata();
  delete incomplete.attentionMask;
  assert.throws(() => buildValidityRecord(incomplete), /attentionMask is required/);
});

test('exact prefix metadata hits only for the same prefix and validity key', () => {
  const key = createValidityKey(metadata());
  const cached = createPrefixReuseMetadata({
    tokenIds: [10, 11, 12],
    validityKey: key,
    promptTemplateHash: 'template-a',
    contextLength: 128,
  });
  const hit = validateExactPrefixReuse({
    metadata: cached,
    requestedTokenIds: [10, 11, 12, 13],
    requestedValidityKey: key,
    requestedTemplateHash: 'template-a',
  });
  assert.deepEqual(hit, { hit: true, status: 'exact', reusedTokenCount: 3, mismatchReason: null });
  assert.equal(validateExactPrefixReuse({
    metadata: cached,
    requestedTokenIds: [10, 99, 12, 13],
    requestedValidityKey: key,
    requestedTemplateHash: 'template-a',
  }).mismatchReason, 'prefix_token_hash_mismatch');
  assert.equal(validateExactPrefixReuse({
    metadata: cached,
    requestedTokenIds: [10, 11, 12, 13],
    requestedValidityKey: createValidityKey(metadata({ modelSha256: 'model-b' })),
    requestedTemplateHash: 'template-a',
  }).mismatchReason, 'validity_key_mismatch');
  assert.equal(markPrefixOutcome(cached, { status: 'miss', mismatchReason: 'prompt_template_mismatch' }).reuseStatus, 'miss');
});

test('RoPE zero shift is identity and values are outside the transform', () => {
  const config = { scheme: 'rope', base: 10000, scale: 1, headDim: 8, frequencyLayout: 'interleaved' };
  const key = [1, -2, 3, -4, 5, -6, 7, -8];
  assert.deepEqual([...rotateKeyVector(key, 0, config)], key);
  const shifted = rotateKeyVector(key, 7, config);
  const originalNorm = key.reduce((sum, value) => sum + value * value, 0);
  const shiftedNorm = [...shifted].reduce((sum, value) => sum + value * value, 0);
  assert.ok(Math.abs(originalNorm - shiftedNorm) < 1e-10);
});

test('RoPE supports negative shifts, batches, and rejects unsupported metadata', () => {
  const config = { scheme: 'rope', base: 10000, scale: 0.5, headDim: 4 };
  const key = [1, 2, 3, 4];
  const forward = rotateKeyVector(key, 9, config);
  const backward = rotateKeyVector(forward, -9, config);
  assert.ok(backward.every((value, index) => Math.abs(value - key[index]) < 1e-10));
  assert.equal(shiftKeyBatch([key, key], [0, 2], config).length, 2);
  assert.throws(() => validateRopeConfig({ scheme: 'alibi', base: 10000, scale: 1, headDim: 4 }), /unsupported_position_scheme/);
  assert.equal(validateRoPEMetadata(config, { ...config, base: 20000 }).valid, false);
});

test('trace schema records exact, approximate, repair, fallback, and is atomic', async () => {
  const trace = createTrace({
    runId: 'run-1',
    modelSha256: 'model-a',
    tokenizerSha256: 'tokenizer-a',
    templateSha256: 'template-a',
    queryId: 'q-1',
    validityStatus: 'exact',
    reuseMode: 'prefix',
    promptTokenCount: 4,
    prefix: { hit: true, reusedTokenCount: 3 },
    timing: { schedulerMs: 0.12 },
    counters: { tokensCached: 3 },
  });
  assert.equal(validateTrace(trace).valid, true);
  const dir = await mkdtemp(join(tmpdir(), 'hacer-phase1-'));
  const jsonPath = join(dir, 'trace.json');
  const jsonlPath = join(dir, 'trace.jsonl');
  await writeTraceAtomic(jsonPath, trace);
  await appendTraceJsonl(jsonlPath, trace);
  assert.equal(JSON.parse(await readFile(jsonPath, 'utf8')).validityStatus, 'exact');
  assert.match(await readFile(jsonlPath, 'utf8'), /"traceSchemaVersion":1/);
  assert.equal(validateTrace({ ...trace, validityStatus: 'invalid' }).valid, false);
});

test('HACERKV manifest enforces aligned non-overlapping pages and validity fields', () => {
  const manifest = createCacheManifest({
    modelSha256: 'a'.repeat(64), tokenizerSha256: 'b'.repeat(64),
    templateSha256: 'c'.repeat(64), chunkSha256: 'd'.repeat(64),
    chunkId: 'chunk-1', tokenCount: 32, layerCount: 2, kvHeads: 4, headDim: 8,
    pageSize: 4096, alignment: 4096,
    pages: [
      { offset: 4096, length: 4096, layer: 0, tokenStart: 0, tokenEnd: 16, kind: 'anchor' },
      { offset: 8192, length: 4096, layer: 1, tokenStart: 16, tokenEnd: 32, kind: 'bulk' },
    ],
  });
  assert.equal(manifest.magic, 'HACERKV\0');
  assert.equal(alignUp(4097, 4096), 8192);
  assert.equal(validateCacheManifest(manifest, { modelSha256: 'a'.repeat(64) }).valid, true);
  assert.equal(validateCacheManifest(manifest, { modelSha256: 'e'.repeat(64) }).reason, 'validity_mismatch:modelSha256');
  assert.equal(estimateReadAmplification(manifest.pages, [{ start: 4096, end: 5000 }]), 4096 / 904);
});

test('hardware scheduler rejects unsafe low-bit actions and returns a traceable fallback', () => {
  const state = {
    pageSize: 4096, flashReadBytesPerSecond: 20_000_000,
    dequantBytesPerSecond: 100_000_000, prefillFlopsPerSecond: 1_000_000,
    freeMemoryBytes: 1_000, schedulerOverheadMs: 1,
  };
  const workload = { tokens: 100, layers: 4, kvBytesPerToken: 64, flopsPerTokenLayer: 10_000,
    anchorBytes: 100, workspaceBytes: 100, baseRisk: 0.5 };
  const cost = estimateActionCost({ state, action: { bulkBits: 4, design: 'B', repairRatio: 0.05 }, workload });
  assert.ok(cost.predictedMs > 0);
  const selected = chooseAction({ state, workload, constraints: { maxQualityLoss: 0.0001, maxLatencyMs: 1 } });
  assert.equal(selected.fallback, true);
  assert.equal(selected.selected.design, 'full');
});
