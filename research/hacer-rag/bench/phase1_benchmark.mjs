import { performance } from 'node:perf_hooks';
import { createValidityKey } from '../src/validityKey.mjs';
import { createPrefixReuseMetadata, validateExactPrefixReuse } from '../src/prefixReuse.mjs';
import { rotateKeyVector } from '../src/ropeScalar.mjs';

const keyMetadata = {
  modelSha256: 'benchmark-model',
  tokenizerSha256: 'benchmark-tokenizer',
  templateSha256: 'benchmark-template',
  architecture: { family: 'decoder-only', layers: 18, kvHeads: 4, headDim: 64 },
  position: { scheme: 'rope', base: 10000, scale: 1, frequencyLayout: 'interleaved' },
  attentionMask: { policy: 'causal', window: null },
  quantization: { key: 'fp16', value: 'fp16', bulk: 'q4_0' },
  tensorLayout: { endian: 'little', order: 'layer-head-token-dim' },
  chunk: { id: 'benchmark:0', contentSha256: 'content', tokenIdsSha256: 'tokens' },
  assembly: { orderSha256: 'order', localPositionOrigin: 0 },
  namespace: { user: 'benchmark', profile: 'default' },
};

const tokenIds = Array.from({ length: 512 }, (_, index) => index % 32000);
const validityKey = createValidityKey(keyMetadata);
const prefix = createPrefixReuseMetadata({
  tokenIds: tokenIds.slice(0, 256),
  validityKey,
  promptTemplateHash: 'benchmark-template',
  contextLength: 2048,
});
const ropeConfig = { scheme: 'rope', base: 10000, scale: 1, headDim: 64, frequencyLayout: 'interleaved' };
const keyVector = Array.from({ length: 64 }, (_, index) => Math.sin(index / 3));

function measure(iterations, fn) {
  const start = performance.now();
  for (let index = 0; index < iterations; index += 1) fn(index);
  return (performance.now() - start) / iterations;
}

const result = {
  benchmark: 'hacer-rag-phase1-host-reference',
  iterations: 10000,
  validityKey: {
    digestPrefix: validityKey.digest.slice(0, 16),
    meanMs: measure(10000, () => createValidityKey(keyMetadata)),
  },
  exactPrefixValidation: {
    prefixTokens: prefix.prefixTokenCount,
    meanMs: measure(10000, () => validateExactPrefixReuse({
      metadata: prefix,
      requestedTokenIds: tokenIds,
      requestedValidityKey: validityKey,
      requestedTemplateHash: 'benchmark-template',
    })),
  },
  ropeRotation: {
    headDim: ropeConfig.headDim,
    meanMs: measure(10000, index => rotateKeyVector(keyVector, index % 16, ropeConfig)),
  },
  note: 'Host reference overhead only; not mobile TTFT, energy, or quality evidence.',
};

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
