import { initLlama, LlamaContext } from 'llama.rn';
import RNFS from 'react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { profileKey } from '../profileManager';
import type { Chunk } from './chunker';
import { normalizeVector } from './vectorStore';

const MODEL_DIR = `${RNFS.ExternalDirectoryPath}/models`;
const MAX_TOKENS = 480; // leave space for encoder special tokens in n_ctx=512
let context: LlamaContext | null = null;
let owner: string | null = null;
let identity = '';
let queue: Promise<unknown> = Promise.resolve();
const cache = new Map<string, Float32Array>();

/** All native encoder calls (including init/release) are serialized. */
function serial<T>(work: () => Promise<T>): Promise<T> {
  const result = queue.catch(() => {}).then(work);
  queue = result.catch(() => {});
  return result;
}

async function ensure(profileId: string): Promise<LlamaContext> {
  const modelId = await AsyncStorage.getItem(profileKey(profileId, '_embedding_model')) || 'all-minilm-l6-v2-q4_k_m';
  const path = `${MODEL_DIR}/${modelId}.gguf`;
  if (!(await RNFS.exists(path))) throw new Error(`Embedding model not found: ${modelId}`);
  const stat = await RNFS.stat(path);
  // File signature invalidates the cache on replacement. It is not a hash.
  const signature = `${modelId}:${stat.size}:${new Date(stat.mtime).getTime()}`;
  if (context && identity === signature && owner === profileId) return context;
  if (context) await context.release();
  context = null; identity = ''; owner = null; cache.clear();
  // Keep encoder batch/context compatible with full non-causal BERT attention.
  context = await initLlama({ model: `file://${path}`, n_ctx: 512, n_batch: 512,
    n_ubatch: 512, n_gpu_layers: 0, n_threads: 4, embedding: true, use_mmap: true });
  identity = signature; owner = profileId;
  return context;
}

export async function initEmbeddingModel(profileId: string): Promise<void> {
  await serial(async () => { await ensure(profileId); });
}
export async function releaseEmbeddingModel(profileId: string): Promise<void> {
  await serial(async () => {
    if (owner === profileId && context) {
      await context.release(); context = null; owner = null; identity = ''; cache.clear();
    }
  });
}
export async function getEmbeddingIdentity(profileId: string): Promise<string> {
  return serial(async () => { await ensure(profileId); return identity; });
}
export function embeddingText(chunk: Chunk): string {
  return `${chunk.sectionHeader ? `${chunk.sectionHeader}\n` : ''}${chunk.text}`;
}

/** Split only units that cannot fit the actual encoder tokenizer. */
export async function prepareEmbeddingChunks(chunks: Chunk[], profileId: string): Promise<Chunk[]> {
  return serial(async () => {
    const ctx = await ensure(profileId);
    const result: Chunk[] = [];
    const fit = async (chunk: Chunk): Promise<void> => {
      if ((await ctx.tokenize(embeddingText(chunk))).tokens.length <= MAX_TOKENS) {
        result.push({ ...chunk, chunkIndex: result.length }); return;
      }
      if (chunk.text.length < 8) throw new Error('Section heading exceeds encoder context; shorten the heading.');
      const middle = Math.floor(chunk.text.length / 2);
      let split = chunk.text.lastIndexOf(' ', middle);
      if (split < middle / 2) split = middle;
      if (/[\uD800-\uDBFF]/.test(chunk.text[split - 1])) split--;
      await fit({ ...chunk, text: chunk.text.slice(0, split).trim(), continued: true });
      await fit({ ...chunk, text: chunk.text.slice(split).trim(), continued: true });
    };
    for (const chunk of chunks) await fit(chunk);
    return result;
  });
}

async function encode(ctx: LlamaContext, text: string): Promise<Float32Array> {
  if (!text.trim()) throw new Error('Cannot embed empty text.');
  if ((await ctx.tokenize(text)).tokens.length > MAX_TOKENS) throw new Error('Embedding input exceeds 480 tokens. Shorten the query or re-ingest the document.');
  const result = await ctx.embedding(text);
  return normalizeVector(new Float32Array(result.embedding));
}

export async function embedQuery(text: string, profileId: string): Promise<{
  vector: Float32Array; cacheHit: boolean; modelIdentity: string;
}> {
  return serial(async () => {
    const ctx = await ensure(profileId);
    // Exact input key: case/numbers/punctuation may affect encoder semantics.
    const cached = cache.get(text);
    if (cached) {
      cache.delete(text); cache.set(text, cached);
      return { vector: cached.slice(), cacheHit: true, modelIdentity: identity };
    }
    const vector = await encode(ctx, text);
    if (text.length <= 2048) {
      cache.set(text, vector);
      if (cache.size > 32) cache.delete(cache.keys().next().value!);
    }
    return { vector: vector.slice(), cacheHit: false, modelIdentity: identity };
  });
}
export async function embedText(text: string, profileId: string): Promise<Float32Array> {
  return (await embedQuery(text, profileId)).vector;
}
export async function embedBatch(texts: string[], profileId: string,
  onProgress?: (done: number, total: number) => void, expectedIdentity?: string,
  cancelled?: () => boolean): Promise<Float32Array[]> {
  return serial(async () => {
    const ctx = await ensure(profileId);
    if (expectedIdentity && expectedIdentity !== identity) throw new Error('Embedding model changed during ingestion. Retry.');
    const vectors: Float32Array[] = [];
    for (const text of texts) {
      if (cancelled?.()) throw new Error('Document ingestion cancelled.');
      vectors.push(await encode(ctx, text));
      onProgress?.(vectors.length, texts.length);
      if (vectors.length % 8 === 0) await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
    return vectors;
  });
}
