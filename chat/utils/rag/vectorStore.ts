import RNFS from 'react-native-fs';
import type { Chunk } from './chunker';
import { nearTerm, positiveInt, terms } from './text';

export const MAX_INDEX_CHUNKS = 2048;
export interface VectorEntry { chunk: Chunk; vector: number[] }
export interface VectorIndex {
  entries: VectorEntry[];
  createdAt: string;
  updatedAt: string;
  version?: number;
  embeddingModel?: string;
}
export interface SearchResult {
  chunk: Chunk;
  /** RRF ranking score, NOT a calibrated relevance probability. */
  score: number;
  denseScore: number;
  bm25Score: number;
  denseRank?: number;
  lexicalRank?: number;
  rrfScore: number;
  lexicalCoverage: number;
  exactTermMatches: number;
  matchedTerms: string[];
}
interface RecordEntry { chunk: Chunk; vector: Float32Array }
interface LexicalIndex {
  lengths: number[];
  average: number;
  postings: Map<string, Map<number, number>>;
  byLength: Map<number, string[]>;
}
interface State {
  entries: RecordEntry[];
  createdAt: string;
  updatedAt: string;
  embeddingModel?: string;
  lexical?: LexicalIndex;
}
const states = new Map<string, State>();
const loads = new Map<string, Promise<void>>();
const writes = new Map<string, Promise<unknown>>();
const epochs = new Map<string, number>();
const pathFor = (id: string) => `${RNFS.ExternalDirectoryPath}/profiles/${id}/rag_index.json`;
const fresh = (): State => ({ entries: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() });

export function normalizeVector(vector: Float32Array): Float32Array {
  if (!vector.length || vector.length > 8192) throw new Error('Invalid embedding dimension.');
  let norm = 0;
  for (const value of vector) {
    if (!Number.isFinite(value)) throw new Error('Embedding contains non-finite values.');
    norm += value * value;
  }
  if (norm <= 0) throw new Error('Embedding model returned a zero vector.');
  const scale = 1 / Math.sqrt(norm);
  return Float32Array.from(vector, value => value * scale);
}

function validateChunk(chunk: Chunk): void {
  if (!chunk || typeof chunk.text !== 'string' || !chunk.text.trim() || chunk.text.length > 16000 ||
      typeof chunk.docId !== 'string' || typeof chunk.docName !== 'string' ||
      !Number.isInteger(chunk.chunkIndex) || chunk.chunkIndex < 0) throw new Error('Invalid document chunk in index.');
}

/** Coalesced lazy load: queries and edits cannot accidentally see an empty index. */
export async function loadIndex(profileId: string): Promise<void> {
  if (states.has(profileId)) return;
  const pending = loads.get(profileId);
  if (pending) return pending;
  const epoch = epochs.get(profileId) || 0;
  const promise = (async () => {
    const path = pathFor(profileId);
    let next = fresh();
    if (await RNFS.exists(path)) {
      const size = Number((await RNFS.stat(path)).size);
      if (size > 64 * 1024 * 1024) throw new Error('Index exceeds the mobile loading limit.');
      const stored: VectorIndex = JSON.parse(await RNFS.readFile(path, 'utf8'));
      if (!Array.isArray(stored.entries) || stored.entries.length > MAX_INDEX_CHUNKS) throw new Error('Invalid or oversized RAG index.');
      let dimension = 0;
      const entries = stored.entries.map(entry => {
        validateChunk(entry.chunk);
        if (!Array.isArray(entry.vector)) throw new Error('Invalid stored embedding.');
        const vector = normalizeVector(new Float32Array(entry.vector));
        if (dimension && vector.length !== dimension) throw new Error('Mixed embedding dimensions; re-index documents.');
        dimension = vector.length;
        return { chunk: entry.chunk, vector };
      });
      next = { entries, createdAt: stored.createdAt || next.createdAt,
        updatedAt: stored.updatedAt || next.updatedAt, embeddingModel: stored.embeddingModel };
    }
    if ((epochs.get(profileId) || 0) === epoch) states.set(profileId, next);
  })();
  loads.set(profileId, promise);
  try { await promise; } finally { if (loads.get(profileId) === promise) loads.delete(profileId); }
}

function stateFor(profileId: string): State {
  const state = states.get(profileId);
  if (!state) throw new Error('RAG index is not loaded.');
  return state;
}

async function mutate(profileId: string, change: (state: State) => State): Promise<void> {
  const operation = (writes.get(profileId) || Promise.resolve()).catch(() => {}).then(async () => {
    await loadIndex(profileId);
    const previous = stateFor(profileId);
    const next = change(previous);
    if (next.entries.length > MAX_INDEX_CHUNKS) throw new Error(`Mobile index limit is ${MAX_INDEX_CHUNKS} chunks. Remove a document first.`);
    const path = pathFor(profileId);
    await RNFS.mkdir(`${RNFS.ExternalDirectoryPath}/profiles/${profileId}`);
    const record: VectorIndex = {
      version: 2, embeddingModel: next.embeddingModel,
      createdAt: next.createdAt, updatedAt: new Date().toISOString(),
      entries: next.entries.map(entry => ({ chunk: entry.chunk, vector: Array.from(entry.vector) })),
    };
    // Write complete data before same-directory replacement. Android RNFS uses
    // renameTo here; never unlink the valid index first. Publish RAM only after
    // persistence succeeds. Mutations are serialized to avoid lost updates.
    const epoch = epochs.get(profileId) || 0;
    try {
      await RNFS.writeFile(`${path}.tmp`, JSON.stringify(record), 'utf8');
      await RNFS.moveFile(`${path}.tmp`, path);
      if ((epochs.get(profileId) || 0) === epoch) {
        states.set(profileId, { ...next, updatedAt: record.updatedAt, lexical: undefined });
      }
    } finally { if (await RNFS.exists(`${path}.tmp`)) await RNFS.unlink(`${path}.tmp`).catch(() => {}); }
  });
  writes.set(profileId, operation);
  try { await operation; } finally { if (writes.get(profileId) === operation) writes.delete(profileId); }
}

export async function saveIndex(profileId: string): Promise<void> { await mutate(profileId, state => state); }

export async function replaceDocumentInIndex(docId: string, chunks: Chunk[], vectors: Float32Array[],
  profileId: string, embeddingModel?: string): Promise<void> {
  if (!chunks.length || chunks.length !== vectors.length) throw new Error('Chunk/vector count mismatch.');
  const added = chunks.map((chunk, i) => {
    validateChunk(chunk);
    if (chunk.docId !== docId || chunk.chunkIndex !== i) throw new Error('Document chunk identities are inconsistent.');
    return { chunk, vector: normalizeVector(vectors[i]) };
  });
  const dimension = added[0].vector.length;
  if (added.some(entry => entry.vector.length !== dimension)) throw new Error('Mixed embedding dimensions.');
  await mutate(profileId, state => {
    const retained = state.entries.filter(entry => entry.chunk.docId !== docId);
    if (retained.some(entry => entry.vector.length !== dimension)) throw new Error('Embedding dimension changed; re-index documents.');
    if (retained.length && state.embeddingModel && embeddingModel && state.embeddingModel !== embeddingModel) {
      throw new Error('Embedding model changed; remove and re-index existing documents.');
    }
    return { ...state, entries: [...retained, ...added], embeddingModel: embeddingModel || state.embeddingModel };
  });
}

export async function removeDocFromIndex(docId: string, profileId: string): Promise<void> {
  await mutate(profileId, state => ({ ...state, entries: state.entries.filter(entry => entry.chunk.docId !== docId) }));
}
export async function clearIndex(profileId: string): Promise<void> {
  await mutate(profileId, state => ({ ...state, entries: [], embeddingModel: undefined }));
}
export function flushInMemoryIndex(profileId?: string): void {
  const ids = profileId ? [profileId] : Array.from(new Set([...states.keys(), ...loads.keys(), ...writes.keys()]));
  for (const id of ids) { epochs.set(id, (epochs.get(id) || 0) + 1); states.delete(id); loads.delete(id); }
}
export function getIndexSize(profileId?: string): number {
  return profileId ? states.get(profileId)?.entries.length || 0 : Array.from(states.values()).reduce((sum, state) => sum + state.entries.length, 0);
}
export function getIndexedDocIds(profileId: string): string[] {
  return Array.from(new Set((states.get(profileId)?.entries || []).map(entry => entry.chunk.docId)));
}
export function getIndexStats(profileId: string) {
  const state = stateFor(profileId);
  return { chunks: state.entries.length, vectorBytes: state.entries.reduce((n, entry) => n + entry.vector.byteLength, 0),
    legacyChunks: state.entries.filter(entry => entry.chunk.chunkerVersion !== 2).length,
    embeddingModel: state.embeddingModel || null, revision: state.updatedAt };
}
export function getIndexedDocuments(profileId: string) {
  const docs = new Map<string, { id: string; name: string; chunkCount: number; addedAt: string }>();
  const state = stateFor(profileId);
  state.entries.forEach(({ chunk }) => {
    const doc = docs.get(chunk.docId) || { id: chunk.docId, name: chunk.docName, chunkCount: 0, addedAt: state.createdAt };
    doc.chunkCount++;
    docs.set(doc.id, doc);
  });
  return Array.from(docs.values());
}

function lexicalFor(state: State): LexicalIndex {
  if (state.lexical) return state.lexical;
  const postings = new Map<string, Map<number, number>>();
  const byLength = new Map<number, string[]>();
  const lengths = state.entries.map(({ chunk }, idx) => {
    const tokens = terms(`${chunk.sectionHeader || ''}\n${chunk.text}`);
    for (const term of tokens) {
      if (!postings.has(term)) {
        postings.set(term, new Map());
        const bucket = byLength.get(term.length) || [];
        bucket.push(term); byLength.set(term.length, bucket);
      }
      const posting = postings.get(term)!;
      posting.set(idx, (posting.get(idx) || 0) + 1);
    }
    return tokens.length;
  });
  state.lexical = { postings, byLength, lengths, average: lengths.reduce((a, b) => a + b, 0) / Math.max(1, lengths.length) || 1 };
  return state.lexical;
}

/** Bounded top-k storage: O(N log k), no sorting/copying the full vector index. */
function top(scores: ArrayLike<number>, k: number): { idx: number; score: number }[] {
  const out: { idx: number; score: number }[] = [];
  for (let idx = 0; idx < scores.length; idx++) {
    const score = scores[idx];
    if (score <= 0) continue;
    let low = 0; let high = out.length;
    while (low < high) { const mid = (low + high) >>> 1; if (out[mid].score >= score) low = mid + 1; else high = mid; }
    if (low < k) { out.splice(low, 0, { idx, score }); if (out.length > k) out.pop(); }
  }
  return out;
}

export function search(queryVector: Float32Array, profileId: string, topK = 3, queryText = ''): SearchResult[] {
  const state = stateFor(profileId);
  const index = state.entries;
  if (!index.length) return [];
  const query = normalizeVector(queryVector);
  if (query.length !== index[0].vector.length) throw new Error('Query/index embedding dimensions differ. Re-index with the selected model.');
  const k = positiveInt(topK, 3, 24);
  const dense = new Float32Array(index.length);
  index.forEach((entry, i) => { let dot = 0; for (let j = 0; j < query.length; j++) dot += query[j] * entry.vector[j]; dense[i] = dot; });
  const lexical = lexicalFor(state);
  const queryTerms = Array.from(new Set(terms(queryText))).slice(0, 32);
  const bm25 = new Float32Array(index.length);
  const matches = new Map<number, Set<string>>();
  const exact = new Uint16Array(index.length);
  let fuzzyChecks = 0;
  for (const term of queryTerms) {
    const alternatives = [term];
    if (!lexical.postings.has(term) && term.length >= 5 && !/\d/.test(term)) {
      for (const length of [term.length, term.length - 1, term.length + 1]) {
        for (const candidate of lexical.byLength.get(length) || []) {
          if (++fuzzyChecks > 4000 || alternatives.length >= 5) break;
          if (nearTerm(term, candidate)) alternatives.push(candidate);
        }
      }
    }
    const termScores = new Map<number, number>();
    for (const alternative of alternatives) {
      const posting = lexical.postings.get(alternative);
      if (!posting) continue;
      const idf = Math.log(1 + (index.length - posting.size + 0.5) / (posting.size + 0.5));
      posting.forEach((tf, idx) => {
        const score = idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lexical.lengths[idx] / lexical.average));
        termScores.set(idx, Math.max(termScores.get(idx) || 0, score * (alternative === term ? 1 : 0.8)));
        const hit = matches.get(idx) || new Set<string>(); hit.add(term); matches.set(idx, hit);
        if (alternative === term) exact[idx]++;
      });
    }
    termScores.forEach((score, idx) => { bm25[idx] += score; });
  }
  const pool = new Map<number, SearchResult>();
  const insert = (idx: number): SearchResult => {
    if (!pool.has(idx)) pool.set(idx, { chunk: index[idx].chunk, score: 0, rrfScore: 0,
      denseScore: dense[idx], bm25Score: bm25[idx], matchedTerms: Array.from(matches.get(idx) || []),
      lexicalCoverage: (matches.get(idx)?.size || 0) / Math.max(1, queryTerms.length), exactTermMatches: exact[idx] });
    return pool.get(idx)!;
  };
  top(dense, k).forEach(({ idx }, rank) => { const item = insert(idx); item.denseRank = rank + 1; item.rrfScore += 1 / (61 + rank); });
  // Zero-match documents receive NO lexical rank, avoiding arbitrary RRF boosts.
  top(bm25, k).forEach(({ idx }, rank) => { const item = insert(idx); item.lexicalRank = rank + 1; item.rrfScore += 1 / (61 + rank); });
  return Array.from(pool.values()).map(item => ({ ...item, score: item.rrfScore }))
    .sort((a, b) => b.score - a.score || b.lexicalCoverage - a.lexicalCoverage || a.chunk.chunkIndex - b.chunk.chunkIndex).slice(0, k);
}
