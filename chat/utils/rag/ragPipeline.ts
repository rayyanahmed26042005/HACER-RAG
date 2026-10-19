import RNFS from 'react-native-fs';
import { chunkText, CHUNKER_VERSION } from './chunker';
import { embedQuery, embedBatch, embeddingText, getEmbeddingIdentity, prepareEmbeddingChunks } from './embedder';
import { clearIndex, loadIndex, getIndexedDocuments, getIndexStats, MAX_INDEX_CHUNKS,
  replaceDocumentInIndex, removeDocFromIndex, search } from './vectorStore';
import { nowMs, positiveInt, terms } from './text';
import { packEvidence, EvidenceSource } from './evidenceSelector';
import type { EvidenceAtom } from './evidenceAtoms';
import { atomizeChunk } from './evidenceAtoms';
import {
  buildEvidenceCertificate,
  renderContractEvidence,
  selectCertifiedEvidence,
  EvidenceCertificate,
  EvidenceSelectionDiagnostic,
} from './evidenceCertificate';

export interface IngestOptions {
  onProgress?: (stage: string, done: number, total: number) => void;
  cancelled?: () => boolean;
}
export interface IngestMetrics {
  documentId: string; documentName: string; rawCharacters: number; rawWords: number;
  chunkCount: number; duplicateChunksRemoved: number; averageChunkWords: number;
  maxChunkWords: number; sectionCount: number; chunkingMs: number;
  embeddingMs: number; persistenceMs: number; totalMs: number;
  encoderSplitChunks: number; vectorBytes: number; embeddingModel: string;
  strategy: 'structure-aware-v2';
}
export interface RetrievalOptions {
  topK?: number; candidateK?: number; maxSelectedChunks?: number; minSimilarity?: number;
  maxContextTokens?: number; evidenceDemand?: 'single_hop' | 'multi_hop' | 'broad';
  deduplicate?: boolean;
  strategy?: 'evidence' | 'fixed_top_k';
  countTokens?: (text: string) => Promise<number>;
  fitsPrompt?: (context: string) => Promise<boolean>;
  policy?: 'ebmrag-v2' | 'mobcert-v1';
}
export interface RetrievalMetrics {
  context: string; strategy: 'ebmrag-v2' | 'fixed-top-k' | 'mobcert-v1';
  embeddingMs: number; indexLoadMs: number; vectorSearchMs: number; contextConstructionMs: number;
  candidateK: number; candidateChunkCount: number; selectedChunkCount: number;
  duplicateChunksRemoved: number; contextChars: number; contextTokenEstimate: number;
  contextTokens: number; tokenCountMethod: 'native' | 'character_estimate';
  requestedContextTokenBudget: number; appliedContextTokenBudget: number; compressionRatio: number;
  topScore: number | null; scoreMargin: number | null;
  queryTermCount: number; matchedQueryTermCount: number; queryTermCoverage: number;
  lexicalHit: boolean; exactTermMatches: number; selectedChunkIds: string[];
  candidateChunkIds: string[]; failureReason: string | null;
  queryEmbeddingCacheHit: boolean; indexChunks: number; vectorBytes: number; legacyChunks: number;
  budgetLimited: boolean; rejectedByBudget: number; tokenChecks: number;
  uncoveredQueryTerms: string[]; sources: EvidenceSource[];
  embeddingIdentityVerified: boolean;
  answerContract?: EvidenceCertificate;
  answerContractAtoms?: EvidenceAtom[];
  evidenceSelectionDiagnostics?: EvidenceSelectionDiagnostic[];
}

const ingestions = new Set<string>();
export async function ingestTextDoc(text: string, docId: string, docName: string, profileId: string,
  options: IngestOptions = {}): Promise<{ chunkCount: number; metrics: IngestMetrics }> {
  if (ingestions.has(profileId)) throw new Error('A document is already being indexed in this profile.');
  ingestions.add(profileId);
  const start = nowMs();
  try {
    await loadIndex(profileId);
    options.onProgress?.('Structure-aware chunking...', 0, 1);
    const chunkStart = nowMs();
    const raw = chunkText(text, docId, docName);
    const seen = new Set<string>();
    const unique = raw.filter(chunk => {
      // Exact normalized whitespace only; different numbers/negations survive.
      const key = `${chunk.sectionHeader || ''}\n${chunk.text.replace(/\s+/g, ' ').trim()}`;
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });
    if (!unique.length) throw new Error('No extractable text. Scanned PDFs require OCR before upload.');
    const chunks = await prepareEmbeddingChunks(unique, profileId);
    const retainedChunks = getIndexedDocuments(profileId).filter(doc => doc.id !== docId).reduce((sum, doc) => sum + doc.chunkCount, 0);
    if (chunks.length + retainedChunks > MAX_INDEX_CHUNKS) throw new Error('Mobile index is full. Remove a document or split this one before indexing.');
    const chunkingMs = nowMs() - chunkStart;
    const modelIdentity = await getEmbeddingIdentity(profileId);
    const indexStats = getIndexStats(profileId);
    if (indexStats.embeddingModel && indexStats.embeddingModel !== modelIdentity && indexStats.chunks) {
      throw new Error('Embedding model changed. Remove old documents and re-index with the selected model.');
    }
    const embeddingStart = nowMs();
    const vectors = await embedBatch(chunks.map(embeddingText), profileId,
      (done, total) => options.onProgress?.('Embedding...', done, total), modelIdentity, options.cancelled);
    if (options.cancelled?.()) throw new Error('Document ingestion cancelled.');
    const embeddingMs = nowMs() - embeddingStart;
    const persistStart = nowMs();
    options.onProgress?.('Saving index...', 0, 1);
    await replaceDocumentInIndex(docId, chunks, vectors, profileId, modelIdentity);
    const persistenceMs = nowMs() - persistStart;
    const sizes = chunks.map(chunk => chunk.text.split(/\s+/).length);
    const metrics: IngestMetrics = {
      documentId: docId, documentName: docName, rawCharacters: text.length,
      rawWords: text.trim().split(/\s+/).length, chunkCount: chunks.length,
      duplicateChunksRemoved: raw.length - unique.length,
      averageChunkWords: sizes.reduce((a, b) => a + b, 0) / sizes.length,
      maxChunkWords: Math.max(...sizes), sectionCount: new Set(chunks.map(chunk => chunk.sectionHeader)).size,
      chunkingMs, embeddingMs, persistenceMs, totalMs: nowMs() - start,
      encoderSplitChunks: chunks.filter(chunk => chunk.continued).length,
      vectorBytes: vectors.reduce((sum, vector) => sum + vector.byteLength, 0),
      embeddingModel: modelIdentity, strategy: `structure-aware-v${CHUNKER_VERSION}` as 'structure-aware-v2',
    };
    await RNFS.writeFile(`${RNFS.ExternalDirectoryPath}/profiles/${profileId}/rag_ingest_latest.json`,
      JSON.stringify(metrics, null, 2), 'utf8').catch(error => console.warn('[RAG] Metrics save failed:', error));
    options.onProgress?.('Done', 1, 1);
    return { chunkCount: chunks.length, metrics };
  } finally { ingestions.delete(profileId); }
}

export async function removeDoc(docId: string, profileId: string): Promise<void> { await removeDocFromIndex(docId, profileId); }
export async function initRAG(profileId: string): Promise<void> { await loadIndex(profileId); }
export async function clearRAG(profileId: string): Promise<void> { await clearIndex(profileId); }
export async function retrieveContext(query: string, profileId: string, options: number | RetrievalOptions = 3): Promise<string | null> {
  return (await retrieveContextDetailed(query, profileId, typeof options === 'number' ? { topK: options } : options)).context || null;
}

function sourcesFromAtoms(atoms: EvidenceAtom[], scores = new Map<string, { denseScore: number; lexicalScore: number }>): EvidenceSource[] {
  const sources = new Map<string, EvidenceSource>();
  for (const atom of atoms) {
    const id = `${atom.docId}:${atom.chunkIndex}`;
    const score = scores.get(id) || { denseScore: 0, lexicalScore: 0 };
    const source = sources.get(id) || {
      id,
      docId: atom.docId,
      docName: atom.docName,
      chunkIndex: atom.chunkIndex,
      sectionHeader: atom.sectionHeader,
      excerpts: [],
      denseScore: score.denseScore,
      lexicalScore: score.lexicalScore,
    };
    source.excerpts.push(atom.text);
    sources.set(id, source);
  }
  return Array.from(sources.values());
}

export async function retrieveContextDetailed(query: string, profileId: string,
  options: RetrievalOptions = {}): Promise<RetrievalMetrics> {
  const loadStart = nowMs();
  await loadIndex(profileId);
  const indexLoadMs = nowMs() - loadStart;
  const stats = getIndexStats(profileId);
  const candidateK = positiveInt(options.candidateK, options.strategy === 'fixed_top_k' ? 3 : 12, 24);
  const requestedBudget = Math.max(0, Math.floor(options.maxContextTokens ?? 256));
  if (!Number.isFinite(requestedBudget)) throw new Error('Invalid RAG token budget.');
  const maxChunks = positiveInt(options.maxSelectedChunks ?? options.topK, 4, 8);
  let embeddingMs = 0; let vectorSearchMs = 0; let cacheHit = false;
  let candidates: ReturnType<typeof search> = [];
  if (stats.chunks && query.trim() && requestedBudget > 0) {
    const embeddingStart = nowMs();
    const encoded = await embedQuery(query, profileId);
    embeddingMs = nowMs() - embeddingStart; cacheHit = encoded.cacheHit;
    if (stats.embeddingModel && stats.embeddingModel !== encoded.modelIdentity) {
      throw new Error('Query/index embedding model differs; re-index documents.');
    }
    const searchStart = nowMs();
    candidates = search(encoded.vector, profileId, candidateK, query)
      .filter(result => result.bm25Score > 0 || result.denseScore >= (options.minSimilarity ?? 0.15));
    vectorSearchMs = nowMs() - searchStart;
  }
  const constructionStart = nowMs();
  const countTokens = options.countTokens || (async text => Math.ceil(text.length / 4));

  if (options.policy === 'mobcert-v1') {
    const candidateAtoms = candidates.flatMap(result => atomizeChunk(result.chunk));
    const candidateScores = new Map(candidates.map(result => [
      `${result.chunk.docId}:${result.chunk.chunkIndex}`,
      { denseScore: result.denseScore, lexicalScore: result.bm25Score },
    ]));
    const selection = selectCertifiedEvidence(query, candidateAtoms, requestedBudget, Math.max(8, maxChunks * 4));
    let selectedAtoms = selection.atoms;
    let contract = buildEvidenceCertificate(query, selectedAtoms, requestedBudget);
    let rendered = renderContractEvidence(selectedAtoms, contract);
    let fits = !options.fitsPrompt || await options.fitsPrompt(rendered.context);
    let budgetLimited = selection.budgetLimited;
    // The certificate/status block has a small token cost that is not part of
    // the raw atom estimate. Remove the last admitted atom until the actual
    // chat-template prompt fits, preserving the same hard context invariant as
    // EBM-RAG v2.
    while (!fits && selectedAtoms.length) {
      selectedAtoms = selectedAtoms.slice(0, -1);
      contract = buildEvidenceCertificate(query, selectedAtoms, requestedBudget);
      rendered = renderContractEvidence(selectedAtoms, contract);
      fits = !options.fitsPrompt || await options.fitsPrompt(rendered.context);
      budgetLimited = true;
    }
    // A status block without an evidence span must not count as usable RAG
    // context; otherwise qa.ts would invoke the generator despite a fully
    // missing contract and could turn the status text into an answer.
    const context = fits && selectedAtoms.length ? rendered.context : '';
    const contextTokens = context ? await countTokens(context) : 0;
    const sources = sourcesFromAtoms(selectedAtoms, candidateScores);
    const queryTerms = Array.from(new Set(terms(query))).slice(0, 32);
    const matchedQueryTerms = Array.from(new Set(contract.demands.flatMap(demand => demand.matchedTerms)))
      .filter(term => queryTerms.includes(term));
    const candidateChars = candidates.reduce((sum, result) => sum + result.chunk.text.length, 0);
    const evidenceChars = selectedAtoms.reduce((sum, atom) => sum + atom.text.length, 0);
    const failureReason = !stats.chunks ? 'empty_index' : !candidates.length ? 'no_relevant_candidates' :
      !selectedAtoms.length ? (contract.missingDemandCount ? 'contract_missing_demands' : 'no_evidence_fits_budget') :
        contract.ambiguousDemandCount ? 'contract_ambiguous_demands' :
          contract.missingDemandCount ? 'contract_missing_demands' : null;
    const contextConstructionMs = nowMs() - constructionStart;
    return {
      context, sources, strategy: 'mobcert-v1',
      embeddingMs, indexLoadMs, vectorSearchMs, contextConstructionMs, candidateK,
      candidateChunkCount: candidates.length, selectedChunkCount: sources.length,
      duplicateChunksRemoved: 0, contextChars: context.length,
      contextTokenEstimate: Math.ceil(context.length / 4), contextTokens,
      tokenCountMethod: options.countTokens ? 'native' : 'character_estimate',
      requestedContextTokenBudget: requestedBudget, appliedContextTokenBudget: requestedBudget,
      compressionRatio: candidateChars ? evidenceChars / candidateChars : 0,
      topScore: candidates[0]?.score ?? null,
      scoreMargin: candidates.length > 1 ? candidates[0].score - candidates[1].score : null,
      queryTermCount: queryTerms.length, matchedQueryTermCount: matchedQueryTerms.length,
      queryTermCoverage: queryTerms.length ? matchedQueryTerms.length / queryTerms.length : 1,
      uncoveredQueryTerms: queryTerms.filter(term => !matchedQueryTerms.includes(term)),
      lexicalHit: candidates.some(result => result.bm25Score > 0),
      exactTermMatches: candidates.reduce((max, result) => Math.max(max, result.exactTermMatches), 0),
      selectedChunkIds: sources.map(source => source.id),
      candidateChunkIds: candidates.map(result => `${result.chunk.docId}:${result.chunk.chunkIndex}`),
      failureReason,
      queryEmbeddingCacheHit: cacheHit, indexChunks: stats.chunks, vectorBytes: stats.vectorBytes,
      legacyChunks: stats.legacyChunks, embeddingIdentityVerified: stats.embeddingModel !== null,
      budgetLimited, rejectedByBudget: budgetLimited ? 1 : 0, tokenChecks: selection.selectionChecks,
      answerContract: contract, answerContractAtoms: selectedAtoms,
      evidenceSelectionDiagnostics: selection.selectionDiagnostics,
    };
  }

  const packed = await packEvidence(query, candidates, {
    maxTokens: requestedBudget, maxChunks, demand: options.evidenceDemand || 'broad',
    deduplicate: options.deduplicate ?? true, strategy: options.strategy,
    countTokens,
    fitsPrompt: options.fitsPrompt,
  });
  const contextConstructionMs = nowMs() - constructionStart;
  return {
    context: packed.context, sources: packed.sources,
    strategy: options.strategy === 'fixed_top_k' ? 'fixed-top-k' : 'ebmrag-v2',
    embeddingMs, indexLoadMs, vectorSearchMs, contextConstructionMs, candidateK,
    candidateChunkCount: candidates.length, selectedChunkCount: packed.sources.length,
    duplicateChunksRemoved: packed.duplicatesRemoved, contextChars: packed.context.length,
    contextTokenEstimate: Math.ceil(packed.context.length / 4), contextTokens: packed.contextTokens,
    tokenCountMethod: options.countTokens ? 'native' : 'character_estimate',
    requestedContextTokenBudget: requestedBudget, appliedContextTokenBudget: requestedBudget,
    compressionRatio: packed.candidateChars ? packed.evidenceChars / packed.candidateChars : 0,
    topScore: candidates[0]?.score ?? null,
    scoreMargin: candidates.length > 1 ? candidates[0].score - candidates[1].score : null,
    queryTermCount: packed.queryTerms.length, matchedQueryTermCount: packed.matchedQueryTerms.length,
    queryTermCoverage: packed.queryTermCoverage, uncoveredQueryTerms: packed.uncoveredQueryTerms,
    lexicalHit: packed.sources.some(source => source.lexicalScore > 0),
    exactTermMatches: candidates.reduce((max, result) => Math.max(max, result.exactTermMatches), 0),
    selectedChunkIds: packed.sources.map(source => source.id),
    candidateChunkIds: candidates.map(result => `${result.chunk.docId}:${result.chunk.chunkIndex}`),
    failureReason: !stats.chunks ? 'empty_index' : !candidates.length ? 'no_relevant_candidates' :
      !packed.sources.length ? 'no_evidence_fits_budget' : null,
    queryEmbeddingCacheHit: cacheHit, indexChunks: stats.chunks, vectorBytes: stats.vectorBytes,
    legacyChunks: stats.legacyChunks, embeddingIdentityVerified: stats.embeddingModel !== null,
    budgetLimited: packed.budgetLimited, rejectedByBudget: packed.rejectedByBudget, tokenChecks: packed.tokenChecks,
  };
}
