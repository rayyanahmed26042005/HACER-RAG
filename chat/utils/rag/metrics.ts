import { terms } from './text';

/** Optional reference labels; absent labels produce null, never fake quality. */
export interface RagLabels { relevantChunkIds?: string[]; referenceAnswer?: string }
export function evaluateRag(retrievedIds: string[], context: string, answer: string, labels?: RagLabels) {
  const gold = labels?.relevantChunkIds ? new Set(labels.relevantChunkIds) : null;
  const ids = Array.from(new Set(retrievedIds));
  const hits = gold ? ids.filter(id => gold.has(id)).length : 0;
  const rank = gold ? ids.findIndex(id => gold.has(id)) : -1;
  const responseTerms = terms(answer);
  const contextTerms = new Set(terms(context));
  let f1: number | null = null;
  if (labels?.referenceAnswer !== undefined) {
    const reference = terms(labels.referenceAnswer);
    const counts = new Map<string, number>();
    reference.forEach(term => counts.set(term, (counts.get(term) || 0) + 1));
    let overlap = 0;
    responseTerms.forEach(term => { if (counts.get(term)) { overlap++; counts.set(term, counts.get(term)! - 1); } });
    f1 = responseTerms.length + reference.length ? 2 * overlap / (responseTerms.length + reference.length) : 1;
  }
  return {
    recallAtK: gold?.size ? hits / gold.size : null,
    precisionAtK: gold ? (ids.length ? hits / ids.length : 0) : null,
    reciprocalRank: gold?.size ? (rank >= 0 ? 1 / (rank + 1) : 0) : null,
    answerTokenF1: f1,
    // Word overlap is NOT entailment, factual correctness or hallucination rate.
    answerContextTokenOverlap: responseTerms.length
      ? responseTerms.filter(term => contextTerms.has(term)).length / responseTerms.length : null,
    groundedness: null,
    correctness: null,
  };
}
