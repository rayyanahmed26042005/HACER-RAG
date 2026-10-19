import type { SearchResult } from './vectorStore';
import { evidenceUnits, matchedTerms, terms } from './text';

export interface EvidenceSource {
  id: string;
  docId: string;
  docName: string;
  chunkIndex: number;
  sectionHeader?: string;
  excerpts: string[];
  denseScore: number;
  lexicalScore: number;
}
export interface PackingOptions {
  maxTokens: number;
  maxChunks: number;
  demand: 'single_hop' | 'multi_hop' | 'broad';
  deduplicate: boolean;
  countTokens: (text: string) => Promise<number>;
  /** Checks the full templated prompt + output reserve, not just excerpts. */
  fitsPrompt?: (context: string) => Promise<boolean>;
  strategy?: 'evidence' | 'fixed_top_k';
}
interface Unit { result: SearchResult; text: string; order: number; matches: string[]; cost: number }

export function renderSources(sources: EvidenceSource[]): string {
  return sources.map((source, i) => `[${i + 1}] ${source.docName}${source.sectionHeader ? ` — ${source.sectionHeader}` : ''}\n${source.excerpts.join('\n')}`).join('\n\n');
}

function sourcesFor(units: Unit[]): EvidenceSource[] {
  const sources = new Map<string, EvidenceSource>();
  // Group in selection order, but preserve original source order within a chunk.
  for (const unit of units) {
    const { chunk } = unit.result;
    const id = `${chunk.docId}:${chunk.chunkIndex}`;
    if (!sources.has(id)) sources.set(id, {
      id, docId: chunk.docId, docName: chunk.docName, chunkIndex: chunk.chunkIndex,
      sectionHeader: chunk.sectionHeader, excerpts: [],
      denseScore: unit.result.denseScore, lexicalScore: unit.result.bm25Score,
    });
  }
  sources.forEach(source => {
    source.excerpts = units.filter(unit => `${unit.result.chunk.docId}:${unit.result.chunk.chunkIndex}` === source.id)
      .sort((a, b) => a.order - b.order).map(unit => unit.text);
  });
  return Array.from(sources.values());
}

/** Coverage-gain per estimated prefill token, with exact admission checks.
 * No summarizer model, no arbitrary text slicing, no word-overlap deletion of
 * conflicting facts. This is an engineering heuristic, not an optimal solver.
 */
export async function packEvidence(query: string, candidates: SearchResult[], options: PackingOptions) {
  const queryTerms = Array.from(new Set(terms(query))).slice(0, 32);
  const enumeration = /\b(list|all|products|offices|names|types|services)\b/i.test(query);
  const units: Unit[] = [];
  let duplicatesRemoved = 0;
  const seen = new Set<string>();
  for (const result of candidates) {
    const segments = options.strategy === 'fixed_top_k' ? [result.chunk.text] : evidenceUnits(result.chunk.text);
    const columnHeader = segments[0]?.includes('|') ? segments[0] : '';
    const sourceUnits = segments.map((segment, order) => {
      // Resolve immediate pronoun/continuation references by retaining their
      // preceding source sentence. Never stitch text from different documents.
      const text = columnHeader && order > 0 && segment.includes('|') ? `${columnHeader}\n${segment}` :
        order > 0 && /^(?:it|its|they|their|this|these|the (?:device|model|system|x\d+))\b/i.test(segment)
          ? `${segments[order - 1]} ${segment}` : segment;
      return { result, text, order,
        matches: matchedTerms(queryTerms, new Set(terms(`${result.chunk.sectionHeader || ''}\n${text}`))),
        cost: Math.max(1, Math.ceil(text.length / 4)) };
    }).sort((a, b) => b.matches.length - a.matches.length || a.order - b.order).slice(0, 12);
    for (const unit of sourceUnits) {
      const key = `${result.chunk.sectionHeader || ''}\n${unit.text.replace(/\s+/g, ' ').trim()}`;
      if (options.deduplicate && seen.has(key)) { duplicatesRemoved++; continue; }
      seen.add(key); units.push(unit);
    }
  }
  const selected: Unit[] = [];
  const covered = new Set<string>();
  const groups = new Set<string>();
  const chunks = new Set<string>();
  let rejectedByBudget = 0;
  let tokenChecks = 0;
  const remaining = [...units];
  // Bound admission calls; all original candidate IDs are retained in metrics.
  while (remaining.length && selected.length < 16 && tokenChecks < 48) {
    const utility = (unit: Unit): number => {
      const fresh = unit.matches.filter(term => !covered.has(term)).length;
      const group = `${unit.result.chunk.docId}:${unit.result.chunk.sectionHeader || unit.result.chunk.chunkIndex}`;
      const diversity = !groups.has(group) ? (enumeration ? 1.4 : 0.25) : 0;
      const relevance = unit.result.rrfScore * 30 + Math.max(0, unit.result.denseScore) * 0.2;
      return (fresh * 2 + unit.matches.length * 0.2 + diversity + relevance) / Math.sqrt(unit.cost + 12);
    };
    if (options.strategy !== 'fixed_top_k') remaining.sort((a, b) => utility(b) - utility(a) || a.order - b.order);
    const unit = remaining.shift()!;
    const chunkId = `${unit.result.chunk.docId}:${unit.result.chunk.chunkIndex}`;
    if (!chunks.has(chunkId) && chunks.size >= options.maxChunks) continue;
    // Exact containment only, within the same document+section. Keep differing
    // quantities and negations even if most words overlap.
    if (options.deduplicate && selected.some(other =>
      other.result.chunk.docId === unit.result.chunk.docId &&
      other.result.chunk.sectionHeader === unit.result.chunk.sectionHeader && other.text.includes(unit.text))) {
      duplicatesRemoved++; continue;
    }
    const context = renderSources(sourcesFor([...selected, unit]));
    tokenChecks++;
    if (await options.countTokens(context) > options.maxTokens ||
        (options.fitsPrompt && !(await options.fitsPrompt(context)))) {
      rejectedByBudget++; continue;
    }
    selected.push(unit); chunks.add(chunkId);
    unit.matches.forEach(term => covered.add(term));
    groups.add(`${unit.result.chunk.docId}:${unit.result.chunk.sectionHeader || unit.result.chunk.chunkIndex}`);
  }
  const sources = sourcesFor(selected);
  const context = renderSources(sources);
  const available = new Set(units.flatMap(unit => unit.matches));
  const rawChars = candidates.reduce((sum, result) => sum + result.chunk.text.length, 0);
  return { context, sources, contextTokens: context ? await options.countTokens(context) : 0,
    duplicatesRemoved, rejectedByBudget, tokenChecks, selectedUnits: selected.length,
    queryTerms, matchedQueryTerms: Array.from(covered),
    uncoveredQueryTerms: queryTerms.filter(term => !covered.has(term)),
    availableTermCoverage: available.size / Math.max(1, queryTerms.length),
    queryTermCoverage: covered.size / Math.max(1, queryTerms.length),
    evidenceChars: selected.reduce((sum, unit) => sum + unit.text.length, 0),
    candidateChars: rawChars,
    budgetLimited: rejectedByBudget > 0 || (remaining.length > 0 && tokenChecks >= 48),
  };
}
