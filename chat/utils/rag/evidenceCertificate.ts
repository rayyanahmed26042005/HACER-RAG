import { nearTerm } from './text';
import type { EvidenceAtom } from './evidenceAtoms';
import { demandLabel, extractDemandSlots, type DemandSlot } from './questionDemands';

export type DemandStatus = 'certified' | 'ambiguous' | 'missing';

export interface DemandCertificate {
  demandId: string;
  label: string;
  status: DemandStatus;
  sourceIds: string[];
  matchedTerms: string[];
  values: string[];
  valueLabels: string[];
  units: string[];
  polarities: Array<'positive' | 'negated' | 'unknown'>;
  conflictingValues: string[];
}

export interface EvidenceCertificate {
  query: string;
  demands: DemandCertificate[];
  selectedAtomIds: string[];
  selectedTokens: number;
  tokenBudget: number;
  certifiedDemandCount: number;
  ambiguousDemandCount: number;
  missingDemandCount: number;
  requiredCoverage: number;
}

export interface SelectionResult {
  atoms: EvidenceAtom[];
  certificate: EvidenceCertificate;
  budgetLimited: boolean;
  selectionChecks: number;
  selectionDiagnostics: EvidenceSelectionDiagnostic[];
}

export type EvidenceRejectionReason =
  | 'structural_header'
  | 'no_demand_match'
  | 'budget_exceeded'
  | 'not_selected_after_cover';

export interface EvidenceSelectionDiagnostic {
  atomId: string;
  status: 'selected' | 'rejected';
  reason?: EvidenceRejectionReason;
  matchedDemandIds: string[];
  estimatedTokens: number;
}

function matches(expected: string, actual: string): boolean {
  return expected === actual || nearTerm(expected, actual);
}

function overlap(expected: string[], actual: string[]): string[] {
  return expected.filter(term => actual.some(candidate => matches(term, candidate)));
}

function demandMatch(demand: DemandSlot, atom: EvidenceAtom): {
  score: number; matchedTerms: string[]; values: string[]; valueLabels: string[]; units: string[];
  polarities: Array<'positive' | 'negated' | 'unknown'>;
} {
  if (atom.structural) {
    return { score: 0, matchedTerms: [], values: [], valueLabels: [], units: [], polarities: [] };
  }
  // Role/entity qualifiers are exact vocabulary constraints. Fuzzy matching
  // would incorrectly treat `officer` as the unrelated `office` attribute.
  // Broad entity requests intentionally retain bounded typo tolerance so a
  // misspelled company/product name can still retrieve its supported facts.
  const entityTerms = demand.entityTerms.length
    ? demand.broadEntity
      ? overlap(demand.entityTerms, atom.terms)
      : demand.entityTerms.filter(term => atom.terms.includes(term))
    : [];
  const attributeTerms = demand.attributeTerms.length
    ? overlap(demand.attributeTerms, [...atom.attributes, ...atom.terms])
    : [];
  const clauseTerms = overlap(demand.terms, atom.terms);
  const numeric = overlap(demand.numericConstraints, atom.numbers);
  const units = overlap(demand.unitConstraints, atom.units);
  const relevantFacts = demand.attributeTerms.flatMap(attribute => atom.facts
    .filter(fact => matches(attribute, fact.attribute)));
  const rawContextTerms = demand.terms.filter(term =>
    !demand.attributeTerms.includes(term) && !demand.entityTerms.includes(term) &&
    !['annual', 'available', 'based', 'full', 'monthly', 'new', 'paid', 'remote', 'time'].includes(term));
  // Prefer a distinguishing qualifier over generic population words. For
  // example, `gym` must separate Gym Allowance from Internet Stipend even
  // though both are described as benefits for employees.
  const specificContextTerms = rawContextTerms.filter(term => !['all', 'employees'].includes(term));
  const contextTerms = specificContextTerms.length ? specificContextTerms : rawContextTerms;
  const contextSensitive = contextTerms.some(term =>
    ['contractor', 'contractors', 'eligible', 'employees', 'equipment', 'gym', 'hourly', 'learning', 'part', 'remote', 'sick']
      .includes(term));
  const contextSatisfied = !contextTerms.length || (contextSensitive
    ? contextTerms.every(term => atom.terms.some(candidate => matches(term, candidate)))
    : contextTerms.some(term => atom.terms.some(candidate => matches(term, candidate)))) ||
    (!contextSensitive && relevantFacts.some(fact => !!fact.value || fact.units.length > 0));
  const entitySatisfied = !demand.entityTerms.length || entityTerms.length > 0;
  const attributeSatisfied = demand.broadEntity
    ? entityTerms.length > 0
    : demand.attributeTerms.length
    ? attributeTerms.length > 0
    : clauseTerms.length > 0;
  // A generic attribute such as “allowance” cannot certify a query about a
  // particular population (for example contractor sick leave) unless the
  // source also matches the query's distinguishing context. Entity-specific
  // lookups may rely on the entity plus attribute match.
  if (!entitySatisfied || !attributeSatisfied || (contextSensitive && !contextSatisfied)) {
    return { score: 0, matchedTerms: [], values: [], valueLabels: [], units: [], polarities: [] };
  }
  const score = entityTerms.length * (demand.broadEntity ? 5 : 3) + attributeTerms.length * 3 +
    Math.min(3, clauseTerms.length) + numeric.length * 2 + units.length;
  if (demand.broadEntity) {
    return {
      score,
      matchedTerms: Array.from(new Set([...entityTerms, ...clauseTerms])),
      values: [],
      valueLabels: [],
      units: [],
      polarities: [],
    };
  }
  // If an attribute-scoped parser found values, use those values only. A
  // single unlabelled number remains a safe fallback for prose such as
  // “the device weighs 2.3 kg”; multiple unscoped numbers are deliberately
  // not assigned to the demand because they may describe other facts.
  // A qualitative table value such as `Office: Lisbon` may share an atom with
  // an unrelated joined year.  Do not turn that year into an office conflict.
  const factNumbers = Array.from(new Set(relevantFacts
    .filter(fact => !fact.value)
    .flatMap(fact => fact.numbers)));
  const valueLabels = Array.from(new Set(relevantFacts
    .filter(fact => !fact.numbers.length && fact.value)
    .map(fact => fact.value!)));
  const factUnits = Array.from(new Set(relevantFacts.flatMap(fact => fact.units)));
  const values = !demand.attributeTerms.length || valueLabels.length
    ? []
    : factNumbers.length || atom.numbers.length !== 1
      ? (factNumbers.length ? factNumbers : numeric)
      : atom.numbers;
  const matchedUnits = factUnits.length ? factUnits : (atom.numbers.length === 1 ? atom.units : units);
  const polarities = relevantFacts.length
    ? Array.from(new Set(relevantFacts.map(fact => fact.polarity)))
    : [atom.polarity];
  return {
    score,
    matchedTerms: Array.from(new Set([...entityTerms, ...attributeTerms, ...clauseTerms])),
    values,
    valueLabels,
    units: matchedUnits,
    polarities,
  };
}

function conflictValues(values: string[], valueLabels: string[], polarities: Array<'positive' | 'negated' | 'unknown'>,
  allowMultipleValues: boolean): string[] {
  const conflicts = new Set<string>();
  if (!allowMultipleValues && new Set(values).size > 1) values.forEach(value => conflicts.add(value));
  if (!allowMultipleValues && new Set(valueLabels).size > 1) valueLabels.forEach(value => conflicts.add(value));
  if (new Set(polarities).size > 1) polarities.forEach(polarity => conflicts.add(`polarity:${polarity}`));
  return Array.from(conflicts);
}

function normalizedTerms(text: string): string[] {
  return (text.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
    .match(/[\p{L}\p{N}]+/gu) || []);
}

function containsTermPhrase(text: string, phrase: string): boolean {
  const haystack = normalizedTerms(text);
  const needle = normalizedTerms(phrase);
  if (!needle.length || needle.length > haystack.length) return false;
  return haystack.some((_, index) => needle.every((term, offset) => haystack[index + offset] === term));
}

function canonicalNumber(value: string): string {
  return value.normalize('NFKD').toLowerCase().replace(/\s+/g, '').replace(/,/g, '');
}

function auditSentences(answer: string): string[] {
  // Keep a citation placed after terminal punctuation attached to the claim:
  // `The company ... . [1]` is common model formatting.
  const parts = answer.split(/(?<=[.!?])\s+(?!\[\d)/).filter(Boolean);
  const sentences: string[] = [];
  for (const part of parts) {
    const previous = sentences[sentences.length - 1];
    // Do not split a citation requirement at ordinary document abbreviations
    // such as “Zephyr Dynamics Inc. uses ...”.
    if (previous && /\b(?:Inc|Ltd|LLC|Co|Dr|Mr|Ms|Mrs|Prof|St|No|vs)\.$/iu.test(previous)) {
      sentences[sentences.length - 1] = `${previous} ${part}`;
    } else {
      sentences.push(part);
    }
  }
  return sentences;
}

export function buildEvidenceCertificate(query: string, selectedAtoms: EvidenceAtom[], tokenBudget = 0): EvidenceCertificate {
  const demands = extractDemandSlots(query);
  const certificates = demands.map(demand => {
    const matchesForDemand = selectedAtoms.map(atom => ({ atom, match: demandMatch(demand, atom) }))
      .filter(item => item.match.score > 0);
    const sourceIds = Array.from(new Set(matchesForDemand.map(item => item.atom.id)));
    const matchedTerms = Array.from(new Set(matchesForDemand.flatMap(item => item.match.matchedTerms)));
    const values = Array.from(new Set(matchesForDemand.flatMap(item => item.match.values)));
    const valueLabels = Array.from(new Set(matchesForDemand.flatMap(item => item.match.valueLabels)));
    const units = Array.from(new Set(matchesForDemand.flatMap(item => item.match.units)));
    const polarities = Array.from(new Set(matchesForDemand.flatMap(item => item.match.polarities)));
    const conflicts = conflictValues(values, valueLabels, polarities, demand.enumeration);
    const status: DemandStatus = !sourceIds.length ? 'missing' : conflicts.length ? 'ambiguous' : 'certified';
    return {
      demandId: demand.id,
      label: demandLabel(demand),
      status,
      sourceIds, matchedTerms, values, valueLabels, units, polarities, conflictingValues: conflicts,
    };
  });
  const certified = certificates.filter(item => item.status === 'certified').length;
  const ambiguous = certificates.filter(item => item.status === 'ambiguous').length;
  return {
    query,
    demands: certificates,
    selectedAtomIds: selectedAtoms.map(atom => atom.id),
    selectedTokens: selectedAtoms.reduce((sum, atom) => sum + atom.estimatedTokens, 0),
    tokenBudget,
    certifiedDemandCount: certified,
    ambiguousDemandCount: ambiguous,
    missingDemandCount: certificates.length - certified - ambiguous,
    requiredCoverage: certificates.length ? (certified + ambiguous) / certificates.length : 1,
  };
}

/**
 * Select a compact evidence cover. This is a deterministic prototype, not a
 * claim of optimal set cover. The certificate is recomputed after every
 * admission so the stopping condition is observable and testable.
 */
export function selectCertifiedEvidence(query: string, candidates: EvidenceAtom[], tokenBudget: number,
  maxAtoms = 16): SelectionResult {
  const safeBudget = Math.max(0, Math.floor(tokenBudget));
  const selected: EvidenceAtom[] = [];
  const remaining = [...candidates];
  let selectedTokens = 0;
  let checks = 0;
  const broadEntity = extractDemandSlots(query).some(demand => demand.broadEntity);
  const decisions = new Map<string, EvidenceSelectionDiagnostic>();
  const demandSlots = extractDemandSlots(query);
  while (remaining.length && selected.length < maxAtoms && checks < 64) {
    const current = buildEvidenceCertificate(query, selected, safeBudget);
    // Do not stop after the first apparently sufficient value if the bounded
    // candidate frontier contains a contradictory value or polarity for the
    // same demand. Keeping the second source makes the ambiguity observable.
    const frontier = buildEvidenceCertificate(query, [...selected, ...remaining], safeBudget);
    const conflictRepairNeeded = frontier.ambiguousDemandCount > current.ambiguousDemandCount;
    // A broad entity request is intentionally covered by multiple supported
    // source atoms. Stop only after the bounded candidate frontier is
    // exhausted or the token budget is full; a single definition is not a
    // sufficient answer when the document contains additional entity facts.
    if (current.missingDemandCount === 0 && !conflictRepairNeeded && !broadEntity) break;
    const scored = remaining.map(atom => {
      const matchedDemandIds = demandSlots.filter(demand => demandMatch(demand, atom).score > 0).map(demand => demand.id);
      if (atom.structural) {
        decisions.set(atom.id, { atomId: atom.id, status: 'rejected', reason: 'structural_header', matchedDemandIds, estimatedTokens: atom.estimatedTokens });
      } else if (!matchedDemandIds.length) {
        decisions.set(atom.id, { atomId: atom.id, status: 'rejected', reason: 'no_demand_match', matchedDemandIds, estimatedTokens: atom.estimatedTokens });
      } else if (selectedTokens + atom.estimatedTokens > safeBudget) {
        decisions.set(atom.id, { atomId: atom.id, status: 'rejected', reason: 'budget_exceeded', matchedDemandIds, estimatedTokens: atom.estimatedTokens });
      }
      const before = new Set(current.demands.filter(demand => demand.status !== 'missing').map(demand => demand.demandId));
      const after = buildEvidenceCertificate(query, [...selected, atom], safeBudget);
      const newlyCovered = after.demands.filter(demand => demand.status !== 'missing' && !before.has(demand.demandId)).length;
      const ambiguityGain = Math.max(0, after.ambiguousDemandCount - current.ambiguousDemandCount);
      const matched = demandSlots.reduce((sum, demand) => sum + demandMatch(demand, atom).score, 0);
      const valueDensity = (newlyCovered * 12 + matched * 0.2 + ambiguityGain * 12) / Math.sqrt(atom.estimatedTokens + 8);
      return { atom, valueDensity, newlyCovered, matched };
    }).filter(item => item.matched > 0 && selectedTokens + item.atom.estimatedTokens <= safeBudget)
      .sort((a, b) => b.valueDensity - a.valueDensity || b.newlyCovered - a.newlyCovered || a.atom.sourceOrder - b.atom.sourceOrder);
    checks++;
    if (!scored.length) break;
    const best = scored[0].atom;
    selected.push(best);
    const matchedDemandIds = demandSlots.filter(demand => demandMatch(demand, best).score > 0).map(demand => demand.id);
    decisions.set(best.id, { atomId: best.id, status: 'selected', matchedDemandIds, estimatedTokens: best.estimatedTokens });
    selectedTokens += best.estimatedTokens;
    remaining.splice(remaining.indexOf(best), 1);
  }
  const certificate = buildEvidenceCertificate(query, selected, safeBudget);
  for (const atom of candidates) {
    if (!decisions.has(atom.id)) {
      decisions.set(atom.id, {
        atomId: atom.id,
        status: 'rejected',
        reason: 'not_selected_after_cover',
        matchedDemandIds: demandSlots.filter(demand => demandMatch(demand, atom).score > 0).map(demand => demand.id),
        estimatedTokens: atom.estimatedTokens,
      });
    }
  }
  return {
    atoms: selected,
    certificate,
    budgetLimited: certificate.missingDemandCount > 0 && remaining.some(atom => atom.estimatedTokens + selectedTokens > safeBudget),
    selectionChecks: checks,
    selectionDiagnostics: Array.from(decisions.values()),
  };
}

export interface AnswerContractAudit {
  status: 'certified' | 'partial' | 'ambiguous' | 'repaired' | 'abstained' | 'unverifiable';
  unsupportedProtectedTokens: string[];
  citedSourceIds: string[];
  requiredCitationCoverage: number;
  demandStatuses: DemandCertificate[];
}

export interface RenderedContractEvidence {
  context: string;
  citationMap: Map<number, string>;
}

/** Render source spans with stable, prompt-visible citation ordinals. */
export function renderContractEvidence(atoms: EvidenceAtom[], certificate: EvidenceCertificate): RenderedContractEvidence {
  const citationMap = new Map<number, string>();
  const excerpts = atoms.map((atom, index) => {
    const citation = index + 1;
    citationMap.set(citation, atom.id);
    return `[${citation}] ${atom.docName}${atom.sectionHeader ? ` — ${atom.sectionHeader}` : ''}\n${atom.text}`;
  });
  const status = certificate.demands.map(demand => {
    const citations = demand.sourceIds.map(sourceId => {
      const citation = Array.from(citationMap.entries()).find(([, id]) => id === sourceId)?.[0];
      return citation ? `[${citation}]` : null;
    }).filter((citation): citation is string => !!citation);
    const suffix = citations.length ? ` supported by ${citations.join(', ')}` : '';
    return `- ${demand.label}: ${demand.status}${suffix}`;
  }).join('\n');
  return {
    context: `Evidence status:\n${status || '- no evidence demands identified'}\n\nExcerpts:\n${excerpts.join('\n\n')}`,
    citationMap,
  };
}

/**
 * Render only facts that are completely certified by the selected source
 * spans. This is intentionally narrow: broad prose, qualitative labels,
 * enumerations, conflicts, negation, and missing units stay on the LLM path.
 * The direct path is a latency optimization, not a replacement for synthesis.
 */
export function buildDirectEvidenceAnswer(certificate: EvidenceCertificate, atoms: EvidenceAtom[]): string | null {
  if (!certificate.demands.length || certificate.demands.some(demand =>
    demand.status !== 'certified' || demand.values.length !== 1 || demand.valueLabels.length ||
    demand.polarities.some(polarity => polarity !== 'positive') || demand.units.length !== 1 ||
    !demand.sourceIds.length)) return null;
  const citationByAtomId = new Map(atoms.map((atom, index) => [atom.id, index + 1]));
  const lines = certificate.demands.map(demand => {
    const citations = demand.sourceIds.map(sourceId => citationByAtomId.get(sourceId))
      .filter((citation): citation is number => citation !== undefined);
    if (!citations.length) return null;
    const label = demand.label.charAt(0).toUpperCase() + demand.label.slice(1);
    const unit = demand.units.length === 1 ? ` ${demand.units[0]}` : ` ${demand.units.join(' ')}`;
    return `${label}: ${demand.values[0]}${unit} [${citations.join(', ')}].`;
  });
  return lines.every((line): line is string => !!line) ? lines.join(' ') : null;
}

function protectedTokens(answer: string): { numbers: string[]; units: string[]; entities: string[]; negated: boolean } {
  // Citation ordinals such as [1] identify a source and are not factual
  // numeric claims. Remove them before checking protected values.
  const prose = answer.replace(/\[(?:\d+\s*,\s*)*\d+\]/g, ' ');
  const numbers = Array.from(new Set((prose.match(/(?<![\p{L}])(?:[$€£]\s*)?\d+(?:(?:[,.]\s*|\s+(?=\d{3}(?:\s|$)))\d+)*(?:\s*[–-]\s*\d+(?:(?:[,.]\s*|\s+(?=\d{3}(?:\s|$)))\d+)*)?(?![\p{L}])/gu) || [])
    .map(value => value.normalize('NFKD').toLowerCase().replace(/\s+/g, ' ').trim())));
  const units = Array.from(new Set((prose.match(/(?<![\p{L}])(?:hours?|days?|weeks?|months?|years?|minutes?|seconds?|ghz|mhz|hz|kg|mg|lbs?|km|cm|mm|gb|mb|tb|usd|eur|gbp|°c|%|g|m|c)(?![\p{L}])/giu) || [])
    .map(value => value.normalize('NFKD').toLowerCase().trim())));
  const firstWord = prose.match(/^\s*([\p{Lu}][\p{L}\p{N}-]*)/u)?.[1]?.toLowerCase();
  const ignoredCapitalized = new Set(['a', 'an', 'the', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'where', 'when', 'how', 'it']);
  const entities = Array.from(new Set((prose.match(/\b[\p{Lu}][\p{L}\p{N}-]*\b/gu) || [])
    .filter(value => value.toLowerCase() !== firstWord || /[A-Z][a-z]+[A-Z0-9]/.test(value))
    .filter(value => !ignoredCapitalized.has(value.toLowerCase()))
    .map(value => value.toLowerCase().trim())));
  return { numbers, units, entities, negated: /\b(?:not|never|no|without|cannot|can't|doesn't|isn't|aren't)\b/i.test(prose) };
}

/**
 * Audit only protected factual tokens. It deliberately does not pretend that
 * lexical overlap proves full entailment; answer correctness still requires
 * human or labeled evaluation.
 */
export function auditAnswerContract(answer: string, contract: EvidenceCertificate,
  atoms: EvidenceAtom[]): AnswerContractAudit {
  const unsupported = new Set<string>();
  const citedOrdinals = new Set<number>();
  const sentences = auditSentences(answer);
  const factualVerb = /\b(?:is|are|was|were|has|have|allows?|approved|weighs?|costs?|lasts?|located|joined|measures?)\b/i;
  for (const sentence of sentences) {
    const protectedSentence = protectedTokens(sentence);
    // Models commonly group citations as `[1, 2]`; treating that as only
    // source 1 incorrectly marks a supported answer unverifiable.
    const ordinals = Array.from(sentence.matchAll(/\[((?:\d+\s*,\s*)*\d+)\]/g))
      .flatMap(match => match[1].split(/\s*,\s*/).map(Number))
      .filter(ordinal => Number.isInteger(ordinal) && ordinal > 0);
    ordinals.forEach(ordinal => citedOrdinals.add(ordinal));
    const citedAtoms = ordinals.map(ordinal => atoms[ordinal - 1]).filter((atom): atom is EvidenceAtom => !!atom);
    const refusal = /\b(?:not found|don't have enough information|no information)\b/i.test(proseForAudit(sentence));
    const hasProtectedClaim = protectedSentence.numbers.length > 0 || protectedSentence.units.length > 0 ||
      protectedSentence.entities.length > 0 || protectedSentence.negated || factualVerb.test(sentence);
    if (refusal || !hasProtectedClaim) continue;
    if (!citedAtoms.length) {
      unsupported.add('citation:missing');
      continue;
    }
    const allowedNumbers = new Set(citedAtoms.flatMap(atom => atom.numbers).map(canonicalNumber));
    const allowedUnits = new Set(citedAtoms.flatMap(atom => atom.units));
    // Single-word place/product labels are not emitted by the conservative
    // atom NER helper. Include source terms for capitalized answer labels so a
    // cited `Office: Lisbon` supports “Lisbon”, while a cited Lisbon span does
    // not authorize “Reykjavik”. This remains lexical auditing, not entailment.
    const allowedEntities = new Set([
      ...citedAtoms.flatMap(atom => atom.entities),
      ...citedAtoms.flatMap(atom => atom.terms),
    ]);
    protectedSentence.numbers.filter(value => !allowedNumbers.has(canonicalNumber(value))).forEach(value => unsupported.add(value));
    protectedSentence.units.filter(value => !allowedUnits.has(value)).forEach(value => unsupported.add(value));
    protectedSentence.entities.filter(value => !allowedEntities.has(value)).forEach(value => unsupported.add(value));
    const citedValueLabels = new Set(citedAtoms.flatMap(atom => atom.facts
      .filter(fact => !!fact.value).map(fact => fact.value!)));
    const knownValueLabels = new Set(contract.demands.flatMap(demand => demand.valueLabels));
    for (const label of knownValueLabels) {
      if (containsTermPhrase(sentence, label) && !Array.from(citedValueLabels).some(value => containsTermPhrase(value, label))) {
        unsupported.add(label);
      }
    }
    const hasPositiveEvidence = citedAtoms.some(atom => atom.polarity === 'positive');
    const hasNegatedEvidence = citedAtoms.some(atom => atom.polarity === 'negated');
    if (protectedSentence.negated && hasPositiveEvidence && !hasNegatedEvidence) unsupported.add('polarity:negated');
    if (!protectedSentence.negated && hasNegatedEvidence && !hasPositiveEvidence) unsupported.add('polarity:positive');
  }
  const citedSourceIds = Array.from(citedOrdinals).map(ordinal => atoms[ordinal - 1]?.id)
    .filter((id): id is string => !!id);
  const certified = contract.demands.filter(demand => demand.status === 'certified' || demand.status === 'ambiguous');
  const coveredByCitation = certified.filter(demand => demand.sourceIds.some(id => citedSourceIds.includes(id))).length;
  const requiredCitationCoverage = certified.length ? coveredByCitation / certified.length : 1;
  const missingCitation = certified.length > 0 && requiredCitationCoverage < 1;
  const hardUnsupported = Array.from(unsupported).filter(token => token !== 'citation:missing');
  const status = hardUnsupported.length ? 'abstained' : missingCitation || unsupported.has('citation:missing')
    ? 'unverifiable' : contract.ambiguousDemandCount ? 'ambiguous'
      : contract.certifiedDemandCount === 0 && contract.missingDemandCount > 0 ? 'abstained'
        : contract.missingDemandCount ? 'partial' : 'certified';
  return {
    status,
    unsupportedProtectedTokens: Array.from(unsupported),
    citedSourceIds,
    requiredCitationCoverage,
    demandStatuses: contract.demands,
  };
}

function proseForAudit(answer: string): string {
  return answer.replace(/\[\d+\]/g, ' ');
}
