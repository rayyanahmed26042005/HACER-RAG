import { evidenceUnits, terms, nearTerm } from './text';
import type { Chunk } from './chunker';

export type EvidencePolarity = 'positive' | 'negated' | 'unknown';

export interface EvidenceFact {
  /** Canonical attribute represented by this source span, such as `weight`. */
  attribute: string;
  numbers: string[];
  units: string[];
  /** A non-numeric value label, used only for conflict detection. */
  value?: string;
  polarity: EvidencePolarity;
}

export interface EvidenceAtom {
  id: string;
  docId: string;
  docName: string;
  chunkIndex: number;
  sourceOrder: number;
  sectionHeader?: string;
  text: string;
  terms: string[];
  entities: string[];
  attributes: string[];
  numbers: string[];
  units: string[];
  temporalMarkers: string[];
  polarity: EvidencePolarity;
  /** Table column labels are structure, not answerable evidence. */
  structural?: boolean;
  /** Attribute-scoped values used for conflict detection. */
  facts: EvidenceFact[];
  /** Stable entity/attribute keys used to detect differing values. */
  factKeys: string[];
  estimatedTokens: number;
}

const ATTRIBUTE_HINTS = new Set([
  'allowance', 'allowances', 'approved', 'approval', 'battery', 'benefit', 'budget', 'capacity',
  'cost', 'date', 'duration', 'employees', 'fee', 'hours', 'joined', 'length',
  'location', 'office', 'price', 'product', 'products', 'role', 'salary',
  'services', 'size', 'status', 'stipend', 'stipends', 'temperature', 'types', 'version', 'warranty',
  'weight', 'width', 'year', 'years',
]);

const UNIT_PATTERN = /(?<![\p{L}])(?:hours?|days?|weeks?|months?|years?|minutes?|seconds?|ghz|mhz|hz|kg|mg|lbs?|km|cm|mm|gb|mb|tb|usd|eur|gbp|°c|%|g|m|c)(?![\p{L}])/giu;
const NUMBER_PATTERN = /(?:[$€£]\s*)?\d+(?:[,.]\d+)*(?:\s*[–-]\s*\d+(?:[,.]\d+)*)?/gu;
const YEAR_PATTERN = /\b(?:19|20)\d{2}\b/g;
const NEGATION_PATTERN = /\b(?:not|never|no|without|cannot|can't|couldn't|doesn't|isn't|aren't|wasn't|weren't|neither|nor)\b/i;

const ATTRIBUTE_ALIASES: Record<string, string[]> = {
  allowance: ['allowance', 'allowances', 'stipend', 'stipends'],
  battery: ['battery', 'last', 'lasts', 'duration'],
  cost: ['cost', 'fee', 'price'],
  date: ['date'],
  duration: ['duration', 'last', 'lasts'],
  hours: ['hour', 'hours'],
  location: ['location', 'located', 'based'],
  office: ['office', 'location', 'located', 'based'],
  price: ['price', 'cost', 'fee'],
  status: ['status', 'approved', 'approval'],
  temperature: ['temperature', 'range'],
  weight: ['weight', 'weigh', 'weighs'],
  width: ['width'],
  year: ['year', 'years', 'joined'],
};

function normalizeValue(value: string): string {
  return value.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
    .replace(/[|:;,()[\]{}]/g, ' ').replace(/\s+/g, ' ').trim();
}

/** Extract capitalized names/product labels without requiring an NER model. */
export function extractEntityLabels(text: string): string[] {
  const labels = new Set<string>();
  // Do not allow a heading and the first word of its body to fuse into one
  // entity merely because `\s` also matches newlines.
  for (const line of text.split(/\r?\n+/)) {
    const matches = line.match(/\b(?:[\p{Lu}][\p{L}\p{N}-]*(?:\s+|$)){1,5}/gu) || [];
    for (const match of matches) {
      const value = match.trim().replace(/[.,;:!?]+$/, '');
      const words = value.split(/\s+/).filter(Boolean);
      // Avoid treating a sentence-initial common word as an entity. Multi-word
      // labels and mixed-case product names remain useful even when one token
      // is sentence-initial.
      if (words.length > 1 || /[A-Z][a-z]+[A-Z0-9]/.test(value) || /\d/.test(value)) {
        labels.add(normalizeValue(value));
      }
    }
  }
  return Array.from(labels);
}

export function extractNumbers(text: string): string[] {
  return Array.from(new Set((text.match(NUMBER_PATTERN) || []).map(value => normalizeValue(value))));
}

export function extractUnits(text: string): string[] {
  return Array.from(new Set(Array.from(text.matchAll(UNIT_PATTERN), match => normalizeValue(match[0]))));
}

function extractAttributes(text: string, textTerms: string[]): string[] {
  const attributes = new Set<string>();
  for (const term of textTerms) if (ATTRIBUTE_HINTS.has(term)) attributes.add(term);
  for (const [attribute, aliases] of Object.entries(ATTRIBUTE_ALIASES)) {
    if (aliases.some(alias => textTerms.includes(alias))) attributes.add(attribute);
  }
  // Table/colon labels are strong attribute signals, even when the vocabulary
  // is domain-specific and not in the compact mobile hint list.
  for (const match of text.matchAll(/(?:^|\|)\s*([^|:\n]{2,40})\s*:/giu)) {
    const label = terms(match[1]);
    label.forEach(term => attributes.add(term));
  }
  return Array.from(attributes);
}

function attributeMatchesLabel(attribute: string, label: string): boolean {
  const labelTerms = terms(label);
  const aliases = ATTRIBUTE_ALIASES[attribute] || [attribute];
  return labelTerms.includes(attribute) || aliases.some(alias => labelTerms.includes(alias));
}

function escapedAliases(attribute: string): string {
  return (ATTRIBUTE_ALIASES[attribute] || [attribute])
    .slice().sort((a, b) => b.length - a.length)
    .map(alias => alias.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
}

function localFactSpan(text: string, attribute: string): { valueText: string; start: number; end: number } | null {
  const match = text.match(new RegExp(`\\b(?:${escapedAliases(attribute)})\\b`, 'iu'));
  if (!match || match.index === undefined) return null;
  const start = match.index;
  let end = text.length;
  const followingText = text.slice(start + match[0].length);
  for (const otherAttribute of Object.keys(ATTRIBUTE_ALIASES)) {
    if (otherAttribute === attribute) continue;
    const nextAttribute = followingText.match(new RegExp(`\\b(?:${escapedAliases(otherAttribute)})\\b`, 'iu'));
    // Words such as `lasts` and `hours` can describe the current battery
    // fact; they are not automatically the start of a new attribute. Treat a
    // later alias as a boundary only when prose introduces it as a separate
    // coordinated fact ("and ... temperature") or after a delimiter.
    const prefix = nextAttribute?.index === undefined ? '' : followingText.slice(
      Math.max(0, nextAttribute.index - 48), nextAttribute.index,
    );
    const startsAnotherFact = /(?:\b(?:and|but|while|whereas)\b[^.!?\n|;]{0,40}|,\s*[^.!?\n|;]{0,24})$/iu.test(prefix);
    if (nextAttribute?.index !== undefined && startsAnotherFact) {
      end = Math.min(end, start + match[0].length + nextAttribute.index);
    }
  }
  for (let index = start + match[0].length; index < text.length; index++) {
    if (index >= end) break;
    const character = text[index];
    const next = text[index + 1] || '';
    // Decimal points belong to the value; sentence punctuation ends it.
    if ((character === '.' || character === '!' || character === '?') && !/\d/.test(next)) {
      end = index;
      break;
    }
    if (character === '\n' || character === '|' || character === ';') {
      end = index;
      break;
    }
  }
  return { valueText: text.slice(start, end), start, end };
}

function boundedFactEnd(text: string, start: number, limit: number): number {
  for (let index = start; index < limit; index++) {
    const character = text[index];
    const next = text[index + 1] || '';
    if ((character === '.' || character === '!' || character === '?') && !/\d/.test(next)) return index;
    if (character === '\n' || character === '|' || character === ';') return index;
  }
  return limit;
}

function localPolarity(text: string, start: number, end: number, attribute: string): EvidencePolarity {
  const aliases = escapedAliases(attribute);
  const sliceStart = Math.max(0, start - 32);
  const target = new RegExp(`\\b(?:${aliases})\\b`, 'iu').exec(text.slice(sliceStart, end));
  if (!target || target.index === undefined) return 'unknown';
  const beforeTarget = text.slice(sliceStart, sliceStart + target.index);
  return /\b(?:not|never|no|without|cannot|can't|couldn't|doesn't|isn't|aren't|wasn't|weren't)\s*$/i.test(beforeTarget)
    ? 'negated' : 'positive';
}

function normalizedFactValue(value: string): string | undefined {
  const normalized = value.normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
    .toLowerCase().replace(/[^\p{L}\p{N}%$€£./+-]+/gu, ' ').trim();
  return normalized || undefined;
}

function splitTableCells(value: string): string[] {
  return value.split('|').map(cell => cell.trim()).filter(Boolean);
}

function looksLikeWhitespaceTableHeader(value: string): boolean {
  const headerTerms = new Set(terms(value));
  const knownColumns = ['name', 'title', 'office', 'joined', 'role', 'location', 'status', 'date'];
  return knownColumns.filter(column => headerTerms.has(column)).length >= 2 &&
    (headerTerms.has('name') || headerTerms.has('title'));
}

function whitespaceTableHeaderAttributes(value: string): string[] {
  const headerTerms = terms(value);
  const attributes = new Set<string>();
  if (headerTerms.includes('office')) attributes.add('office');
  if (headerTerms.includes('joined')) attributes.add('joined');
  if (headerTerms.includes('location')) attributes.add('location');
  if (headerTerms.includes('role') || headerTerms.includes('title')) attributes.add('role');
  if (headerTerms.includes('status')) attributes.add('status');
  return Array.from(attributes);
}

function tableFacts(row: string, header: string, attributes: string[]): EvidenceFact[] {
  const headerCells = splitTableCells(header);
  const rowCells = splitTableCells(row);
  const facts: EvidenceFact[] = [];
  if (header.includes('|')) {
    for (let index = 0; index < Math.min(headerCells.length, rowCells.length); index++) {
      const headerAttributes = extractAttributes(headerCells[index], terms(headerCells[index]));
      const value = rowCells[index];
      for (const attribute of headerAttributes.filter(candidate => attributes.includes(candidate))) {
        facts.push({
          attribute,
          numbers: extractNumbers(value),
          units: extractUnits(value),
          value: extractNumbers(value).length ? undefined : normalizedFactValue(value),
          polarity: NEGATION_PATTERN.test(value) ? 'negated' : 'positive',
        });
      }
    }
  }
  // PDF/text extraction often flattens a table into whitespace-separated
  // lines. Preserve the reliable trailing fields for the common
  // `Name Title Office Joined` layout without pretending that arbitrary
  // whitespace is a complete table parser.
  if (!header.includes('|') && looksLikeWhitespaceTableHeader(header)) {
    const rowTokens = row.split(/\s+/).filter(Boolean);
    const joinedIndex = rowTokens.findIndex(token => /^(?:19|20)\d{2}$/.test(token));
    if (joinedIndex > 0 && attributes.includes('joined')) {
      facts.push({ attribute: 'joined', numbers: [rowTokens[joinedIndex]], units: [], polarity: 'positive' });
    }
    if (joinedIndex > 1 && (attributes.includes('office') || attributes.includes('location'))) {
      const officeValue = rowTokens[joinedIndex - 1];
      facts.push({ attribute: attributes.includes('office') ? 'office' : 'location', numbers: [], units: [], value: normalizedFactValue(officeValue), polarity: 'positive' });
    }
  }
  return facts;
}

/**
 * Associate values with the attribute that introduces them. A source atom can
 * contain several facts (for example `Weight: 2.3 kg. Year: 2024`); treating
 * every number in that atom as a value for every demand creates false
 * ambiguity. The parser is intentionally conservative and falls back to an
 * attribute's local verb phrase when no colon-labelled span is present.
 */
function extractFacts(text: string, attributes: string[]): EvidenceFact[] {
  const facts: EvidenceFact[] = [];
  const labelled = Array.from(text.matchAll(/([\p{L}\p{N}][\p{L}\p{N}\s/()-]{1,39}):\s*/gu));
  for (let i = 0; i < labelled.length; i++) {
    const match = labelled[i];
    const label = match[1].trim();
    const start = (match.index || 0) + match[0].length;
    const nextStart = i + 1 < labelled.length ? (labelled[i + 1].index || text.length) : text.length;
    const valueEnd = boundedFactEnd(text, start, nextStart);
    const valueText = text.slice(start, valueEnd).trim();
    for (const attribute of attributes) {
      if (!attributeMatchesLabel(attribute, label)) continue;
      facts.push({
        attribute,
        numbers: extractNumbers(valueText),
        units: extractUnits(valueText),
        value: extractNumbers(valueText).length ? undefined : normalizedFactValue(valueText),
        polarity: NEGATION_PATTERN.test(valueText) ? 'negated' : 'positive',
      });
    }
  }

  for (const attribute of attributes) {
    if (facts.some(fact => fact.attribute === attribute)) continue;
    const span = localFactSpan(text, attribute);
    if (!span) continue;
    facts.push({
      attribute,
      numbers: extractNumbers(span.valueText),
      units: extractUnits(span.valueText),
      value: extractNumbers(span.valueText).length ? undefined : normalizedFactValue(span.valueText),
      polarity: localPolarity(text, span.start, span.end, attribute),
    });
  }
  return facts;
}

function factKeys(entities: string[], attributes: string[]): string[] {
  const entityKey = entities.length ? entities.sort().join('+') : '*';
  return (attributes.length ? attributes : ['*']).map(attribute => `${entityKey}|${attribute}`);
}

/**
 * Convert a chunk into source-preserving atoms. This is deliberately
 * deterministic and extractive: the atom metadata can guide selection, but
 * the original text is always retained for the generator.
 */
export function atomizeChunk(chunk: Chunk): EvidenceAtom[] {
  const units = evidenceUnits(chunk.text);
  const pipeUnits = units.filter(unit => unit.includes('|'));
  const whitespaceHeader = units.find(unit => looksLikeWhitespaceTableHeader(unit)) || '';
  const candidateTableHeader = pipeUnits[0] || whitespaceHeader;
  const hasTableHeader = (pipeUnits.length > 1 || !!whitespaceHeader) &&
    extractAttributes(candidateTableHeader, terms(candidateTableHeader)).length > 0;
  const firstTableUnit = hasTableHeader ? units.findIndex(unit => unit === candidateTableHeader) : -1;
  const tableHeader = firstTableUnit >= 0 ? units[firstTableUnit] : '';
  return units.map((text, sourceOrder) => {
    const atomTerms = Array.from(new Set(terms(`${chunk.sectionHeader || ''}\n${text}`)));
    const entities = extractEntityLabels(`${chunk.sectionHeader || ''}\n${text}`);
    const headerAttributes = tableHeader && sourceOrder > firstTableUnit &&
      (text.includes('|') || !tableHeader.includes('|'))
      ? (tableHeader.includes('|')
        ? splitTableCells(tableHeader).flatMap(cell => extractAttributes(cell, terms(cell)))
        : whitespaceTableHeaderAttributes(tableHeader))
      : [];
    const attributes = sourceOrder === firstTableUnit
      ? []
      : Array.from(new Set([...extractAttributes(text, atomTerms), ...headerAttributes]));
    const rowFacts = tableHeader && sourceOrder > firstTableUnit &&
      (text.includes('|') || !tableHeader.includes('|'))
      ? tableFacts(text, tableHeader, attributes)
      : [];
    const numbers = extractNumbers(text);
    const unitsFound = extractUnits(text);
    const polarity: EvidencePolarity = NEGATION_PATTERN.test(text) ? 'negated' : 'positive';
    const facts = [...extractFacts(text, attributes), ...rowFacts];
    return {
      id: `${chunk.docId}:${chunk.chunkIndex}:${sourceOrder}`,
      docId: chunk.docId,
      docName: chunk.docName,
      chunkIndex: chunk.chunkIndex,
      sourceOrder,
      sectionHeader: chunk.sectionHeader,
      text,
      terms: atomTerms,
      entities,
      attributes,
      numbers,
      units: unitsFound,
      temporalMarkers: Array.from(new Set(text.match(YEAR_PATTERN) || [])),
      polarity,
      structural: sourceOrder === firstTableUnit,
      facts,
      factKeys: factKeys(entities, attributes),
      estimatedTokens: Math.max(1, Math.ceil(text.length / 4)),
    };
  });
}

export function atomizeChunks(chunks: Chunk[]): EvidenceAtom[] {
  return chunks.flatMap(atomizeChunk);
}

export function termsMatch(expected: string, actual: string): boolean {
  return expected === actual || nearTerm(expected, actual);
}

export function atomSupportsFactKey(atom: EvidenceAtom, key: string): boolean {
  return atom.factKeys.includes(key) || atom.factKeys.some(candidate => {
    const [entity, attribute] = candidate.split('|');
    const [expectedEntity, expectedAttribute] = key.split('|');
    return (expectedEntity === '*' || entity === '*' || termsMatch(expectedEntity, entity)) &&
      (expectedAttribute === '*' || attribute === '*' || termsMatch(expectedAttribute, attribute));
  });
}
