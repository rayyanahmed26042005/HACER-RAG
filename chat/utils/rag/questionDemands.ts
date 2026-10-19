import { terms } from './text';
import { extractEntityLabels, extractNumbers, extractUnits } from './evidenceAtoms';

export interface DemandSlot {
  id: string;
  clause: string;
  terms: string[];
  entityTerms: string[];
  attributeTerms: string[];
  numericConstraints: string[];
  unitConstraints: string[];
  required: boolean;
  enumeration: boolean;
  /** Broad entity-description request such as “What is Zypher?” */
  broadEntity: boolean;
}

const ATTRIBUTE_WORDS = new Set([
  'allowance', 'allowances', 'approved', 'approval', 'battery', 'benefit', 'budget', 'capacity',
  'cost', 'date', 'duration', 'fee', 'hours', 'joined', 'length', 'location',
  'office', 'offices', 'price', 'product', 'products', 'role', 'salary', 'services', 'size',
  'stipend', 'stipends',
  'status', 'temperature', 'types', 'version', 'warranty', 'weight', 'width',
  'year', 'years',
]);
const ENUMERATION_WORDS = new Set(['all', 'list', 'names', 'offices', 'products', 'services', 'types']);
const QUESTION_WORDS = new Set(['what', 'which', 'who', 'where', 'when', 'how', 'does', 'do', 'is', 'are', 'the', 'of', 'and', 'or', 'please', 'tell', 'me', 'about']);
// These terms describe a specific role, population, or policy lookup. They
// must not be treated as a broad entity request merely because the question
// has the shape “Who/What is ...?”.
const NON_BROAD_TERMS = new Set([
  'allowance', 'allowances', 'cmo', 'contractor', 'contractors', 'employee', 'employees',
  'hourly', 'leave', 'officer', 'part', 'sick', 'stipend', 'stipends', 'title',
]);

function splitClauses(query: string): string[] {
  const value = query.replace(/[?!.]+$/g, '').trim();
  if (!value) return [];
  // Keep “battery life and weight” as two demands, but avoid splitting a
  // short phrase such as “research and development” into meaningless slots.
  const pieces = value.split(/\s+(?:and|&)\s+|\s*;\s*|\s*,\s*/i).map(piece => piece.trim()).filter(Boolean);
  return pieces.length > 1 ? pieces : [value];
}

function attributeTerms(clauseTerms: string[]): string[] {
  const explicit = clauseTerms.filter(term => ATTRIBUTE_WORDS.has(term));
  if (explicit.length) return Array.from(new Set(explicit));
  return clauseTerms.filter(term => !QUESTION_WORDS.has(term)).slice(-3);
}

function isBroadEntityClause(clause: string, clauseTerms: string[]): boolean {
  const explicitAttributes = clauseTerms.filter(term => ATTRIBUTE_WORDS.has(term));
  if (explicitAttributes.length || clauseTerms.some(term => NON_BROAD_TERMS.has(term))) return false;
  return /^(?:what|who)\s+(?:is|are)\s+.+$/i.test(clause.trim()) ||
    /^(?:tell\s+me\s+about|describe)\s+.+$/i.test(clause.trim());
}

/**
 * Conservative, no-model demand extraction. It is intentionally allowed to
 * over-split; the benchmark measures the resulting retrieval cost and false
 * missing-slot rate rather than hiding ambiguity in a black-box parser.
 */
export function extractDemandSlots(query: string): DemandSlot[] {
  const clauses = splitClauses(query);
  if (!clauses.length) return [];
  const allEntities = extractEntityLabels(query).flatMap(entity => terms(entity));
  const fullTerms = terms(query);
  return clauses.map((clause, index) => {
    const clauseTerms = terms(clause);
    const broadEntity = isBroadEntityClause(clause, clauseTerms);
    const entityTerms = Array.from(new Set([
      ...extractEntityLabels(clause).flatMap(entity => terms(entity)),
      ...allEntities,
      ...(broadEntity ? clauseTerms.filter(term => !QUESTION_WORDS.has(term)) : []),
    ]));
    // Do not let an entity tail such as “Zephyr Dynamics” become an
    // attribute when a role lookup falls back to its final terms.
    const attributes = broadEntity ? [] : attributeTerms(clauseTerms)
      .filter(attribute => !entityTerms.includes(attribute));
    const enumeration = clauseTerms.some(term => ENUMERATION_WORDS.has(term));
    const usefulTerms = clauseTerms.filter(term => !QUESTION_WORDS.has(term));
    return {
      id: `demand-${index + 1}`,
      clause,
      terms: Array.from(new Set(usefulTerms.length ? usefulTerms : fullTerms)),
      entityTerms,
      attributeTerms: attributes,
      numericConstraints: extractNumbers(clause),
      unitConstraints: extractUnits(clause),
      required: true,
      enumeration,
      broadEntity,
    };
  });
}

export function demandLabel(slot: DemandSlot): string {
  return slot.attributeTerms.join(' ') || (slot.broadEntity ? slot.entityTerms.join(' ') : '') || slot.terms.join(' ') || slot.clause;
}
