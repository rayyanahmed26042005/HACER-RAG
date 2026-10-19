// Shared lexical normalization. Numbers and negations remain significant.
const STOP_WORDS = new Set(('a an the is are was were be been of in on at to for from ' +
  'and or with by as it its they their this that these those what which who where ' +
  'when how please tell me about does do can could would should').split(' '));

export const nowMs = (): number => typeof performance !== 'undefined'
  ? performance.now() : Date.now();

export function terms(text: string): string[] {
  return (text.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
    .match(/[\p{L}\p{N}]+/gu) || [])
    .filter(term => !STOP_WORDS.has(term));
}

/** Linear-space/linear-time edit-distance <= 1 (no vocabulary-sized matrices). */
export function nearTerm(a: string, b: string): boolean {
  if (a === b) return true;
  if (a.length < 5 || b.length < 5 || /\d/.test(a + b) || Math.abs(a.length - b.length) > 1) return false;
  // A common mobile-query typo is a transposition, for example `zypher` for
  // `zephyr`. Treat one two-position transposition as one bounded edit, but
  // keep numeric tokens and short words exact to avoid broad fuzzy matches.
  if (a.length === b.length) {
    const mismatches: number[] = [];
    for (let index = 0; index < a.length; index++) {
      if (a[index] !== b[index]) mismatches.push(index);
    }
    if (mismatches.length === 2) {
      const [first, second] = mismatches;
      if (a[first] === b[second] && a[second] === b[first]) return true;
    }
  }
  let i = 0;
  let j = 0;
  let edits = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { i++; j++; continue; }
    if (++edits > 1) return false;
    if (a.length >= b.length) i++;
    if (b.length >= a.length) j++;
  }
  return edits + a.length - i + b.length - j <= 1;
}

export function matchedTerms(queryTerms: string[], textTerms: Set<string>): string[] {
  return queryTerms.filter(term => textTerms.has(term) ||
    (term.length >= 5 && Array.from(textTerms).some(other => nearTerm(term, other))));
}

/** Source-order units; keep decimal points, abbreviations, table rows and lists. */
export function evidenceUnits(text: string): string[] {
  return text.split(/\n+/).flatMap(line => {
    const value = line.trim();
    if (!value) return [];
    if (value.includes('|') || /\S\s{2,}\S/.test(value) || /^[-*•]\s/.test(value)) return [value];
    return value.split(/(?<=[.!?。！？])\s+(?=[\p{Lu}\p{N}"'])/u)
      .reduce<string[]>((out, part) => {
        if (out.length && /\b(?:Dr|Mr|Ms|Mrs|Prof|Inc|e\.g|i\.e)\.$/.test(out[out.length - 1])) {
          out[out.length - 1] += ` ${part}`;
        } else { out.push(part); }
        return out;
      }, []);
  });
}

export function positiveInt(value: number | undefined, fallback: number, max: number): number {
  return Number.isFinite(value) && value! > 0 ? Math.min(max, Math.floor(value!)) : fallback;
}
