import { evidenceUnits } from './text';

export interface Chunk {
  text: string;
  docId: string;
  chunkIndex: number;
  docName: string;
  sectionHeader?: string;
  chunkerVersion?: number;
  /** A fragment of an oversized source unit; never silently discarded. */
  continued?: boolean;
}

export const CHUNKER_VERSION = 2;
export const MAX_DOCUMENT_CHARS = 2_000_000;
const MAX_BODY_CHARS = 1000;
const MAX_WORDS = 105;

/** Bound unpunctuated/OCR units too, retaining every character and short tail. */
function boundedUnits(text: string): string[] {
  const output: string[] = [];
  for (const unit of evidenceUnits(text)) {
    let remaining = unit;
    while (remaining.length > MAX_BODY_CHARS || remaining.split(/\s+/).length > MAX_WORDS) {
      const words = remaining.match(/\S+\s*/g) || [];
      const wordEnd = words.slice(0, MAX_WORDS).join('').length;
      let end = Math.min(MAX_BODY_CHARS, wordEnd);
      const space = remaining.lastIndexOf(' ', end);
      if (space > end / 2) end = space;
      // Avoid splitting a UTF-16 surrogate pair.
      if (/[\uD800-\uDBFF]/.test(remaining[end - 1])) end--;
      output.push(remaining.slice(0, end).trim());
      remaining = remaining.slice(end).trim();
    }
    if (remaining) output.push(remaining);
  }
  return output;
}

/** Heading breadcrumbs and atomic rows; short facts are as important as prose. */
export function chunkText(text: string, docId: string, docName: string): Chunk[] {
  if (!text?.trim()) return [];
  if (text.length > MAX_DOCUMENT_CHARS) throw new Error('Document exceeds the mobile text limit (2 million characters). Split it into smaller documents.');
  const chunks: Chunk[] = [];
  const headings: string[] = [];
  let body: string[] = [];
  let header = '';
  let tableHeader = '';

  const emit = () => {
    if (!body.length) return;
    chunks.push({
      text: body.join('\n'), docId, docName, chunkIndex: chunks.length,
      sectionHeader: header || undefined, chunkerVersion: CHUNKER_VERSION,
    });
    body = [];
  };
  const add = (unit: string) => {
    const proposed = [...body, unit].join('\n');
    if (body.length && (proposed.length > MAX_BODY_CHARS || proposed.split(/\s+/).length > MAX_WORDS)) {
      emit();
      if (tableHeader && (tableHeader + unit).length < MAX_BODY_CHARS) body.push(tableHeader);
    }
    body.push(unit);
  };

  const lines = text.replace(/\r\n?/g, '\n').replace(/\u0000/g, '').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const markdown = line.match(/^(#{1,6})\s+(.+)$/);
    const numbered = line.match(/^(\d+(?:\.\d+)*\.?)[ \t]+([\p{Lu}][^.!?]{0,75})$/u);
    const colon = line.length < 80 && line.endsWith(':') && !line.includes('|');
    if (markdown || numbered || colon) {
      // Retain headings with no body as evidence (e.g. a product catalogue).
      if (!body.length && header) add(headings[headings.length - 1]);
      emit();
      const depth = markdown ? markdown[1].length : numbered
        ? numbered[1].replace(/\.$/, '').split('.').length : 2;
      headings.splice(Math.min(headings.length, depth - 1));
      headings.push(markdown ? markdown[2] : line);
      header = headings.join(' > ');
      tableHeader = '';
      continue;
    }
    // Markdown table separators carry no facts. Repeat the original column
    // labels when a table crosses chunks instead of severing row semantics.
    if (/^\|?[\s:|-]+\|[\s:|-]*$/.test(line)) continue;
    if (line.includes('|') && /^\|?[\s:|-]+\|/.test(lines[i + 1]?.trim() || '')) tableHeader = line;
    for (const unit of boundedUnits(line)) add(unit);
  }
  if (!body.length && header) add(headings[headings.length - 1]);
  emit();
  return chunks;
}
