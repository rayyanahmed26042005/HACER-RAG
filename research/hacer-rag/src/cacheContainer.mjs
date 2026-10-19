import { createHash } from 'node:crypto';

export const HACERKV_MAGIC = 'HACERKV\0';
export const HACERKV_SCHEMA_VERSION = 1;

const PRECISION_BITS = new Set([2, 4, 8, 16]);

export function sha256Hex(value) {
  return createHash('sha256').update(String(value)).digest('hex');
}

export function alignUp(value, alignment) {
  if (!Number.isInteger(value) || value < 0) throw new Error('value must be a non-negative integer');
  if (!Number.isInteger(alignment) || alignment <= 0) throw new Error('alignment must be positive');
  return Math.ceil(value / alignment) * alignment;
}

function requireHash(name, value) {
  if (!/^[a-f0-9]{64}$/i.test(value || '')) throw new Error(`${name} must be a SHA-256 hex digest`);
  return value.toLowerCase();
}

function pageEnd(page) {
  return page.offset + page.length;
}

/**
 * Build a portable manifest for the proposed HACERKV container.
 *
 * This is deliberately a host-side metadata reference. It does not claim to
 * serialize llama.cpp tensors or provide native KV import.
 */
export function createCacheManifest({
  modelSha256,
  tokenizerSha256,
  templateSha256,
  chunkSha256,
  chunkId,
  tokenCount,
  layerCount,
  kvHeads,
  headDim,
  anchorPrecision = 16,
  bulkPrecision = 4,
  pageSize = 4096,
  alignment = pageSize,
  pages = [],
  rope = {},
  maskPolicyId = 'causal',
}) {
  for (const [name, value] of Object.entries({ modelSha256, tokenizerSha256, templateSha256, chunkSha256 })) {
    requireHash(name, value);
  }
  if (!chunkId || !Number.isInteger(tokenCount) || tokenCount < 0) throw new Error('invalid chunk metadata');
  if (!Number.isInteger(layerCount) || layerCount <= 0) throw new Error('layerCount must be positive');
  if (!Number.isInteger(kvHeads) || kvHeads <= 0 || !Number.isInteger(headDim) || headDim <= 0) {
    throw new Error('invalid KV shape');
  }
  if (!PRECISION_BITS.has(anchorPrecision) || !PRECISION_BITS.has(bulkPrecision)) throw new Error('unsupported precision');
  if (!Number.isInteger(pageSize) || pageSize <= 0 || !Number.isInteger(alignment) || alignment <= 0) {
    throw new Error('invalid page size or alignment');
  }

  const normalizedPages = pages.map((page, index) => {
    const normalized = {
      id: page.id || `page-${index}`,
      offset: page.offset,
      length: page.length,
      layer: page.layer,
      tokenStart: page.tokenStart,
      tokenEnd: page.tokenEnd,
      kind: page.kind || 'bulk',
      crc32c: page.crc32c || null,
    };
    if (!Number.isInteger(normalized.offset) || normalized.offset < 0 || normalized.offset % alignment !== 0) {
      throw new Error(`page ${normalized.id} offset is not aligned`);
    }
    if (!Number.isInteger(normalized.length) || normalized.length <= 0) throw new Error(`page ${normalized.id} length is invalid`);
    if (!Number.isInteger(normalized.layer) || normalized.layer < 0 || normalized.layer >= layerCount) {
      throw new Error(`page ${normalized.id} layer is invalid`);
    }
    if (!Number.isInteger(normalized.tokenStart) || !Number.isInteger(normalized.tokenEnd) ||
        normalized.tokenStart < 0 || normalized.tokenEnd <= normalized.tokenStart || normalized.tokenEnd > tokenCount) {
      throw new Error(`page ${normalized.id} token range is invalid`);
    }
    return normalized;
  }).sort((a, b) => a.offset - b.offset);

  for (let index = 1; index < normalizedPages.length; index += 1) {
    if (normalizedPages[index - 1].offset + normalizedPages[index - 1].length > normalizedPages[index].offset) {
      throw new Error('cache pages overlap');
    }
  }

  return {
    magic: HACERKV_MAGIC,
    schemaVersion: HACERKV_SCHEMA_VERSION,
    modelSha256: modelSha256.toLowerCase(),
    tokenizerSha256: tokenizerSha256.toLowerCase(),
    templateSha256: templateSha256.toLowerCase(),
    rope: {
      id: rope.id || 'standard-rope',
      base: rope.base ?? null,
      scale: rope.scale ?? null,
      scheme: rope.scheme || 'unknown',
    },
    maskPolicyId,
    chunkId,
    chunkSha256: chunkSha256.toLowerCase(),
    tokenCount,
    layerCount,
    kvHeads,
    headDim,
    anchorPrecision,
    bulkPrecision,
    pageSize,
    alignment,
    pages: normalizedPages,
  };
}

export function validateCacheManifest(manifest, expected = {}) {
  if (!manifest || manifest.magic !== HACERKV_MAGIC || manifest.schemaVersion !== HACERKV_SCHEMA_VERSION) {
    return { valid: false, reason: 'schema_mismatch' };
  }
  for (const [field, expectedValue] of Object.entries(expected)) {
    if (expectedValue !== undefined && manifest[field] !== expectedValue) {
      return { valid: false, reason: `validity_mismatch:${field}` };
    }
  }
  try {
    createCacheManifest(manifest);
  } catch (error) {
    return { valid: false, reason: error instanceof Error ? error.message : String(error) };
  }
  return { valid: true, pageCount: manifest.pages.length, bytes: manifest.pages.reduce((sum, page) => Math.max(sum, pageEnd(page)), 0) };
}

export function estimateReadAmplification(pages, requestedRanges) {
  const requested = requestedRanges.reduce((sum, range) => sum + Math.max(0, range.end - range.start), 0);
  const loaded = pages.reduce((sum, page) => {
    const needed = requestedRanges.some(range => pageEnd(page) > range.start && page.offset < range.end);
    return sum + (needed ? page.length : 0);
  }, 0);
  return requested > 0 ? loaded / requested : 1;
}
