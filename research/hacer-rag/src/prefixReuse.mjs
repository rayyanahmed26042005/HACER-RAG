import { sha256 } from './validityKey.mjs';

export const PREFIX_SCHEMA_VERSION = 1;

export function tokenIdsHash(tokenIds) {
  if (!Array.isArray(tokenIds) || tokenIds.some(token => !Number.isInteger(token))) {
    throw new TypeError('tokenIds must be an array of integers');
  }
  return sha256(JSON.stringify(tokenIds));
}

export function createPrefixReuseMetadata({
  tokenIds,
  validityKey,
  promptTemplateHash,
  contextLength,
  cacheFileSchema = 'llama-session-v1',
  tokensCached = 0,
  tokensEvaluated = 0,
  traceId,
}) {
  if (!validityKey?.digest) throw new TypeError('validityKey.digest is required');
  if (!Number.isInteger(contextLength) || contextLength <= 0) throw new TypeError('contextLength must be positive');
  if (!Number.isInteger(tokensCached) || tokensCached < 0) throw new TypeError('tokensCached must be non-negative');
  if (!Number.isInteger(tokensEvaluated) || tokensEvaluated < 0) throw new TypeError('tokensEvaluated must be non-negative');
  return {
    schemaVersion: PREFIX_SCHEMA_VERSION,
    validityDigest: validityKey.digest,
    prefixTokenCount: tokenIds.length,
    prefixTokenIdsSha256: tokenIdsHash(tokenIds),
    promptTemplateHash,
    contextLength,
    cacheFileSchema,
    tokensCached,
    tokensEvaluated,
    traceId: traceId || null,
    reuseStatus: 'candidate',
    mismatchReason: null,
  };
}

export function validateExactPrefixReuse({ metadata, requestedTokenIds, requestedValidityKey, requestedTemplateHash }) {
  if (!metadata || metadata.schemaVersion !== PREFIX_SCHEMA_VERSION) {
    return miss('prefix_metadata_schema_mismatch');
  }
  if (metadata.validityDigest !== requestedValidityKey?.digest) return miss('validity_key_mismatch');
  if (metadata.promptTemplateHash !== requestedTemplateHash) return miss('prompt_template_mismatch');
  if (metadata.prefixTokenIdsSha256 !== tokenIdsHash(requestedTokenIds.slice(0, metadata.prefixTokenCount))) {
    return miss('prefix_token_hash_mismatch');
  }
  if (metadata.prefixTokenCount > requestedTokenIds.length) return miss('cached_prefix_longer_than_request');
  return {
    hit: true,
    status: 'exact',
    reusedTokenCount: metadata.prefixTokenCount,
    mismatchReason: null,
  };
}

export function markPrefixOutcome(metadata, outcome) {
  if (!['exact', 'miss', 'fallback'].includes(outcome.status)) throw new TypeError('invalid prefix outcome');
  return {
    ...metadata,
    reuseStatus: outcome.status,
    mismatchReason: outcome.mismatchReason || null,
    tokensCached: outcome.reusedTokenCount ?? metadata.tokensCached,
  };
}

function miss(reason) {
  return { hit: false, status: 'miss', reusedTokenCount: 0, mismatchReason: reason };
}
