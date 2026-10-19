import { createHash } from 'node:crypto';

export const VALIDITY_SCHEMA_VERSION = 1;

const REQUIRED_FIELDS = [
  'modelSha256', 'tokenizerSha256', 'templateSha256', 'architecture',
  'position', 'attentionMask', 'quantization', 'tensorLayout',
  'chunk', 'assembly', 'namespace',
];

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
}

function canonicalize(value) {
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(canonicalize);
  return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonicalize(value[key])]));
}

export function canonicalJson(value) {
  return JSON.stringify(canonicalize(value));
}

export function sha256(value) {
  const input = typeof value === 'string' || Buffer.isBuffer(value)
    ? value
    : canonicalJson(value);
  return createHash('sha256').update(input).digest('hex');
}

function requireNonEmptyString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError(`${label} must be a non-empty string`);
  }
}

function validateMetadata(metadata) {
  assertObject(metadata, 'metadata');
  for (const field of REQUIRED_FIELDS) {
    if (!(field in metadata)) throw new TypeError(`metadata.${field} is required`);
  }
  for (const field of ['modelSha256', 'tokenizerSha256', 'templateSha256']) {
    requireNonEmptyString(metadata[field], `metadata.${field}`);
  }
  assertObject(metadata.position, 'metadata.position');
  assertObject(metadata.attentionMask, 'metadata.attentionMask');
  assertObject(metadata.quantization, 'metadata.quantization');
  assertObject(metadata.tensorLayout, 'metadata.tensorLayout');
  assertObject(metadata.architecture, 'metadata.architecture');
  assertObject(metadata.chunk, 'metadata.chunk');
  assertObject(metadata.assembly, 'metadata.assembly');
  assertObject(metadata.namespace, 'metadata.namespace');
}

export function buildValidityRecord(metadata) {
  validateMetadata(metadata);
  return canonicalize({ schemaVersion: VALIDITY_SCHEMA_VERSION, ...metadata });
}

export function createValidityKey(metadata) {
  const record = buildValidityRecord(metadata);
  return {
    algorithm: 'sha256-canonical-json-v1',
    digest: sha256(record),
    record,
  };
}

export function compareValidityRecords(expected, actual) {
  const left = buildValidityRecord(expected);
  const right = buildValidityRecord(actual);
  const differences = [];
  const paths = new Set([...flattenPaths(left), ...flattenPaths(right)]);
  for (const path of [...paths].sort()) {
    const a = readPath(left, path);
    const b = readPath(right, path);
    if (canonicalJson(a) !== canonicalJson(b)) differences.push({ path, expected: a, actual: b });
  }
  return differences;
}

export function explainValidityMismatch(expectedKey, actualKey) {
  if (!expectedKey || !actualKey) return [{ path: '$', reason: 'missing_validity_key' }];
  if (expectedKey.digest === actualKey.digest) return [];
  return compareValidityRecords(expectedKey.record, actualKey.record).map(item => ({
    ...item,
    reason: 'validity_field_changed',
  }));
}

function flattenPaths(value, prefix = '$') {
  if (value === null || typeof value !== 'object') return [prefix];
  if (Array.isArray(value)) return value.flatMap((item, index) => flattenPaths(item, `${prefix}[${index}]`));
  const keys = Object.keys(value);
  if (!keys.length) return [prefix];
  return keys.flatMap(key => flattenPaths(value[key], `${prefix}.${key}`));
}

function readPath(value, path) {
  const tokens = path.replace(/^\$/, '').split(/[.[\]]/).filter(Boolean);
  return tokens.reduce((current, token) => current?.[token], value);
}
