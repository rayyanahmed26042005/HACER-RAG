export function validateRopeConfig(config) {
  if (!config || typeof config !== 'object') throw new TypeError('RoPE config is required');
  if (config.scheme !== 'rope') throw new Error(`unsupported_position_scheme:${config.scheme || 'missing'}`);
  if (!Number.isFinite(config.base) || config.base <= 0) throw new TypeError('RoPE base must be positive');
  if (!Number.isFinite(config.scale) || config.scale <= 0) throw new TypeError('RoPE scale must be positive');
  if (!Number.isInteger(config.headDim) || config.headDim <= 0 || config.headDim % 2 !== 0) {
    throw new TypeError('RoPE headDim must be a positive even integer');
  }
  if (config.frequencyLayout && !['interleaved', 'half_split'].includes(config.frequencyLayout)) {
    throw new Error(`unsupported_frequency_layout:${config.frequencyLayout}`);
  }
  return {
    scheme: 'rope',
    base: config.base,
    scale: config.scale,
    headDim: config.headDim,
    frequencyLayout: config.frequencyLayout || 'interleaved',
  };
}

export function rotateKeyVector(key, deltaPosition, config) {
  const rope = validateRopeConfig(config);
  if (!Array.isArray(key) && !ArrayBuffer.isView(key)) throw new TypeError('key must be an array or typed array');
  if (key.length !== rope.headDim) throw new RangeError('key length must equal headDim');
  if (!Number.isInteger(deltaPosition)) throw new TypeError('deltaPosition must be an integer');
  if (key.some(value => !Number.isFinite(value))) throw new TypeError('key contains a non-finite value');
  const out = new Float64Array(key.length);
  const pairCount = rope.headDim / 2;
  for (let pair = 0; pair < pairCount; pair += 1) {
    const exponent = (2 * pair) / rope.headDim;
    const theta = deltaPosition / rope.scale / (rope.base ** exponent);
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const i = rope.frequencyLayout === 'half_split' ? pair : 2 * pair;
    const j = rope.frequencyLayout === 'half_split' ? pair + pairCount : 2 * pair + 1;
    out[i] = key[i] * cos - key[j] * sin;
    out[j] = key[i] * sin + key[j] * cos;
  }
  return out;
}

export function shiftKeyBatch(keys, deltaPositions, config) {
  if (!Array.isArray(keys) || !Array.isArray(deltaPositions) || keys.length !== deltaPositions.length) {
    throw new TypeError('keys and deltaPositions must have equal lengths');
  }
  return keys.map((key, index) => rotateKeyVector(key, deltaPositions[index], config));
}

export function validateRoPEMetadata(local, global) {
  try {
    const a = validateRopeConfig(local);
    const b = validateRopeConfig(global);
    const differences = [];
    for (const field of ['base', 'scale', 'headDim', 'frequencyLayout']) {
      if (a[field] !== b[field]) differences.push({ field, local: a[field], global: b[field] });
    }
    return { valid: differences.length === 0, differences };
  } catch (error) {
    return { valid: false, differences: [{ field: 'scheme', reason: error.message }] };
  }
}
