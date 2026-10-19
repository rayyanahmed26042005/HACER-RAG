// Functional regressions with in-memory storage and a deterministic fake native
// encoder/tokenizer. Tests execute production TS modules; no device timings or
// answer-quality claims are produced. Run: npm run test:rag
const assert = require('node:assert/strict');
const { test, beforeEach } = require('node:test');
const fs = require('node:fs');
const Module = require('node:module');
const ts = require('typescript');
require.extensions['.ts'] = (module, path) => module._compile(ts.transpileModule(fs.readFileSync(path, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
}).outputText, path);

const files = new Map();
let failMove = false;
let loads = 0;
let encoding = false;
let encodes = 0;
let inits = 0;
const model = '/mobile/models/all-minilm-l6-v2-q4_k_m.gguf';
const rnfs = {
  ExternalDirectoryPath: '/mobile',
  exists: async path => files.has(path) || path === model,
  mkdir: async () => {},
  stat: async path => ({ size: (files.get(path) || '').length, mtime: new Date(0) }),
  readFile: async path => { loads++; if (!files.has(path)) throw new Error('missing file'); return files.get(path); },
  writeFile: async (path, content) => { files.set(path, content); },
  moveFile: async (path, dest) => { if (failMove) throw new Error('storage full'); files.set(dest, files.get(path)); files.delete(path); },
  unlink: async path => { files.delete(path); },
};
const native = {
  initLlama: async () => {
    inits++;
    return {
      tokenize: async text => ({ tokens: Array(Math.ceil(text.length / 4)).fill(1) }),
      embedding: async text => {
        assert.equal(encoding, false, 'native encoder calls must not overlap');
        encoding = true; encodes++;
        await new Promise(resolve => setTimeout(resolve, 1));
        encoding = false;
        return { embedding: [1, text.includes('battery') ? 1 : 0, 0.1] };
      },
      release: async () => assert.equal(encoding, false),
    };
  },
};
const originalLoad = Module._load;
Module._load = function (name, parent, main) {
  if (name === 'react-native-fs') return rnfs;
  if (name === 'llama.rn') return native;
  if (name === '@react-native-async-storage/async-storage') return { getItem: async () => null, setItem: async () => {} };
  if (name.endsWith('/profileManager')) return { profileKey: (id, key) => id + key };
  return originalLoad.call(this, name, parent, main);
};
const { chunkText } = require('../utils/rag/chunker.ts');
const store = require('../utils/rag/vectorStore.ts');
const embed = require('../utils/rag/embedder.ts');
const { ingestTextDoc, retrieveContextDetailed } = require('../utils/rag/ragPipeline.ts');
const { packEvidence } = require('../utils/rag/evidenceSelector.ts');
const { createPromptBudget } = require('../utils/rag/promptBudget.ts');
const { buildDocumentPrompt } = require('../utils/rag/promptBudget.ts');
const { evaluateRag } = require('../utils/rag/metrics.ts');
const { classifyEvidenceDemand, selectInferencePlan } = require('../utils/inferenceController.ts');
const { atomizeChunk } = require('../utils/rag/evidenceAtoms.ts');
const { extractDemandSlots } = require('../utils/rag/questionDemands.ts');
const { buildEvidenceCertificate, selectCertifiedEvidence, auditAnswerContract, buildDirectEvidenceAnswer } = require('../utils/rag/evidenceCertificate.ts');
const { evaluateResearchRun } = require('../utils/rag/researchEvaluator.ts');

beforeEach(async () => {
  await embed.releaseEmbeddingModel('p');
  store.flushInMemoryIndex(); files.clear(); failMove = false; loads = 0; encodes = 0; inits = 0;
});
const chunk = (text, docId = 'd', chunkIndex = 0, sectionHeader = '') => ({ text, docId, chunkIndex, sectionHeader, docName: 'handbook.txt', chunkerVersion: 2 });
const result = (text, index, header = '') => ({ chunk: chunk(text, 'd', index, header), score: 0.03,
  rrfScore: 0.03, denseScore: 0.5, bm25Score: 1, lexicalCoverage: 0.5, exactTermMatches: 1, matchedTerms: [] });
const countTokens = async text => Math.ceil(text.length / 4);

test('ingestion preserves short facts, nested headings, decimals, table rows and tails', () => {
  const chunks = chunkText('# Handbook\n## Products\n### AeroSense X7\nWeight: 2.3 kg.\nBattery: 72 hours.\n1.1 Leadership Team\nJorge Fonseca | Chief Legal Officer | Lisbon | 2019\n2. Benefits\nGym: $60/month.', 'd', 'h.txt');
  const text = chunks.map(c => c.text).join('\n');
  for (const fact of ['2.3 kg', '72 hours', 'Jorge Fonseca | Chief Legal Officer | Lisbon', '$60/month']) assert.ok(text.includes(fact), fact);
  assert.ok(chunks.some(c => c.sectionHeader.includes('Handbook > Products > AeroSense X7')));
  assert.ok(chunks.some(c => c.sectionHeader === '1.1 Leadership Team' || c.sectionHeader.endsWith('1.1 Leadership Team')));
});
test('unpunctuated long units are bounded and no short final fact disappears', () => {
  const text = Array.from({ length: 420 }, (_, i) => `w${i}`).join(' ') + ' LASTFACT';
  const chunks = chunkText(text, 'd', 'h');
  assert.ok(chunks.every(c => c.text.length <= 1000));
  assert.equal(chunks.map(c => c.text).join(' ').split(/\s+/).length, 421);
  assert.ok(chunks.at(-1).text.includes('LASTFACT'));
});
test('pure Unicode text remains indexable and bounded', () => {
  const text = '東京気温は三十度。'.repeat(180);
  const chunks = chunkText(text, 'd', 'h');
  assert.ok(chunks.every(c => c.text.length <= 1000));
  assert.equal(chunks.map(c => c.text).join(''), text);
});
test('idempotent document replacement and persistent reload preserve other documents', async () => {
  await store.replaceDocumentInIndex('a', [chunk('old', 'a')], [new Float32Array([1, 0])], 'p');
  await store.replaceDocumentInIndex('b', [chunk('other', 'b')], [new Float32Array([0, 1])], 'p');
  await store.replaceDocumentInIndex('a', [chunk('new', 'a')], [new Float32Array([1, 0])], 'p');
  store.flushInMemoryIndex('p');
  await Promise.all([store.loadIndex('p'), store.loadIndex('p')]);
  assert.equal(loads, 1); assert.equal(store.getIndexSize('p'), 2);
  assert.ok(store.search(new Float32Array([1, 0]), 'p', 3, 'new')[0].chunk.text === 'new');
});
test('failed persistence keeps last good RAM and disk index', async () => {
  await store.replaceDocumentInIndex('d', [chunk('old')], [new Float32Array([1, 0])], 'p');
  const before = files.get('/mobile/profiles/p/rag_index.json'); failMove = true;
  await assert.rejects(store.replaceDocumentInIndex('d', [chunk('replacement')], [new Float32Array([1, 0])], 'p'), /storage full/);
  assert.equal(files.get('/mobile/profiles/p/rag_index.json'), before);
  assert.equal(store.search(new Float32Array([1, 0]), 'p')[0].chunk.text, 'old');
});
test('concurrent commits retain both documents; profile caches stay isolated', async () => {
  await Promise.all(['a', 'b'].map(id => store.replaceDocumentInIndex(id, [chunk(id, id)], [new Float32Array([1, 0])], 'p')));
  await store.loadIndex('other'); assert.equal(store.getIndexSize('other'), 0); assert.equal(store.getIndexSize('p'), 2);
  await store.removeDocFromIndex('a', 'p'); store.flushInMemoryIndex('p'); await store.loadIndex('p');
  assert.deepEqual(store.getIndexedDocIds('p'), ['b']);
});
test('invalid vectors and mismatched dimensions fail explicitly', async () => {
  for (const vector of [[0, 0], [NaN, 1], []]) await assert.rejects(store.replaceDocumentInIndex('d', [chunk('x')], [new Float32Array(vector)], 'p'));
  await store.replaceDocumentInIndex('d', [chunk('x')], [new Float32Array([1, 0])], 'p');
  assert.throws(() => store.search(new Float32Array([1, 0, 0]), 'p'), /dimensions/);
});
test('BM25 zero matches do not change dense ranking', async () => {
  await store.replaceDocumentInIndex('d', [chunk('oranges', 'd', 0), chunk('apples', 'd', 1)],
    [new Float32Array([0, 1]), new Float32Array([1, 0])], 'p');
  const hits = store.search(new Float32Array([1, 0]), 'p', 3, 'zzzzzz');
  assert.equal(hits[0].chunk.chunkIndex, 1); assert.equal(hits[0].lexicalRank, undefined);
});
test('bounded typo matching retrieves model-specific specs, never fuzzes numbers', async () => {
  await store.replaceDocumentInIndex('d', [chunk('AeroSense X7 battery lasts 72 hours and weight is 2.3 kg', 'd', 0), chunk('Unrelated weather report', 'd', 1)],
    [new Float32Array([1, 0.1]), new Float32Array([1, 0.2])], 'p');
  const hits = store.search(new Float32Array([1, 0]), 'p', 8, 'battery weight aerosence X7');
  assert.equal(hits[0].chunk.chunkIndex, 0); assert.ok(hits[0].matchedTerms.includes('aerosence'));
  assert.equal(store.search(new Float32Array([1, 0]), 'p', 8, 'X8')[0].bm25Score, 0);
});
test('encoder lifecycle is serialized and exact query cache is bounded by identity', async () => {
  await Promise.all([embed.initEmbeddingModel('p'), embed.initEmbeddingModel('p')]); assert.equal(inits, 1);
  const queries = await Promise.all([embed.embedQuery('Battery?', 'p'), embed.embedQuery('battery?', 'p')]);
  assert.equal(encodes, 2); assert.equal(queries[1].cacheHit, false);
  assert.equal((await embed.embedQuery('Battery?', 'p')).cacheHit, true);
  await embed.releaseEmbeddingModel('p'); assert.equal((await embed.embedQuery('Battery?', 'p')).cacheHit, false);
});
test('ingest -> persisted reload -> retrieve packs product enumeration across sections', async () => {
  const text = '# Products\n## AeroSense X7\nAeroSense X7 measures air quality.\n## SwarmKit Pro\nSwarmKit Pro coordinates drones.\n## CloudMapper Lite\nCloudMapper Lite maps atmospheric data.';
  const first = await ingestTextDoc(text, 'd', 'handbook.txt', 'p');
  await ingestTextDoc(text, 'd', 'handbook.txt', 'p');
  assert.equal(store.getIndexSize('p'), first.chunkCount);
  store.flushInMemoryIndex('p');
  const retrieved = await retrieveContextDetailed('What are the products?', 'p', { countTokens, maxContextTokens: 220, maxSelectedChunks: 6 });
  for (const name of ['AeroSense X7', 'SwarmKit Pro', 'CloudMapper Lite']) assert.ok(retrieved.context.includes(name), name);
  assert.ok(retrieved.contextTokens <= 220); assert.equal(retrieved.tokenCountMethod, 'native');
  assert.ok(files.has('/mobile/profiles/p/rag_ingest_latest.json'));
});
test('packing preserves conflicting quantities and negation, with strict budgets', async () => {
  const candidates = [result('The device is approved and weighs 2.3 kg.', 0), result('The device is not approved and weighs 3.2 kg.', 1)];
  const packed = await packEvidence('Is the device approved and what is its weight?', candidates,
    { maxTokens: 150, maxChunks: 3, demand: 'multi_hop', deduplicate: true, countTokens });
  assert.ok(packed.context.includes('not approved')); assert.ok(packed.context.includes('2.3 kg')); assert.ok(packed.context.includes('3.2 kg'));
  const empty = await packEvidence('device', candidates, { maxTokens: 1, maxChunks: 3, demand: 'broad', deduplicate: true, countTokens });
  assert.equal(empty.context, ''); assert.ok(empty.budgetLimited);
});
test('oversized top evidence does not block a smaller complementary fact', async () => {
  const packed = await packEvidence('battery life and weight', [result('battery '.repeat(300), 0), result('Weight is 2.3 kg.', 1)],
    { maxTokens: 40, maxChunks: 3, demand: 'multi_hop', deduplicate: true, countTokens });
  assert.ok(packed.context.includes('2.3 kg')); assert.ok(packed.contextTokens <= 40);
});
test('true token budget includes template, question and output reserve', async () => {
  const ctx = { tokenize: async text => ({ tokens: Array(text.length).fill(1) }),
    getFormattedChat: async messages => `<BOS><USER>${messages[0].content}<ASSISTANT>` };
   const budget = await createPromptBudget(ctx, 'What are the products?', 1024, 448);
  const packed = await packEvidence('products', [result('AeroSense X7 is a product.', 0), result('SwarmKit Pro is a product.', 1)],
    { maxTokens: budget.contextBudget, maxChunks: 3, demand: 'broad', deduplicate: true,
      countTokens: budget.countTokens, fitsPrompt: budget.fitsPrompt });
   assert.ok(await budget.countPrompt(packed.context) + budget.outputTokens + 8 <= 1024);
   await assert.rejects(createPromptBudget(ctx, 'x'.repeat(1100), 512, 128), /too long/);
});
test('query classification recognizes short enumeration and multi-part questions', () => {
  assert.equal(classifyEvidenceDemand('What are the products?', 'document'), 'broad');
  assert.equal(classifyEvidenceDemand('Battery life and weight?', 'document'), 'multi_hop');
  assert.equal(classifyEvidenceDemand('What is Zypher?', 'document'), 'broad');
});

test('broad entity demand accepts one-edit transposition and selects multiple supported facts', () => {
  const atoms = [
    ...atomizeChunk(chunk('Zephyr Dynamics builds atmospheric data harvesting systems.', 'd', 0, 'Overview')),
    ...atomizeChunk(chunk('Zephyr Dynamics operates micro-drone swarm technology.', 'd', 1, 'Capabilities')),
    ...atomizeChunk(chunk('Zephyr Dynamics has 847 employees across 6 offices.', 'd', 2, 'Company')),
  ];
  const selected = selectCertifiedEvidence('What is zypher?', atoms, 120, 8);
  assert.equal(selected.certificate.demands[0].status, 'certified');
  assert.equal(selected.certificate.demands[0].label, 'zypher');
  assert.ok(selected.atoms.length >= 2);
  assert.ok(selected.atoms.some(atom => atom.text.includes('atmospheric')));
  assert.ok(selected.atoms.some(atom => atom.text.includes('micro-drone')));
});

test('broad entity prompt asks for a complete supported synthesis', () => {
  const prompt = buildDocumentPrompt('What is Zypher?', '[1] Zephyr builds atmospheric systems. [2] It supports monitoring.');
  assert.match(prompt, /broad entity question/i);
  assert.match(prompt, /all supported facts/i);
  assert.match(prompt, /not only a one-line definition/i);
  assert.match(prompt, /at least two distinct supported facts/i);
  assert.match(prompt, /canonical names and spelling/i);
  assert.match(prompt, /retrieval metadata, not document content/i);
  assert.match(prompt, /never write the placeholder \[n\]/i);
  assert.match(prompt, /exact numeric source marker/i);
});

test('broad document questions receive a larger adaptive output plan than lookups', () => {
  const broad = selectInferencePlan(220, 'document', { mode: 'adaptive' }, 'What is Zypher?');
  const lookup = selectInferencePlan(220, 'document', { mode: 'adaptive' }, 'What is the weight?');
  assert.equal(broad.evidenceDemand, 'broad');
  assert.ok(broad.nPredict > lookup.nPredict);
});

test('MobCert broad requests widen the bounded baseline candidate frontier', () => {
  const plan = selectInferencePlan(220, 'document', { mode: 'baseline', ragPolicy: 'mobcert-v1' }, 'What is Zypher?');
  const historical = selectInferencePlan(220, 'document', { mode: 'baseline' }, 'What is Zypher?');
  assert.equal(plan.candidateK, 8);
  assert.equal(historical.candidateK, 3);
});
test('quality metrics require labels; word overlap cannot claim entailment', () => {
  const metrics = evaluateRag(['a', 'b'], 'The device is not approved.', 'The device is approved.');
  assert.equal(metrics.recallAtK, null); assert.equal(metrics.groundedness, null); assert.equal(metrics.correctness, null);
  const labeled = evaluateRag(['b', 'a'], 'x', '72 hours', { relevantChunkIds: ['a', 'c'], referenceAnswer: '72 hours' });
  assert.equal(labeled.recallAtK, 0.5); assert.equal(labeled.precisionAtK, 0.5); assert.equal(labeled.reciprocalRank, 0.5); assert.equal(labeled.answerTokenF1, 1);
});

test('certificate demand slots preserve separated entity attributes', () => {
  const demands = extractDemandSlots('What is the battery life and weight of AeroSense X7?');
  assert.equal(demands.length, 2);
  assert.ok(demands[0].attributeTerms.includes('battery'));
  assert.ok(demands[1].attributeTerms.includes('weight'));
  assert.ok(demands[0].entityTerms.includes('aerosense'));
  assert.ok(demands[1].entityTerms.includes('x7'));
});

test('certificate selection covers complementary facts within a token budget', () => {
  const atoms = [
    ...atomizeChunk(chunk('AeroSense X7 is our field device.', 'd', 0, 'Products > AeroSense X7')),
    ...atomizeChunk(chunk('Battery life: 72 hours.', 'd', 1, 'Products > AeroSense X7')),
    ...atomizeChunk(chunk('Weight: 2.3 kg.', 'd', 2, 'Products > AeroSense X7')),
  ];
  const selected = selectCertifiedEvidence('What is the battery life and weight of AeroSense X7?', atoms, 80);
  assert.equal(selected.certificate.missingDemandCount, 0);
  assert.equal(selected.certificate.certifiedDemandCount, 2);
  assert.ok(selected.atoms.some(atom => atom.numbers.includes('72')));
  assert.ok(selected.atoms.some(atom => atom.numbers.includes('2.3')));
  assert.ok(selected.certificate.selectedTokens <= 80);
});

test('certificate marks conflicting values and polarity instead of deduplicating them', () => {
  const numericAtoms = [
    ...atomizeChunk(chunk('The device weighs 2.3 kg.', 'd', 0, 'Device')),
    ...atomizeChunk(chunk('The device weighs 3.2 kg.', 'd', 1, 'Device')),
  ];
  const numeric = buildEvidenceCertificate('What is the device weight?', numericAtoms, 80);
  assert.equal(numeric.demands[0].status, 'ambiguous');
  assert.deepEqual(numeric.demands[0].values.sort(), ['2.3', '3.2']);

  const polarityAtoms = [
    ...atomizeChunk(chunk('The device is approved.', 'd', 0, 'Policy')),
    ...atomizeChunk(chunk('The device is not approved.', 'd', 1, 'Policy')),
  ];
  const polarity = buildEvidenceCertificate('Is the device approved?', polarityAtoms, 80);
  assert.equal(polarity.demands[0].status, 'ambiguous');
  assert.ok(polarity.demands[0].polarities.includes('positive'));
  assert.ok(polarity.demands[0].polarities.includes('negated'));
});

test('certificate scopes values to the requested attribute within one atom', () => {
  const atom = atomizeChunk(chunk('Weight: 2.3 kg. Manufacturing year: 2024.', 'd', 0, 'Device'));
  const certificate = buildEvidenceCertificate('What is the device weight?', atom, 80);
  assert.equal(certificate.demands[0].status, 'certified');
  assert.deepEqual(certificate.demands[0].values, ['2.3']);
  assert.deepEqual(certificate.demands[0].conflictingValues, []);
});

test('certificate scopes a value before a second attribute in the same sentence', () => {
  const atom = atomizeChunk(chunk(
    'It weighs 2.3 kg and has an operating temperature range of -60°C to +85°C.',
    'd', 0, 'AeroSense X7',
  ));
  const certificate = buildEvidenceCertificate('What is the weight of the AeroSense X7?', atom, 80);
  assert.equal(certificate.demands[0].status, 'certified');
  assert.deepEqual(certificate.demands[0].values, ['2.3']);
});

test('certificate keeps duration words inside a battery fact', () => {
  const atom = atomizeChunk(chunk(
    'The X7 uses a proprietary QuantumCell battery that lasts 72 hours on a single charge.',
    'd', 0, 'AeroSense X7',
  ));
  const certificate = buildEvidenceCertificate('What is the battery life of the AeroSense X7?', atom, 80);
  assert.equal(certificate.demands[0].status, 'certified');
  assert.deepEqual(certificate.demands[0].values, ['72']);
  assert.deepEqual(certificate.demands[0].units, ['hours']);
  assert.deepEqual(certificate.demands[0].valueLabels, []);
});

test('certificate ignores unrelated numeric columns for a role and office lookup', () => {
  const evidence = atomizeChunk(chunk(
    'Name | Title | Office | Joined\nJorge Fonseca | Chief Legal Officer | Lisbon | 2019',
    'd', 0, 'Leadership',
  ));
  evidence.push(...atomizeChunk(chunk('Employees are required to be in-office a minimum of 2 days.', 'd', 1, 'Remote Work')));
  const certificate = buildEvidenceCertificate(
    'Who is the Chief Legal Officer and which office are they based in?', evidence, 80,
  );
  assert.equal(certificate.ambiguousDemandCount, 0);
  assert.equal(certificate.missingDemandCount, 0);
  assert.deepEqual(certificate.demands[1].valueLabels, ['lisbon']);
});

test('certificate keeps a remote eligibility qualifier scoped to the monthly allowance', () => {
  const evidence = [
    ...atomizeChunk(chunk('New hires receive an equipment stipend of $800 USD.', 'd', 0, 'Benefits')),
    ...atomizeChunk(chunk('Internet Stipend $50/month for remote-eligible employees.', 'd', 1, 'Benefits')),
  ];
  const certificate = buildEvidenceCertificate(
    'What are all the equipment stipends and monthly allowances available for remote-eligible employees?',
    evidence, 80,
  );
  assert.equal(certificate.missingDemandCount, 0);
  assert.equal(certificate.ambiguousDemandCount, 0);
  assert.deepEqual(certificate.demands[1].values, ['$50']);
  assert.deepEqual(certificate.demands[1].units, ['month']);
});

test('certificate keeps similarly typed employee benefits scoped by their qualifiers', () => {
  const evidence = [
    ...atomizeChunk(chunk('Learning Budget $1,500 USD/year per employee.', 'd', 0, 'Benefits')),
    ...atomizeChunk(chunk('Gym Allowance $60/month reimbursed via expense report.', 'd', 1, 'Benefits')),
    ...atomizeChunk(chunk('Internet Stipend $50/month for remote-eligible employees.', 'd', 2, 'Benefits')),
  ];
  const certificate = buildEvidenceCertificate(
    'What is the annual learning budget and gym allowance for employees?', evidence, 80,
  );
  assert.equal(certificate.missingDemandCount, 0);
  assert.equal(certificate.ambiguousDemandCount, 0);
  assert.deepEqual(certificate.demands[0].values, ['$1 500']);
  assert.deepEqual(certificate.demands[1].values, ['$60']);
});

test('certificate scopes polarity to the requested fact', () => {
  const atom = atomizeChunk(chunk('The device is approved but it is not exportable.', 'd', 0, 'Policy'));
  const certificate = buildEvidenceCertificate('Is the device approved?', atom, 80);
  assert.equal(certificate.demands[0].status, 'certified');
  assert.deepEqual(certificate.demands[0].polarities, ['positive']);
});

test('certificate restores table column semantics for a row-level office answer', () => {
  const evidence = atomizeChunk(chunk(
    'Name | Title | Office | Joined\nJorge Fonseca | Chief Legal Officer | Lisbon | 2019',
    'd', 0, 'Leadership',
  ));
  const certificate = buildEvidenceCertificate('Which office is Jorge Fonseca based in?', evidence, 80);
  assert.equal(certificate.demands[0].status, 'certified');
  assert.deepEqual(certificate.demands[0].valueLabels, ['lisbon']);
});

test('certificate restores common flattened PDF table columns', () => {
  const evidence = atomizeChunk(chunk(
    'Name Title Office Joined\nJorge Fonseca Chief Legal Officer Lisbon 2019',
    'd', 0, 'Leadership',
  ));
  const certificate = buildEvidenceCertificate('Which office is Jorge Fonseca based in?', evidence, 80);
  assert.equal(certificate.demands[0].status, 'certified');
  assert.deepEqual(certificate.demands[0].valueLabels, ['lisbon']);
  assert.ok(evidence[0].structural);
});

test('certificate detects conflicting non-numeric values but permits enumeration', () => {
  const evidence = [
    ...atomizeChunk(chunk('Office: Lisbon.', 'd', 0, 'Locations')),
    ...atomizeChunk(chunk('Office: Reykjavik.', 'd', 1, 'Locations')),
  ];
  const conflict = buildEvidenceCertificate('Which office is the headquarters?', evidence, 80);
  assert.equal(conflict.demands[0].status, 'ambiguous');
  assert.deepEqual(conflict.demands[0].valueLabels.sort(), ['lisbon', 'reykjavik']);

  const enumeration = buildEvidenceCertificate('What are the offices?', evidence, 80);
  assert.equal(enumeration.demands[0].status, 'certified');
  assert.deepEqual(enumeration.demands[0].valueLabels.sort(), ['lisbon', 'reykjavik']);
});

test('table headers cannot certify a value without selecting a data row', () => {
  const evidence = atomizeChunk(chunk(
    'Name | Title | Office | Joined\nJorge Fonseca | Chief Legal Officer | Lisbon | 2019',
    'd', 0, 'Leadership',
  ));
  const selected = selectCertifiedEvidence('What are the offices?', evidence, 80);
  assert.equal(selected.certificate.demands[0].status, 'certified');
  assert.ok(selected.atoms.every(atom => !atom.structural));
  assert.deepEqual(selected.certificate.demands[0].valueLabels, ['lisbon']);
});

test('certificate selector does not stop before a bounded conflicting candidate', () => {
  const candidates = [
    ...atomizeChunk(chunk('The device weighs 2.3 kg.', 'd', 0, 'Device')),
    ...atomizeChunk(chunk('The device weighs 3.2 kg.', 'd', 1, 'Device')),
  ];
  const selected = selectCertifiedEvidence('What is the device weight?', candidates, 80);
  assert.equal(selected.certificate.demands[0].status, 'ambiguous');
  assert.equal(selected.atoms.length, 2);
});

test('answer contract rejects unsupported protected values and allows certified partial answers', () => {
  const atoms = atomizeChunk(chunk('Battery life: 72 hours.', 'd', 0, 'AeroSense X7'));
  const contract = buildEvidenceCertificate('What is the battery life and weight of AeroSense X7?', atoms, 40);
  assert.equal(contract.certifiedDemandCount, 1);
  assert.equal(contract.missingDemandCount, 1);
  const partial = auditAnswerContract('Battery life is 72 hours [1].', contract, atoms);
  assert.equal(partial.status, 'partial');
  assert.deepEqual(partial.unsupportedProtectedTokens, []);
  assert.equal(auditAnswerContract('Battery life is 72 hours.', contract, atoms).status, 'unverifiable');
  const hallucinated = auditAnswerContract('Battery life is 99 hours and weight is 4.0 kg [1].', contract, atoms);
  assert.equal(hallucinated.status, 'abstained');
  assert.ok(hallucinated.unsupportedProtectedTokens.includes('99'));
  assert.ok(hallucinated.unsupportedProtectedTokens.includes('4.0'));
});

test('answer contract marks an all-missing question as abstained, not partial', () => {
  const atoms = atomizeChunk(chunk('Office: Lisbon.', 'd', 0, 'Locations'));
  const contract = buildEvidenceCertificate('What is the battery life?', atoms, 40);
  assert.equal(contract.certifiedDemandCount, 0);
  assert.equal(contract.missingDemandCount, 1);
  assert.equal(auditAnswerContract("I don't have enough information in the uploaded documents to answer that.", contract, atoms).status, 'abstained');
});

test('answer contract binds protected values to the cited source span', () => {
  const atoms = [
    ...atomizeChunk(chunk('Battery life: 72 hours.', 'd', 0, 'AeroSense X7')),
    ...atomizeChunk(chunk('Weight: 2.3 kg.', 'd', 1, 'AeroSense X7')),
  ];
  const contract = buildEvidenceCertificate('What is the battery life and weight of AeroSense X7?', atoms, 80);
  const wrongSource = auditAnswerContract('Battery life is 2.3 kg [1].', contract, atoms);
  assert.equal(wrongSource.status, 'abstained');
  assert.ok(wrongSource.unsupportedProtectedTokens.includes('2.3'));
});

test('answer contract never certifies an answer from contradictory evidence', () => {
  const atoms = [
    ...atomizeChunk(chunk('Office: Lisbon.', 'd', 0, 'Locations')),
    ...atomizeChunk(chunk('Office: Reykjavik.', 'd', 1, 'Locations')),
  ];
  const contract = buildEvidenceCertificate('Which office is the headquarters?', atoms, 80);
  const audit = auditAnswerContract('The headquarters is Lisbon [1].', contract, atoms);
  assert.equal(audit.status, 'ambiguous');
});

test('answer contract audits unsupported qualitative labels in cited answers', () => {
  const atoms = atomizeChunk(chunk('Office: Lisbon.', 'd', 0, 'Locations'));
  const contract = buildEvidenceCertificate('Which office is the headquarters?', atoms, 80);
  const correct = auditAnswerContract('The headquarters is Lisbon [1].', contract, atoms);
  assert.equal(correct.status, 'certified');
  const wrong = auditAnswerContract('The headquarters is Reykjavik [1].', contract, atoms);
  assert.equal(wrong.status, 'abstained');
  assert.ok(wrong.unsupportedProtectedTokens.includes('reykjavik'));
});

test('answer contract expands grouped citation ordinals', () => {
  const atoms = [
    ...atomizeChunk(chunk('Battery life: 72 hours.', 'd', 0, 'AeroSense X7')),
    ...atomizeChunk(chunk('Weight: 2.3 kg.', 'd', 1, 'AeroSense X7')),
  ];
  const contract = buildEvidenceCertificate('What is the battery life and weight of AeroSense X7?', atoms, 80);
  const audit = auditAnswerContract('The AeroSense X7 battery lasts 72 hours and weighs 2.3 kg [1, 2].', contract, atoms);
  assert.equal(audit.status, 'certified');
  assert.equal(audit.requiredCitationCoverage, 1);
  assert.deepEqual(audit.unsupportedProtectedTokens, []);
});

test('answer contract does not split company abbreviations into uncited claims', () => {
  const atoms = [
    ...atomizeChunk(chunk('Zephyr Dynamics Inc. operates in Reykjavik.', 'd', 0, 'Overview')),
    ...atomizeChunk(chunk('Zephyr Dynamics Inc. has 847 employees.', 'd', 1, 'Overview')),
  ];
  const contract = buildEvidenceCertificate('What is zypher?', atoms, 80);
  const audit = auditAnswerContract('Zephyr Dynamics Inc. has 847 employees. [1, 2]', contract, atoms);
  assert.equal(audit.status, 'certified');
  assert.deepEqual(audit.unsupportedProtectedTokens, []);
});

test('complete numeric MobCert certificates can answer without generation', () => {
  const atoms = [
    ...atomizeChunk(chunk('Battery life: 72 hours.', 'd', 0, 'AeroSense X7')),
    ...atomizeChunk(chunk('Weight: 2.3 kg.', 'd', 1, 'AeroSense X7')),
  ];
  const certificate = buildEvidenceCertificate('What is the battery life and weight of AeroSense X7?', atoms, 80);
  const answer = buildDirectEvidenceAnswer(certificate, atoms);
  assert.equal(answer, 'Battery: 72 hours [1]. Weight: 2.3 kg [2].');
  assert.equal(auditAnswerContract(answer, certificate, atoms).status, 'certified');
});

test('broad qualitative certificates stay on the generation path', () => {
  const atoms = atomizeChunk(chunk('Zephyr Dynamics operates atmospheric data systems.', 'd', 0, 'Overview'));
  const certificate = buildEvidenceCertificate('What is zypher?', atoms, 80);
  assert.equal(buildDirectEvidenceAnswer(certificate, atoms), null);
});

test('specific population qualifiers prevent generic allowance evidence', () => {
  const atoms = atomizeChunk(chunk('Gym Allowance: $60/month reimbursed via expense report.', 'd', 0, 'Benefits'));
  const certificate = buildEvidenceCertificate('What is the paid sick leave allowance for part-time hourly contractors?', atoms, 80);
  assert.equal(certificate.demands[0].status, 'missing');
  assert.equal(buildDirectEvidenceAnswer(certificate, atoms), null);
});

test('role lookups do not certify from an organization heading alone', () => {
  const atoms = atomizeChunk(chunk('ZEPHYR DYNAMICS INC.', 'd', 0, 'Overview'));
  const certificate = buildEvidenceCertificate('Who is the Chief Marketing Officer (CMO) of Zephyr Dynamics?', atoms, 80);
  assert.equal(certificate.demands[0].status, 'missing');
});

test('direct evidence audits currency thousands separators consistently', () => {
  const atoms = atomizeChunk(chunk('Price: $1,500 USD.', 'd', 0, 'Benefits'));
  const certificate = buildEvidenceCertificate('What is the price?', atoms, 80);
  const answer = buildDirectEvidenceAnswer(certificate, atoms);
  assert.ok(answer);
  assert.equal(auditAnswerContract(answer, certificate, atoms).status, 'certified');
});

test('mobcert retrieval emits source-span status and stays within the prompt budget', async () => {
  const text = '# Products\n## AeroSense X7\nBattery life: 72 hours.\nWeight: 2.3 kg.\n## Other\nUnrelated weather report.';
  await ingestTextDoc(text, 'd', 'handbook.txt', 'p');
  const retrieved = await retrieveContextDetailed('What is the battery life and weight of AeroSense X7?', 'p', {
    policy: 'mobcert-v1', countTokens, maxContextTokens: 120, maxSelectedChunks: 6,
  });
  assert.equal(retrieved.strategy, 'mobcert-v1');
  assert.ok(retrieved.answerContract);
  assert.ok(retrieved.answerContractAtoms?.length);
  assert.equal(retrieved.answerContract?.missingDemandCount, 0);
  assert.ok(retrieved.context.includes('Evidence status:'));
  assert.ok(retrieved.context.includes('72 hours'));
  assert.ok(retrieved.context.includes('2.3 kg'));
  assert.ok(retrieved.contextTokens <= 120);
  assert.ok(retrieved.sources.some(source => source.denseScore > 0 || source.lexicalScore > 0));
});

test('mobcert retrieval preserves table row semantics end to end', async () => {
  await ingestTextDoc(
    '# Leadership\nName | Title | Office | Joined\nJorge Fonseca | Chief Legal Officer | Lisbon | 2019',
    'leadership', 'leadership.txt', 'p',
  );
  const retrieved = await retrieveContextDetailed('Which office is Jorge Fonseca based in?', 'p', {
    policy: 'mobcert-v1', countTokens, maxContextTokens: 140, maxSelectedChunks: 4,
  });
  assert.equal(retrieved.answerContract?.missingDemandCount, 0);
  assert.equal(retrieved.answerContract?.ambiguousDemandCount, 0);
  assert.ok(retrieved.answerContract?.demands[0].valueLabels.includes('lisbon'));
  assert.ok(retrieved.context.includes('Lisbon'));
});

test('mobcert retrieval exposes qualitative conflicts instead of selecting one', async () => {
  await ingestTextDoc(
    '# Locations\nOffice: Lisbon.\nOffice: Reykjavik.',
    'locations', 'locations.txt', 'p',
  );
  const retrieved = await retrieveContextDetailed('Which office is the headquarters?', 'p', {
    policy: 'mobcert-v1', countTokens, maxContextTokens: 140, maxSelectedChunks: 4,
  });
  assert.equal(retrieved.answerContract?.ambiguousDemandCount, 1);
  assert.equal(retrieved.failureReason, 'contract_ambiguous_demands');
  assert.ok(retrieved.answerContract?.demands[0].valueLabels.includes('lisbon'));
  assert.ok(retrieved.answerContract?.demands[0].valueLabels.includes('reykjavik'));
});

test('mobcert retrieval skips generation context when no demand is supported', async () => {
  await ingestTextDoc('# Locations\nOffice: Lisbon.', 'missing', 'missing.txt', 'p');
  const retrieved = await retrieveContextDetailed('What is the battery life?', 'p', {
    policy: 'mobcert-v1', countTokens, maxContextTokens: 140, maxSelectedChunks: 4,
  });
  assert.equal(retrieved.answerContract?.missingDemandCount, 1);
  assert.equal(retrieved.context, '');
  assert.equal(retrieved.failureReason, 'contract_missing_demands');
});

test('research metrics do not call skipped inference fully grounded', () => {
  const metrics = evaluateResearchRun({
    query: 'What is the battery life?',
    response: "I don't have enough information in the uploaded documents to answer that.",
    ragContext: '', policy: 'mobcert-v1', totalLatencyMs: 100,
    ttftMs: null, prefillMs: 0, prefillTokPerSec: 0, decodeMs: 0,
    decodeTokPerSec: 0, promptTokens: 80, generatedTokens: 0, threads: 4,
    primaryBottleneck: 'not_applicable', inferenceSkipped: true,
    selectedChunkCount: 0, candidateChunkCount: 3,
    contractAudit: {
      status: 'abstained', unsupportedProtectedTokens: [], demandsCount: 1,
      demandsSatisfied: 0, demandsMissing: 1, demandsAmbiguous: 0,
    },
  });
  assert.equal(metrics.answerGenerated, false);
  assert.equal(metrics.qualityStatus, 'not_applicable');
  assert.equal(metrics.faithfulness, null);
  assert.equal(metrics.hallucinationRate, null);
  assert.equal(metrics.answerF1, null);
  assert.equal(metrics.ttftMs, null);
  assert.equal(metrics.retrievalEvidenceCoverage, 0);
});

test('research evaluator computes deterministic groundedness, recall, precision, F1 and refusal metrics', async () => {
  const {
    computeRetrievalMetrics,
    computeGroundedness,
    computeTokenF1,
    computeRougeL,
    evaluateResearchRun,
    ZEPHYR_BENCHMARK_QUERIES,
  } = require('../utils/rag/researchEvaluator');

  assert.ok(ZEPHYR_BENCHMARK_QUERIES.length >= 7);

  // 1. Retrieval precision & recall against ground truth
  const gt = 'The learning budget is $1,500 USD/year per employee, and the gym allowance is $60/month reimbursed via expense report.';
  const ctx = 'Learning Budget: $1,500 USD/year per employee\nGym Allowance: $60/month reimbursed via expense report';
  const retM = computeRetrievalMetrics(ctx, gt);
  assert.ok(retM.contextRecall > 0.8);
  assert.ok(retM.contextPrecision > 0.6);

  // 2. Groundedness / Faithfulness
  const groundedResp = 'The annual learning budget is $1,500 USD/year per employee and the gym allowance is $60/month.';
  const groundedScore = computeGroundedness(groundedResp, ctx);
  assert.ok(groundedScore.faithfulness >= 0.8);
  assert.ok(groundedScore.hallucinationRate <= 0.2);

  const hallucinatedResp = 'Employees can travel anywhere on a corporate private jet with free unlimited caviar.';
  const badScore = computeGroundedness(hallucinatedResp, ctx);
  assert.ok(badScore.hallucinationRate >= 0.6);

  // 3. Token F1 & ROUGE-L
  const f1Score = computeTokenF1(groundedResp, gt);
  assert.ok(f1Score.f1 > 0.7);
  const rougeScore = computeRougeL(groundedResp, gt);
  assert.ok(rougeScore > 0.7);

  // 4. evaluateResearchRun end-to-end for mobcert-v1
  const mobcertRun = evaluateResearchRun({
    query: 'What is the annual learning budget and gym allowance for employees?',
    response: groundedResp,
    ragContext: ctx,
    policy: 'mobcert-v1',
    retrievalLatencyMs: 42.5,
    totalLatencyMs: 3500.0,
    ttftMs: 1200.0,
    prefillMs: 800.0,
    prefillTokPerSec: 6.2,
    decodeMs: 2700.0,
    decodeTokPerSec: 3.5,
    promptTokens: 210,
    generatedTokens: 25,
    threads: 4,
    primaryBottleneck: 'token_decode',
    contractAudit: {
      status: 'certified',
      unsupportedProtectedTokens: [],
      demandsCount: 2,
      demandsSatisfied: 2,
      demandsMissing: 0,
      demandsAmbiguous: 0,
    },
  });
  assert.equal(mobcertRun.policy, 'mobcert-v1');
  assert.equal(mobcertRun.contractStatus, 'certified');
  assert.ok(mobcertRun.contextRecall !== null && mobcertRun.contextRecall > 0.8);
  assert.ok(mobcertRun.answerF1 !== null && mobcertRun.answerF1 > 0.7);
  assert.equal(mobcertRun.ttftMs, 1200.0);

  // 5. evaluateResearchRun for negative query refusal
  const refusalRun = evaluateResearchRun({
    query: 'What is the paid sick leave allowance for part-time hourly contractors?',
    response: "I don't have enough information in the uploaded documents to answer that.",
    ragContext: '',
    policy: 'mobcert-v1',
    retrievalLatencyMs: 20.0,
    totalLatencyMs: 30.0,
    ttftMs: null,
    prefillMs: 0,
    prefillTokPerSec: 0,
    decodeMs: 0,
    decodeTokPerSec: 0,
    promptTokens: 10,
    generatedTokens: 0,
    threads: 4,
    primaryBottleneck: 'prompt_prefill',
    inferenceSkipped: true,
    contractAudit: {
      status: 'abstained',
      unsupportedProtectedTokens: [],
    },
  });
  assert.equal(refusalRun.modelAction, 'refused');
  assert.equal(refusalRun.refusalCorrectness, 1.0);
  assert.equal(refusalRun.avoidedInference, true);
  assert.equal(refusalRun.faithfulness, null);
});
