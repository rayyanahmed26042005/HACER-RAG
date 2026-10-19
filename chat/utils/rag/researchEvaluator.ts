/**
 * researchEvaluator.ts
 *
 * Deterministic on-device research evaluation module for physical phone experiments.
 * Computes IEEE-standard evaluation parameters matching evaluate_rag_offline.py:
 * - Context Recall & Context Precision
 * - Retrieval Evidence Coverage
 * - Groundedness / Faithfulness (Context Entailment & Hallucination Rate)
 * - Answer Correctness & Truthness (Token F1, Precision, Recall, ROUGE-L)
 * - Refusal Correctness (Controlled Abstention Verification)
 * - Contract Status & Demand Auditing
 * - Latency & Hardware Telemetry (TTFT, Prefill tok/s, Decode tok/s, Total ms, RAM)
 */

export interface ResearchMetrics {
  policy: 'mobcert-v1' | 'ebmrag-v2';
  queryId?: string;
  expectedAction?: 'answer' | 'refuse';
  modelAction: 'answered' | 'refused';
  
  // 1. Retrieval Evidence Quality
  contextRecall: number | null;
  contextPrecision: number | null;
  retrievalEvidenceCoverage: number | null; // 0 - 100%, labeled answer queries only
  selectedChunkCount: number;
  candidateChunkCount: number;
  compressionRatio: number;
  retrievalLatencyMs: number;

  // 2. Deterministic Groundedness & Faithfulness
  faithfulness: number | null; // 0.0 - 1.0 when an answer was generated
  hallucinationRate: number | null; // 0.0 - 1.0 when an answer was generated

  // 3. Answer Correctness (against reference if available)
  answerF1: number | null;
  answerPrecision: number | null;
  answerRecall: number | null;
  rougeL: number | null;
  hasGroundTruth: boolean;

  // 4. Refusal Correctness
  refusalCorrectness: number | null; // 1.0 or 0.0 if expectedAction === 'refuse'

  // 5. Contract Status (MobCert vs EBM-RAG)
  contractStatus: 'certified' | 'abstained' | 'repaired' | 'unverifiable' | 'ambiguous' | 'partial' | 'uncertified_baseline';
  demandsCount: number;
  demandsSatisfied: number;
  demandsMissing: number;
  demandsAmbiguous: number;
  conflictsDetected: string[];
  avoidedInference: boolean;
  answerGenerated: boolean;
  qualityStatus: 'measured' | 'not_applicable';

  // 6. Latency & Telemetry
  totalLatencyMs: number;
  ttftMs: number | null;
  prefillMs: number;
  prefillTokPerSec: number;
  decodeMs: number;
  decodeTokPerSec: number;
  promptTokens: number;
  generatedTokens: number;
  primaryBottleneck: 'prompt_prefill' | 'token_decode' | 'not_applicable';
  threads: number;
  availableRamMb: number | null;
  batteryTempC: number | null;
  thermalStatus: string;
}

export interface ReferenceQuery {
  id: string;
  category: string;
  mode: 'general' | 'document';
  query: string;
  groundTruth: string;
  targetSection: string;
  expectedAction: 'answer' | 'refuse';
}

export const ZEPHYR_BENCHMARK_QUERIES: ReferenceQuery[] = [
  {
    id: 'q1_single_hop_battery',
    category: 'single_hop',
    mode: 'document',
    query: 'What is the battery life and weight of the AeroSense X7?',
    groundTruth: 'The AeroSense X7 uses a QuantumCell battery that lasts 72 hours on a single charge and weighs 2.3 kg.',
    targetSection: '2.1 AeroSense X7',
    expectedAction: 'answer',
  },
  {
    id: 'q2_single_hop_clo',
    category: 'single_hop',
    mode: 'document',
    query: 'Who is the Chief Legal Officer and which office are they based in?',
    groundTruth: 'Jorge Fonseca is the Chief Legal Officer, based in the Lisbon office.',
    targetSection: '1.1 Leadership Team',
    expectedAction: 'answer',
  },
  {
    id: 'q3_single_hop_bonus',
    category: 'single_hop',
    mode: 'document',
    query: 'What is the annual learning budget and gym allowance for employees?',
    groundTruth: 'The learning budget is $1,500 USD/year per employee, and the gym allowance is $60/month reimbursed via expense report.',
    targetSection: '3.3 Compensation & Benefits',
    expectedAction: 'answer',
  },
  {
    id: 'q4_multi_hop_pricing',
    category: 'multi_hop',
    mode: 'document',
    query: 'Compare the subscription pricing, drone capacity, and customers between the SwarmKit Pro tiers.',
    groundTruth: 'SwarmKit Pro costs $1,200/month for up to 256 drones and $3,800/month for full 2,048-drone capacity. Customers include the Norwegian Meteorological Institute and Kenya Civil Aviation Authority.',
    targetSection: '2.2 SwarmKit Pro',
    expectedAction: 'answer',
  },
  {
    id: 'q5_multi_hop_remote_stipends',
    category: 'multi_hop',
    mode: 'document',
    query: 'What are all the equipment stipends and monthly allowances available for remote-eligible employees?',
    groundTruth: 'A one-time remote work equipment stipend of $800 USD for new hires, and a $50/month internet stipend for remote-eligible employees.',
    targetSection: '3.2 Remote Work Policy & 3.3 Compensation & Benefits',
    expectedAction: 'answer',
  },
  {
    id: 'q6_negative_contractor_leave',
    category: 'negative_insufficient_evidence',
    mode: 'document',
    query: 'What is the paid sick leave allowance for part-time hourly contractors?',
    groundTruth: 'The handbook only specifies policies for full-time employees and does not specify contractor sick leave. The model must refuse to invent information and state insufficient document evidence.',
    targetSection: '3.1 Leave Policy',
    expectedAction: 'refuse',
  },
  {
    id: 'q7_negative_unlisted_executive',
    category: 'negative_insufficient_evidence',
    mode: 'document',
    query: 'Who is the Chief Marketing Officer (CMO) of Zephyr Dynamics?',
    groundTruth: 'The leadership team in Section 1.1 does not list a Chief Marketing Officer. The model must refuse or state it is not mentioned.',
    targetSection: '1.1 Leadership Team',
    expectedAction: 'refuse',
  },
  {
    id: 'q8_general_edge_computing',
    category: 'general_no_rag',
    mode: 'general',
    query: 'Explain what on-device edge AI inference is in two concise sentences.',
    groundTruth: 'Edge AI inference refers to running machine learning models locally on hardware such as mobile phones without sending data to cloud servers.',
    targetSection: 'None (General LLM)',
    expectedAction: 'answer',
  },
  {
    id: 'q9_general_iceland_capital',
    category: 'general_no_rag',
    mode: 'general',
    query: 'What is the capital of Iceland and what language is spoken there?',
    groundTruth: 'Reykjavik is the capital of Iceland, and Icelandic is the official language.',
    targetSection: 'None (General LLM)',
    expectedAction: 'answer',
  },
];

const STOPWORDS = new Set([
  'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'of', 'in', 'on', 'at',
  'to', 'for', 'with', 'by', 'that', 'this', 'it', 'its', 'as', 'from', 'be', 'been',
]);

export function normalizeText(text: string): string {
  if (!text) return '';
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function stripCitationOrdinals(text: string): string {
  return (text || '').replace(/\[\s*\d+\s*\]/g, ' ');
}

export function getTokens(text: string): string[] {
  const norm = normalizeText(text);
  return norm ? norm.split(/\s+/).filter(Boolean) : [];
}

export function isContentToken(token: string): boolean {
  return !STOPWORDS.has(token) && (token.length > 2 || /\d/.test(token));
}

export function getBigrams(tokens: string[]): string[] {
  if (tokens.length < 2) return [];
  const result: string[] = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    result.push(`${tokens[i]} ${tokens[i + 1]}`);
  }
  return result;
}

export function computeRetrievalMetrics(retrievedContext: string, groundTruth: string): {
  contextRecall: number;
  contextPrecision: number;
} {
  const gtTokens = new Set(getTokens(stripCitationOrdinals(groundTruth)).filter(isContentToken));
  if (!gtTokens.size) {
    return { contextRecall: 1.0, contextPrecision: 1.0 };
  }
  const ctxTokens = new Set(getTokens(stripCitationOrdinals(retrievedContext)).filter(isContentToken));
  let matched = 0;
  for (const token of gtTokens) {
    if (ctxTokens.has(token)) matched++;
  }
  const recall = matched / gtTokens.size;
  const precision = matched / Math.max(1, ctxTokens.size);
  return {
    contextRecall: Number(recall.toFixed(4)),
    contextPrecision: Number(precision.toFixed(4)),
  };
}

export function computeGroundedness(response: string, context: string): {
  faithfulness: number;
  hallucinationRate: number;
} {
  const cleanedResponse = stripCitationOrdinals(response);
  const sentences = cleanedResponse.split(/[.!?]+/).map(s => s.trim()).filter(Boolean);
  if (!sentences.length) {
    return { faithfulness: 1.0, hallucinationRate: 0.0 };
  }

  const ctxTokens = getTokens(stripCitationOrdinals(context));
  const ctxWords = new Set(ctxTokens);
  const ctxBigrams = new Set(getBigrams(ctxTokens));

  let groundedSentences = 0;
  let totalContentWords = 0;
  let supportedContentWords = 0;

  for (const sentence of sentences) {
    const sentTokens = getTokens(sentence);
    const sentWords = sentTokens.filter(isContentToken);
    if (!sentWords.length) {
      groundedSentences++;
      continue;
    }
    const sentBigrams = getBigrams(sentTokens);
    let wordsInCtx = 0;
    for (const w of sentWords) {
      if (ctxWords.has(w)) wordsInCtx++;
    }
    let bigramsInCtx = 0;
    for (const bg of sentBigrams) {
      if (ctxBigrams.has(bg)) bigramsInCtx++;
    }

    totalContentWords += sentWords.length;
    supportedContentWords += wordsInCtx;

    const ratio = wordsInCtx / sentWords.length;
    const bigramRatio = sentBigrams.length ? bigramsInCtx / sentBigrams.length : 0;
    if (ratio >= 0.70 || (sentBigrams.length > 0 && bigramRatio >= 0.5)) {
      groundedSentences++;
    }
  }

  const sentenceFaithfulness = groundedSentences / sentences.length;
  const tokenGroundedness = supportedContentWords / Math.max(1, totalContentWords);
  const faithfulness = Number((0.5 * sentenceFaithfulness + 0.5 * tokenGroundedness).toFixed(4));
  const hallucinationRate = Number(Math.max(0, 1.0 - faithfulness).toFixed(4));

  return { faithfulness, hallucinationRate };
}

export function computeTokenF1(prediction: string, groundTruth: string): {
  precision: number;
  recall: number;
  f1: number;
} {
  const predTokens = getTokens(stripCitationOrdinals(prediction));
  const gtTokens = getTokens(stripCitationOrdinals(groundTruth));

  if (!predTokens.length || !gtTokens.length) {
    const exact = normalizeText(prediction) === normalizeText(groundTruth) ? 1.0 : 0.0;
    return { precision: exact, recall: exact, f1: exact };
  }

  const predCounts = new Map<string, number>();
  for (const t of predTokens) {
    predCounts.set(t, (predCounts.get(t) || 0) + 1);
  }

  const gtCounts = new Map<string, number>();
  for (const t of gtTokens) {
    gtCounts.set(t, (gtCounts.get(t) || 0) + 1);
  }

  let overlap = 0;
  for (const [t, count] of predCounts.entries()) {
    if (gtCounts.has(t)) {
      overlap += Math.min(count, gtCounts.get(t)!);
    }
  }

  const precision = overlap / predTokens.length;
  const recall = overlap / gtTokens.length;
  const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1: Number(f1.toFixed(4)),
  };
}

export function computeRougeL(prediction: string, groundTruth: string): number {
  const pTokens = getTokens(stripCitationOrdinals(prediction));
  const gTokens = getTokens(stripCitationOrdinals(groundTruth));
  const m = pTokens.length;
  const n = gTokens.length;
  if (m === 0 || n === 0) return 0.0;

  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (pTokens[i - 1] === gTokens[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  const lcs = dp[m][n];
  const prec = lcs / m;
  const rec = lcs / n;
  const f1 = (prec + rec) > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  return Number(f1.toFixed(4));
}

export function findMatchingReferenceQuery(query: string): ReferenceQuery | null {
  const norm = normalizeText(query);
  for (const item of ZEPHYR_BENCHMARK_QUERIES) {
    const itemNorm = normalizeText(item.query);
    if (norm === itemNorm || norm.includes(itemNorm) || itemNorm.includes(norm)) {
      return item;
    }
    // Also check token overlap
    const queryTokens = new Set(getTokens(query));
    const refTokens = getTokens(item.query);
    const overlap = refTokens.filter(t => queryTokens.has(t)).length;
    if (overlap / refTokens.length >= 0.75) {
      return item;
    }
  }
  return null;
}

export function evaluateResearchRun(params: {
  query: string;
  response: string;
  ragContext?: string;
  policy: 'mobcert-v1' | 'ebmrag-v2';
  retrievalLatencyMs?: number;
  totalLatencyMs: number;
  ttftMs: number | null;
  prefillMs: number;
  prefillTokPerSec: number;
  decodeMs: number;
  decodeTokPerSec: number;
  promptTokens: number;
  generatedTokens: number;
  threads: number;
  primaryBottleneck: 'prompt_prefill' | 'token_decode' | 'not_applicable';
   inferenceSkipped?: boolean;
   directAnswer?: boolean;
  selectedChunkCount?: number;
  candidateChunkCount?: number;
  compressionRatio?: number;
  queryTermCoverage?: number;
  contractAudit?: {
    status: 'certified' | 'abstained' | 'repaired' | 'unverifiable' | 'ambiguous' | 'partial';
    unsupportedProtectedTokens: string[];
    demandsCount?: number;
    demandsSatisfied?: number;
    demandsMissing?: number;
    demandsAmbiguous?: number;
    conflictingDemands?: string[];
  };
  deviceState?: {
    availableMemoryMb?: number | null;
    batteryTempC?: number | null;
    thermalStatusName?: string;
  };
}): ResearchMetrics {
  const {
    query,
    response,
    ragContext = '',
    policy,
    retrievalLatencyMs = 0,
    totalLatencyMs,
    ttftMs,
    prefillMs,
    prefillTokPerSec,
    decodeMs,
    decodeTokPerSec,
    promptTokens,
    generatedTokens,
    threads,
    primaryBottleneck,
    inferenceSkipped = false,
    directAnswer = false,
    selectedChunkCount = 0,
    candidateChunkCount = 0,
    compressionRatio = 0,
    queryTermCoverage = 1.0,
    contractAudit,
    deviceState,
  } = params;

  const isRefusal = /don't have enough information|not explicitly state|does not mention|do not mention|doesn't mention|provided excerpts? do not contain|provided text does not contain|insufficient document evidence|cannot certify|not mentioned in the provided/i.test(response);
  const answerGenerated = (directAnswer || (!inferenceSkipped && generatedTokens > 0)) && !!response.trim() && !isRefusal;
  const matchedRef = findMatchingReferenceQuery(query);

  let contextRecall: number | null = null;
  let contextPrecision: number | null = null;
  let answerF1: number | null = null;
  let answerPrecision: number | null = null;
  let answerRecall: number | null = null;
  let rougeL: number | null = null;
  let refusalCorrectness: number | null = null;

  if (matchedRef) {
    if (matchedRef.expectedAction === 'refuse') {
      refusalCorrectness = isRefusal ? 1.0 : 0.0;
      // A correctly refused question has no answer or positive evidence span
      // to score. Refusal correctness is reported separately.
    } else {
      const retM = computeRetrievalMetrics(ragContext, matchedRef.groundTruth);
      contextRecall = retM.contextRecall;
      contextPrecision = retM.contextPrecision;

      if (answerGenerated) {
        const tokM = computeTokenF1(response, matchedRef.groundTruth);
        answerF1 = tokM.f1;
        answerPrecision = tokM.precision;
        answerRecall = tokM.recall;
        rougeL = computeRougeL(response, matchedRef.groundTruth);
      }
      refusalCorrectness = isRefusal ? 0.0 : 1.0;
    }
  }

  // Deterministic Groundedness & Faithfulness
  let groundednessScore: number | null = null;
  let hallucinationRate: number | null = null;
  if (answerGenerated && ragContext.trim()) {
    const groundedness = computeGroundedness(response, ragContext);
    groundednessScore = groundedness.faithfulness;
    hallucinationRate = groundedness.hallucinationRate;
  }

  // Contract Status & Demands
  let finalContractStatus: ResearchMetrics['contractStatus'] = 'uncertified_baseline';
  let conflictsDetected: string[] = [];
  let demandsCount = 0;
  let demandsSatisfied = 0;
  let demandsMissing = 0;
  let demandsAmbiguous = 0;

  if (policy === 'mobcert-v1') {
    if (contractAudit) {
      finalContractStatus = contractAudit.status;
      conflictsDetected = contractAudit.conflictingDemands || contractAudit.unsupportedProtectedTokens || [];
      demandsCount = contractAudit.demandsCount || 0;
      demandsSatisfied = contractAudit.demandsSatisfied || 0;
      demandsMissing = contractAudit.demandsMissing || 0;
      demandsAmbiguous = contractAudit.demandsAmbiguous || 0;
    } else {
      finalContractStatus = 'uncertified_baseline';
    }
  }

  return {
    policy,
    queryId: matchedRef?.id,
    expectedAction: matchedRef?.expectedAction,
    modelAction: isRefusal ? 'refused' : 'answered',
    contextRecall,
    contextPrecision,
    retrievalEvidenceCoverage: matchedRef && matchedRef.expectedAction === 'answer' && contextRecall !== null
      ? Number((contextRecall * 100).toFixed(1)) : null,
    selectedChunkCount,
    candidateChunkCount,
    compressionRatio: Number(compressionRatio.toFixed(3)),
    retrievalLatencyMs: Number(retrievalLatencyMs.toFixed(1)),
    faithfulness: groundednessScore,
    hallucinationRate,
    answerF1,
    answerPrecision,
    answerRecall,
    rougeL,
    hasGroundTruth: Boolean(matchedRef),
    refusalCorrectness,
    contractStatus: finalContractStatus,
    demandsCount,
    demandsSatisfied,
    demandsMissing,
    demandsAmbiguous,
    conflictsDetected,
    avoidedInference: inferenceSkipped,
    answerGenerated,
    qualityStatus: answerGenerated ? 'measured' : 'not_applicable',
    totalLatencyMs: Number(totalLatencyMs.toFixed(1)),
    ttftMs: ttftMs !== null ? Number(ttftMs.toFixed(1)) : null,
    prefillMs: Number(prefillMs.toFixed(1)),
    prefillTokPerSec: Number(prefillTokPerSec.toFixed(1)),
    decodeMs: Number(decodeMs.toFixed(1)),
    decodeTokPerSec: Number(decodeTokPerSec.toFixed(1)),
    promptTokens,
    generatedTokens,
    primaryBottleneck,
    threads,
    availableRamMb: deviceState?.availableMemoryMb ?? null,
    batteryTempC: deviceState?.batteryTempC ?? null,
    thermalStatus: deviceState?.thermalStatusName || 'normal',
  };
}
