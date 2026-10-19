// chat/tasks/qa.ts

import { LlamaContext } from '../utils/types';
import { retrieveContextDetailed, RetrievalMetrics } from '../utils/rag/ragPipeline';
import { buildDocumentPrompt, createPromptBudget, NO_DOCUMENT_EVIDENCE } from '../utils/rag/promptBudget';
import { evaluateRag } from '../utils/rag/metrics';
import { auditAnswerContract, AnswerContractAudit, buildDirectEvidenceAnswer } from '../utils/rag/evidenceCertificate';
import { ResearchMetrics, evaluateResearchRun } from '../utils/rag/researchEvaluator';
import RNFS from 'react-native-fs';
import {
  InferenceRequestOptions,
  InferencePlan,
  recordInferenceObservation,
  selectInferencePlan,
  BASELINE_INFERENCE_CONFIG,
  getGeneralOutputLimit,
} from '../utils/inferenceController';
import { DeviceState, getDeviceState } from '../utils/deviceTelemetry';

export interface InferenceBenchmarkReport {
  rag?: RetrievalMetrics;
  rag_contract_audit?: AnswerContractAudit;
  rag_evidence_selection_diagnostics?: RetrievalMetrics['evidenceSelectionDiagnostics'];
  rag_quality?: ReturnType<typeof evaluateRag>;
  research_metrics?: ResearchMetrics;
  input_prompt_tokens: number | null;
  output_tokens_requested: number;
  benchmark_request_id?: string;
  benchmark_query_id?: string;
  benchmark_corpus_id?: string;
  execution_path: 'llm_generated' | 'direct_evidence' | 'abstained';
  allow_direct_evidence_answer: boolean;
  inference_skipped: boolean;
  direct_evidence_answer: boolean;
  native_timings_available: boolean;
  native_prompt_truncated: boolean;
  output_limit_reached: boolean;
  tokens_cached: number | null;
  timestamp: string;
  query: string;
  chat_mode: 'general' | 'document';
  profile_id: string | null;

  // RAG and prompt construction stages.
  rag_retrieval_ms: number;
  embedding_ms: number;
  vector_search_ms: number;
  context_construction_ms: number;
  context_chars: number;
  retrieved_chunk_count: number;
  candidate_chunk_count: number;
  rag_strategy: string;
  rag_candidate_k: number;
  rag_duplicate_chunks_removed: number;
  rag_context_token_estimate: number;
  rag_requested_context_budget: number;
  rag_applied_context_budget: number;
  rag_compression_ratio: number;
  retrieved_top_score: number | null;
  retrieved_score_margin: number | null;
  rag_query_term_count: number;
  rag_matched_query_term_count: number;
  rag_query_term_coverage: number;
  rag_lexical_hit: boolean;
  rag_exact_term_matches: number;
  rag_selected_chunk_ids: string[];
  rag_failure_reason: string | null;
  prompt_chars: number;
  prompt_tokenization_ms: number;
  prompt_tokens: number;

  // Native llama.cpp timing split.
  prefill_ms: number;
  prefill_tokens_per_second: number;
  generated_tokens: number;
  decode_ms: number;
  tokens_per_second: number;
  /** Request submission to first streamed token, including retrieval. */
  ttft_ms: number | null;
  /** LLM completion start to first streamed token. */
  llm_ttft_ms: number | null;
  total_inference_ms: number;
  total_warm_latency_ms: number;

  // Explicit controller/configuration record.
  controller_mode: InferencePlan['mode'];
  evidence_demand: InferencePlan['evidenceDemand'];
  controller_workload_bucket: string;
  controller_selection_reason: string;
  controller_state_read_ms: number;
  device_thermal_status: string;
  device_available_memory_mb: number | null;
  device_low_memory: boolean;
  device_battery_level: number | null;
  device_battery_temp_c: number | null;
  device_cpu_frequencies_khz: number[];
  n_threads: number;
  n_predict: number;
  context_token_budget: number | null;
  primary_bottleneck: 'prompt_prefill' | 'token_decode' | 'not_applicable';

  // Quality is intentionally not fabricated by the profiler. The experiment
  // runner must supply retrieval/answer labels separately.
  quality_evaluation: 'not_measured';
  response_text: string;
}

export let latestInferenceBenchmark: InferenceBenchmarkReport | null = null;

const nowMs = (): number => {
  // performance.now() is monotonic on React Native. Date.now() is retained as
  // a fallback for Jest and older JS runtimes.
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
};

const safePercent = (value: number, total: number): string =>
  total > 0 ? ((value / total) * 100).toFixed(1) : '0.0';

/**
 * Conservative display-time repair for the opt-in contract policy. It removes
 * only sentences containing protected values that the selected source spans do
 * not authorize. Full entailment is still a labeled evaluation problem.
 */
function repairUnsupportedContractClaims(response: string, audit: AnswerContractAudit): string {
  if (audit.status !== 'abstained' || !audit.unsupportedProtectedTokens.length) return response;
  if (audit.unsupportedProtectedTokens.some(token => token.startsWith('polarity:'))) return '';
  const bad = audit.unsupportedProtectedTokens.map(token => token.toLowerCase());
  const sentences = response.split(/(?<=[.!?])\s+/).filter(Boolean);
  const retained = sentences.filter(sentence => !bad.some(token => sentence.toLowerCase().includes(token)));
  return retained.join(' ').trim();
}

export const handleQA = async (
  context: LlamaContext,
  input: string,
  profileId?: string | null,
  chatMode: 'general' | 'document' = 'general',
  inferenceOptions: InferenceRequestOptions = {},
): Promise<string> => {
  const globalStart = nowMs();
  latestInferenceBenchmark = null;

  let deviceState: DeviceState | undefined = inferenceOptions.deviceState;
  let controllerStateReadMs = 0;
  if (inferenceOptions.mode === 'adaptive' && !deviceState) {
    const stateStart = nowMs();
    deviceState = await getDeviceState();
    controllerStateReadMs = nowMs() - stateStart;
  }
  const effectiveInferenceOptions: InferenceRequestOptions = {
    ...inferenceOptions,
    deviceState,
  };
  // Select the evidence policy before retrieval. The final plan is selected
  // again after exact prompt tokenization so thread history uses the true
  // workload bucket.
  const preliminaryPlan = selectInferencePlan(
    Math.max(1, Math.ceil(input.length / 4)),
    chatMode,
    effectiveInferenceOptions,
    input,
  );
  const budgetStart = nowMs();
  const promptBudget = chatMode === 'document' ? await createPromptBudget(context, input,
    BASELINE_INFERENCE_CONFIG.n_ctx, preliminaryPlan.nPredict) : null;
  let ragMetrics: RetrievalMetrics | undefined;
  let directEvidenceAnswer = '';

  // ── RAG retrieval ─────────────────────────────────────────────────────────
  let ragContext: string | null = null;
  let retrievalMs = 0;
  let embeddingMs = 0;
  let vectorSearchMs = 0;
  let contextConstructionMs = 0;
  let contextChars = 0;
  let retrievedChunkCount = 0;
  let candidateChunkCount = 0;
  let ragStrategy = 'not_used';
  let ragCandidateK = 0;
  let ragDuplicateChunksRemoved = 0;
  let ragContextTokenEstimate = 0;
  let ragRequestedContextBudget = 0;
  let ragAppliedContextBudget = 0;
  let ragCompressionRatio = 1;
  let retrievedTopScore: number | null = null;
  let retrievedScoreMargin: number | null = null;
  let ragQueryTermCount = 0;
  let ragMatchedQueryTermCount = 0;
  let ragQueryTermCoverage = 0;
  let ragLexicalHit = false;
  let ragExactTermMatches = 0;
  let ragSelectedChunkIds: string[] = [];
  let ragFailureReason: string | null = null;
  let ragContractAudit: AnswerContractAudit | undefined;

  if (chatMode === 'document' && !profileId) throw new Error('Select a profile before searching documents.');
  if (chatMode === 'document' && profileId) {
    const retrievalStart = nowMs();
    try {
      const retrieval = await retrieveContextDetailed(input, profileId, {
        maxContextTokens: Math.min(preliminaryPlan.contextTokenBudget ?? Infinity, promptBudget!.contextBudget),
        deduplicate: preliminaryPlan.deduplicate,
        evidenceDemand: preliminaryPlan.evidenceDemand,
        candidateK: preliminaryPlan.candidateK,
        maxSelectedChunks: preliminaryPlan.maxSelectedChunks,
         // An explicit EBM-RAG-v2 benchmark request must exercise the
         // evidence-budgeted selector even when threads are held at the
         // historical baseline. The old fixed-top-k path remains available
         // only when no policy is requested.
         strategy: effectiveInferenceOptions.ragPolicy === 'ebmrag-v2' || preliminaryPlan.mode === 'adaptive'
           ? 'evidence' : 'fixed_top_k',
        policy: effectiveInferenceOptions.ragPolicy,
        countTokens: promptBudget!.countTokens,
        fitsPrompt: promptBudget!.fitsPrompt,
      });
      ragMetrics = retrieval;
      retrievalMs = nowMs() - retrievalStart;
      if (retrieval) {
        ragContext = retrieval.context;
        embeddingMs = retrieval.embeddingMs;
        vectorSearchMs = retrieval.vectorSearchMs;
        contextConstructionMs = retrieval.contextConstructionMs;
        contextChars = retrieval.contextChars;
        retrievedChunkCount = retrieval.selectedChunkCount;
        candidateChunkCount = retrieval.candidateChunkCount;
        ragStrategy = retrieval.strategy;
        ragCandidateK = retrieval.candidateK;
        ragDuplicateChunksRemoved = retrieval.duplicateChunksRemoved;
        ragContextTokenEstimate = retrieval.contextTokenEstimate;
        ragRequestedContextBudget = retrieval.requestedContextTokenBudget;
        ragAppliedContextBudget = retrieval.appliedContextTokenBudget;
        ragCompressionRatio = retrieval.compressionRatio;
        retrievedTopScore = retrieval.topScore;
        retrievedScoreMargin = retrieval.scoreMargin;
        ragQueryTermCount = retrieval.queryTermCount;
        ragMatchedQueryTermCount = retrieval.matchedQueryTermCount;
        ragQueryTermCoverage = retrieval.queryTermCoverage;
        ragLexicalHit = retrieval.lexicalHit;
        ragExactTermMatches = retrieval.exactTermMatches;
        ragSelectedChunkIds = retrieval.selectedChunkIds;
        ragFailureReason = retrieval.failureReason;
        if (effectiveInferenceOptions.ragPolicy === 'mobcert-v1' &&
            effectiveInferenceOptions.allowDirectEvidenceAnswer !== false && retrieval.answerContract && retrieval.answerContractAtoms) {
          directEvidenceAnswer = buildDirectEvidenceAnswer(retrieval.answerContract, retrieval.answerContractAtoms) || '';
        }
      } else {
        ragStrategy = 'ebmrag-v1';
        ragCandidateK = preliminaryPlan.candidateK || 0;
        ragRequestedContextBudget = preliminaryPlan.contextTokenBudget || 0;
        ragAppliedContextBudget = preliminaryPlan.contextTokenBudget || 0;
        ragFailureReason = 'no_relevant_evidence';
      }
      console.log(
        `[QA] RAG retrieval completed in ${retrievalMs.toFixed(2)}ms:`,
        ragContext ? `YES (${ragContext.length} chars)` : 'NONE',
      );
    } catch (err) {
      retrievalMs = nowMs() - retrievalStart;
      // Loading/corruption/model errors must not masquerade as no evidence.
      throw err;
    }
  }

  // ── Build prompt ──────────────────────────────────────────────────────────
  let finalPrompt = '';
  if (chatMode === 'document') {
    finalPrompt = buildDocumentPrompt(input, ragContext || '');
  } else {
    finalPrompt = 'You are Genix, a concise offline AI assistant running on the user\'s device.\n' +
      'Answer directly in 2-4 sentences. Do not greet the user, repeat the question, or add unnecessary detail. If you don\'t know something, say so honestly.\n\n' +
      'Question: ' + input;
  }

  // A separate native tokenize call is useful for controlled experiments but
  // is profiling overhead in normal chat. The controller uses an estimate in
  // the normal path; native prompt_n remains authoritative in the report.
  const tokenizationStart = nowMs();
  let plannedPromptTokens = Math.max(1, Math.ceil(finalPrompt.length / 4));
  if (promptBudget) {
    plannedPromptTokens = await promptBudget.countPrompt(ragContext || '');
    if (plannedPromptTokens > promptBudget.promptLimit) throw new Error('Document prompt exceeds its context budget.');
  } else if (effectiveInferenceOptions.measurePromptTokens && context.tokenize) {
    try {
      const tokenized = await context.tokenize(finalPrompt);
      if (Array.isArray(tokenized?.tokens) && tokenized.tokens.length > 0) {
        plannedPromptTokens = tokenized.tokens.length;
      }
    } catch (error) {
      console.warn('[QA] Prompt tokenization unavailable; using character estimate:', error);
    }
  }
  const promptTokenizationMs = promptBudget ? Math.max(0, nowMs() - budgetStart - retrievalMs) :
    chatMode === 'general' || effectiveInferenceOptions.measurePromptTokens ? nowMs() - tokenizationStart : 0;
  const plan = selectInferencePlan(plannedPromptTokens, chatMode, effectiveInferenceOptions, input);
  if (promptBudget) plan.nPredict = promptBudget.outputTokens;
  if (chatMode === 'general') {
    // Avoid requesting another 384–512 tokens after a long prompt inside the
    // 512-token context. This caps CPU decode runaway while preserving larger
    // budgets for explicit story/code/detail requests.
    const contextRunway = Math.max(32, BASELINE_INFERENCE_CONFIG.n_ctx - plannedPromptTokens - 8);
    plan.nPredict = Math.min(plan.nPredict, getGeneralOutputLimit(input), contextRunway);
  }

  // ── Run LLM ───────────────────────────────────────────────────────────────
  const llmStart = nowMs();
  let firstTokenAt: number | null = null;
  console.log(
    `[QA] Initiating LLM inference (prompt chars: ${finalPrompt.length}, ` +
    `planned tokens: ${plannedPromptTokens}, threads: ${plan.nThreads}, mode: ${plan.mode})...`,
  );

  const inferenceSkipped = chatMode === 'document' && (!ragContext || !!directEvidenceAnswer);
  const result: Awaited<ReturnType<LlamaContext['completion']>> = inferenceSkipped
    ? { text: directEvidenceAnswer || NO_DOCUMENT_EVIDENCE }
    : await context.completion(
    {
      messages: [{ role: 'user', content: finalPrompt }],
      jinja: true,
      n_threads: plan.nThreads,
      n_predict: plan.nPredict,
      temperature: 0.1,
      top_p: 0.9,
      stop: [
        '<end_of_turn>',
        '<eos>',
        '<|im_end|>',
        '</s>',
        '<|eot_id|>',
      ],
    },
    data => {
      if (firstTokenAt === null && data.token) firstTokenAt = nowMs();
    },
  );

  const llmEnd = nowMs();
  const totalLLMTime = Math.max(0, llmEnd - llmStart);
  const totalWarmLatency = Math.max(0, llmEnd - globalStart);

  let response = result.text?.trim() || '';
  response = response.replace(/(<end_of_turn>|<eos>|<\|im_end\|>|<\/s>|<\|eot_id\|>)/gi, '').trim();
  if (!response) throw new Error('Model returned empty response.');

  if (directEvidenceAnswer && ragMetrics?.answerContract && ragMetrics.answerContractAtoms) {
    ragContractAudit = auditAnswerContract(response, ragMetrics.answerContract, ragMetrics.answerContractAtoms);
  } else if (ragMetrics?.answerContract && ragMetrics.answerContractAtoms) {
    const initialAudit = auditAnswerContract(response, ragMetrics.answerContract, ragMetrics.answerContractAtoms);
    const repaired = repairUnsupportedContractClaims(response, initialAudit);
    if (repaired !== response) {
      response = repaired || NO_DOCUMENT_EVIDENCE;
      const repairedAudit = auditAnswerContract(response, ragMetrics.answerContract, ragMetrics.answerContractAtoms);
      ragContractAudit = { ...repairedAudit, status: repairedAudit.status === 'abstained' ? 'abstained' : 'repaired' };
    } else {
      ragContractAudit = initialAudit;
    }
  }

  const timings = result.timings;
  const promptTokens = timings?.prompt_n ?? result.tokens_evaluated ?? plannedPromptTokens;
  const prefillMs = timings?.prompt_ms ?? 0;
  const prefillTokPerSec = timings?.prompt_per_second ??
    (prefillMs > 0 ? promptTokens / (prefillMs / 1000) : 0);
  const genTokens = timings?.predicted_n ?? result.tokens_predicted ?? 0;
  const decodeMs = Math.max(0, timings?.predicted_ms ?? (totalLLMTime - prefillMs));
  const decodeTokPerSec = timings?.predicted_per_second ??
    (decodeMs > 0 ? genTokens / (decodeMs / 1000) : 0);
  const ttftMs = firstTokenAt === null ? null : Math.max(0, firstTokenAt - globalStart);
  const llmTTFTMs = firstTokenAt === null ? null : Math.max(0, firstTokenAt - llmStart);
  const primaryBottleneck: 'prompt_prefill' | 'token_decode' | 'not_applicable' = inferenceSkipped
    ? 'not_applicable'
    : prefillMs > decodeMs ? 'prompt_prefill' : 'token_decode';

  const report: InferenceBenchmarkReport = {
    rag: ragMetrics,
    rag_contract_audit: ragContractAudit,
    rag_evidence_selection_diagnostics: ragMetrics?.evidenceSelectionDiagnostics,
    rag_quality: ragMetrics ? evaluateRag(ragMetrics.selectedChunkIds, ragContext || '', response, inferenceOptions.ragLabels) : undefined,
    input_prompt_tokens: plannedPromptTokens,
    output_tokens_requested: plan.nPredict,
    benchmark_request_id: effectiveInferenceOptions.benchmarkRequestId,
    benchmark_query_id: effectiveInferenceOptions.benchmarkQueryId,
    benchmark_corpus_id: effectiveInferenceOptions.benchmarkCorpusId,
    inference_skipped: inferenceSkipped,
    direct_evidence_answer: !!directEvidenceAnswer,
    execution_path: inferenceSkipped
      ? (directEvidenceAnswer ? 'direct_evidence' : 'abstained')
      : 'llm_generated',
    allow_direct_evidence_answer: effectiveInferenceOptions.allowDirectEvidenceAnswer !== false,
    native_timings_available: !!result.timings,
    native_prompt_truncated: !!result.truncated,
    output_limit_reached: !!result.stopped_limit,
    tokens_cached: result.tokens_cached ?? null,
    timestamp: new Date().toISOString(),
    query: input,
    chat_mode: chatMode,
    profile_id: profileId ?? null,
    rag_retrieval_ms: retrievalMs,
    embedding_ms: embeddingMs,
    vector_search_ms: vectorSearchMs,
    context_construction_ms: contextConstructionMs,
    context_chars: contextChars,
    retrieved_chunk_count: retrievedChunkCount,
    candidate_chunk_count: candidateChunkCount,
    rag_strategy: ragStrategy,
    rag_candidate_k: ragCandidateK,
    rag_duplicate_chunks_removed: ragDuplicateChunksRemoved,
    rag_context_token_estimate: ragContextTokenEstimate,
    rag_requested_context_budget: ragRequestedContextBudget,
    rag_applied_context_budget: ragAppliedContextBudget,
    rag_compression_ratio: ragCompressionRatio,
    retrieved_top_score: retrievedTopScore,
    retrieved_score_margin: retrievedScoreMargin,
    rag_query_term_count: ragQueryTermCount,
    rag_matched_query_term_count: ragMatchedQueryTermCount,
    rag_query_term_coverage: ragQueryTermCoverage,
    rag_lexical_hit: ragLexicalHit,
    rag_exact_term_matches: ragExactTermMatches,
    rag_selected_chunk_ids: ragSelectedChunkIds,
    rag_failure_reason: ragFailureReason,
    prompt_chars: finalPrompt.length,
    prompt_tokenization_ms: promptTokenizationMs,
    prompt_tokens: promptTokens,
    prefill_ms: prefillMs,
    prefill_tokens_per_second: prefillTokPerSec,
    generated_tokens: genTokens,
    decode_ms: decodeMs,
    tokens_per_second: decodeTokPerSec,
    ttft_ms: ttftMs,
    llm_ttft_ms: llmTTFTMs,
    total_inference_ms: totalLLMTime,
    total_warm_latency_ms: totalWarmLatency,
    controller_mode: plan.mode,
    evidence_demand: plan.evidenceDemand,
    controller_workload_bucket: plan.workloadBucket,
    controller_selection_reason: plan.selectionReason,
    controller_state_read_ms: controllerStateReadMs,
    device_thermal_status: deviceState?.thermalStatusName || 'not-read',
    device_available_memory_mb: deviceState?.availableMemoryMb ?? null,
    device_low_memory: deviceState?.lowMemory ?? false,
    device_battery_level: deviceState?.batteryLevel ?? null,
    device_battery_temp_c: deviceState?.batteryTempC ?? null,
    device_cpu_frequencies_khz: deviceState?.cpuFrequenciesKhz || [],
    n_threads: plan.nThreads,
    n_predict: plan.nPredict,
    context_token_budget: plan.contextTokenBudget ?? null,
    primary_bottleneck: primaryBottleneck,
    quality_evaluation: 'not_measured',
    response_text: response,
  };

  const researchMetrics = evaluateResearchRun({
    query: input,
    response,
    ragContext: ragContext || '',
    policy: effectiveInferenceOptions.ragPolicy || (ragStrategy === 'mobcert-v1' ? 'mobcert-v1' : 'ebmrag-v2'),
    retrievalLatencyMs: retrievalMs,
    totalLatencyMs: totalWarmLatency,
    ttftMs,
    prefillMs,
    prefillTokPerSec,
    decodeMs,
    decodeTokPerSec,
    promptTokens,
    generatedTokens: genTokens,
    threads: plan.nThreads,
    primaryBottleneck,
    inferenceSkipped,
    directAnswer: !!directEvidenceAnswer,
    selectedChunkCount: retrievedChunkCount,
    candidateChunkCount,
    compressionRatio: ragCompressionRatio,
    queryTermCoverage: ragQueryTermCoverage,
    contractAudit: ragContractAudit ? {
      status: ragContractAudit.status,
      unsupportedProtectedTokens: ragContractAudit.unsupportedProtectedTokens,
      demandsCount: ragMetrics?.answerContract?.demands.length,
      demandsSatisfied: ragMetrics?.answerContract?.demands.filter(d => d.status === 'certified').length,
      demandsMissing: ragMetrics?.answerContract?.demands.filter(d => d.status === 'missing').length,
      demandsAmbiguous: ragMetrics?.answerContract?.demands.filter(d => d.status === 'ambiguous').length,
    } : undefined,
    deviceState: {
      availableMemoryMb: deviceState?.availableMemoryMb,
      batteryTempC: deviceState?.batteryTempC,
      thermalStatusName: deviceState?.thermalStatusName,
    },
  });
  report.research_metrics = researchMetrics;

  if (!inferenceSkipped && result.timings) recordInferenceObservation({
    plan,
    totalInferenceMs: totalLLMTime,
    prefillMs,
    decodeMs,
  });
  latestInferenceBenchmark = report;

  console.log('================================================================');
  console.log('⚡ [INFERENCE_BENCHMARK] HARDWARE & INFERENCE PROFILING SPLIT');
  console.log('================================================================');
  console.log(`Policy                    : ${researchMetrics.policy.toUpperCase()}`);
  console.log(`Total Warm Latency        : ${totalWarmLatency.toFixed(2)} ms`);
  console.log(`RAG Retrieval Latency     : ${retrievalMs.toFixed(2)} ms (${safePercent(retrievalMs, totalWarmLatency)}%)`);
  console.log(`Prompt Prefill Latency    : ${prefillMs.toFixed(2)} ms (${safePercent(prefillMs, totalWarmLatency)}%)`);
  console.log(`├── Prompt Tokens (n)     : ${promptTokens}`);
  console.log(`└── Prefill Speed         : ${prefillTokPerSec.toFixed(2)} tokens/sec`);
  console.log(`Token Decode Latency      : ${decodeMs.toFixed(2)} ms (${safePercent(decodeMs, totalWarmLatency)}%)`);
  console.log(`├── Generated Tokens (n)  : ${genTokens}`);
  console.log(`└── Generation Throughput : ${decodeTokPerSec.toFixed(2)} tokens/sec`);
  console.log(`TTFT (request → token)    : ${ttftMs === null ? 'unavailable' : `${ttftMs.toFixed(2)} ms`}`);
  console.log(`Native LLM TTFT           : ${llmTTFTMs === null ? 'unavailable' : `${llmTTFTMs.toFixed(2)} ms`}`);
  console.log(`Primary Bottleneck Target : ${primaryBottleneck}`);
  console.log(`Controller                : ${plan.mode}, ${plan.nThreads} threads`);
  console.log(`Contract Status           : ${researchMetrics.contractStatus}`);
  console.log(`Groundedness/Faithfulness : ${researchMetrics.faithfulness === null ? 'N/A' : `${(researchMetrics.faithfulness * 100).toFixed(1)}%`}`);
  console.log('================================================================');

  try {
    const reportPath = `${RNFS.ExternalDirectoryPath}/inference_benchmark_latest.json`;
    await RNFS.writeFile(reportPath, JSON.stringify(report, null, 2), 'utf8');
    console.log('[QA] Benchmark report persisted to:', reportPath);

    // Append to historical run log for physical phone matched comparison
    const historyPath = `${RNFS.ExternalDirectoryPath}/evaluation_runs_history.json`;
    let history: any[] = [];
    if (await RNFS.exists(historyPath)) {
      try {
        const existing = await RNFS.readFile(historyPath, 'utf8');
        history = JSON.parse(existing);
        if (!Array.isArray(history)) history = [];
      } catch (_) {
        history = [];
      }
    }
    history.push({
      run_id: `run_${Date.now()}`,
      timestamp: report.timestamp,
      query: report.query,
      policy: researchMetrics.policy,
      response: report.response_text,
      research_metrics: researchMetrics,
      telemetry: {
        total_latency_ms: report.total_warm_latency_ms,
        ttft_ms: report.ttft_ms,
        prefill_ms: report.prefill_ms,
        decode_ms: report.decode_ms,
        tokens_per_second: report.tokens_per_second,
      },
    });
    await RNFS.writeFile(historyPath, JSON.stringify(history, null, 2), 'utf8');
  } catch (err) {
    console.warn('[QA] Failed to persist benchmark report or history to disk:', err);
  }

  return response;
};
