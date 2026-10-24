import RNFS from 'react-native-fs';
import type { LlamaContext } from './types';

export type HacerPhase1Request = {
  phase: 'seed' | 'baseline' | 'reuse';
  prefix: string;
  suffix: string;
  sessionPath: string;
  outputPath: string;
  nPredict?: number;
  promptSha256?: string;
  activeNCtx?: number;
};

type CompletionResult = {
  text?: string;
  tokens_evaluated?: number;
  tokens_predicted?: number;
  tokens_cached?: number;
  truncated?: boolean;
  stopped_eos?: boolean;
  stopped_word?: boolean;
  stopped_limit?: boolean | number;
  stopping_word?: string;
  timings?: Record<string, number>;
};

/**
 * Minimal native session probe for HACER Phase 1.
 *
 * This is deliberately separate from the normal QA/RAG path and does not
 * import or change MobCert. It measures exact-prefix session reuse only;
 * independent chunk KV import remains a later native-runtime phase.
 */
export async function runHacerPhase1Probe(
  context: LlamaContext,
  request: HacerPhase1Request,
): Promise<Record<string, unknown>> {
  if (!context.completion || !context.tokenize) throw new Error('llama context lacks completion/tokenize');
  if (request.phase === 'reuse' && !context.loadSession) throw new Error('llama context lacks loadSession');
  if (request.phase === 'seed' && !context.saveSession) throw new Error('llama context lacks saveSession');
  if (!request.prefix || request.suffix === undefined || request.suffix === null || !request.sessionPath || !request.outputPath) {
    throw new Error('prefix, suffix, sessionPath, and outputPath are required');
  }

  const fullPrompt = `${request.prefix}${request.suffix}`;
  const prefixTokens = await context.tokenize(request.prefix);
  const fullTokens = await context.tokenize(fullPrompt);

  // Strict Context Budget Safety Enforced at Native Token Level
  const activeCtx = request.activeNCtx ?? 512;
  const nPredict = request.nPredict ?? 1;
  const maxSafePromptTokens = activeCtx - 4 - nPredict;
  if (request.phase !== 'seed' && fullTokens.tokens.length > maxSafePromptTokens) {
    throw new Error(
      `Context budget safety violation: prompt (${fullTokens.tokens.length} tok) + n_predict (${nPredict}) exceeds n_ctx (${activeCtx}) - 4 limit (${activeCtx - 4}). Headroom deficit: ${fullTokens.tokens.length - maxSafePromptTokens} tokens.`
    );
  }

  const started = Date.now();
  let loaded: { tokens_loaded: number; prompt: string } | null = null;
  let savedTokenCount: number | null = null;
  let completion: CompletionResult;

  if (request.phase === 'seed') {
    completion = await context.completion({
      prompt: request.prefix,
      n_predict: nPredict,
      temperature: 0,
      top_p: 1,
    });
    savedTokenCount = await context.saveSession!(request.sessionPath, { tokenSize: prefixTokens.tokens.length });
  } else {
    if (request.phase === 'reuse') loaded = await context.loadSession!(request.sessionPath);
    completion = await context.completion({
      prompt: fullPrompt,
      n_predict: nPredict,
      temperature: 0,
      top_p: 1,
    });
  }

  // Reject any silent truncation by native engine
  if (completion.truncated) {
    throw new Error(`Native engine reported truncated=true. Workload silently truncated by engine.`);
  }

  const result = {
    schema_version: 1,
    experiment: 'hacer-phase1-exact-prefix-session',
    phase: request.phase,
    active_n_ctx: request.activeNCtx ?? null,
    prefix_token_count: prefixTokens.tokens.length,
    full_prompt_token_count: fullTokens.tokens.length,
    session_path: request.sessionPath,
    session_exists: await RNFS.exists(request.sessionPath),
    tokens_loaded: loaded?.tokens_loaded ?? 0,
    tokens_saved: savedTokenCount,
    tokens_cached: completion.tokens_cached ?? null,
    tokens_evaluated: completion.tokens_evaluated ?? null,
    tokens_predicted: completion.tokens_predicted ?? null,
    truncated: completion.truncated ?? false,
    stopped_eos: completion.stopped_eos ?? false,
    stopped_word: completion.stopped_word ?? false,
    stopped_limit: completion.stopped_limit ?? false,
    stopping_word: completion.stopping_word ?? null,
    text: completion.text ?? null,
    timings: completion.timings ?? null,
    elapsed_wall_ms: Date.now() - started,
    prompt_sha256: request.promptSha256 ?? null,
    exact_prefix_claim: request.phase === 'reuse' &&
      loaded?.tokens_loaded === prefixTokens.tokens.length,
    context_headroom: activeCtx - (fullTokens.tokens.length + nPredict),
    note: 'Native session probe only; no independent chunk KV import and no paper result.',
  };
  await RNFS.writeFile(request.outputPath, JSON.stringify(result, null, 2), 'utf8');
  return result;
}
