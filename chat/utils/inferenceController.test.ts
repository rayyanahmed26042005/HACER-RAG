import {
  recordInferenceObservation,
  resetInferenceController,
  selectInferencePlan,
  getGeneralOutputLimit,
} from './inferenceController';

describe('inference controller', () => {
  beforeEach(() => resetInferenceController());

  it('preserves the four-thread baseline by default', () => {
    const plan = selectInferencePlan(200, 'document');

    expect(plan.mode).toBe('baseline');
    expect(plan.nThreads).toBe(4);
    expect(plan.nPredict).toBe(512);
  });

  it('explores adaptive candidates and then chooses the fastest observed candidate', () => {
    const candidates = [2, 4];
    const options = { mode: 'adaptive' as const, candidateThreads: candidates, allowExploration: true };
    const first = selectInferencePlan(200, 'document', options);
    expect(first.nThreads).toBe(4);
    recordInferenceObservation({ plan: first, totalInferenceMs: 1000, prefillMs: 700, decodeMs: 300 });

    const second = selectInferencePlan(200, 'document', options);
    expect(second.nThreads).toBe(2);
    recordInferenceObservation({ plan: second, totalInferenceMs: 700, prefillMs: 500, decodeMs: 200 });

    const third = selectInferencePlan(200, 'document', options);
    expect(third.nThreads).toBe(2);
    expect(third.selectionReason).toContain('lowest observed');
  });

  it('bounds ordinary general answers to avoid CPU decode runaway', () => {
    expect(getGeneralOutputLimit('What is AI?')).toBe(64);
    expect(getGeneralOutputLimit('Explain what on-device inference is.')).toBe(128);
    expect(getGeneralOutputLimit('Write a detailed story about AI.')).toBe(256);
    expect(selectInferencePlan(383, 'general', { mode: 'adaptive' }, 'What is AI?').nPredict).toBe(64);
  });
});
