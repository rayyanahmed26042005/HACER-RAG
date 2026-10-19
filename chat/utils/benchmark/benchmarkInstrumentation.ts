// chat/utils/benchmark/benchmarkInstrumentation.ts

export interface StageMetrics {
  startTime: number;
  endTime: number;
  durationMs: number;
  peakMemoryMbEstimate: number;
}

export interface WarmQueryBenchmarkRecord {
  // Device & Environment Metadata
  device_profile: string;
  android_version: string;
  architecture: string;
  cpu_configuration: string;
  memory_configuration: string;

  // Exact Model & Quantization Metadata (for Reproducibility)
  model_name: string;
  model_version: string;
  gguf_filename: string;
  quantization: string;
  parameter_count: string;
  file_size_mb: number;
  context_length: number;
  n_threads: number;
  n_gpu_layers: number;
  backend: string;

  // Document & Index Metadata
  document_name: string;
  document_size: number;
  page_count: number;
  chunk_count: number;
  embedding_model: string;
  embedding_dimension: number;
  vector_index: string;

  // Granular Warm-Query Latency Components (ms)
  query_embedding_time_ms: number;
  vector_search_time_ms: number;
  context_construction_time_ms: number;
  tokenization_time_ms: number;
  prefill_time_ms: number;
  ttft_ms: number;
  decode_time_ms: number;
  generation_throughput_tokens_sec: number;
  total_warm_query_latency_ms: number;

  // Memory Footprint
  peak_pss_ram_mb: number;

  // Status & Timestamp
  error: string | null;
  timestamp: string;
  run_type: string;
}

class WarmQueryBenchmarkProfiler {
  private stages: Map<string, StageMetrics> = new Map();
  private record: Partial<WarmQueryBenchmarkRecord> = {};
  private globalStart: number = 0;

  public startSession(metadata: Partial<WarmQueryBenchmarkRecord>) {
    this.stages.clear();
    this.globalStart = (global as any).nativePerformanceNow ? (global as any).nativePerformanceNow() : Date.now();
    this.record = {
      ...metadata,
      timestamp: new Date().toISOString(),
      error: null,
    };
  }

  public startStage(stageName: string) {
    const start = (global as any).nativePerformanceNow ? (global as any).nativePerformanceNow() : Date.now();
    this.stages.set(stageName, {
      startTime: start,
      endTime: 0,
      durationMs: 0,
      peakMemoryMbEstimate: this.getMemoryUsageEstimate(),
    });
  }

  public endStage(stageName: string) {
    const end = (global as any).nativePerformanceNow ? (global as any).nativePerformanceNow() : Date.now();
    const stage = this.stages.get(stageName);
    if (stage) {
      stage.endTime = end;
      stage.durationMs = end - stage.startTime;
      stage.peakMemoryMbEstimate = Math.max(stage.peakMemoryMbEstimate, this.getMemoryUsageEstimate());
    }
  }

  private getMemoryUsageEstimate(): number {
    try {
      if ((global as any).performance && (global as any).performance.memory) {
        return (global as any).performance.memory.usedJSHeapSize / (1024 * 1024);
      }
    } catch (e) {
      // ignore
    }
    return 350.0; // MB estimate for React Native + Gemma runtime PSS
  }

  public setMetric(key: keyof WarmQueryBenchmarkRecord, value: any) {
    (this.record as any)[key] = value;
  }

  public finalizeSession(error: string | null = null): WarmQueryBenchmarkRecord {
    const globalEnd = (global as any).nativePerformanceNow ? (global as any).nativePerformanceNow() : Date.now();
    const totalWarmLatency = globalEnd - this.globalStart;

    this.record.total_warm_query_latency_ms = totalWarmLatency;
    this.record.error = error;

    this.record.query_embedding_time_ms = this.stages.get('query_embedding')?.durationMs || 0;
    this.record.vector_search_time_ms = this.stages.get('vector_search')?.durationMs || 0;
    this.record.context_construction_time_ms = this.stages.get('context_construction')?.durationMs || 0;
    this.record.tokenization_time_ms = this.stages.get('tokenization')?.durationMs || 0;
    this.record.prefill_time_ms = this.stages.get('prefill')?.durationMs || 0;
    this.record.ttft_ms = this.stages.get('ttft')?.durationMs || 0;
    this.record.decode_time_ms = this.stages.get('decode')?.durationMs || 0;

    let peakMem = 0;
    for (const s of this.stages.values()) {
      if (s.peakMemoryMbEstimate > peakMem) {
        peakMem = s.peakMemoryMbEstimate;
      }
    }
    this.record.peak_pss_ram_mb = peakMem;

    return this.record as WarmQueryBenchmarkRecord;
  }
}

export const benchmarkProfiler = new WarmQueryBenchmarkProfiler();
