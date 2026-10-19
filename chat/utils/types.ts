export interface LlamaContext {
  completion: (params: {
    prompt?: string;
    messages?: { role: string; content: string }[];
    jinja?: boolean;
    n_threads?: number;
    n_predict: number;
    temperature: number;
    top_p?: number;
    stop?: string[];
  }, onToken?: (data: { token?: string }) => void) => Promise<{
    text: string;
    content?: string;
    tokens_evaluated?: number;
    tokens_predicted?: number;
    tokens_cached?: number;
    truncated?: boolean;
    stopped_limit?: number | boolean;
    timings?: {
      prompt_n: number;
      prompt_ms: number;
      prompt_per_token_ms?: number;
      prompt_per_second?: number;
      predicted_n: number;
      predicted_ms: number;
      predicted_per_token_ms?: number;
      predicted_per_second?: number;
    };
  }>;
  tokenize?: (text: string) => Promise<{ tokens: number[] }>;
  loadSession?: (filepath: string) => Promise<{ tokens_loaded: number; prompt: string }>;
  saveSession?: (filepath: string, options?: { tokenSize: number }) => Promise<number>;
  getFormattedChat?: (messages: { role: string; content: string }[], template?: string | null,
    params?: { jinja?: boolean }) => Promise<string | { prompt: string }>;
  release: () => Promise<void>;
  embedding: (text: string) => Promise<any>;
}

// Optional structured data for tasks (if they need it)
export interface CalendarEvent {
  title: string;
  date: string; // "YYYY-MM-DD"
  time: string; // "HH:mm"
  location?: string;
  notes?: string;
  reminder?: number;
}
// types.ts
export type RootTabParamList = {
  Models: undefined;
  Chat: undefined;
  Documents: undefined;
  Settings: undefined;
};
export interface AlarmEvent {
  Day?: string;
  Time?: string;
}
