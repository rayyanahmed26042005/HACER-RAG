import { LlamaContext } from './types';

export const getStructuredOutput = async (
  context: LlamaContext,
  input: string,
  prompt: string
): Promise<string> => {
  const result = await context.completion({
    messages: [
      { role: 'system', content: prompt },
      { role: 'user', content: input },
    ],
    n_predict: 500,
    temperature: 0.7,
  });
  return result.text;
};