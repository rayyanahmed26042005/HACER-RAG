import { Platform, Alert, PermissionsAndroid, NativeModules } from 'react-native';
import { LlamaContext } from '../utils/types';

const { DirectCall } = NativeModules; // ✅ Correct native module name
console.log('NativeModules:', NativeModules); // Confirm DirectCall is visible

export const handleCall = async (context: LlamaContext, userInput: string): Promise<string> => {
  try {
    console.log('Raw User Input (Call):', userInput);

    const systemPrompt = `Generate a simple json format for given number given text. Parse the number from the given sentence and give a pure json format like {"Number":""}`;

    const result = await context.completion({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userInput }
      ],
      n_predict: 300,
      temperature: 0.5,
    });

    const raw = result.text.trim();
    console.log('Raw LLM Response (Call):', raw);

    const jsonMatch = raw.match(/\{[\s\S]*?\}/);
    if (!jsonMatch) {
      throw new Error('No valid JSON found');
    }

    const parsed = JSON.parse(jsonMatch[0]);
    const number = parsed.Number;

    if (!number) throw new Error('No phone number provided');

    if (Platform.OS === 'android') {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.CALL_PHONE,
        {
          title: 'Phone Call Permission',
          message: 'This app needs access to your phone to make calls',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        }
      );

      if (granted === PermissionsAndroid.RESULTS.GRANTED) {
        if (!DirectCall || typeof DirectCall.call !== 'function') {
          throw new Error('Native module "DirectCall" not properly linked');
        }

        DirectCall.call(String(number)); 
        return `📞 Calling ${number}...`;
      } else {
        return `⚠ Permission denied to make direct call.`;
      }
    } else {
      return '⚠ Direct call is only supported on Android.';
    }

  } catch (error) {
    console.warn('Call failed:', error);
    return `⚠ Failed to make call. ${error}`;
  }
};