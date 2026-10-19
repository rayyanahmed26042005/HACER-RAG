import { PermissionsAndroid, Platform, NativeModules } from 'react-native';
import { LlamaContext } from '../utils/types';

const { ContactModule } = NativeModules;

export const handleContactCall = async (
  context: LlamaContext,
  userInput: string
): Promise<string> => {
  try {
    console.log('Raw User Input (Contact Call):', userInput);

    const systemPrompt = `Generate a simple json format for given word after the word "call" from the given text. Parse that word from the given sentence and give a pure json format like {"Name":""}`;
    const result = await context.completion({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userInput }
      ],
      n_predict: 300,
      temperature: 0.5
    });

    const raw = result.text.trim();
    console.log('Raw LLM Response (Contact Call):', raw);

    const jsonMatch = raw.match(/\{[\s\S]*?\}/);
    if (!jsonMatch) throw new Error('No valid JSON found');

    const parsed = JSON.parse(jsonMatch[0]);
    const name = parsed.Name?.trim();
    if (!name) throw new Error('No contact name found');

    if (Platform.OS === 'android') {
      const granted = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.READ_CONTACTS,
        PermissionsAndroid.PERMISSIONS.CALL_PHONE
      ]);

      if (
        granted['android.permission.READ_CONTACTS'] !== PermissionsAndroid.RESULTS.GRANTED ||
        granted['android.permission.CALL_PHONE'] !== PermissionsAndroid.RESULTS.GRANTED
      ) {
        return '⚠ Permissions denied for contacts or calling.';
      }

      const contacts = await ContactModule.getContacts();
      console.log('📇 Native Contacts:', contacts);

      const match = contacts.find((c: any) => {
        if (!c.name) return false;
      
        const normalizedName = c.name.trim().toLowerCase();
        const targetName = name.trim().toLowerCase();
      
        // Check if any full word in contact name matches the target name exactly
        const words = normalizedName.split(/\s+/); // split by whitespace
        return words.includes(targetName);
      });
      
      

      if (!match || !match.number) {
        return `❌ No contact found with name ${name}`;
      }

      const number = match.number;
      ContactModule.callNumber(number);
      return `📞 Calling ${match.name} (${number})...`;
    } else {
      return '⚠ Not supported on this platform.';
    }

  } catch (err) {
    console.warn('Contact Call failed:', err);
    return `⚠ Failed to call contact. ${err}`;
  }
};