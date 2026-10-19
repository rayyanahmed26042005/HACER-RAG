import { Linking, Platform, NativeModules } from 'react-native';
import { LlamaContext } from '../utils/types';

const { AppLauncherModule } = NativeModules;

const appNameToPackageMap: Record<string, string> = {
  youtube: 'com.google.android.youtube',
  photos: 'com.google.android.apps.photos',
  calendar: 'com.google.android.calendar',
  gmail: 'com.google.android.gm',
  whatsapp: 'com.whatsapp',
  chrome: 'com.android.chrome',
  googlechrome: 'com.android.chrome',
  facebook: 'com.facebook.katana',
  instagram: 'com.instagram.android',
  camera: 'com.android.camera',
  messages: 'com.google.android.apps.messaging',
};

const fallbackPackages: Record<string, string[]> = {
  youtube: ['com.google.android.youtube', 'com.youtube.android'],
  photos: ['com.google.android.apps.photos', 'com.android.gallery3d'],
  chrome: ['com.android.chrome', 'com.google.chrome'],
};

export const handleOpenApp = async (
  context: LlamaContext,
  userInput: string
): Promise<string> => {
  try {
    console.log('Raw User Input (Open App):', userInput);

    const systemPrompt = `Extract the application name after the word "open" from the given sentence. Respond in pure JSON format like {"App":""}. Use lowercase app names without space, e.g. "youtube", "calendar", "googlephotos".`;

    const result = await context.completion({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userInput },
      ],
      n_predict: 300,
      temperature: 0.5,
    });

    const raw = result.text.trim();
    console.log('Raw LLM Response (Open App):', raw);

    const jsonMatch = raw.match(/\{[\s\S]*?\}/);
    if (!jsonMatch) throw new Error('No valid JSON found');

    const parsed = JSON.parse(jsonMatch[0]);
    console.log('Parsed JSON:', parsed);
    if (!parsed || typeof parsed.App !== 'string') {
      throw new Error(`Invalid or missing "App" key in parsed JSON: ${JSON.stringify(parsed)}`);
    }
    const userAppName = parsed.App?.toLowerCase().trim();

    if (!userAppName) throw new Error('No app name found');

    let appPackages = [appNameToPackageMap[userAppName]];
    if (fallbackPackages[userAppName]) {
      appPackages = [...new Set([...appPackages, ...fallbackPackages[userAppName]])];
    }

    if (!appPackages[0]) {
      return `Try again....... I couldn't find a known app for "${userAppName}". Please install "${userAppName}"`;
    }

    if (Platform.OS === 'android') {
      for (const appPackage of appPackages) {
        console.log(`Checking package: ${appPackage}`);
        const isAvailable = await AppLauncherModule.isAppInstalled(appPackage);
        console.log(`isAppInstalled(${appPackage}): ${isAvailable}`);
        if (isAvailable) {
          await AppLauncherModule.launchApp(appPackage);
          return `Opening ${userAppName}...`;
        }
      }
      return `App "${userAppName}" is not installed or cannot be launched.Please try after installing "${userAppName}"`;
    } else if (Platform.OS === 'ios') {
      const appUrlScheme = userAppName === 'chrome' ? 'googlechrome://' : `${userAppName}://`;
      const supported = await Linking.canOpenURL(appUrlScheme);
      if (!supported) {
        return `App "${userAppName}" is not installed. Try another App`;
      }
      await Linking.openURL(appUrlScheme);
      return `Opening ${userAppName}...`;
    }

    return 'Platform not supported for launching apps.';
  } catch (err) {
    // console.warn('Try again.....', err);
    return `Try again..... Failed to open App : ${err}`;
  }
};


