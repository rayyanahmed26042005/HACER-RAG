import { LlamaContext, AlarmEvent } from '../utils/types';
import RTNMyAlarm from '../RTNMyAlarm/js/NativeMyAlarm';
import moment from 'moment';

export const handleAlarm = async (
    context: LlamaContext,
    userInput: string
): Promise<string> => {
    let completionResponse: any = null;

    try {
        console.log('Raw User Input:', userInput);

        // Clear and specific prompt for a single alarm
        const systemPrompt = `
        You are a strict JSON generator. Extract a day of the week and a time (in 24-hour HH:mm format) from the user's sentence.
        
        Reply ONLY with raw JSON. Do NOT include markdown, explanations, or extra text.
        
        If either value is missing or invalid, use null.
        
        Examples:
        Input: "Set an alarm on Monday at 8 AM" → {"Day":"Monday","Time":"08:00"}`;

        userInput = userInput.charAt(0).toUpperCase() + userInput.slice(1).toLowerCase();

        completionResponse = await context.completion({
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: userInput },
            ],
            n_predict: 500,
            temperature: 0.7,
        }).catch((err) => {
            throw new Error(`LLM completion failed: ${err.message}`);
        });

        console.log('Raw Completion Response:', completionResponse);

        let llmResponse: string;
        if (completionResponse && typeof completionResponse.text === 'string') {
            llmResponse = completionResponse.text;
        } else if (typeof completionResponse === 'string') {
            llmResponse = completionResponse;
        } else {
            throw new Error('Invalid LLM response: no text property or string response');
        }

        console.log('Raw LLM Response:', llmResponse);

        const jsonMatch = llmResponse.match(/\{[\s\S]*?\}/);
        if (!jsonMatch) {
            throw new Error(`No valid JSON found. Raw output: "${llmResponse}"`);
        }

        const jsonText = cleanJsonString(jsonMatch[0]);
        console.log('Cleaned JSON:', jsonText);

        let alarmData: Partial<AlarmEvent>;
        try {
            alarmData = JSON.parse(jsonText);
        } catch (parseError: any) {
            throw new Error(`JSON Parse error: ${parseError.message}. Raw JSON: "${jsonText}"`);
        }
        console.log('Parsed Alarm Data:', alarmData);

        const { Day: rawDay, Time: rawTime } = alarmData;

        if (!rawDay || !rawTime) {
            throw new Error(`Invalid alarm data: Day="${rawDay}", Time="${rawTime}"`);
        }

        const time = normalizeTime(rawTime);
        const alarmMoment = moment(time, ['HH:mm', 'hh:mm A'], true);
        if (!alarmMoment.isValid()) {
            throw new Error(`Invalid time format "${time}" after normalization`);
        }

        const hour = alarmMoment.hour();
        const minute = alarmMoment.minute();
        const days = getDaysFromDayName(rawDay);
        if (days.length === 0) {
            throw new Error(`No valid day mapped for "${rawDay}"`);
        }

        console.log(`Setting alarm: hour=${hour}, minute=${minute}, days=${days}`);

        if (!RTNMyAlarm) {
            throw new Error('RTNMyAlarm native module not found');
        }

        try {
            const result = await RTNMyAlarm.setAlarm(hour, minute, days);
            console.log('Alarm intent triggered successfully:', result);
        } catch (nativeError: any) {
            if (nativeError.code === 'NO_COMPATIBLE_APP') {
                throw new Error(nativeError.message);
            }
            throw new Error(`Failed to set alarm via native module: ${nativeError.message}`);
        }

        return `⏰ Alarm set at ${time} on ${rawDay} (Day: ${days.join(', ')})`;
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.error('Alarm setup error:', errorMessage);

        const now = moment();
        const fallback: AlarmEvent = {
            Day: now.format('dddd'),
            Time: now.format('HH:mm'),
        };

        const alarmMoment = moment(fallback.Time, ['HH:mm'], true);
        const hour = alarmMoment.hour();
        const minute = alarmMoment.minute();
        const days = getDaysFromDayName(fallback.Day);

        if (RTNMyAlarm) {
            try {
                const result = await RTNMyAlarm.setAlarm(hour, minute, days);
                console.log('Fallback alarm set successfully:', result);
            } catch (fallbackError: any) {
                console.error('Fallback setup error:', fallbackError);
            }
        }

        if (errorMessage.includes('No compatible clock app found')) {
            return `⚠️ ${errorMessage}. Please open Google Clock settings and ensure it’s enabled, or install another app that supports AlarmClock intents.`;
        }
        return `⚠️ Failed to parse response. Set fallback alarm at ${fallback.Time} on ${fallback.Day}\nError: ${errorMessage}\nRaw LLM Output: "${completionResponse?.text || completionResponse || 'unknown'}"`;
    }
};

const getDaysFromDayName = (dayName?: string): number[] => {
    const daysMap: { [key: string]: number } = {
        'sunday': 1,
        'monday': 2,
        'tuesday': 3,
        'wednesday': 4,
        'thursday': 5,
        'friday': 6,
        'saturday': 7,
        'satuday': 7,
        'satu': 7,
        'fri': 6,
    };
    const normalizedDay = dayName?.toLowerCase().trim();
    return normalizedDay && daysMap[normalizedDay] ? [daysMap[normalizedDay]] : [moment().isoWeekday()];
};

const cleanJsonString = (jsonStr: string): string => {
    return jsonStr
        .replace(/\u00A0/g, ' ')
        .replace(/[\u200B-\u200D\uFEFF]/g, '')
        .replace(/<\/?s>/g, '')
        .replace(/,\s*}/g, '}')
        .replace(/,\s*]/g, ']')
        .trim();
};

const normalizeTime = (rawTime?: string): string => {
    if (!rawTime) return '09:00';
    let time = rawTime.trim();
    if (/^\d{1,2}:\d{2}\s?(AM|PM)?$/.test(time)) {
        const momentTime = moment(time, ['h:mm A', 'HH:mm'], true);
        if (momentTime.isValid()) {
            return momentTime.format('HH:mm');
        }
    }
    // Default to 24-hour format if no AM/PM is specified
    return time.replace(/^(\d):/, '0$1:'); // Pad single-digit hours
};