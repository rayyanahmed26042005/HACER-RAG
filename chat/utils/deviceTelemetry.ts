import { NativeModules, Platform } from 'react-native';

export interface DeviceState {
  availableProcessors: number;
  availableMemoryMb: number | null;
  totalMemoryMb: number | null;
  lowMemory: boolean;
  thermalStatus: number | null;
  thermalStatusName: string;
  batteryLevel: number | null;
  batteryTempC: number | null;
  cpuFrequenciesKhz: number[];
}

const telemetryModule = NativeModules.DeviceTelemetryModule as {
  getSnapshot?: () => Promise<Partial<DeviceState>>;
} | undefined;

let cachedState: DeviceState | null = null;
let cachedAt = 0;
const CACHE_MS = 5000;

const fallbackState = (): DeviceState => ({
  availableProcessors: 4,
  availableMemoryMb: null,
  totalMemoryMb: null,
  lowMemory: false,
  thermalStatus: null,
  thermalStatusName: Platform.OS === 'android' ? 'unavailable' : 'unsupported',
  batteryLevel: null,
  batteryTempC: null,
  cpuFrequenciesKhz: [],
});

export async function getDeviceState(): Promise<DeviceState> {
  const now = Date.now();
  if (cachedState && now - cachedAt < CACHE_MS) return cachedState;

  if (!telemetryModule?.getSnapshot) {
    cachedState = fallbackState();
    cachedAt = now;
    return cachedState;
  }

  try {
    const raw = await telemetryModule.getSnapshot();
    cachedState = {
      ...fallbackState(),
      ...raw,
      availableProcessors: Math.max(1, Math.round(raw.availableProcessors || 4)),
      cpuFrequenciesKhz: Array.isArray(raw.cpuFrequenciesKhz) ? raw.cpuFrequenciesKhz : [],
    };
  } catch (error) {
    console.warn('[DEVICE_TELEMETRY] Falling back to conservative defaults:', error);
    cachedState = fallbackState();
  }
  cachedAt = now;
  return cachedState;
}

export function clearDeviceStateCache(): void {
  cachedState = null;
  cachedAt = 0;
}
