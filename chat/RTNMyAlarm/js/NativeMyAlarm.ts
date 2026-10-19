import { TurboModule, TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  setAlarm(hour: number, minute: number, days: number[]): void;
}

export default TurboModuleRegistry.getEnforcing<Spec>('RTNMyAlarm');
