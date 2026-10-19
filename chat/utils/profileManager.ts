// chat/utils/profileManager.ts
// Fully offline, local-only profile management.
// Profiles stored in AsyncStorage (lightweight metadata only).
// Heavy data (RAG indexes) stored in RNFS per-profile directories.

import AsyncStorage from '@react-native-async-storage/async-storage';
import RNFS from 'react-native-fs';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Profile {
  id: string;
  name: string;
  pinHash?: string;       // optional SHA-256 hash of 4-digit PIN
  pinEnabled: boolean;    // whether this profile has lock enabled
  avatarColor: string;    // random pastel color for avatar circle
  createdAt: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const PROFILES_LIST_KEY = 'profiles_list';
const ACTIVE_PROFILE_KEY = 'active_profile_id';

// All AsyncStorage keys a profile owns — used for thorough cleanup
const PROFILE_DATA_SUFFIXES = [
  '_docs_meta',
  '_selected_model',
  '_task_history',
  '_chat_messages',
  '_embedding_model',
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Generate a scoped AsyncStorage key for a given profile. */
export function profileKey(profileId: string, suffix: string): string {
  return `${profileId}${suffix}`;
}

/** Get the RNFS directory path for a profile's data. */
export function profileDir(profileId: string): string {
  return `${RNFS.ExternalDirectoryPath}/profiles/${profileId}`;
}

/** Simple hash for PIN (not cryptographic, but sufficient for local offline use). */
function hashPin(pin: string): string {
  // Simple djb2 hash — adequate for local PIN comparison
  let hash = 5381;
  for (let i = 0; i < pin.length; i++) {
    hash = (hash * 33) ^ pin.charCodeAt(i);
  }
  return (hash >>> 0).toString(16);
}

/** Generate a random pastel color for avatar. */
function randomAvatarColor(): string {
  const hue = Math.floor(Math.random() * 360);
  return `hsl(${hue}, 65%, 65%)`;
}

/** Generate a unique profile ID. */
function generateId(): string {
  const rand = Math.random().toString(36).substring(2, 10);
  return `p_${Date.now()}_${rand}`;
}

// ─── CRUD ─────────────────────────────────────────────────────────────────────

/** Get all profiles. */
export async function getProfiles(): Promise<Profile[]> {
  const raw = await AsyncStorage.getItem(PROFILES_LIST_KEY);
  if (!raw) return [];
  return JSON.parse(raw);
}

/** Create a new profile. Returns the created profile. */
export async function createProfile(
  name: string,
  pin?: string
): Promise<Profile> {
  const profiles = await getProfiles();

  const profile: Profile = {
    id: generateId(),
    name: name.trim(),
    pinHash: pin ? hashPin(pin) : undefined,
    pinEnabled: !!pin,
    avatarColor: randomAvatarColor(),
    createdAt: new Date().toISOString(),
  };

  profiles.push(profile);
  await AsyncStorage.setItem(PROFILES_LIST_KEY, JSON.stringify(profiles));

  // Create the profile's RNFS directory
  const dir = profileDir(profile.id);
  const exists = await RNFS.exists(dir);
  if (!exists) {
    await RNFS.mkdir(dir);
  }

  return profile;
}

/**
 * Delete a profile and ALL its data.
 * Cleans both RNFS (index files) and AsyncStorage (metadata keys).
 */
export async function deleteProfile(profileId: string): Promise<void> {
  // 1. Delete RNFS profile directory (contains rag_index.json, etc.)
  const dir = profileDir(profileId);
  const dirExists = await RNFS.exists(dir);
  if (dirExists) {
    await RNFS.unlink(dir);
  }

  // 2. Remove all AsyncStorage keys owned by this profile
  const keysToRemove = PROFILE_DATA_SUFFIXES.map(s => profileKey(profileId, s));
  await AsyncStorage.multiRemove(keysToRemove);

  // 3. Remove from profiles list
  const profiles = await getProfiles();
  const updated = profiles.filter(p => p.id !== profileId);
  await AsyncStorage.setItem(PROFILES_LIST_KEY, JSON.stringify(updated));

  // 4. If this was the active profile, clear it
  const activeId = await AsyncStorage.getItem(ACTIVE_PROFILE_KEY);
  if (activeId === profileId) {
    await AsyncStorage.removeItem(ACTIVE_PROFILE_KEY);
  }
}

/** Verify a PIN against a profile. Returns true if correct or no PIN set. */
export async function verifyPin(
  profileId: string,
  pin: string
): Promise<boolean> {
  const profiles = await getProfiles();
  const profile = profiles.find(p => p.id === profileId);
  if (!profile) return false;
  if (!profile.pinEnabled || !profile.pinHash) return true;
  return hashPin(pin) === profile.pinHash;
}

/** Enable or update PIN for a profile. */
export async function setProfilePin(
  profileId: string,
  pin: string | null
): Promise<void> {
  const profiles = await getProfiles();
  const idx = profiles.findIndex(p => p.id === profileId);
  if (idx === -1) return;

  if (pin) {
    profiles[idx].pinEnabled = true;
    profiles[idx].pinHash = hashPin(pin);
  } else {
    profiles[idx].pinEnabled = false;
    profiles[idx].pinHash = undefined;
  }

  await AsyncStorage.setItem(PROFILES_LIST_KEY, JSON.stringify(profiles));
}

// ─── Active Profile ───────────────────────────────────────────────────────────

/** Get the currently active profile ID (or null if none). */
export async function getActiveProfileId(): Promise<string | null> {
  return AsyncStorage.getItem(ACTIVE_PROFILE_KEY);
}

/** Set the active profile ID. */
export async function setActiveProfileId(profileId: string): Promise<void> {
  await AsyncStorage.setItem(ACTIVE_PROFILE_KEY, profileId);
}

/** Clear the active profile (logout). */
export async function clearActiveProfile(): Promise<void> {
  await AsyncStorage.removeItem(ACTIVE_PROFILE_KEY);
}
