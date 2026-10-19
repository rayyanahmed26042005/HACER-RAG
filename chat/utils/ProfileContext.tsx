// chat/utils/ProfileContext.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { getActiveProfileId, setActiveProfileId, clearActiveProfile } from './profileManager';
import { flushInMemoryIndex } from './rag/vectorStore';
import { releaseEmbeddingModel } from './rag/embedder';
import { initRAG } from './rag/ragPipeline';

interface ProfileContextValue {
  profileId: string | null;
  switchProfile: (id: string) => Promise<void>;
  logout: () => Promise<void>;
  isLoading: boolean;
}

const ProfileContext = createContext<ProfileContextValue | undefined>(undefined);

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [profileId, setProfileIdState] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const init = async () => {
      try {
        const active = await getActiveProfileId();
        setProfileIdState(active);
      } catch (err) {
        console.error('Failed to load active profile', err);
      } finally {
        setIsLoading(false);
      }
    };
    init();
  }, []);

  // Bug #9 fix: Single authoritative initialization point for RAG + embedding
  useEffect(() => {
    if (profileId) {
      initRAG(profileId).catch(e => console.warn('RAG init failed:', e));
    }
  }, [profileId]);

  const switchProfile = async (id: string) => {
    // 1. Flush in-memory RAG data for the old profile to prevent leaks
    if (profileId) {
      flushInMemoryIndex(profileId);
      await releaseEmbeddingModel(profileId);
    }

    // 2. Set the new active profile in AsyncStorage and state
    await setActiveProfileId(id);
    setProfileIdState(id);
    // Note: the useEffect above will auto-init RAG + embedding for the new profile
  };

  const logout = async () => {
    // 1. Flush data for current profile
    if (profileId) {
      flushInMemoryIndex(profileId);
      await releaseEmbeddingModel(profileId);
    }

    // 2. Clear active profile from AsyncStorage and state
    await clearActiveProfile();
    setProfileIdState(null);
  };

  return (
    <ProfileContext.Provider value={{ profileId, switchProfile, logout, isLoading }}>
      {children}
    </ProfileContext.Provider>
  );
}

export function useProfile() {
  const context = useContext(ProfileContext);
  if (context === undefined) {
    throw new Error('useProfile must be used within a ProfileProvider');
  }
  return context;
}
