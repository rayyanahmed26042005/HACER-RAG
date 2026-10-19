import React, { useState, useEffect } from 'react';
import { Text, StyleSheet, TouchableOpacity, View, Alert, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useProfile } from '../utils/ProfileContext';
import Icon from 'react-native-vector-icons/MaterialIcons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import RNFS from 'react-native-fs';
import {
  InferenceMode,
  RagPolicy,
  DEFAULT_CHAT_INFERENCE_MODE,
  resetInferenceController,
} from '../utils/inferenceController';
import { profileKey } from '../utils/profileManager';

const Setting1 = ({ navigation }: { navigation: any }) => {
  const { logout, profileId } = useProfile();
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>(DEFAULT_CHAT_INFERENCE_MODE);
  const [ragPolicy, setRagPolicy] = useState<RagPolicy>('mobcert-v1');
  const [fixedThreads, setFixedThreads] = useState<number>(4);
  const [lastBenchmarkJson, setLastBenchmarkJson] = useState<string | null>(null);

  const INFERENCE_MODE_KEY = profileId ? profileKey(profileId, '_inference_mode') : '@genix_inference_mode';
  const RAG_POLICY_KEY = profileId ? profileKey(profileId, '_rag_policy') : '@genix_rag_policy';
  const FIXED_THREADS_KEY = profileId ? profileKey(profileId, '_fixed_threads') : '@genix_fixed_threads';

  useEffect(() => {
    const loadSettings = async () => {
      try {
        if (INFERENCE_MODE_KEY) {
          const mode = await AsyncStorage.getItem(INFERENCE_MODE_KEY);
          if (mode === 'baseline' || mode === 'fixed' || mode === 'adaptive') {
            setInferenceMode(mode);
          }
        }
        if (RAG_POLICY_KEY) {
          const policy = await AsyncStorage.getItem(RAG_POLICY_KEY);
          if (policy === 'mobcert-v1' || policy === 'ebmrag-v2') {
            setRagPolicy(policy as RagPolicy);
          }
        }
        if (FIXED_THREADS_KEY) {
          const threads = await AsyncStorage.getItem(FIXED_THREADS_KEY);
          if (threads && Number.isFinite(Number(threads))) {
            setFixedThreads(Number(threads));
          }
        }
        const benchPath = `${RNFS.ExternalDirectoryPath}/inference_benchmark_latest.json`;
        const exists = await RNFS.exists(benchPath);
        if (exists) {
          const content = await RNFS.readFile(benchPath, 'utf8');
          setLastBenchmarkJson(content);
        }
      } catch (e) {
        console.warn('Error loading settings:', e);
      }
    };
    loadSettings();
  }, [profileId]);

  const updateMode = async (mode: InferenceMode, threads = fixedThreads) => {
    setInferenceMode(mode);
    setFixedThreads(threads);
    try {
      if (INFERENCE_MODE_KEY) await AsyncStorage.setItem(INFERENCE_MODE_KEY, mode);
      if (FIXED_THREADS_KEY) await AsyncStorage.setItem(FIXED_THREADS_KEY, String(threads));
    } catch (e) {
      console.warn('Failed to save settings:', e);
    }
  };

  const updateRagPolicy = async (policy: RagPolicy) => {
    setRagPolicy(policy);
    try {
      if (RAG_POLICY_KEY) await AsyncStorage.setItem(RAG_POLICY_KEY, policy);
    } catch (e) {
      console.warn('Failed to save RAG policy:', e);
    }
  };

  const handleResetController = () => {
    Alert.alert(
      'Reset Inference Policy History',
      'This will clear all observed EMA latency observations for the adaptive controller, returning it to initial exploration state.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => {
            resetInferenceController();
            Alert.alert('Done', 'Adaptive controller observations have been reset.');
          },
        },
      ],
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Icon name="arrow-back" size={24} color="#E5E7EB" />
        </TouchableOpacity>
        <Text style={styles.title}>Settings</Text>
        <View style={{ width: 32 }} />
      </View>

      <ScrollView style={styles.content}>
        <Text style={styles.sectionHeader}>Inference Engine & Benchmarking</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Default Inference Policy</Text>
          <Text style={styles.cardDesc}>
            Used for conversational queries and document RAG:
          </Text>

          <TouchableOpacity
            style={[styles.optionRow, inferenceMode === 'adaptive' && styles.optionRowActive]}
            onPress={() => updateMode('adaptive')}
          >
            <Icon name="bolt" size={20} color={inferenceMode === 'adaptive' ? '#10B981' : '#9CA3AF'} />
            <View style={styles.optionTextContainer}>
              <Text style={[styles.optionTitle, inferenceMode === 'adaptive' && { color: '#10B981' }]}>
                Adaptive Controller (Optimized)
              </Text>
              <Text style={styles.optionSub}>
                Dynamic thread policy, whole-chunk context budgeting, and 82% overlap deduplication.
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.optionRow, inferenceMode === 'baseline' && styles.optionRowActive]}
            onPress={() => updateMode('baseline')}
          >
            <Icon name="timer" size={20} color={inferenceMode === 'baseline' ? '#A78BFA' : '#9CA3AF'} />
            <View style={styles.optionTextContainer}>
              <Text style={[styles.optionTitle, inferenceMode === 'baseline' && { color: '#A78BFA' }]}>
                Historical Baseline (Fixed 4T)
              </Text>
              <Text style={styles.optionSub}>
                Fixed 4 threads, 512 context limit, full Top-3 chunk injection without deduplication.
              </Text>
            </View>
          </TouchableOpacity>

          <View style={styles.fixedThreadsContainer}>
            <Text style={styles.fixedThreadsLabel}>Fixed Thread Count (Controlled Sweeps):</Text>
            <View style={styles.threadButtonsRow}>
              {[1, 2, 4, 6, 8].map((t) => (
                <TouchableOpacity
                  key={t}
                  style={[
                    styles.threadBtn,
                    inferenceMode === 'fixed' && fixedThreads === t && styles.threadBtnActive,
                  ]}
                  onPress={() => updateMode('fixed', t)}
                >
                  <Text
                    style={[
                      styles.threadBtnText,
                      inferenceMode === 'fixed' && fixedThreads === t && styles.threadBtnTextActive,
                    ]}
                  >
                    {t}T
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </View>

        {/* RAG Research Policy Card */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>RAG Policy (Research Comparison)</Text>
          <Text style={styles.cardDesc}>
            Switch between the deterministic MobCert contract and the EBM-RAG baseline:
          </Text>

          <TouchableOpacity
            style={[styles.optionRow, ragPolicy === 'mobcert-v1' && styles.optionRowActive]}
            onPress={() => updateRagPolicy('mobcert-v1')}
          >
            <Icon name="verified-user" size={20} color={ragPolicy === 'mobcert-v1' ? '#10B981' : '#9CA3AF'} />
            <View style={styles.optionTextContainer}>
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Text style={[styles.optionTitle, ragPolicy === 'mobcert-v1' && { color: '#10B981' }]}>
                  MobCert v1 (Source-Span Contract)
                </Text>
              </View>
              <Text style={styles.optionSub}>
                Extracts deterministic evidence atoms, verifies numeric/polarity/table scoping, flags contradictions, and enforces controlled abstention.
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.optionRow, ragPolicy === 'ebmrag-v2' && styles.optionRowActive]}
            onPress={() => updateRagPolicy('ebmrag-v2')}
          >
            <Icon name="bolt" size={20} color={ragPolicy === 'ebmrag-v2' ? '#3B82F6' : '#9CA3AF'} />
            <View style={styles.optionTextContainer}>
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Text style={[styles.optionTitle, ragPolicy === 'ebmrag-v2' && { color: '#3B82F6' }]}>
                  EBM-RAG v2 (Energy-Budgeted Baseline)
                </Text>
              </View>
              <Text style={styles.optionSub}>
                Whole-chunk packing with 82% overlap deduplication, dynamic budget allocation, without deterministic contract certification.
              </Text>
            </View>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.actionCard} onPress={handleResetController}>
          <Icon name="refresh" size={22} color="#F59E0B" />
          <View style={{ marginLeft: 12, flex: 1 }}>
            <Text style={styles.actionCardTitle}>Reset Adaptive EMA History</Text>
            <Text style={styles.actionCardSubtitle}>Clear learned thread latencies to start fresh benchmark runs</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionCard}
          onPress={async () => {
            try {
              const histPath = `${RNFS.ExternalDirectoryPath}/evaluation_runs_history.json`;
              if (await RNFS.exists(histPath)) {
                await RNFS.unlink(histPath);
                Alert.alert('History Cleared', 'Evaluation runs history has been reset.');
              } else {
                Alert.alert('History Empty', 'No past evaluation runs recorded.');
              }
            } catch (err) {
              Alert.alert('Error', String(err));
            }
          }}
        >
          <Icon name="delete-sweep" size={22} color="#EF4444" />
          <View style={{ marginLeft: 12, flex: 1 }}>
            <Text style={[styles.actionCardTitle, { color: '#F87171' }]}>Clear Evaluation History</Text>
            <Text style={styles.actionCardSubtitle}>Reset /sdcard/.../evaluation_runs_history.json for fresh runs</Text>
          </View>
        </TouchableOpacity>

        {Boolean(lastBenchmarkJson) && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Latest Benchmark Record</Text>
            <Text style={styles.cardDesc}>
              Location: /sdcard/Android/data/com.chat/files/inference_benchmark_latest.json
            </Text>
            <View style={styles.jsonPreview}>
              <Text style={styles.jsonText} numberOfLines={12}>
                {lastBenchmarkJson}
              </Text>
            </View>
          </View>
        )}

        <Text style={[styles.sectionHeader, { marginTop: 24 }]}>Account & Profile</Text>
        <TouchableOpacity
          style={styles.modelCard}
          onPress={async () => {
            await logout();
          }}
        >
          <Text style={styles.modelName}>Switch Profile</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
};

export default Setting1;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B0F19',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#1F2937',
    backgroundColor: '#101626',
  },
  backButton: {
    padding: 4,
  },
  title: {
    fontSize: 20,
    color: '#E5E7EB',
    fontWeight: 'bold',
    fontFamily: 'sans-serif',
  },
  content: {
    padding: 16,
  },
  sectionHeader: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#9CA3AF',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 12,
  },
  card: {
    backgroundColor: '#161F30',
    borderRadius: 14,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: '#243048',
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#F9FAFB',
    marginBottom: 4,
  },
  cardDesc: {
    fontSize: 12,
    color: '#94A3B8',
    marginBottom: 12,
  },
  optionRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#1E293B',
    padding: 12,
    borderRadius: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#334155',
  },
  optionRowActive: {
    borderColor: '#10B981',
    backgroundColor: '#0F2922',
  },
  optionTextContainer: {
    marginLeft: 10,
    flex: 1,
  },
  optionTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 2,
  },
  optionSub: {
    fontSize: 11,
    color: '#94A3B8',
    lineHeight: 16,
  },
  fixedThreadsContainer: {
    marginTop: 6,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#243048',
  },
  fixedThreadsLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#D1D5DB',
    marginBottom: 8,
  },
  threadButtonsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  threadBtn: {
    flex: 1,
    marginHorizontal: 3,
    backgroundColor: '#1E293B',
    paddingVertical: 8,
    borderRadius: 8,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155',
  },
  threadBtnActive: {
    backgroundColor: '#6D28D9',
    borderColor: '#A78BFA',
  },
  threadBtnText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#9CA3AF',
  },
  threadBtnTextActive: {
    color: '#FFFFFF',
  },
  actionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#161F30',
    borderRadius: 14,
    padding: 14,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: '#243048',
  },
  actionCardTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#F9FAFB',
  },
  actionCardSubtitle: {
    fontSize: 11,
    color: '#94A3B8',
    marginTop: 2,
  },
  jsonPreview: {
    backgroundColor: '#0B0F19',
    borderRadius: 8,
    padding: 10,
    marginTop: 6,
    borderWidth: 1,
    borderColor: '#1E293B',
  },
  jsonText: {
    fontFamily: 'monospace',
    fontSize: 10,
    color: '#A78BFA',
  },
  modelCard: {
    backgroundColor: '#1E293B',
    padding: 15,
    borderRadius: 12,
    marginBottom: 25,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155',
  },
  modelName: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
});
