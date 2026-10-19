
import React, { useState, useEffect, useRef } from 'react';
import { Image, Modal, TouchableOpacity as RNTouchableOpacity } from 'react-native';
import MaskedView from '@react-native-masked-view/masked-view';
import LinearGradient from 'react-native-linear-gradient';
import {
  View,
  Text,
  TextInput,
  FlatList,
  StyleSheet,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  LogBox,
  Keyboard,
  Animated,
  ScrollView,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { initLlama, LlamaContext } from 'llama.rn';
import RNFS from 'react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';
// RAG retrieval now handled inside qa.ts — no longer needed here
import Icon from 'react-native-vector-icons/MaterialIcons';
import AudioWaveform from './audiowaveformCode';
import { routeTask } from '../utils/taskRouter';
import { runHacerPhase1Probe } from '../utils/hacerPhase1';
import { useProfile } from '../utils/ProfileContext';
import { Profile, getProfiles, profileKey } from '../utils/profileManager';
import { latestInferenceBenchmark, InferenceBenchmarkReport } from '../tasks/qa';
import {
  BASELINE_INFERENCE_CONFIG,
  DEFAULT_CHAT_INFERENCE_MODE,
  InferenceMode,
  InferenceRequestOptions,
  RagPolicy,
} from '../utils/inferenceController';

// Use platform-specific paths for MODELS_DIR
const MODELS_DIR = RNFS.ExternalDirectoryPath + '/models';

// Keys generated dynamically below using profileKey

const getCurrentTime = () => {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const getCurrentDateTime = () => {
  const now = new Date();
  return now.toLocaleString();
};

type Message = {
  text: string;
  type: 'user' | 'bot' | 'loading';
  time: string;
  responseTime?: string;
  benchmark?: InferenceBenchmarkReport | null;
};

type Task = {
  id: string;
  input: string;
  taskType: string;
  dateTime: string;
};

const CustomSwitch = ({
  value,
  onValueChange,
}: {
  value: boolean;
  onValueChange: (val: boolean) => void;
}) => {
  const animatedValue = useRef(new Animated.Value(value ? 1 : 0)).current;

  useEffect(() => {
    Animated.timing(animatedValue, {
      toValue: value ? 1 : 0,
      duration: 200,
      useNativeDriver: false,
    }).start();
  }, [value]);

  const toggleTranslate = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: [3, 37],
  });

  const backgroundColor = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: ['#374151', '#8B5CF6'],
  });

  return (
    <RNTouchableOpacity
      activeOpacity={0.8}
      onPress={() => onValueChange(!value)}
    >
      <Animated.View style={[styles.switchContainer, { backgroundColor }]}>
        {value ? (
          <Text style={[styles.switchText, styles.switchTextOn]}>ON</Text>
        ) : (
          <Text style={[styles.switchText, styles.switchTextOff]}>OFF</Text>
        )}
        <Animated.View
          style={[
            styles.switchKnob,
            { transform: [{ translateX: toggleTranslate }] },
          ]}
        />
      </Animated.View>
    </RNTouchableOpacity>
  );
};

type ChatSession = {
  id: string;
  title: string;
  createdAt: string;
};

const ChatScreen = ({ navigation, route }: { navigation: any; route: any }) => {
  const insets = useSafeAreaInsets();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [context, setContext] = useState<LlamaContext | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [modelStatus, setModelStatus] = useState('Loading...');
  const [currentModel, setCurrentModel] = useState('');
  const [errorCount, setErrorCount] = useState(0);
  const [modelError, setModelError] = useState<string | null>(null);
  const [taskHistory, setTaskHistory] = useState<Task[]>([]);
  const [isHistoryModalVisible, setIsHistoryModalVisible] = useState(false);
  const modelId = route.params?.selectedModelId;
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const contextRef = useRef<LlamaContext | null>(null);
  const currentNCtxRef = useRef<number>(BASELINE_INFERENCE_CONFIG.n_ctx);
  const flatListRef = useRef<FlatList>(null);
  const { profileId } = useProfile();
  const [chatMode, setChatMode] = useState<'general' | 'document'>('general');
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>(DEFAULT_CHAT_INFERENCE_MODE);
  const [ragPolicy, setRagPolicy] = useState<RagPolicy>('mobcert-v1');
  const [fixedThreads, setFixedThreads] = useState<number>(4);
  const [showInferenceModal, setShowInferenceModal] = useState<boolean>(false);
  const [activeBenchmarkDetail, setActiveBenchmarkDetail] = useState<InferenceBenchmarkReport | null>(null);

  // Drawer state & animations
  const drawerAnimation = useRef(new Animated.Value(0)).current;
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [activeProfile, setActiveProfile] = useState<Profile | null>(null);

  // Multi-chat sessions state
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // Direct document ingest states
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState('');
  const isCreatingNewChat = useRef(false);

  const formatModelName = (name: string) => {
    if (!name) return 'Select Model';
    let clean = name;
    if (clean.toLowerCase().includes('gemma-2-2b')) return 'Gemma 2 2B';
    if (clean.toLowerCase().includes('tinyllama')) return 'TinyLlama 1.1B';
    if (clean.toLowerCase().includes('stablelm-2-zephyr')) return 'StableLM 2 Zephyr 1.6B';
    if (clean.toLowerCase().includes('deepseek-r1-distill-qwen-1.5b')) return 'DeepSeek R1 1.5B';
    if (clean.toLowerCase().includes('deepseek-coder-1.3b')) return 'DeepSeek Coder 1.3B';
    if (clean.toLowerCase().includes('phi-2')) return 'Phi-2';
    if (clean.toLowerCase().includes('opengpt-3')) return 'OpenGPT-3';
    if (clean.toLowerCase().includes('phi-3.5-mini')) return 'Phi-3.5 Mini';
    if (clean.toLowerCase().includes('qwen2.5-1.5b')) return 'Qwen 2.5 1.5B';
    if (clean.toLowerCase().includes('qwen2.5-3b')) return 'Qwen 2.5 3B';
    if (clean.toLowerCase().includes('llama-3.2-1b')) return 'Llama 3.2 1B';
    if (clean.toLowerCase().includes('llama-3.2-3b')) return 'Llama 3.2 3B';
    if (clean.toLowerCase().includes('all-minilm')) return 'MiniLM L6 v2';

    clean = clean.replace(/-it-|-chat-|-instruct-/gi, ' ');
    clean = clean.replace(/-q\d_[a-z_0-9]+/gi, ''); 
    clean = clean.replace(/\.gguf$/i, '');
    clean = clean.replace(/-/g, ' ');
    return clean.replace(/\b\w/g, c => c.toUpperCase());
  };

  // Bug #16 fix: guard against null profileId in key generation
  const SELECTED_MODEL_KEY = profileId ? profileKey(profileId, '_selected_model') : '';
  const TASK_HISTORY_KEY = profileId ? profileKey(profileId, '_task_history') : '';
  const CHAT_MESSAGES_KEY = profileId ? profileKey(profileId, '_chat_messages') : '';
  const CHAT_MODE_KEY = profileId ? profileKey(profileId, '_chat_mode') : '';
  const INFERENCE_MODE_KEY = profileId ? profileKey(profileId, '_inference_mode') : '@genix_inference_mode';
  const RAG_POLICY_KEY = profileId ? profileKey(profileId, '_rag_policy') : '@genix_rag_policy';
  const FIXED_THREADS_KEY = profileId ? profileKey(profileId, '_fixed_threads') : '@genix_fixed_threads';
  const SESSIONS_LIST_KEY = profileId ? profileKey(profileId, '_chat_sessions') : '';

  useEffect(() => {
    LogBox.ignoreLogs(['new NativeEventEmitter']);
  }, []);

  useEffect(() => {
    const initializeModel = async () => {
      try {
        // Use the model ID from params if available, otherwise check AsyncStorage
        let selectedModelId = modelId || (await AsyncStorage.getItem(SELECTED_MODEL_KEY));
        if (!selectedModelId) {
          throw new Error('No model selected. Please select a model to start chatting.');
        }

        // Safety guard: the embedding model (MiniLM) must never be loaded as a
        // chat LLM — it only outputs raw token strings, not natural language.
        if (selectedModelId === 'all-minilm-l6-v2-q4_k_m') {
          await AsyncStorage.removeItem(SELECTED_MODEL_KEY);
          throw new Error(
            'The embedding model (MiniLM) cannot be used for chat. Please go to Models and select a chat model (e.g. TinyLlama, Gemma, Phi).'
          );
        }

        const modelPath = `${MODELS_DIR}/${selectedModelId}.gguf`;
        const exists = await RNFS.exists(modelPath).catch((err) => {
          console.error('RNFS.exists error:', err);
          return false;
        });
        if (!exists) {
          throw new Error(`The selected model could not be found. Please download a model to continue.`);
        }

        setModelStatus(`Loading ${selectedModelId}...`);
        setCurrentModel(selectedModelId);

        const llamaContext = await initLlama({
          model: `file://${modelPath}`,
          n_ctx: BASELINE_INFERENCE_CONFIG.n_ctx,
          n_gpu_layers: BASELINE_INFERENCE_CONFIG.n_gpu_layers,
          n_threads: BASELINE_INFERENCE_CONFIG.n_threads,
        }).catch((err) => {
          throw new Error(`Failed to initialize the model. Please try again or select a different model.`);
        });

        if (!llamaContext) {
          throw new Error('Model initialization failed. Please try again.');
        }
        // Bug #11 fix: keep ref in sync for cleanup
        contextRef.current = llamaContext;
        currentNCtxRef.current = BASELINE_INFERENCE_CONFIG.n_ctx;
        setContext(llamaContext);
        setModelStatus(`Model loaded: ${selectedModelId}`);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        setModelStatus(`Error: ${errorMessage}`);
        setModelError(errorMessage);
        setErrorCount((prev) => prev + 1);
        console.error('Model initialization error:', errorMessage);
      }
    };

    const loadTaskHistory = async () => {
      try {
        const history = await AsyncStorage.getItem(TASK_HISTORY_KEY);
        if (history) {
          setTaskHistory(JSON.parse(history));
        }
      } catch (error) {
        console.error('Error loading task history:', error);
      }
    };

    const loadChatMode = async () => {
      try {
        if (CHAT_MODE_KEY) {
          const savedMode = await AsyncStorage.getItem(CHAT_MODE_KEY);
          if (savedMode === 'general' || savedMode === 'document') {
            setChatMode(savedMode);
          } else {
            setChatMode('general');
          }
        }
      } catch (error) {
        console.error('Error loading chat mode:', error);
      }
    };

    const loadInferenceSettings = async () => {
      try {
        if (INFERENCE_MODE_KEY) {
          const savedMode = await AsyncStorage.getItem(INFERENCE_MODE_KEY);
          if (savedMode === 'baseline' || savedMode === 'fixed' || savedMode === 'adaptive') {
            setInferenceMode(savedMode);
          }
        }
        if (RAG_POLICY_KEY) {
          const savedPolicy = await AsyncStorage.getItem(RAG_POLICY_KEY);
          if (savedPolicy === 'mobcert-v1' || savedPolicy === 'ebmrag-v2') {
            setRagPolicy(savedPolicy as RagPolicy);
          }
        }
        if (FIXED_THREADS_KEY) {
          const savedThreads = await AsyncStorage.getItem(FIXED_THREADS_KEY);
          if (savedThreads && Number.isFinite(Number(savedThreads))) {
            setFixedThreads(Number(savedThreads));
          }
        }
      } catch (error) {
        console.warn('Error loading inference settings:', error);
      }
    };

    initializeModel();
    loadTaskHistory();
    loadChatMode();
    loadInferenceSettings();

    // Automated on-device benchmark polling listener
    const triggerPath = `${RNFS.ExternalDirectoryPath}/benchmark_trigger.json`;
    const resultPath = `${RNFS.ExternalDirectoryPath}/benchmark_result.json`;
    let isBenchmarking = false;

    const writeBenchmarkResult = async (path: string, payload: unknown) => {
      const tempPath = `${path}.tmp`;
      try {
        if (await RNFS.exists(tempPath)) {
          await RNFS.unlink(tempPath).catch(() => {});
        }
        await RNFS.writeFile(tempPath, JSON.stringify(payload, null, 2), 'utf8');
        if (await RNFS.exists(path)) {
          await RNFS.unlink(path).catch(() => {});
        }
        await RNFS.moveFile(tempPath, path);
      } catch (err) {
        await RNFS.writeFile(path, JSON.stringify(payload, null, 2), 'utf8').catch(() => {});
      }
    };

    const benchmarkInterval = setInterval(async () => {
      if (isBenchmarking || !contextRef.current) return;
      let activeRequestId: string | null = null;
      let activeResultPath: string | null = null;
      try {
        const exists = await RNFS.exists(triggerPath);
        if (exists) {
          isBenchmarking = true;
          const content = await RNFS.readFile(triggerPath, 'utf8');
          await RNFS.unlink(triggerPath).catch(() => {});
          const trigger = JSON.parse(content);
          console.log('[BENCHMARK_TRIGGER] Received benchmark trigger:', trigger);
          const requestId = typeof trigger.requestId === 'string' ? trigger.requestId : null;
          const resultPath = typeof trigger.resultPath === 'string'
            ? trigger.resultPath
            : `${RNFS.ExternalDirectoryPath}/benchmark_result.json`;
          activeRequestId = requestId;
          activeResultPath = resultPath;
          const statusPath = typeof trigger.statusPath === 'string'
            ? trigger.statusPath
            : `${resultPath}.status`;
          if (!requestId) throw new Error('Benchmark trigger is missing requestId.');
          await writeBenchmarkResult(statusPath, {
            status: 'started',
            request_id: requestId,
            started_at: new Date().toISOString(),
          });
          const q = trigger.query || 'Tell me about zypher';
          const mode = trigger.chatMode || 'document';
          const profId = trigger.profileId || profileId;
          const requestedInferenceMode: InferenceMode =
            trigger.inferenceMode === 'adaptive' || trigger.inferenceMode === 'fixed'
              ? trigger.inferenceMode
              : 'baseline';
          const inferenceOptions: InferenceRequestOptions = {
            mode: requestedInferenceMode,
            allowExploration: Boolean(trigger.allowExploration),
            measurePromptTokens: Boolean(trigger.measurePromptTokens),
            nThreads: Number.isFinite(Number(trigger.nThreads)) ? Number(trigger.nThreads) : undefined,
            contextTokenBudget: Number.isFinite(Number(trigger.contextTokenBudget))
              ? Number(trigger.contextTokenBudget)
              : undefined,
            maxOutputTokens: Number.isFinite(Number(trigger.maxOutputTokens))
              ? Number(trigger.maxOutputTokens)
              : undefined,
            ragPolicy: (trigger.ragPolicy === 'mobcert-v1' || trigger.ragPolicy === 'ebmrag-v2')
              ? trigger.ragPolicy
              : ragPolicy,
            allowDirectEvidenceAnswer: trigger.allowDirectEvidenceAnswer === true,
            benchmarkRequestId: requestId,
            benchmarkQueryId: typeof trigger.queryId === 'string' ? trigger.queryId : undefined,
            benchmarkCorpusId: typeof trigger.corpusId === 'string' ? trigger.corpusId : undefined,
          };

          // Safe Dynamic Context Reconfiguration
          const requestedNCtx = Number(trigger.nCtx || trigger.phase1NCtx);
          if (Number.isFinite(requestedNCtx) && requestedNCtx > 0 && requestedNCtx !== currentNCtxRef.current) {
            console.log(`[BENCHMARK_TRIGGER] Reconfiguring context n_ctx: ${currentNCtxRef.current} -> ${requestedNCtx}`);
            const oldContext = contextRef.current;
            contextRef.current = null;
            setContext(null);
            if (oldContext && oldContext.release) {
              await oldContext.release().catch((e: any) => console.warn('[BENCHMARK_TRIGGER] Error releasing old context:', e));
            }
            let selectedModelId = modelId || (await AsyncStorage.getItem(SELECTED_MODEL_KEY)) || currentModel || 'gemma-2-2b-it-Q4_K_M';
            const modelPath = `${MODELS_DIR}/${selectedModelId}.gguf`;
            const newContext = await initLlama({
              model: `file://${modelPath}`,
              n_ctx: requestedNCtx,
              n_batch: 512,
              n_ubatch: 512,
              n_gpu_layers: BASELINE_INFERENCE_CONFIG.n_gpu_layers,
              n_threads: BASELINE_INFERENCE_CONFIG.n_threads,
              use_mmap: true,
            });
            if (!newContext) {
              throw new Error(`Failed to re-initialize model with n_ctx=${requestedNCtx}`);
            }
            contextRef.current = newContext;
            currentNCtxRef.current = requestedNCtx;
            setContext(newContext);
            console.log(`[BENCHMARK_TRIGGER] Context successfully reconfigured to n_ctx=${requestedNCtx}`);
          }

          if (trigger.action === 'reconfigure_context') {
            await writeBenchmarkResult(resultPath, {
              status: 'completed',
              request_id: requestId,
              action: 'reconfigure_context',
              active_n_ctx: currentNCtxRef.current,
            });
            isBenchmarking = false;
            return;
          }

          if (trigger.phase1 === 'seed' || trigger.phase1 === 'baseline' || trigger.phase1 === 'reuse') {
            const phase1OutputPath = trigger.phase1OutputPath || `${RNFS.ExternalDirectoryPath}/hacer_phase1_result.json`;
            const phase1Result = await runHacerPhase1Probe(contextRef.current, {
              phase: trigger.phase1,
              prefix: String(trigger.phase1Prefix || ''),
              suffix: String(trigger.phase1Suffix || ''),
              sessionPath: String(trigger.phase1SessionPath || `${RNFS.ExternalDirectoryPath}/hacer_phase1_session.bin`),
              outputPath: phase1OutputPath,
              nPredict: Number.isFinite(Number(trigger.phase1NPredict)) ? Number(trigger.phase1NPredict) : 1,
              promptSha256: typeof trigger.promptSha256 === 'string' ? trigger.promptSha256 : undefined,
              activeNCtx: currentNCtxRef.current,
            });
            await writeBenchmarkResult(resultPath, {
              status: 'completed',
              request_id: requestId,
              query: q,
              chatMode: mode,
              active_n_ctx: currentNCtxRef.current,
              phase1: phase1Result,
            });
            isBenchmarking = false;
            return;
          }

          const ans = await routeTask(q, contextRef.current, profId, undefined, mode, inferenceOptions);
          const out = {
            status: 'completed',
            request_id: requestId,
            query: q,
            chatMode: mode,
            inferenceOptions,
            answer: ans,
            benchmark: latestInferenceBenchmark,
          };
          await writeBenchmarkResult(resultPath, out);
          console.log('[BENCHMARK_TRIGGER] Benchmark results written to:', resultPath);
          isBenchmarking = false;
        }
      } catch (e) {
        console.warn('[BENCHMARK_TRIGGER] Error processing trigger:', e);
        try {
          if (activeRequestId && activeResultPath) {
            await writeBenchmarkResult(activeResultPath, {
              status: 'failed',
              request_id: activeRequestId,
              error: e instanceof Error ? e.message : String(e),
              finished_at: new Date().toISOString(),
            });
          }
        } catch (_) {
          // Preserve polling loop
        }
        isBenchmarking = false;
      }
    }, 1500);

    return () => {
      clearInterval(benchmarkInterval);
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
        typingTimeoutRef.current = null;
      }
      // Bug #11 fix: use contextRef instead of stale closure over context state
      if (contextRef.current) {
        try {
          contextRef.current.release();
          contextRef.current = null;
          setContext(null);
        } catch (releaseError) {
          console.error('Error releasing context:', releaseError);
        }
      }
    };
  }, [modelId, profileId]);

  // Load active profile metadata
  useEffect(() => {
    const fetchProfile = async () => {
      if (profileId) {
        try {
          const profiles = await getProfiles();
          const active = profiles.find((p) => p.id === profileId);
          if (active) {
            setActiveProfile(active);
          }
        } catch (e) {
          console.error('Error fetching active profile:', e);
        }
      }
    };
    fetchProfile();
  }, [profileId]);

  // Load chat sessions on mount/profile switch
  useEffect(() => {
    const loadSessions = async () => {
      if (!profileId || !SESSIONS_LIST_KEY) return;
      try {
        const rawSessions = await AsyncStorage.getItem(SESSIONS_LIST_KEY);
        let sessionsList: ChatSession[] = [];
        if (rawSessions) {
          sessionsList = JSON.parse(rawSessions);
        }

        // Migration Check: If no sessions exist, check if there are legacy messages in CHAT_MESSAGES_KEY
        if (sessionsList.length === 0) {
          const legacyMessages = await AsyncStorage.getItem(CHAT_MESSAGES_KEY);
          if (legacyMessages) {
            const parsedMsgs = JSON.parse(legacyMessages);
            if (Array.isArray(parsedMsgs) && parsedMsgs.length > 0) {
              const legacySession: ChatSession = {
                id: 'session_legacy',
                title: 'Recent Chat',
                createdAt: new Date().toISOString(),
              };
              sessionsList = [legacySession];
              await AsyncStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(sessionsList));
              await AsyncStorage.setItem(
                profileKey(profileId, `_chat_messages_session_legacy`),
                JSON.stringify(parsedMsgs)
              );
            }
          }
        }

        // If still empty, create a default first session
        if (sessionsList.length === 0) {
          const defaultSession: ChatSession = {
            id: `session_${Date.now()}`,
            title: 'New Chat',
            createdAt: new Date().toISOString(),
          };
          sessionsList = [defaultSession];
          await AsyncStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(sessionsList));
        }

        setSessions(sessionsList);
        setActiveSessionId(sessionsList[0].id);
      } catch (e) {
        console.error('Error loading chat sessions:', e);
      }
    };

    loadSessions();
  }, [profileId, SESSIONS_LIST_KEY]);

  // Load messages for the active session
  useEffect(() => {
    const loadMessagesForSession = async () => {
      if (!profileId || !activeSessionId) return;
      try {
        const key = profileKey(profileId, `_chat_messages_${activeSessionId}`);
        const rawMsgs = await AsyncStorage.getItem(key);
        if (rawMsgs) {
          setMessages(JSON.parse(rawMsgs));
        } else {
          setMessages([]);
        }
      } catch (e) {
        console.error('Error loading messages for session:', e);
      }
    };

    loadMessagesForSession();
  }, [profileId, activeSessionId]);

  const handleToggleChange = async (val: boolean) => {
    const newMode = val ? 'document' : 'general';
    setChatMode(newMode);
    try {
      if (CHAT_MODE_KEY) {
        await AsyncStorage.setItem(CHAT_MODE_KEY, newMode);
      }
    } catch (error) {
      console.error('Error saving chat mode:', error);
    }
  };

  const selectInferenceMode = async (mode: InferenceMode, threads = fixedThreads) => {
    setInferenceMode(mode);
    setFixedThreads(threads);
    setShowInferenceModal(false);
    try {
      if (INFERENCE_MODE_KEY) await AsyncStorage.setItem(INFERENCE_MODE_KEY, mode);
      if (FIXED_THREADS_KEY) await AsyncStorage.setItem(FIXED_THREADS_KEY, String(threads));
    } catch (e) {
      console.warn('Failed to save inference settings:', e);
    }
  };

  const handleToggleRagPolicy = async () => {
    const nextPolicy: RagPolicy = ragPolicy === 'mobcert-v1' ? 'ebmrag-v2' : 'mobcert-v1';
    setRagPolicy(nextPolicy);
    try {
      if (RAG_POLICY_KEY) {
        await AsyncStorage.setItem(RAG_POLICY_KEY, nextPolicy);
      }
    } catch (e) {
      console.warn('Failed to save RAG policy:', e);
    }
  };

  const LoadingDots = () => {
    const [dots, setDots] = useState('');

    useEffect(() => {
      const interval = setInterval(() => {
        setDots((prev) => (prev.length < 3 ? prev + '.' : ''));
      }, 400);
      return () => clearInterval(interval);
    }, []);

    return <Text style={styles.loadingText}>Generating{dots}</Text>;
  };

  const saveTaskToHistory = async (input: string, taskType: string) => {
    try {
      const newTask: Task = {
        id: Math.random().toString(36).substr(2, 9),
        input,
        taskType,
        dateTime: getCurrentDateTime(),
      };
      const updatedHistory = [newTask, ...taskHistory];
      setTaskHistory(updatedHistory);
      await AsyncStorage.setItem(TASK_HISTORY_KEY, JSON.stringify(updatedHistory));
    } catch (error) {
      console.error('Error saving task to history:', error);
    }
  };

  const clearTaskHistory = async () => {
    try {
      setTaskHistory([]);
      await AsyncStorage.setItem(TASK_HISTORY_KEY, JSON.stringify([]));
    } catch (error) {
      console.error('Error clearing task history:', error);
      Alert.alert('Error', 'Failed to clear task history');
    }
  };

  const clearChat = async () => {
    Keyboard.dismiss();
    try {
      setMessages([]);
      await AsyncStorage.setItem(CHAT_MESSAGES_KEY, JSON.stringify([]));
    } catch (error) {
      console.error('Error clearing chat history:', error);
    }
  };

  const displayWordByWord = (response: string, startTime: number) => {
    const words = response.split(' ');
    let currentIndex = 0;
    const benchSnapshot = latestInferenceBenchmark ? { ...latestInferenceBenchmark } : null;

    const typeNextWord = () => {
      const sessionMessagesKey = profileKey(profileId!, `_chat_messages_${activeSessionId}`);
      if (currentIndex >= words.length) {
        const responseDuration = ((Date.now() - startTime) / 1000).toFixed(2) + 's';
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last) {
            last.responseTime = responseDuration;
            if (benchSnapshot) {
              last.benchmark = benchSnapshot;
            }
          }
          if (sessionMessagesKey) {
            AsyncStorage.setItem(sessionMessagesKey, JSON.stringify(updated)).catch(console.error);
          }
          return updated;
        });
        setIsLoading(false);
        typingTimeoutRef.current = null;
        return;
      }

      const partial = words.slice(0, currentIndex + 1).join(' ');
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          text: partial,
          type: 'bot',
          time: getCurrentTime(),
        };
        if (sessionMessagesKey) {
          AsyncStorage.setItem(sessionMessagesKey, JSON.stringify(updated)).catch(console.error);
        }
        return updated;
      });

      currentIndex++;
      typingTimeoutRef.current = setTimeout(typeNextWord, 60 + Math.random() * 40);
    };

    typeNextWord();
  };

  const handleStopGeneration = () => {
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
      typingTimeoutRef.current = null;
    }
    setMessages((prev) => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (last?.type === 'loading') {
        updated.pop();
      }
      return updated;
    });
    setIsLoading(false);
  };

  const handleSend = async () => {
    if (!inputText.trim() || !context || isLoading || isListening || !activeSessionId) return;

    const userQueryText = inputText.trim();
    const userMessage = {
      text: userQueryText,
      type: 'user' as const,
      time: getCurrentTime(),
    };

    setInputText('');
    setIsLoading(true);

    const sessionMessagesKey = profileKey(profileId!, `_chat_messages_${activeSessionId}`);

    let updatedMsgs: Message[] = [];
    setMessages((prev) => {
      updatedMsgs = [...prev, userMessage, { text: '...', type: 'loading' as const, time: getCurrentTime() }];
      if (sessionMessagesKey) {
        AsyncStorage.setItem(sessionMessagesKey, JSON.stringify(updatedMsgs)).catch(console.error);
      }
      return updatedMsgs;
    });

    const activeSession = sessions.find((s) => s.id === activeSessionId);
    if (activeSession && activeSession.title === 'New Chat') {
      const generatedTitle =
        userQueryText.substring(0, 26) + (userQueryText.length > 26 ? '...' : '');
      const updatedSessions = sessions.map((s) =>
        s.id === activeSessionId ? { ...s, title: generatedTitle } : s
      );
      setSessions(updatedSessions);
      if (SESSIONS_LIST_KEY) {
        AsyncStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(updatedSessions)).catch(console.error);
      }
    }

    try {
      const startTime = Date.now();
      const response = await routeTask(
        userMessage.text,
        context,
        profileId,
        saveTaskToHistory,
        chatMode,
        {
          mode: inferenceMode,
          nThreads: inferenceMode === 'fixed' ? fixedThreads : undefined,
          ragPolicy: ragPolicy,
        },
      );

      if (!response || typeof response !== 'string') {
        throw new Error('Invalid response from routeTask');
      }

      displayWordByWord(response, startTime);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error('Send error:', errorMessage);
      setErrorCount((prev) => prev + 1);

      setMessages((prev) => {
        const last = prev[prev.length - 1];
        let updated: Message[] = [];
        if (last?.type === 'loading') {
          updated = [...prev];
          updated[updated.length - 1] = {
            text: `❌ Error: ${errorMessage}`,
            type: 'bot',
            time: getCurrentTime(),
          };
        } else {
          updated = [...prev, {
            text: `❌ Error: ${errorMessage}`,
            type: 'bot',
            time: getCurrentTime(),
          }];
        }
        if (sessionMessagesKey) {
          AsyncStorage.setItem(sessionMessagesKey, JSON.stringify(updated)).catch(console.error);
        }
        return updated;
      });

      Alert.alert('Error', `Failed to generate response: ${errorMessage}`);
      setIsLoading(false);
    }
  };

  const startNewChat = async () => {
    if (!profileId || !SESSIONS_LIST_KEY || isCreatingNewChat.current) return;
    
    // Check if current messages are already empty
    if (messages.length === 0) {
      closeDrawer();
      return;
    }

    // Also check if the active session is already empty (named "New Chat")
    const activeSession = sessions.find((s) => s.id === activeSessionId);
    if (activeSession && activeSession.title === 'New Chat') {
      closeDrawer();
      return;
    }

    try {
      isCreatingNewChat.current = true;
      const newSession: ChatSession = {
        id: `session_${Date.now()}`,
        title: 'New Chat',
        createdAt: new Date().toISOString(),
      };
      const updatedSessions = [newSession, ...sessions];
      setSessions(updatedSessions);
      await AsyncStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(updatedSessions));
      setActiveSessionId(newSession.id);
      setMessages([]);
      closeDrawer();
    } catch (e) {
      console.error('Error starting new chat:', e);
      Alert.alert('Error', 'Failed to start a new chat');
    } finally {
      isCreatingNewChat.current = false;
    }
  };

  const deleteSession = async (sessionId: string) => {
    Alert.alert(
      'Delete Chat',
      'Are you sure you want to delete this chat session?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              if (profileId) {
                const sessionMessagesKey = profileKey(profileId, `_chat_messages_${sessionId}`);
                await AsyncStorage.removeItem(sessionMessagesKey);
              }

              const updatedSessions = sessions.filter((s) => s.id !== sessionId);
              if (profileId && SESSIONS_LIST_KEY) {
                if (updatedSessions.length === 0) {
                  const defaultSession: ChatSession = {
                    id: `session_${Date.now()}`,
                    title: 'New Chat',
                    createdAt: new Date().toISOString(),
                  };
                  const finalSessions = [defaultSession];
                  setSessions(finalSessions);
                  await AsyncStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(finalSessions));
                  setActiveSessionId(defaultSession.id);
                  setMessages([]);
                } else {
                  setSessions(updatedSessions);
                  await AsyncStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(updatedSessions));
                  if (activeSessionId === sessionId) {
                    setActiveSessionId(updatedSessions[0].id);
                  }
                }
              }
            } catch (e) {
              console.error('Error deleting session:', e);
              Alert.alert('Error', 'Failed to delete chat session');
            }
          },
        },
      ]
    );
  };

  const selectSession = (sessionId: string) => {
    setActiveSessionId(sessionId);
    closeDrawer();
  };

  // Cleaned up handlePickDocument and extractText from ChatScreen - now fully managed in DocumentsScreen

  const openDrawer = () => {
    setIsDrawerOpen(true);
    Animated.timing(drawerAnimation, {
      toValue: 1,
      duration: 250,
      useNativeDriver: true,
    }).start();
  };

  const closeDrawer = () => {
    Animated.timing(drawerAnimation, {
      toValue: 0,
      duration: 200,
      useNativeDriver: true,
    }).start(() => {
      setIsDrawerOpen(false);
    });
  };

  const navigateTo = (screenName: string) => {
    closeDrawer();
    navigation.navigate(screenName);
  };

  const startListening = () => {
    if (isLoading || !context) {
      console.warn('Cannot start listening: Loading or no context');
      return;
    }
    setIsListening(true);
  };

  const stopListening = () => {
    setIsListening(false);
    setInputText('Transcribed text');
  };

  const cancelListening = () => {
    setIsListening(false);
  };

  const renderMessage = ({ item }: { item: Message }) => (
    <View
      style={[
        styles.messageContainer,
        item.type === 'user'
          ? styles.userMessage
          : item.type === 'loading'
            ? styles.loadingMessage
            : styles.botMessage,
      ]}
    >
      <View style={styles.messageRow}>
        {item.type === 'bot' && (
          <Image
            source={require('./logo.png')}
            style={styles.botIcon}
          />
        )}
        <View style={styles.textContainer}>
          {item.type === 'loading' ? (
            <LoadingDots />
          ) : (
            <Text style={styles.messageText}>{item.text}</Text>
          )}
        </View>
      </View>
      <View style={styles.messageTime}>
        <Text style={styles.messageTimeText}>{item.time}</Text>
        {item.responseTime && (
          <Text style={styles.responseTime}> · {item.responseTime}</Text>
        )}
      </View>
      {Boolean(item.benchmark) && (
        <RNTouchableOpacity
          style={[
            styles.benchmarkPill,
            item.benchmark?.research_metrics?.policy === 'mobcert-v1'
              ? styles.benchmarkPillMobCert
              : styles.benchmarkPillEbmRag,
          ]}
          onPress={() => setActiveBenchmarkDetail(item.benchmark || null)}
          activeOpacity={0.7}
        >
          <Icon
            name={
              item.benchmark?.research_metrics?.policy === 'mobcert-v1'
                ? "verified-user"
                : "bolt"
            }
            size={12}
            color={
              item.benchmark?.research_metrics?.policy === 'mobcert-v1'
                ? "#10B981"
                : "#38BDF8"
            }
            style={{ marginRight: 4 }}
          />
          <Text style={styles.benchmarkPillText}>
            {item.benchmark?.research_metrics?.policy === 'mobcert-v1' ? 'MobCert' : 'EBM-RAG'} · {
              item.benchmark?.research_metrics
                ? (item.benchmark.research_metrics.contractStatus === 'certified'
                    ? 'Certified'
                    : item.benchmark.research_metrics.contractStatus === 'abstained'
                    ? 'Abstained'
                    : item.benchmark.research_metrics.contractStatus === 'repaired'
                    ? 'Repaired'
                    : 'Uncertified')
                : (item.benchmark?.controller_mode?.toUpperCase() || 'ADAPTIVE')
            } · TTFT {item.benchmark?.ttft_ms ? `${(item.benchmark.ttft_ms / 1000).toFixed(2)}s` : 'N/A'} · {item.benchmark?.total_warm_latency_ms ? `${(item.benchmark.total_warm_latency_ms / 1000).toFixed(2)}s` : 'N/A'}
          </Text>
          <Icon name="info-outline" size={12} color="#9CA3AF" style={{ marginLeft: 4 }} />
        </RNTouchableOpacity>
      )}
    </View>
  );

  const renderTask = ({ item }: { item: Task }) => (
    <View style={styles.taskItem}>
      <Text style={styles.taskType}>{item.taskType}</Text>
      <Text style={styles.taskText}>{item.input}</Text>
      <Text style={styles.taskDateTime}>{item.dateTime}</Text>
    </View>
  );

  return (
    <View style={{ flex: 1, position: 'relative', backgroundColor: '#0B0F19', paddingTop: insets.top }}>
      <View style={{ flex: 1, backgroundColor: '#0B0F19' }}>
        <KeyboardAvoidingView
          style={styles.container}
          behavior="padding"
          keyboardVerticalOffset={Platform.OS === 'ios' ? 80 : 16}
        >
          <View style={styles.header}>
            <View style={styles.headerTopRow}>
              <View style={styles.headerLeft}>
                <RNTouchableOpacity
                  style={styles.headerButton}
                  onPress={openDrawer}
                >
                  <View style={styles.menuIconContainer}>
                    <View style={[styles.menuLine, { width: 22 }]} />
                    <View style={[styles.menuLine, { width: 22 }]} />
                    <View style={[styles.menuLine, { width: 12 }]} />
                  </View>
                </RNTouchableOpacity>
              </View>

              <View style={styles.headerCenter}>
                <RNTouchableOpacity onPress={() => navigateTo('Models')}>
                  <Text style={styles.headerTitleText}>{formatModelName(currentModel)}</Text>
                </RNTouchableOpacity>
              </View>

              <View style={styles.headerRightSide}>
                <RNTouchableOpacity
                  style={styles.headerButton}
                  onPress={() => navigateTo('Settings')}
                >
                  <Icon name="face" size={26} color="#E5E7EB" />
                </RNTouchableOpacity>
              </View>
            </View>
          </View>

          {!context && !modelError ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#8B5CF6" />
              <Text style={styles.loadingText}>{modelStatus}</Text>
            </View>
          ) : !context && modelError ? (
            <View style={styles.welcomeContainer}>
              <Text style={styles.errorText}>{modelError}</Text>
              <RNTouchableOpacity
                style={styles.selectModelButton}
                onPress={() => navigation.navigate('Models')}
              >
                <Text style={styles.selectModelButtonText}>Go to Models Screen</Text>
              </RNTouchableOpacity>
            </View>
          ) : messages.length === 0 ? (
            <View style={styles.welcomeContainer}>
              <MaskedView
                maskElement={
                  <View style={styles.maskContainer}>
                    <Text style={[styles.welcomeText, { backgroundColor: 'transparent' }]}>
                      How may I help you?
                    </Text>
                  </View>
                }
              >
                <LinearGradient
                  colors={['rgba(232, 74, 127, 1)', 'rgba(122, 95, 255, 1)']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                >
                  <Text style={[styles.welcomeText, { opacity: 0 }]}>
                    How may I help you?
                  </Text>
                </LinearGradient>
              </MaskedView>
            </View>
          ) : (
            <FlatList
              ref={flatListRef}
              data={messages}
              renderItem={renderMessage}
              keyExtractor={(_, index) => index.toString()}
              style={{ flex: 1 }}
              contentContainerStyle={styles.messagesContainer}
              onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
              keyboardShouldPersistTaps="handled"
            />
          )}

          {isIngesting && (
            <Modal transparent={true} visible={true} animationType="fade">
              <View style={styles.ingestModalOverlay}>
                <View style={styles.ingestModalContent}>
                  <ActivityIndicator size="large" color="#8B5CF6" />
                  <Text style={styles.ingestModalText}>{ingestStatus}</Text>
                </View>
              </View>
            </Modal>
          )}

          <Modal
            animationType="slide"
            transparent={true}
            visible={isHistoryModalVisible}
            onRequestClose={() => setIsHistoryModalVisible(false)}
          >
            <View style={styles.modalOverlay}>
              <View style={styles.modalContainer}>
                <View style={styles.modalHeader}>
                  <Text style={styles.modalTitle}>Task History</Text>
                  <RNTouchableOpacity
                    style={styles.clearButton}
                    onPress={clearTaskHistory}
                  >
                    <Text style={styles.clearButtonText}>Clear</Text>
                  </RNTouchableOpacity>
                  <RNTouchableOpacity onPress={() => setIsHistoryModalVisible(false)}>
                    <Icon name="close" size={24} color="#fff" />
                  </RNTouchableOpacity>
                </View>
                <FlatList
                  data={taskHistory}
                  renderItem={renderTask}
                  keyExtractor={(item) => item.id}
                  contentContainerStyle={styles.taskList}
                  ListEmptyComponent={
                    <Text style={styles.emptyText}>No tasks yet</Text>
                  }
                />
              </View>
            </View>
          </Modal>

          {/* Inference Mode Selection Modal */}
          <Modal
            animationType="slide"
            transparent={true}
            visible={showInferenceModal}
            onRequestClose={() => setShowInferenceModal(false)}
          >
            <View style={styles.modalOverlay}>
              <View style={styles.inferenceModalContainer}>
                <View style={styles.modalHeader}>
                  <Text style={styles.modalTitle}>Inference Policy & Threads</Text>
                  <RNTouchableOpacity onPress={() => setShowInferenceModal(false)}>
                    <Icon name="close" size={24} color="#fff" />
                  </RNTouchableOpacity>
                </View>
                <Text style={styles.inferenceSubtitle}>
                  Select how on-device LLM inference and RAG budgeting are executed on your phone:
                </Text>

                <RNTouchableOpacity
                  style={[
                    styles.modeOptionCard,
                    inferenceMode === 'adaptive' && styles.modeOptionCardActive
                  ]}
                  onPress={() => selectInferenceMode('adaptive')}
                >
                  <View style={styles.modeOptionHeader}>
                    <Icon name="bolt" size={20} color="#10B981" />
                    <Text style={styles.modeOptionTitle}>⚡ Adaptive Policy (Optimized)</Text>
                  </View>
                  <Text style={styles.modeOptionDesc}>
                    Dynamic thread exploration/exploitation (EMA latency tracking), memory & thermal awareness, whole-chunk context budgeting, and 82% overlap deduplication.
                  </Text>
                </RNTouchableOpacity>

                <RNTouchableOpacity
                  style={[
                    styles.modeOptionCard,
                    inferenceMode === 'baseline' && styles.modeOptionCardActive
                  ]}
                  onPress={() => selectInferenceMode('baseline')}
                >
                  <View style={styles.modeOptionHeader}>
                    <Icon name="timer" size={20} color="#9CA3AF" />
                    <Text style={styles.modeOptionTitle}>⏱️ Historical Baseline (Fixed 4T)</Text>
                  </View>
                  <Text style={styles.modeOptionDesc}>
                    Fixed 4 threads, 512 context limit, full Top-3 chunk injection without deduplication. Replicates historical application baseline for controlled comparison.
                  </Text>
                </RNTouchableOpacity>

                <View style={styles.fixedThreadsSection}>
                  <Text style={styles.fixedThreadsTitle}>🔧 Fixed Thread Sweeps (Controlled Baseline):</Text>
                  <View style={styles.threadButtonsRow}>
                    {[1, 2, 4, 6, 8].map(t => (
                      <RNTouchableOpacity
                        key={t}
                        style={[
                          styles.threadButton,
                          inferenceMode === 'fixed' && fixedThreads === t && styles.threadButtonActive
                        ]}
                        onPress={() => selectInferenceMode('fixed', t)}
                      >
                        <Text style={[
                          styles.threadButtonText,
                          inferenceMode === 'fixed' && fixedThreads === t && styles.threadButtonTextActive
                        ]}>
                          {t}T
                        </Text>
                      </RNTouchableOpacity>
                    ))}
                  </View>
                </View>
              </View>
            </View>
          </Modal>

          {/* Detailed Benchmark Breakdown Modal */}
          {Boolean(activeBenchmarkDetail) && (
            <Modal
              animationType="fade"
              transparent={true}
              visible={true}
              onRequestClose={() => setActiveBenchmarkDetail(null)}
            >
              <View style={styles.modalOverlay}>
                <View style={styles.benchmarkDetailContainer}>
                  <View style={styles.modalHeader}>
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                      <Icon name="analytics" size={22} color="#8B5CF6" style={{ marginRight: 8 }} />
                      <Text style={styles.modalTitle}>Inference Profiling</Text>
                    </View>
                    <RNTouchableOpacity onPress={() => setActiveBenchmarkDetail(null)}>
                      <Icon name="close" size={24} color="#fff" />
                    </RNTouchableOpacity>
                  </View>

                  <ScrollView style={{ maxHeight: 460 }} showsVerticalScrollIndicator={true}>
                    {/* Policy & Status Banner */}
                    <View style={styles.researchBanner}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                          <Icon
                            name={activeBenchmarkDetail?.research_metrics?.policy === 'mobcert-v1' ? "verified-user" : "bolt"}
                            size={18}
                            color={activeBenchmarkDetail?.research_metrics?.policy === 'mobcert-v1' ? "#10B981" : "#38BDF8"}
                            style={{ marginRight: 6 }}
                          />
                          <Text style={styles.researchPolicyTitle}>
                            {activeBenchmarkDetail?.research_metrics?.policy === 'mobcert-v1'
                              ? 'MobCert-v1 (Source-Span Contract)'
                              : 'EBM-RAG-v2 (Baseline)'}
                          </Text>
                        </View>
                        <View
                          style={[
                            styles.statusBadge,
                            activeBenchmarkDetail?.research_metrics?.contractStatus === 'certified'
                              ? styles.statusCertified
                              : activeBenchmarkDetail?.research_metrics?.contractStatus === 'abstained'
                              ? styles.statusAbstained
                              : activeBenchmarkDetail?.research_metrics?.contractStatus === 'repaired'
                              ? styles.statusRepaired
                              : styles.statusUncertified,
                          ]}
                        >
                          <Text style={styles.statusBadgeText}>
                            {activeBenchmarkDetail?.research_metrics?.contractStatus?.toUpperCase() || 'UNCERTIFIED'}
                          </Text>
                        </View>
                      </View>
                    </View>

                    {/* Section 1: Research Quality Parameters */}
                    <Text style={styles.metricSectionTitle}>IEEE Research Evaluation Parameters</Text>
                    
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Groundedness / Faithfulness:</Text>
                      <Text style={[styles.metricValue, { color: '#34D399', fontWeight: 'bold' }]}>
                         {activeBenchmarkDetail?.research_metrics?.faithfulness !== null && activeBenchmarkDetail?.research_metrics?.faithfulness !== undefined
                           ? `${(activeBenchmarkDetail.research_metrics.faithfulness * 100).toFixed(1)}%`
                           : 'N/A'}
                      </Text>
                    </View>

                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Hallucination Rate:</Text>
                      <Text style={[styles.metricValue, { color: (activeBenchmarkDetail?.research_metrics?.hallucinationRate || 0) > 0 ? '#F87171' : '#6EE7B7' }]}>
                        {activeBenchmarkDetail?.research_metrics?.hallucinationRate !== null && activeBenchmarkDetail?.research_metrics?.hallucinationRate !== undefined
                           ? `${(activeBenchmarkDetail.research_metrics.hallucinationRate * 100).toFixed(1)}%`
                           : 'N/A'}
                      </Text>
                    </View>

                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Retrieval Evidence Coverage:</Text>
                      <Text style={styles.metricValue}>
                        {activeBenchmarkDetail?.research_metrics?.retrievalEvidenceCoverage !== null && activeBenchmarkDetail?.research_metrics?.retrievalEvidenceCoverage !== undefined
                           ? `${activeBenchmarkDetail.research_metrics.retrievalEvidenceCoverage.toFixed(1)}%`
                           : 'N/A'}
                      </Text>
                    </View>

                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Context Recall / Precision:</Text>
                      <Text style={styles.metricValue}>
                        {activeBenchmarkDetail?.research_metrics?.contextRecall !== null && activeBenchmarkDetail?.research_metrics?.contextRecall !== undefined
                          ? `${(activeBenchmarkDetail.research_metrics.contextRecall * 100).toFixed(1)}% / ${(activeBenchmarkDetail.research_metrics.contextPrecision! * 100).toFixed(1)}%`
                          : 'Labeled queries only'}
                      </Text>
                    </View>

                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Answer Correctness (Token F1):</Text>
                      <Text style={[styles.metricValue, { color: '#60A5FA' }]}>
                        {activeBenchmarkDetail?.research_metrics?.answerF1 !== null && activeBenchmarkDetail?.research_metrics?.answerF1 !== undefined
                          ? `${(activeBenchmarkDetail.research_metrics.answerF1 * 100).toFixed(1)}% (ROUGE-L: ${activeBenchmarkDetail.research_metrics.rougeL?.toFixed(3)})`
                          : 'Awaiting reference label'}
                      </Text>
                    </View>

                    {activeBenchmarkDetail?.research_metrics?.refusalCorrectness !== null && activeBenchmarkDetail?.research_metrics?.refusalCorrectness !== undefined && (
                      <View style={styles.metricRow}>
                        <Text style={styles.metricLabel}>Refusal Correctness:</Text>
                        <Text style={[styles.metricValue, { color: activeBenchmarkDetail.research_metrics.refusalCorrectness === 1 ? '#34D399' : '#F87171' }]}>
                          {(activeBenchmarkDetail.research_metrics.refusalCorrectness * 100).toFixed(0)}% ({activeBenchmarkDetail.research_metrics.modelAction})
                        </Text>
                      </View>
                    )}

                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Avoided LLM Inference:</Text>
                      <Text style={[styles.metricValue, { color: activeBenchmarkDetail?.inference_skipped ? '#F59E0B' : '#E2E8F0' }]}>
                        {activeBenchmarkDetail?.inference_skipped ? 'YES (LLM inference avoided; energy not measured)' : 'No (Full Generation)'}
                      </Text>
                    </View>

                    {activeBenchmarkDetail?.research_metrics && activeBenchmarkDetail.research_metrics.policy === 'mobcert-v1' && (
                      <>
                        <View style={styles.metricRow}>
                          <Text style={styles.metricLabel}>Demands Verified:</Text>
                          <Text style={styles.metricValue}>
                            {activeBenchmarkDetail.research_metrics.demandsSatisfied} / {activeBenchmarkDetail.research_metrics.demandsCount} satisfied
                            {activeBenchmarkDetail.research_metrics.demandsMissing > 0 ? ` (${activeBenchmarkDetail.research_metrics.demandsMissing} missing)` : ''}
                          </Text>
                        </View>
                        <View style={styles.metricRow}>
                          <Text style={styles.metricLabel}>Contradictions / Conflicts:</Text>
                          <Text style={[styles.metricValue, { color: activeBenchmarkDetail.research_metrics.conflictsDetected.length ? '#F87171' : '#6EE7B7' }]}>
                            {activeBenchmarkDetail.research_metrics.conflictsDetected.length
                              ? activeBenchmarkDetail.research_metrics.conflictsDetected.join(', ')
                              : 'None detected'}
                          </Text>
                        </View>
                      </>
                    )}

                    {/* Section 2: Latency & Hardware Throughput */}
                    <View style={styles.sectionDivider} />
                    <Text style={styles.metricSectionTitle}>Latency, TTFT & llama.cpp Execution</Text>
                    
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Total Warm Latency:</Text>
                      <Text style={[styles.metricValue, { color: '#6EE7B7', fontWeight: 'bold' }]}>
                        {activeBenchmarkDetail?.total_warm_latency_ms?.toFixed(1)} ms ({((activeBenchmarkDetail?.total_warm_latency_ms || 0) / 1000).toFixed(2)}s)
                      </Text>
                    </View>
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>TTFT (First Token):</Text>
                      <Text style={[styles.metricValue, { color: '#FCD34D' }]}>
                        {activeBenchmarkDetail?.ttft_ms ? `${activeBenchmarkDetail.ttft_ms.toFixed(1)} ms (${(activeBenchmarkDetail.ttft_ms / 1000).toFixed(2)}s)` : 'N/A'}
                      </Text>
                    </View>
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Prompt Prefill:</Text>
                      <Text style={styles.metricValue}>
                        {activeBenchmarkDetail?.prefill_ms?.toFixed(1)} ms ({activeBenchmarkDetail?.prompt_tokens} tokens @ {activeBenchmarkDetail?.prefill_tokens_per_second?.toFixed(1)} tok/s)
                      </Text>
                    </View>
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Token Decode:</Text>
                      <Text style={styles.metricValue}>
                        {activeBenchmarkDetail?.decode_ms?.toFixed(1)} ms ({activeBenchmarkDetail?.generated_tokens} tokens @ {activeBenchmarkDetail?.tokens_per_second?.toFixed(1)} tok/s)
                      </Text>
                    </View>
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Primary Bottleneck:</Text>
                      <Text style={[styles.metricValue, { color: '#FCD34D' }]}>
                        {activeBenchmarkDetail?.primary_bottleneck === 'not_applicable'
                          ? 'N/A (Inference skipped)'
                          : activeBenchmarkDetail?.primary_bottleneck === 'prompt_prefill' ? 'Prompt Prefill' : 'Token Decode'}
                      </Text>
                    </View>

                    {/* Section 3: RAG Retrieval Details */}
                    {activeBenchmarkDetail?.chat_mode === 'document' && (
                      <>
                        <View style={styles.sectionDivider} />
                        <Text style={styles.metricSectionTitle}>RAG Retrieval Details</Text>
                        <View style={styles.metricRow}>
                          <Text style={styles.metricLabel}>Retrieval Latency:</Text>
                          <Text style={styles.metricValue}>{activeBenchmarkDetail.rag_retrieval_ms?.toFixed(1)} ms</Text>
                        </View>
                        <View style={styles.metricRow}>
                          <Text style={styles.metricLabel}>Selected Chunks:</Text>
                          <Text style={styles.metricValue}>
                            {activeBenchmarkDetail.retrieved_chunk_count} kept of {activeBenchmarkDetail.candidate_chunk_count} candidates
                          </Text>
                        </View>
                        <View style={styles.metricRow}>
                          <Text style={styles.metricLabel}>Evidence Tokens / Budget:</Text>
                          <Text style={styles.metricValue}>
                            {activeBenchmarkDetail.rag?.contextTokens || 0} / {activeBenchmarkDetail.rag_applied_context_budget || 256}
                          </Text>
                        </View>
                        <View style={styles.metricRow}>
                          <Text style={styles.metricLabel}>Compression Ratio:</Text>
                          <Text style={styles.metricValue}>
                            {((activeBenchmarkDetail.rag_compression_ratio || 0) * 100).toFixed(1)}%
                          </Text>
                        </View>
                      </>
                    )}

                    {/* Section 4: Device Hardware Telemetry */}
                    <View style={styles.sectionDivider} />
                    <Text style={styles.metricSectionTitle}>Device Hardware Telemetry</Text>
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Threads & Mode:</Text>
                      <Text style={styles.metricValue}>
                        {activeBenchmarkDetail?.controller_mode?.toUpperCase()} ({activeBenchmarkDetail?.n_threads}T)
                      </Text>
                    </View>
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Available RAM:</Text>
                      <Text style={styles.metricValue}>
                        {activeBenchmarkDetail?.device_available_memory_mb ? `${activeBenchmarkDetail.device_available_memory_mb.toFixed(0)} MB` : 'N/A'}
                      </Text>
                    </View>
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Battery Temp:</Text>
                      <Text style={styles.metricValue}>
                        {activeBenchmarkDetail?.device_battery_temp_c ? `${activeBenchmarkDetail.device_battery_temp_c.toFixed(1)} °C` : 'N/A'}
                      </Text>
                    </View>
                    <View style={styles.metricRow}>
                      <Text style={styles.metricLabel}>Thermal Status:</Text>
                      <Text style={styles.metricValue}>{activeBenchmarkDetail?.device_thermal_status || 'normal'}</Text>
                    </View>

                    {/* Copy / Export Action */}
                    <RNTouchableOpacity
                      style={styles.copyJsonButton}
                      onPress={() => {
                        Alert.alert(
                          'Benchmark Record Saved',
                          `Full JSON report is saved at:\n/sdcard/Android/data/com.chat/files/inference_benchmark_latest.json\n\nRun history appended at:\n/sdcard/Android/data/com.chat/files/evaluation_runs_history.json`,
                          [{ text: 'OK' }]
                        );
                      }}
                      activeOpacity={0.7}
                    >
                      <Icon name="save" size={16} color="#FFFFFF" style={{ marginRight: 6 }} />
                      <Text style={styles.copyJsonButtonText}>View Report Path & Storage Info</Text>
                    </RNTouchableOpacity>
                  </ScrollView>
                </View>
              </View>
            </Modal>
          )}

          {isListening && (
            <View style={styles.listeningContainer}>
              <AudioWaveform />
              <Text style={styles.listeningText}>🎙️ Listening...</Text>
              <View style={styles.controls}>
                <RNTouchableOpacity onPress={stopListening} style={styles.controlButton}>
                  <Icon name="check-circle" size={28} color="#fff" />
                </RNTouchableOpacity>
                <RNTouchableOpacity onPress={cancelListening} style={styles.controlButton}>
                  <Icon name="cancel" size={28} color="#fff" />
                </RNTouchableOpacity>
              </View>
            </View>
          )}

          <View style={styles.inputWrapper}>
            <TextInput
              style={styles.input}
              value={inputText}
              onChangeText={setInputText}
              placeholder={chatMode === 'document' ? "Ask about your documents..." : "Type your message..."}
              placeholderTextColor="#888"
              editable={!!context && !isLoading && !isListening}
              multiline={true}
              scrollEnabled={true}
            />
            <View style={styles.inputActionRow}>
              <View style={styles.leftActions}>
                <RNTouchableOpacity
                  style={styles.addButton}
                  onPress={() => navigateTo('Documents')}
                >
                  <Icon name="add" size={22} color="#FFFFFF" />
                </RNTouchableOpacity>

                <RNTouchableOpacity
                  style={[
                    styles.actionPill,
                    chatMode === 'document' ? styles.actionPillActive : styles.actionPillInactive
                  ]}
                  onPress={() => handleToggleChange(chatMode === 'general')}
                >
                  <Icon
                    name={chatMode === 'document' ? "description" : "chat"}
                    size={14}
                    color={chatMode === 'document' ? "#FFFFFF" : "#9CA3AF"}
                  />
                  <Text
                    style={[
                      styles.actionPillText,
                      chatMode === 'document' ? styles.actionPillTextActive : styles.actionPillTextInactive
                    ]}
                  >
                    {chatMode === 'document' ? "Document" : "General"}
                  </Text>
                </RNTouchableOpacity>

                {chatMode === 'document' && (
                  <RNTouchableOpacity
                    style={[
                      styles.actionPill,
                      ragPolicy === 'mobcert-v1' ? styles.actionPillMobCert : styles.actionPillEbmRag
                    ]}
                    onPress={handleToggleRagPolicy}
                    activeOpacity={0.7}
                  >
                    <Icon
                      name={ragPolicy === 'mobcert-v1' ? "verified-user" : "bolt"}
                      size={14}
                      color={ragPolicy === 'mobcert-v1' ? "#10B981" : "#38BDF8"}
                    />
                    <Text
                      style={[
                        styles.actionPillText,
                        ragPolicy === 'mobcert-v1' ? styles.actionPillTextMobCert : styles.actionPillTextEbmRag
                      ]}
                    >
                      {ragPolicy === 'mobcert-v1' ? "MobCert-v1" : "EBM-RAG-v2"}
                    </Text>
                  </RNTouchableOpacity>
                )}

                <RNTouchableOpacity
                  style={[
                    styles.actionPill,
                    inferenceMode === 'adaptive'
                      ? styles.actionPillAdaptive
                      : inferenceMode === 'fixed'
                      ? styles.actionPillFixed
                      : styles.actionPillBaseline
                  ]}
                  onPress={() => setShowInferenceModal(true)}
                >
                  <Icon
                    name={inferenceMode === 'adaptive' ? "bolt" : inferenceMode === 'fixed' ? "tune" : "timer"}
                    size={14}
                    color={inferenceMode === 'adaptive' ? "#10B981" : inferenceMode === 'fixed' ? "#A78BFA" : "#9CA3AF"}
                  />
                  <Text
                    style={[
                      styles.actionPillText,
                      inferenceMode === 'adaptive'
                        ? styles.actionPillTextAdaptive
                        : inferenceMode === 'fixed'
                        ? styles.actionPillTextFixed
                        : styles.actionPillTextBaseline
                    ]}
                  >
                    {inferenceMode === 'adaptive' ? "Adaptive" : inferenceMode === 'fixed' ? `${fixedThreads}T Fixed` : "Baseline (4T)"}
                  </Text>
                </RNTouchableOpacity>
              </View>

              <View style={styles.rightActions}>
                <RNTouchableOpacity
                  style={styles.micButton}
                  onPress={startListening}
                  disabled={!context || isLoading || isListening}
                >
                  <Icon
                    name="mic"
                    size={20}
                    color={!context || isLoading || isListening ? '#6B7280' : '#FFFFFF'}
                  />
                </RNTouchableOpacity>

                {isLoading ? (
                  <RNTouchableOpacity
                    style={styles.stopButton}
                    onPress={handleStopGeneration}
                  >
                    <Icon name="stop" size={20} color="#FFFFFF" />
                  </RNTouchableOpacity>
                ) : (
                  <RNTouchableOpacity
                    style={[
                      styles.sendButtonCircle,
                      (!context || !inputText.trim() || isListening) && styles.sendButtonCircleDisabled
                    ]}
                    onPress={handleSend}
                    disabled={!context || !inputText.trim() || isListening}
                  >
                    <Icon
                      name="arrow-upward"
                      size={20}
                      color="#0B0F19"
                    />
                  </RNTouchableOpacity>
                )}
              </View>
            </View>
          </View>
        </KeyboardAvoidingView>
      </View>

      {/* Drawer Backdrop Overlay */}
      {isDrawerOpen && (
        <Animated.View
          style={[
            styles.drawerBackdrop,
            {
              opacity: drawerAnimation.interpolate({
                inputRange: [0, 1],
                outputRange: [0, 1],
              }),
            },
          ]}
        >
          <RNTouchableOpacity activeOpacity={1} style={{ flex: 1 }} onPress={closeDrawer} />
        </Animated.View>
      )}

      {/* Drawer Panel */}
      {isDrawerOpen && (
        <Animated.View
          style={[
            styles.drawerContainer,
            {
              transform: [
                {
                  translateX: drawerAnimation.interpolate({
                    inputRange: [0, 1],
                    outputRange: [-280, 0],
                  }),
                },
              ],
            },
          ]}
        >
          <View style={styles.drawerHeader}>
            <Text style={styles.drawerBrand}>Genix</Text>
          </View>

          <View style={styles.drawerNav}>
            <RNTouchableOpacity
              style={styles.drawerNavItem}
              onPress={() => {
                closeDrawer();
                setIsHistoryModalVisible(true);
              }}
            >
              <Icon name="assignment" size={22} color="#9CA3AF" style={styles.drawerNavIcon} />
              <Text style={styles.drawerNavText}>Task History</Text>
            </RNTouchableOpacity>

            <RNTouchableOpacity style={styles.drawerNavItem} onPress={() => navigateTo('Models')}>
              <Icon name="folder" size={22} color="#9CA3AF" style={styles.drawerNavIcon} />
              <Text style={styles.drawerNavText}>Models</Text>
            </RNTouchableOpacity>

            <RNTouchableOpacity style={styles.drawerNavItem} onPress={() => navigateTo('Documents')}>
              <Icon name="description" size={22} color="#9CA3AF" style={styles.drawerNavIcon} />
              <Text style={styles.drawerNavText}>Documents</Text>
            </RNTouchableOpacity>

            <RNTouchableOpacity style={styles.drawerNavItem} onPress={() => navigateTo('Settings')}>
              <Icon name="settings" size={22} color="#9CA3AF" style={styles.drawerNavIcon} />
              <Text style={styles.drawerNavText}>Settings</Text>
            </RNTouchableOpacity>
          </View>

          <View style={styles.drawerDivider} />

          <Text style={styles.drawerSectionHeader}>Recents</Text>

          <FlatList
            data={sessions}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <RNTouchableOpacity
                style={[
                  styles.drawerSessionItem,
                  item.id === activeSessionId && styles.drawerSessionItemActive,
                ]}
                onPress={() => selectSession(item.id)}
                onLongPress={() => deleteSession(item.id)}
              >
                <Icon
                  name="chat-bubble-outline"
                  size={16}
                  color={item.id === activeSessionId ? '#FFFFFF' : '#9CA3AF'}
                  style={styles.drawerSessionIcon}
                />
                <Text
                  style={[
                    styles.drawerSessionText,
                    item.id === activeSessionId && styles.drawerSessionTextActive,
                  ]}
                  numberOfLines={1}
                >
                  {item.title}
                </Text>
              </RNTouchableOpacity>
            )}
            style={styles.drawerSessionsList}
            contentContainerStyle={{ paddingBottom: 20 }}
            keyboardShouldPersistTaps="handled"
          />

          <View style={styles.drawerFooter}>
            <RNTouchableOpacity style={styles.drawerProfileBtn} onPress={() => navigateTo('Settings')}>
              <View style={[styles.drawerAvatar, { backgroundColor: activeProfile?.avatarColor || '#8B5CF6' }]}>
                <Text style={styles.drawerAvatarText}>
                  {activeProfile?.name?.charAt(0).toUpperCase() || 'A'}
                </Text>
              </View>
            </RNTouchableOpacity>

            <RNTouchableOpacity style={styles.drawerNewChatBtn} onPress={startNewChat}>
              <Icon name="add" size={18} color="#0B0F19" />
              <Text style={styles.drawerNewChatBtnText}>New chat</Text>
            </RNTouchableOpacity>
          </View>
        </Animated.View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B0F19',
  },
  header: {
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 8,
    flexDirection: 'column',
    alignItems: 'stretch',
  },
  headerTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
  },
  headerLeft: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'center',
  },
  headerCenter: {
    flex: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerRightSide: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  headerBottomRow: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'center',
    paddingLeft: 4,
    marginTop: 4,
  },
  headerButton: {
    padding: 8,
    marginLeft: 4,
  },
  iconImage: {
    width: 24,
    height: 24,
    resizeMode: 'contain',
    tintColor: '#E5E7EB',
  },
  modelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    maxWidth: '100%',
    paddingHorizontal: 8,
  },
  modelName: {
    fontSize: 15.5,
    color: '#9CA3AF',
    fontFamily: 'sans-serif',
    textAlign: 'center',
    maxWidth: 110,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 8,
    fontSize: 17,
    color: '#FFFFFF',
    fontFamily: 'sans-serif',
  },
  welcomeContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: 120, // Prevents overlap with sticky absolute input
  },
  maskContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  welcomeText: {
    fontSize: 28,
    fontWeight: '600',
    color: '#E5E7EB',
    fontFamily: 'sans-serif',
    textAlign: 'center',
  },
  errorText: {
    fontSize: 18,
    color: '#FF6F91',
    fontFamily: 'sans-serif',
    textAlign: 'center',
    marginBottom: 20,
    paddingHorizontal: 20,
  },
  selectModelButton: {
    backgroundColor: '#8B5CF6',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 24,
  },
  selectModelButtonText: {
    fontSize: 16,
    color: '#F9FAFB',
    fontFamily: 'sans-serif',
  },
  messagesContainer: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 24, // simplified as input floats
  },
  messageContainer: {
    marginBottom: 14,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#7F15EA',
    paddingHorizontal: 18,
    paddingVertical: 14,
    borderRadius: 20,
    borderBottomRightRadius: 0,
    maxWidth: '80%',
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: 'transparent',
    paddingHorizontal: 0,
    paddingVertical: 0,
    maxWidth: '90%',
  },
  loadingMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#1E293B',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 20,
    borderBottomLeftRadius: 0,
    maxWidth: '80%',
  },
  messageRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  textContainer: {
    flexShrink: 1,
  },
  messageText: {
    fontSize: 15,
    color: '#E5E7EB',
    lineHeight: 24,
    fontFamily: 'sans-serif',
  },
  messageTime: {
    flexDirection: 'row',
    marginTop: 4,
    marginLeft: 24,
  },
  messageTimeText: {
    fontSize: 13,
    color: '#9CA3AF',
    fontFamily: 'sans-serif',
  },
  responseTime: {
    fontSize: 13,
    color: '#9CA3AF',
    fontFamily: 'sans-serif',
  },
  inputWrapper: {
    backgroundColor: '#1E293B',
    borderRadius: 24,
    paddingHorizontal: 16,
    paddingVertical: 10,
    marginHorizontal: 16,
    marginBottom: Platform.OS === 'ios' ? 10 : 20,
    borderWidth: 1,
    borderColor: '#374151',
  },
  switchContainer: {
    width: 64,
    height: 30,
    borderRadius: 15,
    padding: 3,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    position: 'relative',
  },
  switchKnob: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#FFFFFF',
    position: 'absolute',
    top: 3,
    left: 0,
  },
  switchText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#FFFFFF',
    position: 'absolute',
    fontFamily: 'sans-serif',
    top: 6,
  },
  switchTextOn: {
    left: 10,
  },
  switchTextOff: {
    right: 8,
  },
  input: {
    color: '#F9FAFB',
    fontSize: 16.5,
    minHeight: 45,
    maxHeight: 120,
    fontFamily: 'sans-serif',
    paddingVertical: Platform.OS === 'ios' ? 12 : 8,
    textAlignVertical: 'top',
  },
  inputActionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 6,
    paddingTop: 6,
    borderTopWidth: 0.5,
    borderColor: '#374151',
  },
  leftActions: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  rightActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  addButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#374151',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  actionPill: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 16,
    marginRight: 8,
  },
  actionPillActive: {
    backgroundColor: '#8B5CF6',
  },
  actionPillInactive: {
    backgroundColor: '#374151',
  },
  actionPillText: {
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 4,
  },
  actionPillTextActive: {
    color: '#FFFFFF',
  },
  actionPillTextInactive: {
    color: '#9CA3AF',
  },
  modelPill: {
    backgroundColor: '#374151',
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 16,
    maxWidth: 100,
  },
  modelPillText: {
    fontSize: 12,
    color: '#E5E7EB',
    fontWeight: '500',
  },
  micButton: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: '#374151',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  sendButtonCircle: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonCircleDisabled: {
    backgroundColor: '#4B5563',
  },
  stopButton: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: '#EF4444',
    justifyContent: 'center',
    alignItems: 'center',
  },
  listeningContainer: {
    position: 'absolute',
    bottom: 120,
    left: 20,
    right: 20,
    backgroundColor: '#111827',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 5,
    zIndex: 10,
  },
  listeningText: {
    marginTop: 10,
    fontSize: 17,
    color: '#8B5CF6',
    fontWeight: '500',
    fontFamily: 'Inter',
  },
  controls: {
    flexDirection: 'row',
    marginTop: 16,
  },
  controlButton: {
    backgroundColor: '#8B5CF6',
    borderRadius: 24,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginHorizontal: 8,
  },
  botIcon: {
    width: 18,
    height: 18,
    marginRight: 6,
    resizeMode: 'contain',
    marginTop: 4,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContainer: {
    backgroundColor: '#1F2937',
    borderRadius: 16,
    width: '90%',
    maxHeight: '70%',
    padding: 20,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 20,
    color: '#E5E7EB',
    fontFamily: 'sans-serif',
  },
  taskList: {
    paddingBottom: 16,
  },
  taskItem: {
    backgroundColor: '#2D3748',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  taskType: {
    fontSize: 14,
    color: '#8B5CF6',
    fontWeight: 'bold',
    fontFamily: 'sans-serif',
    marginBottom: 4,
  },
  taskText: {
    fontSize: 16,
    color: '#E5E7EB',
    fontFamily: 'sans-serif',
  },
  taskDateTime: {
    fontSize: 13,
    color: '#9CA3AF',
    marginTop: 4,
    fontFamily: 'sans-serif',
  },
  emptyText: {
    fontSize: 16,
    color: '#9CA3AF',
    textAlign: 'center',
    marginTop: 20,
    fontFamily: 'sans-serif',
  },
  clearButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 12,
    marginRight: 10,
  },
  clearButtonText: {
    fontSize: 17,
    color: '#ff6363',
    fontFamily: 'sans-serif',
    fontWeight: '600',
  },
  // Custom side drawer styles
  drawerBackdrop: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.6)',
    zIndex: 1000,
  },
  drawerContainer: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    width: 280,
    backgroundColor: '#0B0F19',
    borderRightWidth: 1,
    borderColor: '#1F2937',
    zIndex: 1001,
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
    flexDirection: 'column',
  },
  drawerHeader: {
    paddingHorizontal: 20,
    paddingTop: 35,
    marginBottom: 26,
  },
  drawerBrand: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#F9FAFB',
    fontFamily: 'serif',
  },
  drawerNav: {
    paddingHorizontal: 10,
    marginBottom: 8,
  },
  drawerNavItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 12,
    borderRadius: 8,
    marginBottom: 4,
  },
  drawerNavIcon: {
    marginRight: 12,
  },
  drawerNavText: {
    fontSize: 16,
    color: '#E5E7EB',
    fontWeight: '500',
  },
  drawerDivider: {
    height: 1,
    backgroundColor: '#1F2937',
    marginHorizontal: 20,
    marginVertical: 12,
  },
  drawerSectionHeader: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#6B7280',
    textTransform: 'uppercase',
    paddingHorizontal: 22,
    marginBottom: 8,
    letterSpacing: 1,
  },
  drawerSessionsList: {
    flex: 1,
  },
  drawerSessionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 22,
    borderRadius: 8,
    marginHorizontal: 10,
    marginBottom: 2,
  },
  drawerSessionItemActive: {
    backgroundColor: '#1E293B',
  },
  drawerSessionIcon: {
    marginRight: 10,
  },
  drawerSessionText: {
    fontSize: 14.5,
    color: '#9CA3AF',
    flex: 1,
  },
  drawerSessionTextActive: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
  drawerFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 12,
    borderTopWidth: 1,
    borderColor: '#1F2937',
    paddingBottom: Platform.OS === 'ios' ? 30 : 20,
  },
  drawerProfileBtn: {
    padding: 2,
  },
  drawerAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  drawerAvatarText: {
    fontSize: 18,
    color: '#111827',
    fontWeight: 'bold',
  },
  drawerNewChatBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 24,
  },
  drawerNewChatBtnText: {
    fontSize: 14,
    color: '#0B0F19',
    fontWeight: '600',
    marginLeft: 4,
  },
  // Ingest modal overlay styles
  ingestModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  ingestModalContent: {
    backgroundColor: '#1E293B',
    padding: 24,
    borderRadius: 16,
    alignItems: 'center',
    width: '80%',
  },
  ingestModalText: {
    marginTop: 16,
    color: '#E5E7EB',
    fontSize: 16,
    textAlign: 'center',
    fontWeight: '500',
  },
  headerTitleText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#E5E7EB',
    fontFamily: 'serif',
  },
  menuIconContainer: {
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'flex-start',
  },
  menuLine: {
    height: 2.2,
    backgroundColor: '#E5E7EB',
    borderRadius: 1,
    marginVertical: 2.5,
  },
  actionPillAdaptive: {
    backgroundColor: '#064E3B',
    borderColor: '#059669',
  },
  actionPillFixed: {
    backgroundColor: '#2E1065',
    borderColor: '#7C3AED',
  },
  actionPillBaseline: {
    backgroundColor: '#1F2937',
    borderColor: '#374151',
  },
  actionPillTextAdaptive: {
    color: '#10B981',
    fontWeight: '600',
  },
  actionPillTextFixed: {
    color: '#C4B5FD',
    fontWeight: '600',
  },
  actionPillTextBaseline: {
    color: '#9CA3AF',
    fontWeight: '500',
  },
  benchmarkPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1E293B',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    marginTop: 4,
    marginLeft: 24,
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#334155',
  },
  benchmarkPillText: {
    fontSize: 11,
    color: '#CBD5E1',
    fontFamily: 'monospace',
  },
  inferenceModalContainer: {
    backgroundColor: '#101626',
    borderRadius: 16,
    padding: 20,
    width: '92%',
    maxWidth: 420,
    borderWidth: 1,
    borderColor: '#1F2937',
  },
  inferenceSubtitle: {
    fontSize: 13,
    color: '#9CA3AF',
    marginBottom: 16,
    lineHeight: 18,
  },
  modeOptionCard: {
    backgroundColor: '#1E293B',
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1.5,
    borderColor: '#334155',
  },
  modeOptionCardActive: {
    borderColor: '#10B981',
    backgroundColor: '#0F2922',
  },
  modeOptionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  modeOptionTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginLeft: 8,
  },
  modeOptionDesc: {
    fontSize: 12,
    color: '#94A3B8',
    lineHeight: 17,
  },
  fixedThreadsSection: {
    marginTop: 6,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#1F2937',
  },
  fixedThreadsTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#D1D5DB',
    marginBottom: 10,
  },
  threadButtonsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  threadButton: {
    flex: 1,
    marginHorizontal: 3,
    backgroundColor: '#1E293B',
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155',
  },
  threadButtonActive: {
    backgroundColor: '#6D28D9',
    borderColor: '#A78BFA',
  },
  threadButtonText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#9CA3AF',
  },
  threadButtonTextActive: {
    color: '#FFFFFF',
  },
  benchmarkDetailContainer: {
    backgroundColor: '#0F172A',
    borderRadius: 16,
    padding: 20,
    width: '92%',
    maxWidth: 440,
    borderWidth: 1,
    borderColor: '#334155',
  },
  sectionDivider: {
    height: 1,
    backgroundColor: '#1E293B',
    marginVertical: 10,
  },
  metricSectionTitle: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#A78BFA',
    marginBottom: 8,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 4,
  },
  metricLabel: {
    fontSize: 12,
    color: '#94A3B8',
  },
  metricValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#E2E8F0',
    textAlign: 'right',
  },
  researchBanner: {
    backgroundColor: '#1E293B',
    borderRadius: 10,
    padding: 10,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  researchPolicyTitle: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#F8FAFC',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  statusBadgeText: {
    fontSize: 10,
    fontWeight: 'bold',
  },
  statusCertified: {
    backgroundColor: 'rgba(16, 185, 129, 0.2)',
    borderColor: '#10B981',
    borderWidth: 1,
  },
  statusAbstained: {
    backgroundColor: 'rgba(245, 158, 11, 0.2)',
    borderColor: '#F59E0B',
    borderWidth: 1,
  },
  statusRepaired: {
    backgroundColor: 'rgba(6, 182, 212, 0.2)',
    borderColor: '#06B6D4',
    borderWidth: 1,
  },
  statusUncertified: {
    backgroundColor: 'rgba(156, 163, 175, 0.2)',
    borderColor: '#6B7280',
    borderWidth: 1,
  },
  benchmarkPillMobCert: {
    borderColor: '#059669',
    backgroundColor: '#064E3B',
  },
  benchmarkPillEbmRag: {
    borderColor: '#0284C7',
    backgroundColor: '#0C4A6E',
  },
  actionPillMobCert: {
    backgroundColor: '#064E3B',
    borderColor: '#059669',
  },
  actionPillEbmRag: {
    backgroundColor: '#0C4A6E',
    borderColor: '#0284C7',
  },
  actionPillTextMobCert: {
    color: '#10B981',
    fontWeight: '600',
  },
  actionPillTextEbmRag: {
    color: '#38BDF8',
    fontWeight: '600',
  },
  copyJsonButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#374151',
    paddingVertical: 10,
    borderRadius: 8,
    marginTop: 14,
    marginBottom: 6,
  },
  copyJsonButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#FFFFFF',
  },
});

export default ChatScreen;
