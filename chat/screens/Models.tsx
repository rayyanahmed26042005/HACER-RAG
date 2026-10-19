import React, { useState, useEffect } from 'react';
import { View, FlatList, Alert, StyleSheet, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Button, Card, ProgressBar, Text, IconButton } from 'react-native-paper';
import RNFS from 'react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';

type Model = {
  id: string;
  name: string;
  size: number;
  requiredRAM: number;
  downloadUrl: string;
  localPath: string | null;
  isDownloaded: boolean;
  isDownloading: boolean;
  progress: number;
  description: string;
};
import { useProfile } from '../utils/ProfileContext';
import { profileKey } from '../utils/profileManager';
import { downloadManager } from '../utils/downloadManager';

const MODELS_DIR = RNFS.ExternalDirectoryPath + '/models';
// Removed global SELECTED_MODEL_KEY
const EMBEDDING_MODEL_ID = 'all-minilm-l6-v2-q4_k_m';
// Removed global EMBEDDING_MODEL_KEY

const initialModels: Model[] = [
  {
    id: 'gemma-2-2b-it-Q4_K_M',
    name: 'Gemma 2 2B (Recommended)',
    size: 1434085216, // actual file size in bytes (~1.43 GB)
    requiredRAM: 3, // Math.ceil(1434085216 / 500000000) = 3
    downloadUrl: 'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Instruction-tuned Gemma 2B for chat, reasoning, and general tasks under low RAM.'
  },
  {
    id: 'tinyllama-1.1b',
    name: 'TinyLlama 1.1B',
    size: 550000000,
    requiredRAM: 2,
    downloadUrl: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Chat, lightweight tasks'
  },
  {
    id: 'stablelm-2-zephyr-1_6b-Q4_K_M',
    name: 'StableLM 2 Zephyr 1.6B',
    size: 1713507840, // ~1.60 GB
    requiredRAM: 4, // Math.ceil(1713507840 / 500000000)
    downloadUrl: 'https://huggingface.co/brittlewis12/stablelm-2-zephyr-1_6b-GGUF/resolve/main/stablelm-2-zephyr-1_6b.Q4_K_M.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Chat and reasoning model. Blend of Stability AI’s StableLM and Zephyr fine-tuning.'
  },
  {
    id: 'DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M',
    name: 'DeepSeek R1 1.5B',
    size: 1572134400, // ~1.46 GB
    requiredRAM: 4, // Math.ceil(1572134400 / 500000000)
    downloadUrl: 'https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B.Q4_K_M.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Chat & reasoning optimized. Distilled from DeepSeek Qwen model.'
  },
  {
    id: 'phi-2-Q4_K_M',
    name: 'Phi-2',
    size: 1426854400, // ~1.33 GB
    requiredRAM: 3, // Math.ceil(1426854400 / 500000000)
    downloadUrl: 'https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'General-purpose reasoning & text. Trained by Microsoft.'
  },
  {
    id: 'deepseek-coder-1.3b-instruct',
    name: 'DeepSeek Coder 1.3B',
    size: 853000000, // Approx. for Q4_K_M
    requiredRAM: 2, // Math.ceil(853000000 / 500000000)
    downloadUrl: 'https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q4_K_M.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Code generation and instruction following. Optimized for programming tasks.'
  },
  {
    id: 'OpenGPT-3-Q5_K_S',
    name: 'OpenGPT-3',
    size: 1711144960, // ~1.59 GB
    requiredRAM: 4, // Math.ceil(1711144960 / 500000000)
    downloadUrl: 'https://huggingface.co/mradermacher/OpenGPT-3-GGUF/resolve/main/OpenGPT-3.Q5_K_S.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Compact GPT-3-style model. Great for general chat and logic.'
  },
  {
    id: 'Phi-3.5-mini-instruct.Q4_K_M',
    name: 'Phi-3.5 Mini',
    size: 2393232608,
    requiredRAM: 5, // Math.ceil(2393232608 / 500000000) = 5
    downloadUrl: 'https://huggingface.co/MaziyarPanahi/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct.Q4_K_M.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Reasoning (code & math). Multilingual'
  },
  {
    id: 'qwen2.5-1.5b-instruct-q8_0',
    name: 'Qwen 2.5 1.5B',
    size: 1894532128,
    requiredRAM: 4, // Math.ceil(1894532128 / 500000000) = 4
    downloadUrl: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Instruction following, Role-play, Multilingual'
  },
  {
    id: 'qwen2.5-3b-instruct-q5_k_m',
    name: 'Qwen 2.5 3B',
    size: 2438740384,
    requiredRAM: 5, // Math.ceil(2438740384 / 500000000) = 5
    downloadUrl: 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q5_k_m.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Instructions, Role-play, Multilingual'
  },
  {
    id: 'llama-3.2-1b-instruct-q8_0',
    name: 'Llama 3.2 1B',
    size: 1321079200,
    requiredRAM: 3, // Math.ceil(1321079200 / 500000000) = 3
    downloadUrl: 'https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/resolve/main/llama-3.2-1b-instruct-q8_0.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Instruction following, Summarization, Rewriting'
  },
  {
    id: 'Llama-3.2-3B-Instruct-Q6_K',
    name: 'Llama 3.2 3B',
    size: 2643853856,
    requiredRAM: 6, // Math.ceil(2643853856 / 500000000) = 6
    downloadUrl: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Instruction following, Summarization, Rewriting'
  },
  {
    id: 'all-minilm-l6-v2-q4_k_m',
    name: 'MiniLM L6 v2 (Embedding Model)',
    size: 45000000,
    requiredRAM: 1,
    downloadUrl: 'https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf',
    localPath: null,
    isDownloaded: false,
    isDownloading: false,
    progress: 0,
    description: 'Required for document search (RAG). Semantic embedding model. Download this to use the Documents feature.',
  },
];
const ModelsScreen = ({ navigation }: { navigation: any }) => {
  const [models, setModels] = useState<Model[]>(initialModels);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeDownload, setActiveDownload] = useState<string | null>(null);
  const activeDownloadRef = React.useRef<string | null>(null);
  const { profileId } = useProfile();

  const SELECTED_MODEL_KEY = profileId ? profileKey(profileId, '_selected_model') : 'selected_model';
  const EMBEDDING_MODEL_KEY = profileId ? profileKey(profileId, '_embedding_model') : 'embedding_model_id';
  const MODELS_KEY = profileId ? profileKey(profileId, '_models') : 'models';

  // Load saved models and selection
  const getExpectedModelPath = (model: Model) => {
    const ext = model.downloadUrl.split('.').pop()?.split('?')[0] || 'gguf';
    return `${MODELS_DIR}/${model.id}.${ext}`;
  };

  const verifyModelFiles = async (modelList: Model[]) => {
    return Promise.all(
      modelList.map(async model => {
        const expectedPath = model.localPath || getExpectedModelPath(model);
        const exists = await RNFS.exists(expectedPath);
        return {
          ...model,
          localPath: exists ? expectedPath : model.localPath,
          isDownloaded: exists,
          progress: exists ? 100 : (model.isDownloading ? model.progress : 0),
          isDownloading: exists ? false : model.isDownloading,
        };
      })
    );
  };

  const saveModels = async (updatedModels: Model[]) => {
    await AsyncStorage.setItem(MODELS_KEY, JSON.stringify(updatedModels));
  };

  // Load saved models and selection
  useEffect(() => {
    const loadModels = async () => {
      try {
        await RNFS.mkdir(MODELS_DIR);
        const [savedModels, savedSelected] = await Promise.all([
          AsyncStorage.getItem(MODELS_KEY),
          AsyncStorage.getItem(SELECTED_MODEL_KEY)
        ]);

        let modelList = initialModels;
        if (savedModels) {
          try {
            const savedList = JSON.parse(savedModels);
            if (Array.isArray(savedList)) {
              modelList = initialModels.map(initModel => {
                const savedModel = savedList.find(s => s.id === initModel.id);
                if (savedModel) {
                  return {
                    ...initModel,
                    localPath: savedModel.localPath,
                    isDownloaded: savedModel.isDownloaded,
                    isDownloading: savedModel.isDownloading,
                    progress: savedModel.progress,
                  };
                }
                return initModel;
              });
            }
          } catch (e) {
            console.error('Failed to parse saved models:', e);
          }
        }
        const verifiedModels = await verifyModelFiles(modelList);

        setModels(verifiedModels);
        await saveModels(verifiedModels);
        setSelectedModelId(savedSelected);
      } catch (error) {
        Alert.alert('Error', 'Failed to load models');
      } finally {
        setIsLoading(false);
      }
    };

    loadModels();
  }, [MODELS_KEY]);

  // Listen to background downloads progress
  useEffect(() => {
    const listener = (data: any) => {
      setModels(prev =>
        prev.map(m => {
          if (m.id !== data.modelId) return m;
          const isDone = data.progress === 100 && !data.isDownloading;
          return {
            ...m,
            progress: data.progress,
            isDownloading: data.isDownloading,
            isDownloaded: isDone ? true : m.isDownloaded,
            localPath: isDone ? (m.localPath || getExpectedModelPath(m)) : m.localPath,
          };
        })
      );
      if (data.isDownloading) {
        setActiveDownload(data.modelId);
        activeDownloadRef.current = data.modelId;
      } else {
        setActiveDownload(prev => prev === data.modelId ? null : prev);
        if (activeDownloadRef.current === data.modelId) {
          activeDownloadRef.current = null;
        }
        // Re-verify files when downloading stops to ensure completeness shows up
        AsyncStorage.getItem(MODELS_KEY).then(saved => {
          const list = saved ? JSON.parse(saved) : initialModels;
          verifyModelFiles(list).then(verified => {
            setModels(verified);
            saveModels(verified);
          });
        });
      }
    };

    downloadManager.registerListener(listener);

    return () => {
      downloadManager.unregisterListener(listener);
    };
  }, [MODELS_KEY]);

  const handleDownloadError = async (modelId: string, error?: any) => {
    setModels(prev => {
      const model = prev.find(m => m.id === modelId);
      if (model?.localPath) {
        // Fire & forget unlink
        RNFS.exists(model.localPath).then(exists => {
          if (exists) RNFS.unlink(model.localPath!);
        }).catch(err => console.error('Failed to delete partial download:', err));
      } else if (model) {
        const fallbackPath = getExpectedModelPath(model);
        RNFS.exists(fallbackPath).then(exists => {
          if (exists) RNFS.unlink(fallbackPath);
        }).catch(err => console.error('Failed to delete partial download:', err));
      }

      const updatedModels = prev.map(m =>
        m.id === modelId ? { ...m, isDownloading: false, progress: 0, localPath: null, isDownloaded: false } : m
      );
      saveModels(updatedModels);
      return updatedModels;
    });

    setActiveDownload(null);
    activeDownloadRef.current = null;
    if (error?.message !== 'Download has been aborted') {
       Alert.alert('Error', 'Download failed or was interrupted\n' + (error?.message || ''));
    }
  };

  const handleDownload = async (modelId: string) => {
    if (activeDownloadRef.current) {
      return;
    }

    const model = models.find(m => m.id === modelId)!;
    activeDownloadRef.current = modelId;
    setActiveDownload(modelId);

    downloadManager.startDownload(
      modelId,
      model.downloadUrl,
      model.size,
      MODELS_KEY,
      () => {
        activeDownloadRef.current = null;
        AsyncStorage.getItem(MODELS_KEY).then(saved => {
          const list = saved ? JSON.parse(saved) : initialModels;
          verifyModelFiles(list).then(verified => {
            setModels(verified);
            saveModels(verified);
          });
        });
      }
    );
  };

  const handleDelete = async (modelId: string) => {
    try {
      const model = models.find(m => m.id === modelId)!;
      if (model.isDownloading) {
        Alert.alert('Cannot delete while downloading');
        return;
      }

      const filePath = model.localPath || getExpectedModelPath(model);
      const exists = await RNFS.exists(filePath);
      if (exists) {
        await RNFS.unlink(filePath);
      }

      let latestUpdatedModels: Model[] = [];
      setModels(prev => {
        latestUpdatedModels = prev.map(m =>
          m.id === modelId ? {
            ...m,
            isDownloaded: false,
            localPath: null,
            progress: 0,
            isDownloading: false,
          } : m
        );
        return latestUpdatedModels;
      });

      await saveModels(latestUpdatedModels);

      if (selectedModelId === modelId) {
        await AsyncStorage.removeItem(SELECTED_MODEL_KEY);
        setSelectedModelId(null);
      }
    } catch (error: any) {
      Alert.alert('Error', 'Failed to delete model');
    }
  };

  const selectModel = async (modelId: string) => {
    // If the user selected the embedding model, do NOT overwrite the chat
    // model selection key. Store it separately and bail out early.
    if (modelId === EMBEDDING_MODEL_ID) {
      await AsyncStorage.setItem(EMBEDDING_MODEL_KEY, modelId);
      Alert.alert(
        'Embedding Model Ready',
        'This model will be used for document search (RAG). It cannot be used as a chat model.'
      );
      return;
    }

    // Only save as the selected chat model if it is an actual LLM
    setSelectedModelId(modelId);
    await AsyncStorage.setItem(SELECTED_MODEL_KEY, modelId);
    navigation.navigate('Chat', { selectedModelId: modelId });
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#8B5CF6" />
        <Text style={styles.loadingText}>Loading models...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <IconButton
          icon="arrow-left"
          iconColor="#9CA3AF"
          size={24}
          onPress={() => navigation.goBack()}
          style={styles.backButton}
        />
        <Text style={styles.headerTitle}>MODELS</Text>
        <View style={{ width: 48 }} />
      </View>
      <FlatList
        data={models}
        renderItem={({ item }) => (
          <Card style={[
            styles.card,
            selectedModelId === item.id && styles.selectedCard
          ]}>
            <Card.Title
              title={item.name}
              titleStyle={styles.cardTitle}
              subtitle={`${(item.size / 1024 / 1024).toFixed(2)}MB | ${item.requiredRAM}GB RAM${item.id === EMBEDDING_MODEL_ID ? ' · EMBEDDING' : ''}`}
              subtitleStyle={styles.cardSubtitle}
              right={() => selectedModelId === item.id && (
                <IconButton icon="check" iconColor="#8B5CF6" size={24} />
              )}
            />
            <Card.Content>
              <Text style={styles.description}>{item.description}</Text>
            </Card.Content>
            <Card.Content>
              {item.isDownloading && (
                <>
                  <ProgressBar
                    progress={item.progress / 100}
                    color="#8B5CF6"
                  />
                  <Text style={styles.progressText}>{item.progress.toFixed(0)}%</Text>
                </>
              )}
            </Card.Content>
            <Card.Actions style={styles.cardActions}>
              {item.isDownloaded ? (
                <>
                  <Button
                    mode="outlined"
                    onPress={() => handleDelete(item.id)}
                    style={styles.button}
                    labelStyle={styles.buttonLabel}
                    disabled={false}
                  >
                    Delete
                  </Button>
                  <Button
                    mode="contained"
                    onPress={() => selectModel(item.id)}
                    style={styles.button}
                    labelStyle={styles.buttonLabel}
                    disabled={false}
                  >
                    {selectedModelId === item.id ? 'Selected' : 'Select'}
                  </Button>
                </>
              ) : item.isDownloading ? (
                <>
                  <Button
                    mode="outlined"
                    onPress={async () => {
                      const model = models.find(m => m.id === item.id);
                      if (model) {
                        downloadManager.stopDownload(
                          item.id,
                          model.downloadUrl,
                          MODELS_KEY,
                          () => {
                            activeDownloadRef.current = null;
                            AsyncStorage.getItem(MODELS_KEY).then(saved => {
                              if (saved) {
                                const list = JSON.parse(saved);
                                verifyModelFiles(list).then(verified => setModels(verified));
                              }
                            });
                          }
                        );
                      }
                    }}
                    style={styles.cancelButton}
                    labelStyle={styles.cancelButtonLabel}
                  >
                    Cancel
                  </Button>
                  <Button
                    mode="contained"
                    style={styles.button}
                    labelStyle={styles.buttonLabel}
                    loading={true}
                    disabled={true}
                  >
                    Downloading
                  </Button>
                </>
              ) : (
                <Button
                  mode="contained"
                  onPress={() => handleDownload(item.id)}
                  style={styles.button}
                  labelStyle={styles.buttonLabel}
                  disabled={!!activeDownload}
                >
                  Download
                </Button>
              )}
            </Card.Actions>
          </Card>
        )}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContainer}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B0F19',
  },
  header: {
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#1F2937',
    backgroundColor: '#101626',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  backButton: {
    margin: 0,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'grey',
    fontFamily: 'sans-serif',
  },
  listContainer: {
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  card: {
    backgroundColor: '#1E293B',
    marginBottom: 16,
    borderRadius: 12,
  },
  selectedCard: {
    borderWidth: 2,
    borderColor: '#8B5CF6',
  },
  cardTitle: {
    fontSize: 16,
    color: '#E5E7EB',
    fontFamily: 'sans-serif',
  },
  cardSubtitle: {
    fontSize: 14,
    color: '#9CA3AF',
    fontFamily: 'sans-serif',
  },
  description: {
    fontSize: 14,
    color: '#9CA3AF',
    fontFamily: 'sans-serif',
    marginTop: 4,
  },
  progressText: {
    fontSize: 14,
    color: '#E5E7EB',
    fontFamily: 'sans-serif',
    marginTop: 4,
  },
  button: {
    marginLeft: 8,
    borderRadius: 24,
    backgroundColor: '#8B5CF6',
    borderColor: '#8B5CF6',
  },
  buttonLabel: {
    fontSize: 14,
    color: '#F9FAFB',
    fontFamily: 'sans-serif',
  },
  cancelButton: {
    marginLeft: 8,
    borderRadius: 24,
    borderColor: '#EF4444', 
  },
  cancelButtonLabel: {
    fontSize: 14,
    color: '#EF4444',
    fontFamily: 'sans-serif',
  },
  cardActions: {
    justifyContent: 'flex-end',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0B0F19',
  },
  loadingText: {
    marginTop: 8,
    fontSize: 17,
    color: '#E5E7EB',
    fontFamily: 'sans-serif',
  },
});

export default ModelsScreen;