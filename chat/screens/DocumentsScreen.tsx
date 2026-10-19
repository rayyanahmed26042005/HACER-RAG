// chat/screens/DocumentsScreen.tsx

import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { pick, types, isErrorWithCode, errorCodes } from '@react-native-documents/picker';
import RNFS from 'react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { ingestTextDoc, removeDoc } from '../utils/rag/ragPipeline';
import { getIndexSize, getIndexedDocuments, loadIndex } from '../utils/rag/vectorStore';
import { useProfile } from '../utils/ProfileContext';
import { profileKey } from '../utils/profileManager';
import Icon from 'react-native-vector-icons/MaterialIcons';

interface DocMeta {
  id: string;
  name: string;
  addedAt: string;
  chunkCount: number;
}

const DocumentsScreen = ({ navigation }: { navigation: any }) => {
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState('');
  const [indexSize, setIndexSize] = useState(0);
  const [embeddingReady, setEmbeddingReady] = useState(false);
  const [isModelDownloaded, setIsModelDownloaded] = useState(false);
  const [isCheckingModel, setIsCheckingModel] = useState(true);
  const { profileId } = useProfile();
  const activeProfile = useRef(profileId);
  activeProfile.current = profileId;
  const ingestionActive = useRef(false);
  useEffect(() => () => { activeProfile.current = null; }, []);

  // Bug #12 fix: compute key dynamically so it never goes stale
  const getDocsMetaKey = useCallback(() => {
    return profileId ? profileKey(profileId, '_docs_meta') : 'rag_docs_metadata';
  }, [profileId]);

  const checkEmbeddingModelStatus = useCallback(async () => {
    if (!profileId) return;
    try {
      setIsCheckingModel(true);
      const selectedModel = await AsyncStorage.getItem(profileKey(profileId, '_embedding_model')) || 'all-minilm-l6-v2-q4_k_m';
      const modelPath = `${RNFS.ExternalDirectoryPath}/models/${selectedModel}.gguf`;
      const exists = await RNFS.exists(modelPath);
      setIsModelDownloaded(exists);
      
      if (exists) {
        // Automatically default/select MiniLM as the embedding model if not selected
        const selected = await AsyncStorage.getItem(profileKey(profileId, '_embedding_model'));
        if (!selected) {
          await AsyncStorage.setItem(profileKey(profileId, '_embedding_model'), 'all-minilm-l6-v2-q4_k_m');
        }
        // The encoder is initialized lazily during ingestion/retrieval.
        setEmbeddingReady(true);
      } else {
        setEmbeddingReady(false);
      }
    } catch (err) {
      console.warn('Error checking embedding model status:', err);
      setEmbeddingReady(false);
    } finally {
      setIsCheckingModel(false);
    }
  }, [profileId]);

  // Load doc metadata and index size
  const bootstrap = useCallback(async () => {
    if (!profileId) return;
    try {
      await loadIndex(profileId);
      if (activeProfile.current !== profileId) return;
      setIndexSize(getIndexSize(profileId));
      setDocs(getIndexedDocuments(profileId));
    } catch (err: any) {
      console.error('Bootstrap docs meta failed:', err);
    }
  }, [profileId]);

  useEffect(() => {
    if (!profileId) return;
    bootstrap();
  }, [profileId, bootstrap]);

  useEffect(() => {
    if (!profileId) return;
    
    // Initial check
    checkEmbeddingModelStatus();

    // Re-check when screen gains focus
    const unsubscribe = navigation.addListener('focus', () => {
      checkEmbeddingModelStatus();
    });

    return unsubscribe;
  }, [profileId, navigation, checkEmbeddingModelStatus]);

  const saveDocsMeta = async (updated: DocMeta[]) => {
    const key = getDocsMetaKey();
    await AsyncStorage.setItem(key, JSON.stringify(updated));
    setDocs(updated);
  };

  const extractText = async (uri: string, name: string): Promise<string> => {
    const lower = name.toLowerCase();

    if (lower.endsWith('.txt')) {
      return await RNFS.readFile(uri, 'utf8');
    }

    if (lower.endsWith('.pdf')) {
      // react-native-pdf-extractor
      const { BaseExtractor } = require('react-native-pdf-extractor');
      await BaseExtractor.setUri(uri);
      const textLines = await BaseExtractor.getText();
      return textLines.join('\n') || '';
    }

    throw new Error(`Unsupported file type: ${name}`);
  };

  const handlePickDocument = async () => {
    if (!profileId || ingestionActive.current) return;
    const ingestProfile = profileId;
    if (!embeddingReady) {
      Alert.alert('Not ready', 'Embedding model is not loaded yet.');
      return;
    }

    ingestionActive.current = true;
    try {
      const results = await pick({
        type: [types.plainText, types.pdf],
      });

      const { uri, name, size } = results[0];
      if (!name || !uri) return;
      if (activeProfile.current !== ingestProfile) return;
      if (size && size > 16 * 1024 * 1024) throw new Error('Split files larger than 16 MiB before uploading to this mobile index.');

      const docId = docs.find(doc => doc.name === name)?.id || `doc_${Date.now()}`;
      setIsIngesting(true);
      setIngestStatus('Reading file...');

      const text = await extractText(uri, name);

      if (!text?.trim()) {
        throw new Error('No extractable text found. Scanned PDFs need OCR before upload.');
      }

      const { chunkCount, metrics } = await ingestTextDoc(text, docId, name, ingestProfile, {
        cancelled: () => activeProfile.current !== ingestProfile,
        onProgress: (stage, done, total) => {
          setIngestStatus(
            total > 1 ? `${stage} ${done}/${total}` : stage
          );
        },
      });
      if (activeProfile.current !== ingestProfile) return;

      const newDoc: DocMeta = {
        id: docId,
        name,
        addedAt: new Date().toLocaleString(),
        chunkCount,
      };

      const updated = [newDoc, ...docs.filter(doc => doc.id !== docId)];
      await saveDocsMeta(updated);
      setIndexSize(getIndexSize(profileId!));
      setIngestStatus('');
      Alert.alert('Done', `"${name}" indexed with ${chunkCount} chunks across ${metrics.sectionCount} sections in ${(metrics.totalMs / 1000).toFixed(1)}s.`);
    } catch (err: any) {
      if (isErrorWithCode(err) && err.code === errorCodes.OPERATION_CANCELED) return;
      Alert.alert('Error', err.message || 'Failed to ingest document.');
    } finally {
      ingestionActive.current = false;
      setIsIngesting(false);
      setIngestStatus('');
    }
  };

  const handleDeleteDoc = (doc: DocMeta) => {
    Alert.alert(
      'Remove Document',
      `Remove "${doc.name}" from the index?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: async () => {
            if (!profileId) return;
            try {
              await removeDoc(doc.id, profileId);
              if (activeProfile.current !== profileId) return;
              await saveDocsMeta(docs.filter(d => d.id !== doc.id));
              setIndexSize(getIndexSize(profileId));
            } catch (error) { Alert.alert('Error', String(error)); }
          },
        },
      ]
    );
  };

  const renderDoc = ({ item }: { item: DocMeta }) => (
    <View style={styles.docItem}>
      <View style={styles.docInfo}>
        <Text style={styles.docName} numberOfLines={1}>{item.name}</Text>
        <Text style={styles.docMeta}>
          {item.chunkCount} chunks · {item.addedAt}
        </Text>
      </View>
      <TouchableOpacity
        style={styles.deleteButton}
        onPress={() => handleDeleteDoc(item)}
      >
        <Text style={styles.deleteText}>Remove</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Icon name="arrow-back" size={24} color="#E5E7EB" />
        </TouchableOpacity>
        <View style={styles.headerCenter}>
          <Text style={styles.headerTitle}>DOCUMENTS</Text>
          <Text style={styles.indexInfo}>{indexSize} vectors in index</Text>
        </View>
        <View style={{ width: 40 }} />
      </View>

      {isIngesting ? (
        <View style={styles.ingestingContainer}>
          <ActivityIndicator size="large" color="#8B5CF6" />
          <Text style={styles.ingestStatus}>{ingestStatus}</Text>
        </View>
      ) : (
        <>
          {isCheckingModel ? (
            <View style={styles.bannerContainer}>
              <ActivityIndicator size="small" color="#8B5CF6" style={{ marginRight: 12 }} />
              <View style={styles.bannerTextContainer}>
                <Text style={styles.bannerTitle}>Checking embedding model...</Text>
              </View>
            </View>
          ) : !isModelDownloaded ? (
            <View style={styles.bannerContainer}>
              <View style={styles.bannerIconContainer}>
                <Icon name="warning" size={24} color="#F59E0B" />
              </View>
              <View style={styles.bannerTextContainer}>
                <Text style={styles.bannerTitle}>Embedding Model Needed</Text>
                <Text style={styles.bannerSub}>
                  The embedding model is not downloaded. Please download it to enable document upload and search.
                </Text>
              </View>
              <TouchableOpacity
                style={styles.bannerButton}
                onPress={() => navigation.navigate('Models')}
              >
                <Text style={styles.bannerButtonText}>Download</Text>
              </TouchableOpacity>
            </View>
          ) : !embeddingReady ? (
            <View style={styles.bannerContainer}>
              <ActivityIndicator size="small" color="#8B5CF6" style={{ marginRight: 12 }} />
              <View style={styles.bannerTextContainer}>
                <Text style={styles.bannerTitle}>Loading Model...</Text>
                <Text style={styles.bannerSub}>Initializing embedding engine...</Text>
              </View>
            </View>
          ) : null}

          <TouchableOpacity
            style={[
              styles.uploadButton,
              (!embeddingReady || !isModelDownloaded) && styles.uploadButtonDisabled,
            ]}
            onPress={handlePickDocument}
            disabled={!embeddingReady || !isModelDownloaded}
          >
            <Text style={styles.uploadButtonText}>
              {embeddingReady && isModelDownloaded ? '+ Upload Document' : 'Embedding model not ready'}
            </Text>
          </TouchableOpacity>

          <FlatList
            data={docs}
            renderItem={renderDoc}
            keyExtractor={item => item.id}
            contentContainerStyle={styles.listContainer}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <Text style={styles.emptyText}>No documents indexed yet.</Text>
                <Text style={styles.emptySubText}>
                  Upload a .txt or .pdf file to get started.
                </Text>
              </View>
            }
          />
        </>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B0F19',
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#1F2937',
    backgroundColor: '#101626',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  backButton: {
    padding: 4,
  },
  headerCenter: {
    alignItems: 'center',
    flex: 1,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'grey',
    fontFamily: 'sans-serif',
  },
  indexInfo: {
    fontSize: 13,
    color: '#8B5CF6',
    marginTop: 4,
    fontFamily: 'sans-serif',
  },
  uploadButton: {
    backgroundColor: '#8B5CF6',
    margin: 20,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  uploadButtonDisabled: {
    backgroundColor: '#374151',
  },
  uploadButtonText: {
    fontSize: 16,
    color: '#F9FAFB',
    fontFamily: 'sans-serif',
    fontWeight: '600',
  },
  ingestingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  ingestStatus: {
    marginTop: 16,
    fontSize: 16,
    color: '#E5E7EB',
    fontFamily: 'sans-serif',
  },
  listContainer: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  docItem: {
    flexDirection: 'row',
    backgroundColor: '#1E293B',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    alignItems: 'center',
  },
  docInfo: {
    flex: 1,
  },
  docName: {
    fontSize: 15,
    color: '#E5E7EB',
    fontFamily: 'sans-serif',
    fontWeight: '600',
  },
  docMeta: {
    fontSize: 13,
    color: '#9CA3AF',
    marginTop: 4,
    fontFamily: 'sans-serif',
  },
  deleteButton: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EF4444',
  },
  deleteText: {
    fontSize: 13,
    color: '#EF4444',
    fontFamily: 'sans-serif',
  },
  emptyContainer: {
    alignItems: 'center',
    marginTop: 80,
  },
  emptyText: {
    fontSize: 17,
    color: '#9CA3AF',
    fontFamily: 'sans-serif',
  },
  emptySubText: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 8,
    fontFamily: 'sans-serif',
  },
  bannerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1E293B',
    margin: 20,
    marginBottom: 0,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#374151',
  },
  bannerIconContainer: {
    marginRight: 12,
  },
  bannerTextContainer: {
    flex: 1,
  },
  bannerTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#F9FAFB',
    fontFamily: 'sans-serif',
  },
  bannerSub: {
    fontSize: 13,
    color: '#9CA3AF',
    marginTop: 2,
    fontFamily: 'sans-serif',
  },
  bannerButton: {
    backgroundColor: '#8B5CF6',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    marginLeft: 8,
  },
  bannerButtonText: {
    fontSize: 13,
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontFamily: 'sans-serif',
  },
});

export default DocumentsScreen;
