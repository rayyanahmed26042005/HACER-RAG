import RNFS from 'react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Alert } from 'react-native';

const MODELS_DIR = RNFS.ExternalDirectoryPath + '/models';

export interface DownloadProgress {
  modelId: string;
  progress: number;
  isDownloading: boolean;
}

type ProgressListener = (data: DownloadProgress) => void;

class DownloadManager {
  private activeJobs = new Map<string, number>(); // modelId -> jobId
  private progressListeners = new Set<ProgressListener>();
  private activeDownloads = new Map<string, number>(); // modelId -> progress

  registerListener(listener: ProgressListener) {
    this.progressListeners.add(listener);
    // Send initial states
    this.activeDownloads.forEach((progress, modelId) => {
      listener({ modelId, progress, isDownloading: true });
    });
  }

  unregisterListener(listener: ProgressListener) {
    this.progressListeners.delete(listener);
  }

  private notify(modelId: string, progress: number, isDownloading: boolean) {
    const data = { modelId, progress, isDownloading };
    if (isDownloading) {
      this.activeDownloads.set(modelId, progress);
    } else {
      this.activeDownloads.delete(modelId);
    }
    this.progressListeners.forEach(l => l(data));
  }

  isModelDownloading(modelId: string): boolean {
    return this.activeJobs.has(modelId);
  }

  getActiveDownloadModelId(): string | null {
    if (this.activeJobs.size > 0) {
      return Array.from(this.activeJobs.keys())[0];
    }
    return null;
  }

  async startDownload(
    modelId: string,
    downloadUrl: string,
    size: number,
    modelsKey: string,
    onFinish: () => void
  ) {
    if (this.activeJobs.has(modelId)) return;

    try {
      await RNFS.mkdir(MODELS_DIR);
      const ext = downloadUrl.split('.').pop()?.split('?')[0] || 'gguf';
      const localPath = `${MODELS_DIR}/${modelId}.${ext}`;

      const options = {
        fromUrl: downloadUrl,
        toFile: localPath,
        progress: (res: any) => {
          const total = res.contentLength || size || 1;
          const progress = Math.min(Math.floor((res.bytesWritten / total) * 100), 99);
          this.notify(modelId, progress, true);
        },
        progressDivider: 1,
        connectionTimeout: 30000,
        readTimeout: 30000,
        background: true,
        cacheable: false
      };

      const ret = RNFS.downloadFile(options);
      this.activeJobs.set(modelId, ret.jobId);
      this.notify(modelId, 0, true);

      const result = await ret.promise;
      this.activeJobs.delete(modelId);

      if (result.statusCode !== 200) {
        throw new Error(`Server returned status code ${result.statusCode}`);
      }

      // Update models list in AsyncStorage
      const savedModels = await AsyncStorage.getItem(modelsKey);
      const list = savedModels ? JSON.parse(savedModels) : [];
      let updated;
      const found = list.some((m: any) => m.id === modelId);
      if (found) {
        updated = list.map((m: any) =>
          m.id === modelId
            ? { ...m, isDownloaded: true, isDownloading: false, localPath, progress: 100 }
            : m
        );
      } else {
        updated = [...list, { id: modelId, isDownloaded: true, isDownloading: false, localPath, progress: 100 }];
      }
      await AsyncStorage.setItem(modelsKey, JSON.stringify(updated));

      this.notify(modelId, 100, false);
      onFinish();
    } catch (error: any) {
      console.error('DownloadManager error:', error);
      this.activeJobs.delete(modelId);
      this.notify(modelId, 0, false);

      // Clean up partial download file
      const ext = downloadUrl.split('.').pop()?.split('?')[0] || 'gguf';
      const localPath = `${MODELS_DIR}/${modelId}.${ext}`;
      const exists = await RNFS.exists(localPath);
      if (exists) {
        await RNFS.unlink(localPath).catch(err => console.error('Unlink failed:', err));
      }

      // Update list in AsyncStorage to mark as not downloading
      const savedModels = await AsyncStorage.getItem(modelsKey);
      if (savedModels) {
        const list = JSON.parse(savedModels);
        const updated = list.map((m: any) =>
          m.id === modelId
            ? { ...m, isDownloading: false, progress: 0, localPath: null }
            : m
        );
        await AsyncStorage.setItem(modelsKey, JSON.stringify(updated));
      }

      if (error?.message !== 'Download has been aborted') {
        Alert.alert('Download Failed', `Failed to download model:\n${error?.message || ''}`);
      }
      onFinish();
    }
  }

  async stopDownload(modelId: string, downloadUrl: string, modelsKey: string, onFinish: () => void) {
    const jobId = this.activeJobs.get(modelId);
    if (jobId) {
      RNFS.stopDownload(jobId);
      this.activeJobs.delete(modelId);
      this.notify(modelId, 0, false);

      // Clean up partial file
      const ext = downloadUrl.split('.').pop()?.split('?')[0] || 'gguf';
      const localPath = `${MODELS_DIR}/${modelId}.${ext}`;
      const exists = await RNFS.exists(localPath);
      if (exists) {
        await RNFS.unlink(localPath).catch(err => console.error('Unlink failed:', err));
      }

      // Update list in AsyncStorage
      const savedModels = await AsyncStorage.getItem(modelsKey);
      if (savedModels) {
        const list = JSON.parse(savedModels);
        const updated = list.map((m: any) =>
          m.id === modelId
            ? { ...m, isDownloading: false, progress: 0, localPath: null }
            : m
        );
        await AsyncStorage.setItem(modelsKey, JSON.stringify(updated));
      }
      onFinish();
    }
  }
}

export const downloadManager = new DownloadManager();
