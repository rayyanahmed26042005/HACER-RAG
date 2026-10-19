// chat/types/react-native-fs.d.ts
// Type declarations for react-native-fs

declare module 'react-native-fs' {
  export const ExternalDirectoryPath: string;
  export const DocumentDirectoryPath: string;
  export const CachesDirectoryPath: string;
  export const MainBundlePath: string;
  export const TemporaryDirectoryPath: string;

  export function exists(path: string): Promise<boolean>;
  export function mkdir(path: string): Promise<void>;
  export function readFile(path: string, encoding?: string): Promise<string>;
  export function writeFile(path: string, content: string, encoding?: string): Promise<void>;
  export function unlink(path: string): Promise<void>;
  export function moveFile(path: string, destination: string): Promise<void>;
  export function stat(path: string): Promise<{ size: number; mtime: Date }>;
  export function readDir(path: string): Promise<
    {
      name: string;
      path: string;
      size: number;
      isFile: () => boolean;
      isDirectory: () => boolean;
    }[]
  >;

  export interface DownloadBeginCallbackResult {
    jobId: number;
    statusCode: number;
    contentLength: number;
    headers: Record<string, string>;
  }

  export interface DownloadProgressCallbackResult {
    jobId: number;
    contentLength: number;
    bytesWritten: number;
  }

  export interface DownloadResult {
    jobId: number;
    statusCode: number;
    bytesWritten: number;
  }

  export function downloadFile(options: {
    fromUrl: string;
    toFile: string;
    begin?: (res: DownloadBeginCallbackResult) => void;
    progress?: (res: DownloadProgressCallbackResult) => void;
    background?: boolean;
    progressDivider?: number;
    connectionTimeout?: number;
    readTimeout?: number;
    cacheable?: boolean;
    signal?: AbortSignal;
  }): { jobId: number; promise: Promise<DownloadResult> };

  export function stopDownload(jobId: number): void;

  const RNFS: {
    ExternalDirectoryPath: string;
    DocumentDirectoryPath: string;
    CachesDirectoryPath: string;
    MainBundlePath: string;
    TemporaryDirectoryPath: string;
    exists: typeof exists;
    mkdir: typeof mkdir;
    readFile: typeof readFile;
    writeFile: typeof writeFile;
    unlink: typeof unlink;
    moveFile: typeof moveFile;
    stat: typeof stat;
    readDir: typeof readDir;
    downloadFile: typeof downloadFile;
    stopDownload: typeof stopDownload;
  };

  export default RNFS;
}
