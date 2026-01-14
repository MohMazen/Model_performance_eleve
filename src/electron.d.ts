export {};

declare global {
  interface Window {
    electron: {
      selectFile: () => Promise<string | null>;
      runAnalysis: (filePath: string | null) => Promise<{
        success: boolean;
        report: string;
        images: string[];
        outputDir: string;
      }>;
      onAnalysisLog: (callback: (message: string) => void) => void;
      onAnalysisError: (callback: (message: string) => void) => void;
      removeAnalysisListeners: () => void;
    };
  }
}
