
import { useState, useCallback } from "react";
import { UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FileUploadProps {
  onUpload: (file: File) => void;
  isLoading: boolean;
}

export const FileUpload = ({ onUpload, isLoading }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        const file = files[0];
        setSelectedFile(file);
      }
    },
    []
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        const file = files[0];
        setSelectedFile(file);
      }
    },
    []
  );

  const handleUpload = useCallback(() => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  }, [selectedFile, onUpload]);

  return (
    <div className="space-y-4">
      <div
        className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer transition-colors ${
          isDragging ? "border-primary bg-primary/5" : "border-gray-300"
        }`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="w-16 h-16 mb-4 rounded-full bg-primary/10 flex items-center justify-center">
          <UploadCloud className="h-8 w-8 text-primary" />
        </div>
        <p className="text-sm font-medium mb-1">
          <span className="text-primary">Click to upload</span> or drag and drop
        </p>
        <p className="text-xs text-gray-500 mb-4">SVG, PNG, JPG or PDF (max. 10MB)</p>
        
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".jpg,.jpeg,.png,.pdf"
          onChange={handleFileChange}
          disabled={isLoading}
        />
        <label
          htmlFor="file-upload"
          className="inline-flex items-center justify-center text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-9 rounded-md px-3 cursor-pointer"
        >
          Select file
        </label>
      </div>

      {selectedFile && (
        <div className="flex flex-col space-y-2">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
              <UploadCloud className="h-4 w-4 text-primary" />
            </div>
            <div className="text-sm font-medium truncate flex-1">
              {selectedFile.name}
            </div>
            <div className="text-xs text-gray-500">
              {(selectedFile.size / 1024).toFixed(0)} KB
            </div>
          </div>
          <Button 
            className="w-full" 
            onClick={handleUpload} 
            disabled={isLoading}
          >
            {isLoading ? "Processing..." : "Extract Invoice Data"}
          </Button>
        </div>
      )}
    </div>
  );
};
