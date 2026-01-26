import React, { useState, useRef } from 'react';
import styled from 'styled-components';

const Page = styled.section`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1.25rem;
  color: #e5e5e5;
`;

const Title = styled.h1`
  color: #ffffff;
  font-weight: 800;
  margin: 0 0 0.75rem 0;
  font-size: clamp(1.5rem, 3vw, 2rem);
  text-align: center;
`;

const Description = styled.p`
  text-align: center;
  margin: 0 auto 2rem;
  max-width: 760px;
  line-height: 1.7;
  color: #b3b3b3;
`;

const UploadContainer = styled.div`
  background: #1f1f1f;
  border: 2px dashed #333;
  border-radius: 10px;
  padding: 3rem 2rem;
  text-align: center;
  margin-bottom: 2rem;
  transition: border-color 0.3s ease, background 0.3s ease;
  cursor: pointer;

  &:hover {
    border-color: #e50914;
    background: #252525;
  }

  &.dragover {
    border-color: #e50914;
    background: #252525;
  }
`;

const UploadIcon = styled.div`
  font-size: 3rem;
  color: #666;
  margin-bottom: 1rem;
`;

const UploadText = styled.div`
  color: #e5e5e5;
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
`;

const UploadSubtext = styled.div`
  color: #999;
  font-size: 0.9rem;
`;

const HiddenInput = styled.input`
  display: none;
`;

const Button = styled.button`
  padding: 0.9rem 1.25rem;
  background: #e50914;
  color: #ffffff;
  font-weight: 800;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s ease;
  margin-top: 1rem;
  &:hover { 
    background: #f6121d; 
  }
  &:disabled {
    background: #666;
    cursor: not-allowed;
  }
`;

const VideoPreview = styled.div`
  background: #1f1f1f;
  border: 1px solid #333;
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 2rem;
`;

const VideoInfo = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
`;

const VideoName = styled.div`
  color: #ffffff;
  font-weight: 600;
`;

const VideoSize = styled.div`
  color: #999;
  font-size: 0.9rem;
`;

const VideoElement = styled.video`
  width: 100%;
  max-width: 600px;
  height: auto;
  border-radius: 8px;
  background: #000;
`;

const ProcessingContainer = styled.div`
  background: #1f1f1f;
  border: 1px solid #333;
  border-radius: 10px;
  padding: 2rem;
  text-align: center;
  margin-bottom: 2rem;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: #333;
  border-radius: 4px;
  overflow: hidden;
  margin: 1rem 0;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #e50914, #ff6b6b);
  width: ${props => props.progress}%;
  transition: width 0.3s ease;
`;

const SubtitleOutput = styled.div`
  background: #1f1f1f;
  border: 1px solid #333;
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 2rem;
`;

const SubtitleText = styled.div`
  background: #141414;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 1rem;
  color: #e5e5e5;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.6;
  white-space: pre-wrap;
  max-height: 300px;
  overflow-y: auto;
`;

const DownloadButton = styled(Button)`
  background: #28a745;
  &:hover {
    background: #218838;
  }
`;

const RemoveButton = styled.button`
  background: transparent;
  border: 1px solid #666;
  color: #999;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  margin-left: 1rem;
  &:hover {
    border-color: #e50914;
    color: #e50914;
  }
`;

export default function AISubtitling() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [subtitles, setSubtitles] = useState('');
  const [currentFileId, setCurrentFileId] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // API configuration
  const SUBTITLE_API_BASE = process.env.REACT_APP_SUBTITLE_API_BASE || 'http://localhost:5001';

  // Debug API configuration
  React.useEffect(() => {
    console.log('🔧 API Configuration:');
    console.log('  - SUBTITLE_API_BASE:', SUBTITLE_API_BASE);
    console.log('  - Environment variable:', process.env.REACT_APP_SUBTITLE_API_BASE);
  }, []);

  const handleFileSelect = (file) => {
    console.log('File selected:', file);
    if (file && file.type.startsWith('video/')) {
      console.log('Valid video file selected:', file.name, file.type, file.size);
      setSelectedFile(file);
      setSubtitles('');
      setCurrentFileId('');
      setProgress(0);
      setProgressMessage('');
    } else {
      console.error('Invalid file type:', file?.type);
      alert('Please select a valid video file.');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const generateSubtitles = async () => {
    if (!selectedFile) return;

    setProcessing(true);
    setProgress(0);
    setProgressMessage('Starting upload and processing...');

    try {
      // Step 1: Upload the video file (backend will automatically extract audio)
      const formData = new FormData();
      formData.append('video', selectedFile);

      console.log('🚀 Starting upload...');
      console.log('  - File:', selectedFile.name);
      console.log('  - Size:', selectedFile.size, 'bytes');
      console.log('  - Type:', selectedFile.type);
      console.log('  - API URL:', `${SUBTITLE_API_BASE}/upload`);

      // Check if backend is reachable first
      try {
        const healthCheck = await fetch(`${SUBTITLE_API_BASE}/health`, { method: 'GET' });
        console.log('✅ Backend health check:', healthCheck.status);
      } catch (healthError) {
        console.error('❌ Backend not reachable:', healthError);
        throw new Error('Backend service is not available. Please ensure the backend is running on port 5001.');
      }

      // Create AbortController for timeout handling on large uploads
      const abortController = new AbortController();
      const timeoutId = setTimeout(() => abortController.abort(), 30 * 60 * 1000); // 30 minutes timeout

      const uploadResponse = await fetch(`${SUBTITLE_API_BASE}/upload`, {
        method: 'POST',
        body: formData,
        signal: abortController.signal,
        // Remove default timeout for large file uploads
        keepalive: false
      });

      clearTimeout(timeoutId); // Clear timeout if upload completes

      console.log('📤 Upload response status:', uploadResponse.status);

      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text();
        console.error('Upload error response:', errorText);
        throw new Error(`Upload failed: ${uploadResponse.status} - ${errorText}`);
      }

      const uploadResult = await uploadResponse.json();
      console.log('Upload result:', uploadResult);
      
      if (!uploadResult.file_id) {
        throw new Error('No file ID returned from upload');
      }

      const fileId = uploadResult.file_id;
      setCurrentFileId(fileId);

      // Start polling for upload progress
      const pollUploadProgress = async () => {
        try {
          const progressResponse = await fetch(`${SUBTITLE_API_BASE}/progress/${fileId}`);
          if (progressResponse.ok) {
            const progressData = await progressResponse.json();
            setProgress(progressData.progress * 0.5); // Upload takes 50% of total progress
            setProgressMessage(progressData.message);

            if (progressData.progress === 100) {
              // Upload complete, now start subtitle generation
              startSubtitleGeneration(fileId);
              return;
            } else if (progressData.progress === -1) {
              throw new Error(progressData.message);
            } else {
              // Continue polling
              setTimeout(pollUploadProgress, 2000);
            }
          } else {
            throw new Error('Failed to get progress');
          }
        } catch (error) {
          console.error('Progress polling error:', error);
          setProcessing(false);
          alert('Error tracking upload progress: ' + error.message);
        }
      };

      // Start polling after a short delay
      setTimeout(pollUploadProgress, 1000);

    } catch (error) {
      console.error('❌ Upload error:', error);
      setProcessing(false);
      
      // Handle different types of errors
      if (error.name === 'AbortError') {
        setProgressMessage('❌ Upload timeout. File too large or connection too slow.');
      } else if (error.message.includes('Backend service is not available')) {
        setProgressMessage('❌ Backend service unavailable. Please start the backend service.');
      } else if (error.message.includes('Upload failed')) {
        setProgressMessage('❌ Upload failed. Please try again with a smaller file.');
      } else if (error.message.includes('fetch')) {
        setProgressMessage('❌ Network error. Please check your connection and try again.');
      } else {
        setProgressMessage(`❌ Error: ${error.message}`);
      }
    }
  };

  const startSubtitleGeneration = async (fileId) => {
    try {
      console.log('Starting subtitle generation for file ID:', fileId);
      setProgressMessage('Starting subtitle generation...');
      
      const subtitleResponse = await fetch(`${SUBTITLE_API_BASE}/generate-subtitles`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ file_id: fileId })
      });

      console.log('Subtitle generation response status:', subtitleResponse.status);

      if (!subtitleResponse.ok) {
        const errorText = await subtitleResponse.text();
        console.error('Subtitle generation error:', errorText);
        throw new Error(`Subtitle generation failed: ${subtitleResponse.status} - ${errorText}`);
      }

      const subtitleResult = await subtitleResponse.json();
      console.log('Subtitle result:', subtitleResult);

      // Start polling for subtitle generation progress
      const pollSubtitleProgress = async () => {
        try {
          const progressResponse = await fetch(`${SUBTITLE_API_BASE}/progress/${fileId}`);
          if (progressResponse.ok) {
            const progressData = await progressResponse.json();
            // Subtitle generation takes the remaining 50% of progress
            setProgress(50 + (progressData.progress * 0.5));
            setProgressMessage(progressData.message);

            if (progressData.progress === 100) {
              // Fetch the final subtitles
              const subtitlesResponse = await fetch(`${SUBTITLE_API_BASE}/subtitles/${fileId}`);
              if (subtitlesResponse.ok) {
                const subtitlesData = await subtitlesResponse.json();
                setSubtitles(subtitlesData.subtitles || 'No subtitles generated');
              }
              setProcessing(false);
              setProgress(100);
              setProgressMessage('Subtitles generated successfully!');
              return;
            } else if (progressData.progress === -1) {
              throw new Error(progressData.message);
            } else {
              // Continue polling
              setTimeout(pollSubtitleProgress, 2000);
            }
          } else {
            throw new Error('Failed to get progress');
          }
        } catch (error) {
          console.error('Progress polling error:', error);
          setProcessing(false);
          alert('Error tracking subtitle generation: ' + error.message);
        }
      };

      // Start polling after a short delay
      setTimeout(pollSubtitleProgress, 1000);

    } catch (error) {
      console.error('Subtitle generation error:', error);
      alert('Error generating subtitles: ' + error.message);
      setSubtitles('Error: ' + error.message);
      setProcessing(false);
    }
  };

  const downloadSubtitles = () => {
    if (!subtitles) return;

    const blob = new Blob([subtitles], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedFile.name.split('.')[0]}_subtitles.srt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const removeFile = () => {
    setSelectedFile(null);
    setSubtitles('');
    setCurrentFileId('');
    setProgress(0);
    setProgressMessage('');
    setProcessing(false);
  };



  return (
    <Page>
      <Title>AI Subtitling</Title>
      <Description>
        Upload a video file and our AI will automatically generate accurate subtitles. 
        Simply upload your video and click "Generate Subtitles" to start the complete process.
      </Description>

      {!selectedFile ? (
        <UploadContainer
          className={dragOver ? 'dragover' : ''}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          <UploadIcon>🎬</UploadIcon>
          <UploadText>Drag and drop your video file here</UploadText>
          <UploadSubtext>or click to browse and select a file</UploadSubtext>
          <UploadSubtext style={{ marginTop: '0.5rem' }}>
            Supported formats: MP4, AVI, MOV, MKV (Max 5GB)
          </UploadSubtext>
          <HiddenInput
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileInputChange}
          />
        </UploadContainer>
      ) : (
        <VideoPreview>
          <VideoInfo>
            <VideoName>{selectedFile.name}</VideoName>
            <div>
              <VideoSize>{formatFileSize(selectedFile.size)}</VideoSize>
              <RemoveButton onClick={removeFile}>Remove</RemoveButton>
            </div>
          </VideoInfo>
          <VideoElement controls>
            <source src={URL.createObjectURL(selectedFile)} type={selectedFile.type} />
            Your browser does not support the video tag.
          </VideoElement>
          
          <Button 
            onClick={generateSubtitles} 
            disabled={processing}
            style={{ 
              background: processing ? '#666' : '#e50914',
              marginTop: '1rem'
            }}
          >
            {processing ? 'Processing...' : 'Generate Subtitles'}
          </Button>
        </VideoPreview>
      )}

      {processing && (
        <ProcessingContainer>
          <h3 style={{ color: '#fff', margin: '0 0 1rem 0' }}>
            Processing Video & Generating Subtitles
          </h3>
          <p style={{ color: '#999', margin: '0 0 1rem 0' }}>
            {progressMessage || 'Processing your video and generating subtitles...'}
          </p>
          <ProgressBar>
            <ProgressFill progress={progress} />
          </ProgressBar>
          <p style={{ color: '#999', margin: '0.5rem 0 0 0' }}>
            {Math.round(progress)}% complete
          </p>
          
          <div style={{ marginTop: '1rem', padding: '1rem', background: '#262626', borderRadius: '6px' }}>
            <h4 style={{ color: '#fff', margin: '0 0 0.5rem 0', fontSize: '0.9rem' }}>Current Process:</h4>
            <div style={{ color: '#b3b3b3', fontSize: '0.85rem', lineHeight: '1.4' }}>
              {progress < 50 ? (
                <>
                  <div>🔄 Uploading video file...</div>
                  <div>🎵 Extracting audio (MP3)</div>
                  <div>☁️ Preparing the transcription service</div>
                </>
              ) : (
                <>
                  <div>✅ Video uploaded successfully</div>
                  <div>✅ Audio extracted</div>
                  <div>🤖 Transcription service processing speech...</div>
                  <div>📝 Generating SRT subtitles</div>
                </>
              )}
            </div>
          </div>
        </ProcessingContainer>
      )}

      {subtitles && (
        <SubtitleOutput>
          <h3 style={{ color: '#fff', margin: '0 0 1rem 0' }}>Generated Subtitles</h3>
          <SubtitleText>{subtitles}</SubtitleText>
          <DownloadButton onClick={downloadSubtitles}>
            Download SRT File
          </DownloadButton>
        </SubtitleOutput>
      )}
    </Page>
  );
}