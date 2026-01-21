import React, { useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const resolveHighlightApiBase = () => {
  const envValue = process.env.REACT_APP_HIGHLIGHT_TRAILER_API;
  if (envValue) {
    return envValue.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined') {
    const { protocol, hostname } = window.location;
    const localHosts = new Set(['localhost', '127.0.0.1', '0.0.0.0']);
    const isLanHost = /^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1]))/.test(hostname) || hostname.endsWith('.local');
    if (localHosts.has(hostname) || isLanHost) {
      return `${protocol}//${hostname}:5013`;
    }
    return `${protocol}//${hostname}`;
  }
  return '';
};

const HIGHLIGHT_API_BASE = resolveHighlightApiBase();

const buildUploadUrl = () => {
  if (HIGHLIGHT_API_BASE) {
    return `${HIGHLIGHT_API_BASE}/api/highlight-trailer/upload`;
  }
  return '/api/highlight-trailer/upload';
};

const buildAbsoluteUrl = (maybePath) => {
  if (!maybePath) return '';
  if (maybePath.startsWith('http')) return maybePath;
  const prefix = HIGHLIGHT_API_BASE || '';
  if (!prefix) return maybePath;
  return `${prefix}${maybePath.startsWith('/') ? maybePath : `/${maybePath}`}`;
};

const Page = styled.section`
  max-width: 1100px;
  margin: 0 auto;
  padding: 2.5rem 1.25rem 3rem;
  color: var(--text);
`;

const Card = styled.div`
  background: linear-gradient(135deg, rgba(16, 10, 28, 0.92), rgba(10, 7, 18, 0.95));
  border: 1px solid rgba(var(--accent-2-rgb), 0.25);
  border-radius: 24px;
  padding: clamp(1.5rem, 3vw, 2.5rem);
  box-shadow: 0 25px 60px rgba(4, 11, 24, 0.65);
`;

const Heading = styled.h1`
  text-align: center;
  margin: 0 0 0.75rem 0;
  font-size: clamp(1.9rem, 3.5vw, 2.6rem);
  font-weight: 800;
  color: #ffffff;
`;

const Subheading = styled.p`
  text-align: center;
  margin: 0 auto 2.2rem;
  max-width: 720px;
  line-height: 1.7;
  color: #b6c5ef;
`;

const FormGrid = styled.form`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1.4rem;
  margin-bottom: 2rem;
`;

const Field = styled.label`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: #bfc8f3;
`;

const Select = styled.select`
  padding: 0.85rem 1rem;
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: rgba(0, 0, 0, 0.85);
  color: #ffffff;
  font-size: 1rem;
  outline: none;
`;

const NumberInput = styled.input`
  padding: 0.85rem 1rem;
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: rgba(0, 0, 0, 0.85);
  color: #ffffff;
  font-size: 1rem;
  outline: none;
`;

const UploadPane = styled.div`
  grid-column: 1 / -1;
  border: 1px dashed rgba(255, 255, 255, 0.18);
  border-radius: 18px;
  padding: 1.5rem;
  text-align: center;
  background: rgba(0, 0, 0, 0.65);
`;

const UploadInput = styled.input`
  margin-top: 0.75rem;
`;

const PrimaryButton = styled.button`
  grid-column: 1 / -1;
  justify-self: center;
  padding: 0.95rem 2.4rem;
  border-radius: 999px;
  border: none;
  background: linear-gradient(135deg, var(--button-accent), var(--button-accent-2));
  color: var(--on-button-accent);
  font-weight: 700;
  font-size: 1rem;
  cursor: ${props => (props.disabled ? 'not-allowed' : 'pointer')};
  opacity: ${props => (props.disabled ? 0.55 : 1)};
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 40px rgba(var(--button-accent-rgb), 0.28);
  }
`;

const StatusMessage = styled.div`
  text-align: center;
  color: var(--text-subtle);
  font-weight: 600;
  margin-top: 0.5rem;
`;

const PreviewWrapper = styled.div`
  margin-top: 2rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.25rem;
`;

const PreviewCard = styled.div`
  border: 1px solid rgba(99, 102, 241, 0.25);
  border-radius: 18px;
  background: rgba(5, 10, 22, 0.65);
  padding: 1rem;
  box-shadow: inset 0 0 0 1px rgba(99, 102, 241, 0.1);
`;

const PreviewTitle = styled.h3`
  margin: 0 0 0.75rem 0;
  font-size: 1rem;
  color: #dce7ff;
`;

const VideoFrame = styled.video`
  width: 100%;
  border-radius: 12px;
  background: #010511;
  box-shadow: 0 12px 34px rgba(4, 10, 22, 0.75);
`;

const InlineActions = styled.div`
  margin-top: 0.75rem;
  display: flex;
  justify-content: flex-end;
`;

const DownloadLink = styled.a`
  color: var(--text-subtle);
  text-decoration: none;
  font-weight: 600;
  &:hover {
    text-decoration: underline;
  }
`;

const Timeline = styled.div`
  margin: 2rem 0 1rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
`;

const TimelineStep = styled.div`
  padding: 0.9rem 1rem;
  border-radius: 16px;
  border: ${({ $state }) => {
    if ($state === 'done') return '1px solid rgba(34, 197, 94, 0.6)';
    if ($state === 'active') return '1px solid rgba(96, 165, 250, 0.8)';
    return '1px solid rgba(99, 102, 241, 0.25)';
  }};
  background: ${({ $state }) => {
    if ($state === 'done') return 'rgba(21, 83, 45, 0.6)';
    if ($state === 'active') return 'rgba(18, 37, 74, 0.85)';
    return 'rgba(5, 10, 22, 0.45)';
  }};
  color: ${({ $state }) => ($state === 'pending' ? '#9fb2e4' : '#dce7ff')};
  font-size: 0.9rem;
  line-height: 1.5;
  min-height: 110px;
`;

const StepTitle = styled.div`
  font-weight: 700;
  margin-bottom: 0.35rem;
`;

const StepHint = styled.div`
  font-size: 0.85rem;
  opacity: 0.85;
`;

function HighlightTrailer() {
  const [file, setFile] = useState(null);
  const [videoType, setVideoType] = useState('match');
  const [duration, setDuration] = useState('30');
  const [status, setStatus] = useState('');
  const [preview, setPreview] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [stage, setStage] = useState('idle');
  const [uploadPercent, setUploadPercent] = useState(0);
  const [processingSeconds, setProcessingSeconds] = useState(0);

  const isGeneric = videoType === 'generic';
  const outputLabel = isGeneric ? 'Trailer' : 'Highlight';
  const stageOrder = useMemo(() => ['idle', 'uploading', 'processing', 'assembling', 'completed'], []);
  const stepConfig = useMemo(() => ([
    {
      key: 'uploading',
      title: 'Uploading source',
      hint: uploadPercent > 0 ? `${uploadPercent}% complete` : 'Sending file to highlight service'
    },
    {
      key: 'processing',
      title: 'Vision analysis',
      hint: stage === 'processing' ? `Analyzing frames... ${processingSeconds}s elapsed` : 'Vision models find high-energy sequences'
    },
    {
      key: 'assembling',
      title: isGeneric ? 'Cutting trailer' : 'Stitching highlight',
      hint: stage === 'assembling' ? 'Packaging final render...' : 'FFmpeg trims + concatenates matched segments'
    },
    {
      key: 'completed',
      title: `${outputLabel} ready`,
      hint: downloadUrl ? 'Press play below' : 'Awaiting output'
    }
  ]), [uploadPercent, downloadUrl, stage, processingSeconds, outputLabel, isGeneric]);

  useEffect(() => {
    let intervalId;
    if (stage === 'processing') {
      intervalId = setInterval(() => {
        setProcessingSeconds(prev => prev + 1);
      }, 1000);
    } else if (stage !== 'uploading') {
      setProcessingSeconds(0);
    }
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [stage]);

  const handleFile = e => {
    setFile(e.target.files[0]);
    setPreview(e.target.files[0] ? URL.createObjectURL(e.target.files[0]) : null);
    setStatus('');
    setDownloadUrl(null);
    setStage('idle');
    setUploadPercent(0);
  };

  const handleSubmit = async e => {
    e.preventDefault();
    if (!file) return;
    setStatus('Uploading...');
    setStage('uploading');
    setUploadPercent(0);
    setProcessingSeconds(0);
    setDownloadUrl(null);
    try {
      const formData = new FormData();
      formData.append('video', file);
      formData.append('videoType', videoType);
      formData.append('duration', duration);
      const res = await axios.post(buildUploadUrl(), formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadPercent(percent);
            setStatus(`Uploading... ${percent}%`);
            if (percent >= 100) {
              setStage('processing');
              setStatus('Analyzing frames with vision models...');
            }
          }
        }
      });
      setStage('assembling');
      setStatus(isGeneric ? 'Building trailer clips...' : 'Stitching highlight clips...');
      setDownloadUrl(buildAbsoluteUrl(res.data && res.data.downloadUrl));
      setStage('completed');
      setStatus(`${outputLabel} ready!`);
    } catch (err) {
      console.error(err);
      setStage('error');
      setStatus('Upload failed. Please try again.');
    }
  };

  return (
    <Page>
      <Card>
        <Heading>AI Based Highlight & Trailer</Heading>
        <Subheading>
          Upload a match cut for highlight curation or drop in generic footage for trailer assembly. Configure the output duration and preview the generated cut directly inside the workspace.
        </Subheading>
        <FormGrid onSubmit={handleSubmit}>
          <Field>
            Select type
            <Select value={videoType} onChange={e => setVideoType(e.target.value)}>
              <option value="match">Match Video (Highlights)</option>
              <option value="generic">Generic Content (Trailer)</option>
            </Select>
          </Field>
          <Field>
            Output duration (seconds)
            <NumberInput
              type="number"
              min="10"
              max="300"
              value={duration}
              onChange={e => setDuration(e.target.value)}
            />
          </Field>
          <UploadPane>
            <strong>Upload source footage</strong>
            <p style={{ margin: '0.35rem 0 0', color: '#9fb2e4' }}>MP4, MOV, MKV — up to 2GB.</p>
            <UploadInput type="file" accept="video/*" onChange={handleFile} />
            {preview && (
              <div style={{ marginTop: '1rem', color: '#9fb2e4', fontSize: '0.9rem' }}>
                Selected: {file ? file.name : ''}
              </div>
            )}
          </UploadPane>
          <PrimaryButton type="submit" disabled={!file}>
            {file ? (isGeneric ? 'Generate trailer' : 'Generate highlight') : 'Choose a video first'}
          </PrimaryButton>
        </FormGrid>
        {status && <StatusMessage>{status}</StatusMessage>}
        <Timeline>
          {stepConfig.map(step => {
            const currentIdx = stageOrder.indexOf(stage);
            const stepIdx = stageOrder.indexOf(step.key);
            let stepState = 'pending';
            if (stepIdx !== -1 && currentIdx !== -1) {
              if (stepIdx < currentIdx) stepState = 'done';
              else if (stepIdx === currentIdx) stepState = 'active';
            }
            if (stage === 'error') {
              if (step.key === 'uploading') {
                stepState = 'active';
              } else if (stepIdx > stageOrder.indexOf('uploading')) {
                stepState = 'pending';
              }
            }
            return (
              <TimelineStep key={step.key} $state={stepState}>
                <StepTitle>{step.title}</StepTitle>
                <StepHint>{step.hint}</StepHint>
              </TimelineStep>
            );
          })}
        </Timeline>
        <PreviewWrapper>
          {preview && (
            <PreviewCard>
              <PreviewTitle>Source preview</PreviewTitle>
              <VideoFrame controls src={preview} />
            </PreviewCard>
          )}
          {downloadUrl && (
            <PreviewCard>
              <PreviewTitle>Generated highlight</PreviewTitle>
              <VideoFrame controls src={downloadUrl} />
              <InlineActions>
                <DownloadLink href={downloadUrl} download>
                  Download MP4
                </DownloadLink>
              </InlineActions>
            </PreviewCard>
          )}
        </PreviewWrapper>
      </Card>
    </Page>
  );
}

export default HighlightTrailer;
