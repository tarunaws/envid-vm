import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const CATEGORY_OPTIONS = [
  { key: 'Explicit Nudity', label: 'Explicit Nudity' },
  { key: 'Suggestive', label: 'Suggestive' },
  { key: 'Violence', label: 'Violence' },
  { key: 'Visually Disturbing', label: 'Visually Disturbing' },
  { key: 'Rude Gestures', label: 'Rude Gestures' },
  { key: 'Alcohol', label: 'Alcohol' },
  { key: 'Tobacco', label: 'Tobacco / Smoking' },
  { key: 'Drugs', label: 'Drugs & Paraphernalia' },
  { key: 'Weapons', label: 'Weapons' },
  { key: 'Hate Symbols', label: 'Hate Symbols' },
  { key: 'Gambling', label: 'Gambling' },
];

const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;

const resolveModerationApiBase = () => {
  const envValue = process.env.REACT_APP_MODERATION_API_BASE;
  if (envValue) {
    return envValue.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined') {
    const { protocol, hostname } = window.location;
    const normalizedHost = hostname === '0.0.0.0' ? '127.0.0.1' : hostname;
    const localHosts = new Set(['localhost', '127.0.0.1', '::1']);

    if (hostname === '0.0.0.0' || localHosts.has(normalizedHost)) {
      return `${protocol}//${normalizedHost}:5006`.replace(/\/$/, '');
    }

    return `${protocol}//${hostname}`.replace(/\/$/, '');
  }
  return '';
};

const Page = styled.section`
  max-width: 1100px;
  margin: 0 auto;
  padding: 2.75rem 1.5rem 3.25rem;
  color: #dce7ff;
`;

const Title = styled.h1`
  color: #ffffff;
  font-weight: 800;
  margin: 0 0 0.75rem 0;
  font-size: clamp(1.85rem, 4vw, 2.6rem);
  text-align: center;
`;

const Lead = styled.p`
  margin: 0 auto 2rem auto;
  max-width: 780px;
  text-align: center;
  line-height: 1.8;
  color: #b8c9f5;
`;

const UploadCard = styled.label`
  display: block;
  background: rgba(0, 0, 0, 0.92);
  border-radius: 16px;
  padding: 2.4rem 2.1rem;
  border: 2px dashed rgba(255, 255, 255, 0.18);
  text-align: center;
  cursor: pointer;
  transition: border-color 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
  box-shadow: 0 20px 44px rgba(7, 14, 28, 0.55);

  &:hover,
  &.dragover {
    border-color: rgba(255, 255, 255, 0.45);
    box-shadow: 0 26px 56px rgba(17, 36, 64, 0.6);
    background: rgba(0, 0, 0, 0.96);
  }
`;

const UploadIcon = styled.div`
  font-size: 3.2rem;
  color: var(--text);
  margin-bottom: 1rem;
`;

const UploadTitle = styled.div`
  font-size: 1.1rem;
  color: #f8fafc;
  font-weight: 600;
  margin-bottom: 0.5rem;
`;

const UploadHint = styled.div`
  color: #94a3e6;
  font-size: 0.95rem;
`;

const HiddenInput = styled.input`
  display: none;
`;

const ActionsRow = styled.div`
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  gap: 0.85rem;
  margin-top: 1.6rem;
`;

const PrimaryButton = styled.button`
  padding: 0.85rem 1.45rem;
  border-radius: 14px;
  border: none;
  background: linear-gradient(135deg, var(--button-accent), var(--button-accent-2));
  color: var(--on-button-accent);
  font-weight: 700;
  font-size: 0.98rem;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 34px rgba(var(--button-accent-rgb), 0.32);
  }

  &:disabled {
    opacity: 0.55;
    cursor: not-allowed;
    box-shadow: none;
  }
`;

const SecondaryButton = styled.button`
  padding: 0.85rem 1.4rem;
  border-radius: 14px;
  border: 1px solid rgba(148, 163, 255, 0.4);
  background: rgba(15, 23, 42, 0.9);
  color: #dbeafe;
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    border-color: rgba(var(--accent-rgb), 0.55);
    box-shadow: 0 16px 30px rgba(var(--accent-rgb), 0.2);
  }

  &:disabled {
    opacity: 0.55;
    cursor: not-allowed;
    box-shadow: none;
  }
`;

const StatusBanner = styled.div`
  margin-top: 1.25rem;
  padding: 1rem 1.2rem;
  border-radius: 14px;
  border: 1px solid ${({ $error }) => ($error ? 'rgba(248, 113, 113, 0.55)' : 'rgba(var(--accent-rgb), 0.45)')};
  background: ${({ $error }) => ($error ? 'rgba(60, 17, 34, 0.65)' : 'rgba(15, 48, 35, 0.6)')};
  color: ${({ $error }) => ($error ? '#fecaca' : '#bbf7d0')};
  line-height: 1.6;
`;

const OptionsGrid = styled.div`
  margin-top: 2rem;
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
`;

const OptionCard = styled.label`
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.9rem 1rem;
  border-radius: 12px;
  border: 1px solid rgba(99, 102, 241, 0.28);
  background: rgba(14, 22, 40, 0.85);
  cursor: pointer;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;

  &:hover {
    border-color: rgba(var(--accent-rgb), 0.6);
    box-shadow: 0 12px 28px rgba(var(--accent-rgb), 0.16);
  }
`;

const OptionCheckbox = styled.input`
  width: 18px;
  height: 18px;
  accent-color: var(--text);
`;

const ConfidenceRow = styled.div`
  margin-top: 1.6rem;
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  align-items: center;
  color: var(--text-muted);
`;

const Slider = styled.input`
  width: 240px;
`;

const PreviewWrap = styled.div`
  margin-top: 1.6rem;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  align-items: center;
  color: #9fb4ec;
`;

const PreviewVideo = styled.video`
  max-width: min(520px, 100%);
  border-radius: 14px;
  border: 1px solid rgba(99, 102, 241, 0.28);
  box-shadow: 0 18px 36px rgba(6, 15, 30, 0.55);
`;

const ResultsGrid = styled.div`
  margin-top: 2.5rem;
  display: grid;
  gap: 1.5rem;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
`;

const Card = styled.div`
  background: linear-gradient(166deg, rgba(15, 28, 50, 0.92), rgba(29, 47, 78, 0.88));
  border-radius: 18px;
  border: 1px solid rgba(79, 70, 229, 0.28);
  padding: 1.4rem 1.5rem;
  box-shadow: 0 20px 46px rgba(7, 15, 30, 0.55);
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
`;

const CardTitle = styled.h2`
  margin: 0;
  font-size: 1.2rem;
  font-weight: 800;
  color: #f8fafc;
`;

const SummaryList = styled.ul`
  margin: 0;
  padding-left: 1.05rem;
  color: #c8d5ff;
  line-height: 1.6;
`;

const SummaryItem = styled.li`
  margin-bottom: 0.35rem;
`;

const EventsTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
  color: #dce7ff;
`;

const EventsHeadCell = styled.th`
  text-align: left;
  padding: 0.6rem;
  background: rgba(30, 64, 175, 0.32);
  border-bottom: 1px solid rgba(79, 70, 229, 0.35);
`;

const EventsCell = styled.td`
  padding: 0.55rem 0.6rem;
  border-bottom: 1px solid rgba(79, 70, 229, 0.2);
`;

const EmptyState = styled.div`
  margin-top: 2rem;
  padding: 1.5rem;
  border-radius: 14px;
  border: 1px dashed rgba(99, 102, 241, 0.35);
  color: #9fb4ec;
  text-align: center;
`;

const Badge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-size: 0.78rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  background: rgba(56, 189, 248, 0.18);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: #7dd3fc;
`;

const DownloadLink = styled.a`
  color: #7dd3fc;
  font-weight: 700;
  text-decoration: none;
  &:hover {
    color: #bae6fd;
  }
`;

export default function ContentModeration() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [selectedCategories, setSelectedCategories] = useState(() => CATEGORY_OPTIONS.map((option) => option.key));
  const [minConfidence, setMinConfidence] = useState(75);
  const [processing, setProcessing] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const uploadProgressRef = useRef(0);
  const uploadProgressResetRef = useRef(null);

  const API_BASE = useMemo(() => resolveModerationApiBase(), []);
  const REQUEST_TIMEOUT = useMemo(() => {
    const envValue = Number(process.env.REACT_APP_MODERATION_TIMEOUT_MS);
    if (Number.isFinite(envValue) && envValue > 0) {
      return envValue;
    }
    return DEFAULT_TIMEOUT_MS;
  }, []);

  useEffect(() => () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    if (uploadProgressResetRef.current) {
      clearTimeout(uploadProgressResetRef.current);
      uploadProgressResetRef.current = null;
    }
  }, [previewUrl]);

  const handleFileSelection = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      setError('Please choose a video file.');
      return;
    }
    if (file.size > 2 * 1024 * 1024 * 1024) {
      setError('File too large. Maximum supported size is 2GB.');
      return;
    }
    setSelectedFile(file);
    setResult(null);
    setError('');
    setStatus('Ready to analyse.');
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  }, [previewUrl]);

  const onFileInputChange = (event) => {
    const file = event.target.files?.[0];
    handleFileSelection(file);
  };

  const onDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    handleFileSelection(file);
  };

  const toggleCategory = (category) => {
    setSelectedCategories((prev) => {
      if (prev.includes(category)) {
        return prev.filter((item) => item !== category);
      }
      return [...prev, category];
    });
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setResult(null);
    setError('');
    setStatus('');
    setUploadProgress(null);
    uploadProgressRef.current = 0;
    if (uploadProgressResetRef.current) {
      clearTimeout(uploadProgressResetRef.current);
      uploadProgressResetRef.current = null;
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl('');
    }
  };

  const analyseVideo = async () => {
    if (!selectedFile) {
      setError('Select a video to analyse.');
      return;
    }

    setProcessing(true);
    setStatus('Uploading video to moderation service...');
    setError('');
    setResult(null);
    if (uploadProgressResetRef.current) {
      clearTimeout(uploadProgressResetRef.current);
      uploadProgressResetRef.current = null;
    }
  setUploadProgress(1);
  uploadProgressRef.current = 1;

    const formData = new FormData();
    formData.append('video', selectedFile);
    if (selectedCategories.length && selectedCategories.length !== CATEGORY_OPTIONS.length) {
      formData.append('categories', selectedCategories.join(','));
    }
    if (minConfidence) {
      formData.append('min_confidence', String(minConfidence));
    }

    try {
      const endpoint = API_BASE ? `${API_BASE}/moderate` : '/moderate';
      const { data } = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: REQUEST_TIMEOUT,
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        onUploadProgress: (event) => {
          if (!event.total || !Number.isFinite(event.total)) {
            uploadProgressRef.current = Math.max(uploadProgressRef.current || 0, 1);
            setUploadProgress((prev) => {
              const baseline = prev ?? 0;
              return baseline >= 1 ? baseline : 1;
            });
            return;
          }
          const rawPercent = Math.round((event.loaded / event.total) * 100);
          const boundedPercent = Math.min(99, Math.max(1, rawPercent));
          if (boundedPercent <= uploadProgressRef.current) {
            return;
          }
          uploadProgressRef.current = boundedPercent;
          setUploadProgress(boundedPercent);
        },
      });
      uploadProgressRef.current = 100;
      setUploadProgress(100);
      setResult(data);
      setStatus('Moderation timeline generated successfully.');
    } catch (requestError) {
      const backendError = requestError.response?.data?.error || requestError.message;
      setError(`Failed to analyse video: ${backendError}`);
      setStatus('Moderation request failed.');
    } finally {
      setProcessing(false);
      uploadProgressResetRef.current = setTimeout(() => {
        setUploadProgress(null);
        uploadProgressResetRef.current = null;
      }, 400);
      uploadProgressRef.current = 0;
    }
  };

  const moderationEvents = result?.moderationEvents || [];
  const summary = result?.summary;
  const requestMeta = result?.request;
  const metadata = result?.metadata;

  return (
    <Page>
      <Title>Content Moderation Lab</Title>
      <Lead>
        Upload editorial cuts or social promos and flag safety issues in seconds. The analyser spots alcohol, tobacco,
        violence, weapons, hate imagery, and more—returning precise timestamps so your reviewers can jump straight to
        the risky frames.
      </Lead>

      <UploadCard htmlFor="moderation-video-input" onDrop={onDrop} onDragOver={(event) => event.preventDefault()}>
        <UploadIcon>🎯</UploadIcon>
        <UploadTitle>{selectedFile ? 'Replace video asset' : 'Drag & drop video or click to browse'}</UploadTitle>
        <UploadHint>MP4, MOV, MKV, WEBM — up to 2GB.</UploadHint>
        <HiddenInput
          id="moderation-video-input"
          type="file"
          accept="video/*"
          onChange={onFileInputChange}
        />
      </UploadCard>

      {selectedFile && (
        <ConfidenceRow>
          <span><strong>Selected:</strong> {selectedFile.name}</span>
          <span><strong>Size:</strong> {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</span>
          <span><strong>Type:</strong> {selectedFile.type || 'Unknown'}</span>
        </ConfidenceRow>
      )}

      {previewUrl && (
        <PreviewWrap>
          <PreviewVideo src={previewUrl} controls preload="metadata" />
          <span>Preview generated from the uploaded asset.</span>
        </PreviewWrap>
      )}

      <ConfidenceRow>
        <span><strong>Confidence threshold:</strong> {minConfidence}%</span>
        <Slider
          type="range"
          min="0"
          max="100"
          step="1"
          value={minConfidence}
          onChange={(event) => setMinConfidence(Number(event.target.value))}
        />
      </ConfidenceRow>

      <OptionsGrid>
        {CATEGORY_OPTIONS.map((option) => {
          const checked = selectedCategories.includes(option.key);
          return (
            <OptionCard key={option.key}>
              <OptionCheckbox
                type="checkbox"
                checked={checked}
                onChange={() => toggleCategory(option.key)}
              />
              <div>
                <div>{option.label}</div>
                <Badge>{checked ? 'Included' : 'Excluded'}</Badge>
              </div>
            </OptionCard>
          );
        })}
      </OptionsGrid>

      <ActionsRow>
        <PrimaryButton type="button" onClick={analyseVideo} disabled={processing || !selectedFile}>
          {processing ? 'Analysing…' : 'Analyse video'}
        </PrimaryButton>
        <SecondaryButton type="button" onClick={clearSelection} disabled={processing || !selectedFile}>
          Clear selection
        </SecondaryButton>
      </ActionsRow>

      {status && !error && <StatusBanner>{status}</StatusBanner>}
      {error && <StatusBanner $error>{error}</StatusBanner>}

      {processing && uploadProgress !== null && (
        <StatusBanner>
          Upload progress: {uploadProgress}%
        </StatusBanner>
      )}

      {result && (
        <ResultsGrid>
          <Card>
            <CardTitle>Summary</CardTitle>
            <SummaryList>
              <SummaryItem><strong>Job ID:</strong> {result.jobId}</SummaryItem>
              <SummaryItem><strong>Categories:</strong> {Array.isArray(requestMeta?.selectedCategories) ? requestMeta.selectedCategories.join(', ') : 'All recognised categories'}</SummaryItem>
              <SummaryItem><strong>Minimum confidence:</strong> {typeof requestMeta?.minConfidence === 'number' ? `${requestMeta.minConfidence}%` : 'Default'}</SummaryItem>
              <SummaryItem><strong>Total findings:</strong> {summary?.totalFindings || 0}</SummaryItem>
              {typeof metadata?.analysisDurationSeconds === 'number' && (
                <SummaryItem><strong>Analysis time:</strong> {metadata.analysisDurationSeconds}s</SummaryItem>
              )}
              {result.video?.objectKey && (
                <SummaryItem>
                  <strong>S3 object key:</strong> {result.video.objectKey}
                </SummaryItem>
              )}
            </SummaryList>
            {summary?.categories && Object.keys(summary.categories).length > 0 && (
              <>
                <Badge>Categories</Badge>
                <SummaryList>
                  {Object.entries(summary.categories).map(([category, count]) => (
                    <SummaryItem key={category}>{category}: {count}</SummaryItem>
                  ))}
                </SummaryList>
              </>
            )}
            {result.jobId && (
              <SummaryItem>
                <strong>Download report:</strong>{' '}
                <DownloadLink
                  href={(API_BASE ? `${API_BASE}/result/${result.jobId}` : `/result/${result.jobId}`)}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  JSON export
                </DownloadLink>
              </SummaryItem>
            )}
          </Card>

          <Card>
            <CardTitle>Moderation Timeline</CardTitle>
            {moderationEvents.length === 0 ? (
              <EmptyState>No moderation events matched the current filters.</EmptyState>
            ) : (
              <EventsTable>
                <thead>
                  <tr>
                    <EventsHeadCell>Timecode</EventsHeadCell>
                    <EventsHeadCell>Category</EventsHeadCell>
                    <EventsHeadCell>Label</EventsHeadCell>
                    <EventsHeadCell>Confidence</EventsHeadCell>
                  </tr>
                </thead>
                <tbody>
                  {moderationEvents.map((event, index) => (
                    <tr key={`${event.timestamp?.milliseconds}-${event.label}-${index}`}>
                      <EventsCell>{event.timestamp?.timecode || '00:00:00.000'}</EventsCell>
                      <EventsCell>{event.category || '—'}</EventsCell>
                      <EventsCell>{event.label || '—'}</EventsCell>
                      <EventsCell>{event.confidence ? `${event.confidence}%` : '—'}</EventsCell>
                    </tr>
                  ))}
                </tbody>
              </EventsTable>
            )}
          </Card>
        </ResultsGrid>
      )}

      {!result && !processing && !error && !selectedFile && (
        <EmptyState>
          Choose a clip, keep the relevant categories enabled, and we’ll produce a reviewer-ready moderation timeline in one click.
        </EmptyState>
      )}
    </Page>
  );
}
