import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const DEFAULT_PROFILES = [
  {
    id: 'action_enthusiast',
    label: 'Action Enthusiast',
    summary: 'High-intensity pacing, heroic set pieces, and adrenaline-fueled score cues.',
  },
  {
    id: 'family_viewer',
    label: 'Family Viewer',
    summary: 'Humor, warmth, ensemble moments, and inclusive storytelling arcs.',
  },
  {
    id: 'thriller_buff',
    label: 'Thriller Buff',
    summary: 'Mystery hooks, dramatic reveals, and escalating tension beats.',
  },
  {
    id: 'romance_devotee',
    label: 'Romance Devotee',
    summary: 'Intimate character moments, sweeping vistas, and emotive dialogue.',
  },
];

const DEFAULT_OPTIONS = {
  languages: ['en', 'es', 'fr', 'hi'],
  subtitleLanguages: ['en', 'es', 'fr', 'hi'],
  durations: [15, 30, 45, 60, 90],
  outputFormats: ['mp4', 'mov'],
};

const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000;

const formatFileSize = (bytes) => {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return null;
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const precision = unitIndex === 0 ? 0 : unitIndex === units.length - 1 ? 2 : 1;
  return `${size.toFixed(precision)} ${units[unitIndex]}`;
};

const resolveTrailerApiBase = () => {
  const envValue = process.env.REACT_APP_TRAILER_API_BASE;
  if (envValue) {
    return envValue.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined') {
    const { protocol, hostname } = window.location;
    const normalized = hostname === '0.0.0.0' ? '127.0.0.1' : hostname;
    const localHosts = new Set(['localhost', '127.0.0.1', '::1']);

    if (localHosts.has(normalized)) {
      return `${protocol}//${normalized}:5007`.replace(/\/$/, '');
    }

    return `${protocol}//${hostname}`.replace(/\/$/, '');
  }
  return '';
};

const Page = styled.section`
  max-width: 1100px;
  margin: 0 auto;
  padding: 2.75rem 1.5rem 3.5rem;
  color: #dce7ff;
`;

const Title = styled.h1`
  color: #ffffff;
  font-weight: 800;
  margin: 0 0 0.75rem 0;
  font-size: clamp(1.9rem, 4vw, 2.8rem);
  text-align: center;
`;

const Lead = styled.p`
  margin: 0 auto 2rem auto;
  max-width: 780px;
  text-align: center;
  line-height: 1.8;
  color: #b9c6f7;
`;

const UploadCard = styled.label`
  display: block;
  background: rgba(0, 0, 0, 0.92);
  border-radius: 16px;
  padding: 2.4rem 2.2rem;
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

const MetaRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1.4rem;
  color: #cbd5f5;
  font-size: 0.95rem;
`;

const SectionHeading = styled.h2`
  color: #ffffff;
  font-weight: 800;
  margin: 3rem 0 1.25rem;
  font-size: clamp(1.4rem, 3vw, 1.75rem);
`;

const ProfilesGrid = styled.div`
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
`;

const ProfileCard = styled.button`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  text-align: left;
  gap: 0.65rem;
  padding: 1rem 1.1rem;
  border-radius: 16px;
  border: 1px solid ${({ $active }) => ($active ? 'rgba(255, 255, 255, 0.55)' : 'rgba(255, 255, 255, 0.18)')};
  background: ${({ $active }) => ($active ? 'rgba(0, 0, 0, 0.9)' : 'rgba(0, 0, 0, 0.78)')};
  color: #dbe4ff;
  cursor: pointer;
  transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.45);
    box-shadow: 0 18px 34px rgba(0, 0, 0, 0.65);
  }
`;

const ProfileTitle = styled.span`
  font-weight: 700;
  font-size: 1.05rem;
  color: #ffffff;
`;

const ProfileSummary = styled.p`
  margin: 0;
  color: #c0cdff;
  line-height: 1.6;
  font-size: 0.92rem;
`;

const OptionsPanel = styled.div`
  display: grid;
  gap: 1rem;
  margin-top: 1.5rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
`;

const OptionGroup = styled.label`
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  font-size: 0.92rem;
  color: #b6c4f6;
`;

const SelectInput = styled.select`
  padding: 0.75rem 0.9rem;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: rgba(0, 0, 0, 0.92);
  color: #f1f5ff;
  font-size: 0.95rem;
  font-weight: 600;
  outline: none;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;

  &:focus {
    border-color: rgba(255, 255, 255, 0.45);
    box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.08);
  }
`;

const CheckboxRow = styled.label`
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-top: 0.4rem;
  font-size: 0.92rem;
  color: #bfcdfc;
`;

const ActionsRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.9rem;
  margin-top: 2.4rem;
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
    box-shadow: 0 16px 34px rgba(var(--button-accent-rgb), 0.35);
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
  border: 1px solid rgba(255, 255, 255, 0.22);
  background: rgba(0, 0, 0, 0.9);
  color: #dbeafe;
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    border-color: rgba(255, 255, 255, 0.45);
    box-shadow: 0 16px 30px rgba(0, 0, 0, 0.65);
  }

  &:disabled {
    opacity: 0.55;
    cursor: not-allowed;
    box-shadow: none;
  }
`;

const DownloadButton = styled(PrimaryButton)`
  text-decoration: none;
`;

const DownloadRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.85rem;
  margin-top: 1rem;
`;

const FileMeta = styled.span`
  color: #9fb4ff;
  font-size: 0.9rem;
`;

const StatusBanner = styled.div`
  margin-top: 1.25rem;
  padding: 1rem 1.2rem;
  border-radius: 14px;
  border: 1px solid ${({ $error }) => ($error ? 'rgba(248, 113, 113, 0.55)' : 'rgba(56, 189, 248, 0.45)')};
  background: ${({ $error }) => ($error ? 'rgba(60, 17, 34, 0.65)' : 'rgba(15, 48, 35, 0.6)')};
  color: ${({ $error }) => ($error ? '#fecaca' : '#bbf7d0')};
  line-height: 1.6;
`;

const VideoPreview = styled.video`
  margin-top: 2rem;
  width: min(640px, 100%);
  border-radius: 16px;
  box-shadow: 0 20px 44px rgba(9, 17, 31, 0.6);
`;

const RenderedVideo = styled(VideoPreview)`
  margin-top: 1rem;
`;

const JobSection = styled.section`
  margin-top: 3rem;
  padding: 1.6rem 1.4rem;
  border-radius: 18px;
  border: 1px solid rgba(99, 102, 241, 0.22);
  background: linear-gradient(160deg, rgba(13, 24, 46, 0.9), rgba(25, 40, 70, 0.84));
  box-shadow: 0 24px 46px rgba(6, 15, 30, 0.55);
`;

const SectionTitle = styled.h3`
  margin: 0 0 1.1rem 0;
  color: #f8fafc;
  font-size: 1.2rem;
  font-weight: 700;
`;

const MetricGrid = styled.div`
  display: grid;
  gap: 0.85rem;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
`;

const MetricCard = styled.div`
  padding: 0.9rem 1rem;
  border-radius: 14px;
  background: rgba(16, 32, 58, 0.85);
  border: 1px solid rgba(79, 70, 229, 0.3);
  color: #dbe4ff;
`;

const MetricLabel = styled.div`
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(148, 163, 255, 0.7);
`;

const MetricValue = styled.div`
  font-size: 1.05rem;
  font-weight: 700;
  margin-top: 0.3rem;
`;

const SceneList = styled.ul`
  margin: 1.2rem 0 0 0;
  padding-left: 1.2rem;
  display: grid;
  gap: 0.75rem;
`;

const SceneItem = styled.li`
  color: #c0ccff;
  line-height: 1.6;
`;

const SubsectionHeading = styled.h4`
  margin: 1.6rem 0 0.6rem;
  color: #e2e8ff;
  font-size: 1rem;
  font-weight: 700;
`;

const DownloadInlineLink = styled.a`
  margin-left: 0.6rem;
  color: #7dd3fc;
  font-size: 0.9rem;
  font-weight: 600;
  text-decoration: none;
  transition: color 0.2s ease;

  &:hover {
    color: #bae6fd;
    text-decoration: underline;
  }
`;

const ProviderPill = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  background: rgba(56, 189, 248, 0.16);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: #bae6fd;
  font-size: 0.85rem;
  font-weight: 600;
`;

const PillGroup = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
`;

const ProvidersRow = styled.div`
  margin-top: 1rem;
`; 

const ScrollContainer = styled.div`
  margin-top: 1rem;
  overflow-x: auto;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  min-width: 480px;
`;

const TableHead = styled.thead`
  background: rgba(27, 40, 72, 0.75);
`;

const TableHeaderCell = styled.th`
  text-align: left;
  padding: 0.75rem;
  font-size: 0.85rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: rgba(186, 214, 255, 0.85);
  border-bottom: 1px solid rgba(99, 102, 241, 0.35);
`;

const TableBodyCell = styled.td`
  padding: 0.75rem;
  border-bottom: 1px solid rgba(99, 102, 241, 0.18);
  color: #dbe4ff;
  font-size: 0.95rem;
`;

const EmptyState = styled.div`
  margin-top: 1.4rem;
  color: #9fb4ff;
  font-style: italic;
`;

export default function AIBasedTrailer() {
  const [profiles, setProfiles] = useState(DEFAULT_PROFILES);
  const [selectedProfileId, setSelectedProfileId] = useState(DEFAULT_PROFILES[0]?.id ?? null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [duration, setDuration] = useState(60);
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [subtitleLanguage, setSubtitleLanguage] = useState('en');
  const [outputFormat, setOutputFormat] = useState('mp4');
  const [includeCaptions, setIncludeCaptions] = useState(true);
  const [includeStoryboard, setIncludeStoryboard] = useState(true);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [processing, setProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const uploadProgressRef = useRef(0);
  const uploadProgressResetRef = useRef(null);
  const [job, setJob] = useState(null);

  const API_BASE = useMemo(() => resolveTrailerApiBase(), []);
  const REQUEST_TIMEOUT = useMemo(() => {
    const envValue = Number(process.env.REACT_APP_TRAILER_TIMEOUT_MS);
    if (Number.isFinite(envValue) && envValue > 0) {
      return envValue;
    }
    return DEFAULT_TIMEOUT_MS;
  }, []);

  const buildDeliverableUrl = useCallback((path, options = {}) => {
    if (!path) return null;
    const normalized = path.startsWith('/') ? path : `/${path}`;
    const base = API_BASE || '';
    let url = `${base}${normalized}`;
    if (options.download) {
      url += normalized.includes('?') ? '&download=true' : '?download=true';
    }
    return url;
  }, [API_BASE]);

  useEffect(() => {
    let cancelled = false;

    const fetchProfilesAndOptions = async () => {
      try {
        const endpoint = API_BASE ? `${API_BASE}/profiles` : '/profiles';
        const { data } = await axios.get(endpoint, { timeout: REQUEST_TIMEOUT });
        if (cancelled) return;
        if (Array.isArray(data?.profiles) && data.profiles.length > 0) {
          setProfiles(data.profiles);
          setSelectedProfileId((prev) => prev && data.profiles.some((item) => item.id === prev) ? prev : data.profiles[0].id);
        }
        if (data?.defaults) {
          const defaults = {
            languages: data.defaults.languages || DEFAULT_OPTIONS.languages,
            subtitleLanguages: data.defaults.subtitleLanguages || data.defaults.languages || DEFAULT_OPTIONS.subtitleLanguages,
            durations: data.defaults.durations || DEFAULT_OPTIONS.durations,
            outputFormats: data.defaults.outputFormats || DEFAULT_OPTIONS.outputFormats,
          };
          setOptions((prev) => ({ ...prev, ...defaults }));

          if (Array.isArray(defaults.durations) && defaults.durations.length > 0) {
            setDuration((prev) => (defaults.durations.includes(prev) ? prev : defaults.durations[0]));
          }
          if (Array.isArray(defaults.languages) && defaults.languages.length > 0) {
            setTargetLanguage((prev) => (defaults.languages.includes(prev) ? prev : defaults.languages[0]));
          }
          if (Array.isArray(defaults.subtitleLanguages) && defaults.subtitleLanguages.length > 0) {
            setSubtitleLanguage((prev) => (defaults.subtitleLanguages.includes(prev) ? prev : defaults.subtitleLanguages[0]));
          }
          if (Array.isArray(defaults.outputFormats) && defaults.outputFormats.length > 0) {
            setOutputFormat((prev) => (defaults.outputFormats.includes(prev) ? prev : defaults.outputFormats[0]));
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError('Unable to refresh profiles from backend. Using defaults.');
        }
      }

      try {
        const endpoint = API_BASE ? `${API_BASE}/options` : '/options';
        const { data } = await axios.get(endpoint, { timeout: REQUEST_TIMEOUT });
        if (cancelled) return;
        if (data) {
          setOptions((prev) => ({ ...prev, ...data }));
        }
      } catch (err) {
        // ignore secondary errors
      }
    };

    fetchProfilesAndOptions();

    return () => {
      cancelled = true;
    };
  }, [API_BASE, REQUEST_TIMEOUT]);

  useEffect(() => () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    if (uploadProgressResetRef.current) {
      clearTimeout(uploadProgressResetRef.current);
      uploadProgressResetRef.current = null;
    }
  }, [previewUrl, setStatus, setJob, setUploadProgress]);

  const handleFileSelection = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      setError('Please choose a video file.');
      return;
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError('');
    setStatus('');
    setJob(null);
    setUploadProgress(null);
    if (uploadProgressResetRef.current) {
      clearTimeout(uploadProgressResetRef.current);
      uploadProgressResetRef.current = null;
    }
    uploadProgressRef.current = 0;
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

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setStatus('');
    setError('');
    setJob(null);
    setUploadProgress(null);
    if (uploadProgressResetRef.current) {
      clearTimeout(uploadProgressResetRef.current);
      uploadProgressResetRef.current = null;
    }
    uploadProgressRef.current = 0;
  };

  const submitRequest = async () => {
    if (!selectedFile) {
      setError('Choose a video asset to create an AI-based trailer.');
      return;
    }
    if (!selectedProfileId) {
      setError('Select a viewer profile.');
      return;
    }

    setProcessing(true);
    setError('');
  setStatus('Uploading to Personalized Trailer service...');
    setJob(null);
    if (uploadProgressResetRef.current) {
      clearTimeout(uploadProgressResetRef.current);
      uploadProgressResetRef.current = null;
    }
    setUploadProgress(1);
    uploadProgressRef.current = 1;

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('profile_id', selectedProfileId);
    formData.append('max_duration', String(duration));
    formData.append('target_language', targetLanguage);
    formData.append('subtitle_language', subtitleLanguage);
    formData.append('output_format', outputFormat);
    formData.append('include_captions', includeCaptions ? 'true' : 'false');
    formData.append('include_storyboard', includeStoryboard ? 'true' : 'false');

    try {
      const endpoint = API_BASE ? `${API_BASE}/generate` : '/generate';
      const { data } = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: REQUEST_TIMEOUT,
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        onUploadProgress: (event) => {
          if (!event.total || !Number.isFinite(event.total)) {
            uploadProgressRef.current = Math.max(uploadProgressRef.current || 0, 1);
            setUploadProgress((prev) => (prev == null || prev < 1 ? 1 : prev));
            return;
          }
          const rawPercent = Math.round((event.loaded / event.total) * 100);
          const bounded = Math.min(99, Math.max(1, rawPercent));
          if (bounded <= uploadProgressRef.current) {
            return;
          }
          uploadProgressRef.current = bounded;
          setUploadProgress(bounded);
        },
      });

      uploadProgressRef.current = 100;
      setUploadProgress(100);
  setStatus('Personalized Trailer plan generated.');
      setJob(data?.job ?? null);
    } catch (requestError) {
      const backendError = requestError.response?.data?.error || requestError.message;
      setError(`Failed to create AI-based trailer: ${backendError}`);
  setStatus('Personalized Trailer request failed.');
    } finally {
      setProcessing(false);
      uploadProgressResetRef.current = setTimeout(() => {
        setUploadProgress(null);
        uploadProgressResetRef.current = null;
      }, 400);
      uploadProgressRef.current = 0;
    }
  };

  const activeProfile = profiles.find((profile) => profile.id === selectedProfileId) || null;
  const personalization = job?.personalization;
  const analysis = job?.analysis;
  const providers = job?.providers;
  const assembly = job?.assembly;
  const deliverables = job?.deliverables;
  const masterDeliverable = deliverables?.master;
  const masterStreamUrl = useMemo(
    () => buildDeliverableUrl(masterDeliverable?.downloadUrl),
    [buildDeliverableUrl, masterDeliverable?.downloadUrl]
  );
  const masterDownloadUrl = useMemo(
    () => buildDeliverableUrl(masterDeliverable?.downloadUrl, { download: true }),
    [buildDeliverableUrl, masterDeliverable?.downloadUrl]
  );
  const masterSizeLabel = useMemo(
    () => formatFileSize(masterDeliverable?.sizeBytes),
    [masterDeliverable?.sizeBytes]
  );

  return (
    <Page>
  <Title>Personalized Trailer Lab</Title>
      <Lead>
        Upload a hero cut, pick your audience, and instantly receive a tailored trailer blueprint with scene picks,
        assembly notes, and localization assets.
      </Lead>

      <UploadCard htmlFor="ai-based-trailer-video-input" onDrop={onDrop} onDragOver={(event) => event.preventDefault()}>
        <UploadIcon>🎞️</UploadIcon>
        <UploadTitle>{selectedFile ? 'Replace hero footage' : 'Drag & drop video or click to browse'}</UploadTitle>
        <UploadHint>MP4, MOV, MKV — up to 2GB.</UploadHint>
        <HiddenInput id="ai-based-trailer-video-input" type="file" accept="video/*" onChange={onFileInputChange} />
      </UploadCard>

      {selectedFile && (
        <MetaRow>
          <span><strong>Selected:</strong> {selectedFile.name}</span>
          <span><strong>Size:</strong> {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</span>
          <span><strong>Type:</strong> {selectedFile.type || 'Unknown'}</span>
        </MetaRow>
      )}

      <SectionHeading>Viewer profile</SectionHeading>
      <ProfilesGrid>
        {profiles.map((profile) => (
          <ProfileCard
            key={profile.id}
            type="button"
            onClick={() => setSelectedProfileId(profile.id)}
            $active={profile.id === selectedProfileId}
          >
            <ProfileTitle>{profile.label}</ProfileTitle>
            <ProfileSummary>{profile.summary}</ProfileSummary>
          </ProfileCard>
        ))}
      </ProfilesGrid>

      <SectionHeading>Trailer settings</SectionHeading>
      <OptionsPanel>
        <OptionGroup>
          <span>Maximum runtime (seconds)</span>
          <SelectInput value={duration} onChange={(event) => setDuration(Number(event.target.value))}>
            {options.durations.map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </SelectInput>
        </OptionGroup>
        <OptionGroup>
          <span>Target language</span>
          <SelectInput value={targetLanguage} onChange={(event) => setTargetLanguage(event.target.value)}>
            {options.languages.map((lang) => (
              <option key={lang} value={lang}>{lang.toUpperCase()}</option>
            ))}
          </SelectInput>
        </OptionGroup>
        <OptionGroup>
          <span>Subtitle language</span>
          <SelectInput value={subtitleLanguage} onChange={(event) => setSubtitleLanguage(event.target.value)}>
            {options.subtitleLanguages.map((lang) => (
              <option key={lang} value={lang}>{lang.toUpperCase()}</option>
            ))}
          </SelectInput>
          <CheckboxRow>
            <input
              type="checkbox"
              checked={includeCaptions}
              onChange={(event) => setIncludeCaptions(event.target.checked)}
            />
            Include captions file
          </CheckboxRow>
        </OptionGroup>
        <OptionGroup>
          <span>Master output format</span>
          <SelectInput value={outputFormat} onChange={(event) => setOutputFormat(event.target.value)}>
            {options.outputFormats.map((fmt) => (
              <option key={fmt} value={fmt}>{fmt.toUpperCase()}</option>
            ))}
          </SelectInput>
          <CheckboxRow>
            <input
              type="checkbox"
              checked={includeStoryboard}
              onChange={(event) => setIncludeStoryboard(event.target.checked)}
            />
            Generate storyboard summary
          </CheckboxRow>
        </OptionGroup>
      </OptionsPanel>

      <ActionsRow>
        <PrimaryButton type="button" onClick={submitRequest} disabled={processing}>
          {processing ? 'Generating plan…' : 'Generate trailer plan'}
        </PrimaryButton>
        <SecondaryButton type="button" onClick={clearSelection} disabled={processing}>
          Reset
        </SecondaryButton>
      </ActionsRow>

      {(status || error) && (
        <StatusBanner $error={Boolean(error)}>
          {error || status}
          {processing && uploadProgress !== null && (
            <div style={{ marginTop: '0.6rem', fontSize: '0.9rem' }}>
              Upload progress: {uploadProgress}%
            </div>
          )}
        </StatusBanner>
      )}

      {previewUrl && (
        <VideoPreview src={previewUrl} controls preload="metadata" />
      )}

      {job && (
        <>
          <JobSection>
            <SectionTitle>Job summary</SectionTitle>
            <MetricGrid>
              <MetricCard>
                <MetricLabel>Job ID</MetricLabel>
                <MetricValue>{job.jobId}</MetricValue>
              </MetricCard>
              <MetricCard>
                <MetricLabel>Profile</MetricLabel>
                <MetricValue>{activeProfile?.label || job.input?.profile?.label}</MetricValue>
              </MetricCard>
              <MetricCard>
                <MetricLabel>Duration target</MetricLabel>
                <MetricValue>{job.input?.maxDurationSeconds}s</MetricValue>
              </MetricCard>
              <MetricCard>
                <MetricLabel>Pipeline mode</MetricLabel>
                <MetricValue>{job.mode}</MetricValue>
              </MetricCard>
            </MetricGrid>

            {providers && (
              <ProvidersRow>
                <SectionTitle>Providers</SectionTitle>
                <PillGroup>
                  {Object.entries(providers).map(([key, details]) => (
                    <ProviderPill key={key}>
                      <span style={{ fontWeight: 700 }}>{key}</span>
                      <span style={{ opacity: 0.7 }}>{details.mode}</span>
                    </ProviderPill>
                  ))}
                </PillGroup>
              </ProvidersRow>
            )}
          </JobSection>

          {(() => {
            const variantKeys = deliverables ? Object.keys(deliverables).filter(k => k.startsWith('variant_')) : [];
            console.log('🎬 Deliverables:', deliverables);
            console.log('🎬 Variant keys found:', variantKeys);
            console.log('🎬 Number of variants:', variantKeys.length);
            return variantKeys.length > 0;
          })() ? (
            <JobSection>
              <SectionTitle>Rendered Trailer Variants ({Object.keys(deliverables).filter(k => k.startsWith('variant_')).length} versions)</SectionTitle>
              {Object.entries(deliverables)
                .filter(([key]) => key.startsWith('variant_'))
                .sort(([keyA], [keyB]) => keyA.localeCompare(keyB))
                .map(([key, variant]) => {
                  console.log('🎬 Rendering variant:', key, variant?.name);
                  const variantStreamUrl = buildDeliverableUrl(variant?.downloadUrl);
                  const variantDownloadUrl = buildDeliverableUrl(variant?.downloadUrl, { download: true });
                  const variantSizeLabel = formatFileSize(variant?.sizeBytes);
                  
                  return (
                    <div key={key} style={{ marginBottom: '2rem' }}>
                      <h4 style={{ 
                        margin: '1rem 0 0.5rem 0',
                        fontSize: '1.1rem',
                        fontWeight: 600,
                        color: '#1a1a1a'
                      }}>
                        {variant?.name || key}
                      </h4>
                      {variant?.description && (
                        <p style={{ 
                          margin: '0 0 0.5rem 0',
                          fontSize: '0.9rem',
                          color: '#666'
                        }}>
                          {variant.description}
                        </p>
                      )}
                      {variant?.distribution && (
                        <p style={{ 
                          margin: '0 0 0.8rem 0',
                          fontSize: '0.85rem',
                          color: '#888',
                          fontStyle: 'italic'
                        }}>
                          Distribution: {Object.entries(variant.distribution).map(([k, v]) => `${k}: ${v}`).join(', ')}
                        </p>
                      )}
                      {variantStreamUrl && (
                        <>
                          <RenderedVideo controls src={variantStreamUrl} preload="metadata" />
                          <DownloadRow>
                            <DownloadButton
                              as="a"
                              href={variantDownloadUrl || variantStreamUrl}
                              download
                            >
                              Download {variant?.name || key}
                            </DownloadButton>
                            {variantSizeLabel && <FileMeta>≈ {variantSizeLabel}</FileMeta>}
                          </DownloadRow>
                        </>
                      )}
                    </div>
                  );
                })}
            </JobSection>
          ) : masterStreamUrl ? (
            <JobSection>
              <SectionTitle>Rendered trailer</SectionTitle>
              <RenderedVideo controls src={masterStreamUrl} preload="metadata" />
              <DownloadRow>
                <DownloadButton
                  as="a"
                  href={(masterDownloadUrl || masterStreamUrl) ?? undefined}
                  download
                >
                  Download trailer
                </DownloadButton>
                {masterSizeLabel && <FileMeta>≈ {masterSizeLabel}</FileMeta>}
              </DownloadRow>
            </JobSection>
          ) : (
            masterDeliverable?.note && (
              <JobSection>
                <SectionTitle>Rendered trailer</SectionTitle>
                <EmptyState>{masterDeliverable.note}</EmptyState>
              </JobSection>
            )
          )}

          {analysis && (
            <JobSection>
              <SectionTitle>Scene intelligence</SectionTitle>
              <MetricGrid>
                <MetricCard>
                  <MetricLabel>Total runtime analysed</MetricLabel>
                  <MetricValue>{analysis.totalDuration}s</MetricValue>
                </MetricCard>
                <MetricCard>
                  <MetricLabel>Detected people</MetricLabel>
                  <MetricValue>{analysis.metrics?.detectedPeople}</MetricValue>
                </MetricCard>
                <MetricCard>
                  <MetricLabel>Detected objects</MetricLabel>
                  <MetricValue>{analysis.metrics?.detectedObjects}</MetricValue>
                </MetricCard>
                <MetricCard>
                  <MetricLabel>Dominant emotions</MetricLabel>
                  <MetricValue>{analysis.dominantEmotions?.join(', ')}</MetricValue>
                </MetricCard>
              </MetricGrid>

              {analysis.scenes?.length ? (
                <ScrollContainer>
                  <Table>
                    <TableHead>
                      <tr>
                        <TableHeaderCell>Scene</TableHeaderCell>
                        <TableHeaderCell>Window (s)</TableHeaderCell>
                        <TableHeaderCell>Emotions</TableHeaderCell>
                        <TableHeaderCell>Labels</TableHeaderCell>
                      </tr>
                    </TableHead>
                    <tbody>
                      {analysis.scenes.map((scene) => (
                        <tr key={scene.sceneId}>
                          <TableBodyCell>{scene.sceneId}</TableBodyCell>
                          <TableBodyCell>{scene.start}–{scene.end}</TableBodyCell>
                          <TableBodyCell>{scene.emotions.join(', ')}</TableBodyCell>
                          <TableBodyCell>{scene.labels.join(', ')}</TableBodyCell>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </ScrollContainer>
              ) : (
                <EmptyState>No scenes detected.</EmptyState>
              )}
            </JobSection>
          )}

          {personalization && (
            <JobSection>
              <SectionTitle>Personalization ranking</SectionTitle>
              <MetricGrid>
                <MetricCard>
                  <MetricLabel>Selected scenes</MetricLabel>
                  <MetricValue>{personalization.selectedScenes?.length ?? 0}</MetricValue>
                </MetricCard>
                <MetricCard>
                  <MetricLabel>Estimated runtime</MetricLabel>
                  <MetricValue>{personalization.estimatedDuration ?? 0}s</MetricValue>
                </MetricCard>
              </MetricGrid>
              {personalization.rankedScenes?.length ? (
                <>
                  <SubsectionHeading>Top ranked scenes</SubsectionHeading>
                  <SceneList>
                    {personalization.rankedScenes.slice(0, 6).map((scene) => (
                      <SceneItem key={scene.sceneId}>
                        <strong>{scene.sceneId}</strong> — score {scene.score} ({scene.duration}s)
                      </SceneItem>
                    ))}
                  </SceneList>
                </>
              ) : (
                <EmptyState>No ranked scenes available.</EmptyState>
              )}
              {personalization.selectedScenes?.length ? (
                <>
                  <SubsectionHeading>Selected trailer sequence</SubsectionHeading>
                <SceneList>
                  {personalization.selectedScenes.map((scene) => (
                    <SceneItem key={scene.sceneId}>
                      <strong>{scene.sceneId}</strong> — score {scene.score} ({scene.duration}s)
                    </SceneItem>
                  ))}
                </SceneList>
                </>
              ) : (
                <EmptyState>No scenes selected for the trailer window.</EmptyState>
              )}
            </JobSection>
          )}

          {assembly && (
            <JobSection>
              <SectionTitle>Assembly plan</SectionTitle>
              <MetricGrid>
                <MetricCard>
                  <MetricLabel>Estimated duration</MetricLabel>
                  <MetricValue>{assembly.estimatedDuration}s</MetricValue>
                </MetricCard>
                <MetricCard>
                  <MetricLabel>Renditions</MetricLabel>
                  <MetricValue>{assembly.renditions?.length ?? 0}</MetricValue>
                </MetricCard>
              </MetricGrid>
              {assembly.timeline?.length ? (
                <SceneList>
                  {assembly.timeline.map((clip) => (
                    <SceneItem key={`${clip.sceneId}-${clip.in}`}>
                      {clip.sceneId}: {clip.in}s → {clip.out}s (transition: {clip.transition}, audio cue: {clip.audioCue})
                    </SceneItem>
                  ))}
                </SceneList>
              ) : (
                <EmptyState>No timeline entries generated.</EmptyState>
              )}
            </JobSection>
          )}

          {deliverables && (
            <JobSection>
              <SectionTitle>Deliverables</SectionTitle>
              <SceneList>
                {Object.entries(deliverables).map(([key, details]) => {
                  let description = '';

                  if (key === 'summary' && details) {
                    const parts = [];
                    if (details.targetLanguage) parts.push(`target ${details.targetLanguage}`);
                    if (details.subtitleLanguage) parts.push(`subtitles ${details.subtitleLanguage}`);
                    if (details.generatedAt) {
                      const generatedDate = new Date(details.generatedAt);
                      if (!Number.isNaN(generatedDate.valueOf())) {
                        parts.push(`generated ${generatedDate.toLocaleString()}`);
                      } else {
                        parts.push(`generated ${details.generatedAt}`);
                      }
                    }
                    description = parts.join(' • ');
                  }

                  if (!description) {
                    if (details.path) {
                      description = `stored at ${details.path}`;
                    }
                    if (details.note) {
                      description = description ? `${description} — ${details.note}` : details.note;
                    }
                  }

                  if (key === 'captions' && details.language) {
                    description = description ? `${description} (language: ${details.language})` : `language: ${details.language}`;
                  }

                  if (key === 'storyboard' && Number.isFinite(details.frameCount)) {
                    description = description ? `${description} (${details.frameCount} frames)` : `${details.frameCount} frames`;
                  }

                  if (!description) {
                    description = 'ready';
                  }

                  let inlineDownloadUrl = null;
                  if (key !== 'master') {
                    const downloadHref = buildDeliverableUrl(details.downloadUrl, { download: true });
                    inlineDownloadUrl = downloadHref || buildDeliverableUrl(details.downloadUrl);
                  }

                  return (
                    <SceneItem key={key}>
                      <strong>{key}</strong> — {description}
                      {inlineDownloadUrl && (
                        <DownloadInlineLink
                          href={inlineDownloadUrl}
                          download
                          rel="noopener noreferrer"
                        >
                          Download
                        </DownloadInlineLink>
                      )}
                    </SceneItem>
                  );
                })}
              </SceneList>
            </JobSection>
          )}
        </>
      )}
    </Page>
  );
}
