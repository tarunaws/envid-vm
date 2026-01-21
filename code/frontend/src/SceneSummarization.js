import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const DEFAULT_TIMEOUT_MS = 3 * 60 * 60 * 1000; // 3 hours

const resolveRequestTimeout = () => {
  const envValue = process.env.REACT_APP_SCENE_TIMEOUT_MS;
  if (!envValue) {
    return DEFAULT_TIMEOUT_MS;
  }
  const parsed = Number(envValue);
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed;
  }
  return DEFAULT_TIMEOUT_MS;
};

const resolveSceneApiBase = () => {
  const envValue = process.env.REACT_APP_SCENE_API_BASE;

  const normalise = (value) => value.replace(/\/$/, '');

  if (envValue) {
    const trimmed = envValue.trim();
    if (typeof window !== 'undefined') {
      try {
        const url = new URL(trimmed, window.location.origin);
        const sameHost = url.hostname === window.location.hostname;
        if (sameHost && window.location.protocol === 'https:' && url.protocol === 'http:') {
          url.protocol = window.location.protocol;
          if (!url.port && window.location.port) {
            url.port = window.location.port;
          }
        }
        return normalise(url.href);
      } catch (error) {
        return normalise(trimmed);
      }
    }
    return normalise(trimmed);
  }

  if (typeof window !== 'undefined') {
    const { protocol, hostname, port } = window.location;
    const localHosts = new Set(['localhost', '127.0.0.1', '0.0.0.0']);
    const isLanHost = /^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1]))/.test(hostname) || hostname.endsWith('.local');
    if (localHosts.has(hostname) || isLanHost) {
      return `${protocol}//${hostname}:5004`;
    }
    const origin = `${protocol}//${hostname}${port ? `:${port}` : ''}`;
    return origin;
  }
  return '';
};

const Page = styled.section`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.75rem 1.5rem 3.5rem;
  color: #dce7ff;
`;

const Title = styled.h1`
  color: #ffffff;
  font-weight: 800;
  margin: 0 0 0.75rem 0;
  font-size: clamp(1.8rem, 3.6vw, 2.5rem);
  text-align: center;
`;

const Lead = styled.p`
  margin: 0 auto 2.5rem auto;
  max-width: 760px;
  text-align: center;
  line-height: 1.8;
  color: #b8c9f5;
`;

const UploadCard = styled.label`
  display: block;
  background: rgba(0, 0, 0, 0.92);
  border-radius: 18px;
  padding: 2.6rem 2.2rem;
  border: 2px dashed rgba(255, 255, 255, 0.18);
  text-align: center;
  cursor: pointer;
  transition: border-color 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
  box-shadow: 0 20px 48px rgba(7, 14, 28, 0.55);

  &:hover,
  &.dragover {
    border-color: rgba(255, 255, 255, 0.45);
    box-shadow: 0 26px 58px rgba(17, 36, 64, 0.6);
    background: rgba(0, 0, 0, 0.96);
  }
`;

const UploadIcon = styled.div`
  font-size: 3.4rem;
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
  flex-wrap: wrap;
  gap: 0.85rem;
  margin-top: 1.75rem;
`;

const PrimaryButton = styled.button`
  padding: 0.8rem 1.45rem;
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
  padding: 0.8rem 1.4rem;
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
  margin-top: 1.4rem;
  padding: 1rem 1.2rem;
  border-radius: 14px;
  border: 1px solid ${({ $error }) => ($error ? 'rgba(248, 113, 113, 0.55)' : 'rgba(96, 165, 250, 0.45)')};
  background: ${({ $error }) => ($error ? 'rgba(60, 17, 34, 0.65)' : 'rgba(15, 48, 35, 0.6)')};
  color: ${({ $error }) => ($error ? '#fecaca' : '#bbf7d0')};
  line-height: 1.6;
`;

const ResultGrid = styled.div`
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

const SummaryText = styled.p`
  margin: 0;
  color: var(--text-muted);
  line-height: 1.75;
`;

const HighlightList = styled.ul`
  margin: 0;
  padding-left: 1.25rem;
  color: #dce7ff;
  line-height: 1.65;
`;

const HighlightItem = styled.li`
  margin-bottom: 0.45rem;
`;

const SectionSubhead = styled.h3`
  margin: 0.85rem 0 0.6rem;
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: rgba(191, 219, 254, 0.85);
`;

const TagList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
`;

const Tag = styled.span`
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  background: rgba(56, 189, 248, 0.18);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: #7dd3fc;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
`;

const MetadataList = styled.ul`
  margin: 0;
  padding-left: 1.05rem;
  color: #c8d5ff;
  line-height: 1.6;
`;

const MetadataItem = styled.li`
  margin-bottom: 0.35rem;
`;

const AudioPlayer = styled.audio`
  width: 100%;
`;

const AudioNote = styled.p`
  margin: 0;
  color: #8ea2d6;
  font-size: 0.85rem;
`;

const SsmlPre = styled.pre`
  margin: 0;
  padding: 1.2rem 1.3rem;
  border-radius: 14px;
  background: rgba(8, 18, 34, 0.92);
  border: 1px solid rgba(56, 189, 248, 0.22);
  color: #e0ecff;
  font-size: 0.88rem;
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  overflow-x: hidden;
  font-family: 'Source Code Pro', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  max-height: 440px;
  overflow-y: auto;
`;

const JsonPre = styled.pre`
  margin: 0;
  padding: 1rem 1.1rem;
  border-radius: 12px;
  background: rgba(10, 22, 40, 0.92);
  border: 1px solid rgba(79, 70, 229, 0.32);
  color: #cfd9ff;
  font-size: 0.82rem;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  overflow-x: hidden;
`;

const FileMeta = styled.div`
  margin-top: 1.1rem;
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  color: #9fb4ec;
  font-size: 0.9rem;
`;

const VoiceSelect = styled.select`
  padding: 0.6rem 0.85rem;
  border-radius: 12px;
  border: 1px solid rgba(99, 102, 241, 0.25);
  background: rgba(8, 18, 34, 0.85);
  color: #e2e9f5;
  font-weight: 600;
  font-size: 0.92rem;
  cursor: pointer;
  min-width: 220px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;

  &:focus {
    outline: none;
    border-color: rgba(56, 189, 248, 0.6);
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.12);
  }
`;

const VoiceLabel = styled.span`
  font-size: 0.9rem;
  color: #cbd5f5;
`;

const VoiceRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  margin-top: 1.2rem;
`;

const PreviewImage = styled.img`
  max-width: min(520px, 100%);
  border-radius: 14px;
  border: 1px solid rgba(99, 102, 241, 0.28);
  box-shadow: 0 18px 36px rgba(6, 15, 30, 0.55);
`;

const PreviewVideo = styled.video`
  max-width: min(520px, 100%);
  border-radius: 14px;
  border: 1px solid rgba(99, 102, 241, 0.28);
  box-shadow: 0 18px 36px rgba(6, 15, 30, 0.55);
`;

const PreviewWrap = styled.div`
  margin-top: 1.6rem;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  align-items: center;
  color: #9fb4ec;
`;

const SectionDivider = styled.hr`
  margin: 2.5rem 0 2rem;
  border: none;
  border-top: 1px solid rgba(56, 189, 248, 0.18);
`;

const TimeoutNote = styled.p`
  margin-top: 0.75rem;
  color: rgba(148, 198, 255, 0.85);
  font-size: 0.85rem;
  line-height: 1.5;
`;

const ProgressContainer = styled.div`
  margin-top: 1.1rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
`;

const ProgressTrack = styled.div`
  width: 100%;
  background: rgba(15, 23, 42, 0.7);
  border-radius: 999px;
  border: 1px solid rgba(59, 130, 246, 0.35);
  height: 10px;
  overflow: hidden;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: ${({ $error }) => ($error ? 'linear-gradient(135deg, #f87171, #ef4444)' : 'linear-gradient(135deg, #38bdf8, #6366f1)')};
  width: ${({ $percent }) => `${Math.max(0, Math.min(100, $percent || 0))}%`};
  transition: width 0.2s ease-out;
`;

const ProgressLabel = styled.span`
  font-size: 0.82rem;
  color: rgba(191, 219, 254, 0.8);
`;

const PhaseList = styled.ul`
  margin: 0.6rem 0 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
`;

const PhaseItem = styled.li`
  display: flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.45rem 0.9rem;
  border-radius: 999px;
  border: 1px solid ${({ $status }) => (PHASE_STATUS_THEME[$status] || PHASE_STATUS_THEME.pending).border};
  background: ${({ $status }) => (PHASE_STATUS_THEME[$status] || PHASE_STATUS_THEME.pending).background};
  color: ${({ $status }) => (PHASE_STATUS_THEME[$status] || PHASE_STATUS_THEME.pending).color};
  font-size: 0.82rem;
  font-weight: 600;
  transition: background 0.2s ease, border-color 0.2s ease;
`;

const PhaseDot = styled.span`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${({ $status }) => (PHASE_STATUS_THEME[$status] || PHASE_STATUS_THEME.pending).dot};
  box-shadow: 0 0 0 2px ${({ $status }) => (PHASE_STATUS_THEME[$status] || PHASE_STATUS_THEME.pending).glow};
`;

const metadataLabel = (entry) => {
  const label = entry.label || 'Unknown';
  const confidence = typeof entry.confidence === 'number' ? entry.confidence : 0;
  return `${label} (${Math.round(confidence)}%)`;
};

const formatDuration = (ms) => {
  if (!Number.isFinite(ms) || ms <= 0) {
    return '';
  }
  const totalSeconds = Math.ceil(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const parts = [];
  if (hours) parts.push(`${hours}h`);
  if (minutes) parts.push(`${minutes}m`);
  if (!hours && seconds) parts.push(`${seconds}s`);
  if (!parts.length) {
    parts.push(`${seconds}s`);
  }
  return parts.join(' ');
};

const PROCESSING_PHASES = [
  { key: 'uploading', label: 'Upload media' },
  { key: 'analysing', label: 'Detect scene intelligence' },
  { key: 'summarising', label: 'Craft narrative summary' },
  { key: 'synthesising', label: 'Render voiceover preview' },
  { key: 'complete', label: 'Summary ready' },
];

const PHASE_LABEL_MAP = {
  idle: 'Waiting for media selection',
  uploading: 'Uploading media to the scene lab',
  analysing: 'Analysing scene intelligence with our vision stack',
  summarising: 'Crafting narrative summary with our language studio',
  synthesising: 'Rendering voiceover with neural speech synthesis',
  complete: 'Scene summary ready',
  error: 'Scene analysis encountered an issue',
};

const PHASE_STATUS_THEME = {
  complete: {
    border: 'rgba(34, 197, 94, 0.65)',
    background: 'rgba(21, 128, 61, 0.22)',
    color: '#bbf7d0',
    dot: '#34d399',
    glow: 'rgba(16, 185, 129, 0.45)',
  },
  active: {
    border: 'rgba(59, 130, 246, 0.6)',
    background: 'rgba(30, 64, 175, 0.28)',
    color: '#cbd5f5',
    dot: '#60a5fa',
    glow: 'rgba(96, 165, 250, 0.35)',
  },
  pending: {
    border: 'rgba(148, 163, 254, 0.25)',
    background: 'rgba(15, 23, 42, 0.55)',
    color: 'rgba(191, 219, 254, 0.78)',
    dot: 'rgba(148, 163, 254, 0.6)',
    glow: 'rgba(129, 140, 248, 0.18)',
  },
  error: {
    border: 'rgba(248, 113, 113, 0.7)',
    background: 'rgba(127, 29, 29, 0.32)',
    color: '#fecaca',
    dot: '#f87171',
    glow: 'rgba(248, 113, 113, 0.35)',
  },
};

const voiceOptions = [
  { value: 'Joanna', label: 'Joanna — Warm narrator' },
  { value: 'Matthew', label: 'Matthew — Documentary tone' },
  { value: 'Amy', label: 'Amy — Conversational upbeat' },
  { value: 'Brian', label: 'Brian — Cinematic storyteller' },
  { value: 'Lupe', label: 'Lupe — Spanish bilingual' },
  { value: 'Raveena', label: 'Raveena — Indian English' },
];

export default function SceneSummarization() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [voiceId, setVoiceId] = useState(voiceOptions[0].value);
  const [dragOver, setDragOver] = useState(false);
  const [mediaDurationSec, setMediaDurationSec] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [processingPhase, setProcessingPhase] = useState('idle');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [failurePhase, setFailurePhase] = useState(null);

  const phaseTimersRef = useRef([]);
  const uploadFallbackTimerRef = useRef(null);
  const lastPhaseRef = useRef('idle');
  const processingPhaseRef = useRef('idle');
  const timelineScheduledRef = useRef(false);

  const SCENE_API_BASE = useMemo(() => resolveSceneApiBase(), []);
  const REQUEST_TIMEOUT = useMemo(() => resolveRequestTimeout(), []);
  const selectedFileType = selectedFile?.type || '';
  const derivedTimeoutMs = useMemo(() => {
    if (selectedFileType.startsWith('video/') && mediaDurationSec) {
      return Math.ceil(mediaDurationSec * 5 * 1000);
    }
    return REQUEST_TIMEOUT;
  }, [selectedFileType, mediaDurationSec, REQUEST_TIMEOUT]);
  const formattedTimeout = useMemo(() => {
    if (!derivedTimeoutMs) {
      return null;
    }
    return formatDuration(derivedTimeoutMs);
  }, [derivedTimeoutMs]);
  const usingDurationTimeout = selectedFileType.startsWith('video/') && Boolean(mediaDurationSec);

  const clearPhaseTimers = useCallback(() => {
    phaseTimersRef.current.forEach((timerId) => clearTimeout(timerId));
    phaseTimersRef.current = [];
    timelineScheduledRef.current = false;
  }, []);

  const clearUploadFallbackTimer = useCallback(() => {
    if (uploadFallbackTimerRef.current) {
      clearTimeout(uploadFallbackTimerRef.current);
      uploadFallbackTimerRef.current = null;
    }
  }, []);

  const updateProcessingPhase = useCallback((nextPhase) => {
    setProcessingPhase((prev) => (prev === nextPhase ? prev : nextPhase));
    if (nextPhase === 'error') {
      setFailurePhase(lastPhaseRef.current);
      return;
    }
    if (nextPhase === 'idle') {
      lastPhaseRef.current = 'idle';
      setFailurePhase(null);
      return;
    }
    if (nextPhase === 'complete') {
      lastPhaseRef.current = 'complete';
      setFailurePhase(null);
      return;
    }
    lastPhaseRef.current = nextPhase;
    setFailurePhase(null);
  }, [setProcessingPhase, setFailurePhase]);

  useEffect(() => {
    processingPhaseRef.current = processingPhase;
  }, [processingPhase]);

  useEffect(() => () => {
    clearPhaseTimers();
    clearUploadFallbackTimer();
  }, [clearPhaseTimers, clearUploadFallbackTimer]);

  const scheduleProcessingPhases = useCallback(() => {
    if (timelineScheduledRef.current) {
      return;
    }
    clearPhaseTimers();
    if (processingPhaseRef.current === 'complete' || processingPhaseRef.current === 'error') {
      return;
    }

    const timers = [
      setTimeout(() => {
        if (processingPhaseRef.current === 'complete' || processingPhaseRef.current === 'error') return;
        updateProcessingPhase('summarising');
        setProcessingProgress((prev) => (prev < 70 ? 70 : prev));
        setStatus('Crafting narrative summary with our language studio...');
      }, 4500),
      setTimeout(() => {
        if (processingPhaseRef.current === 'complete' || processingPhaseRef.current === 'error') return;
        updateProcessingPhase('synthesising');
        setProcessingProgress((prev) => (prev < 85 ? 85 : prev));
        setStatus('Rendering voiceover with neural speech synthesis...');
      }, 9000),
      setTimeout(() => {
        if (processingPhaseRef.current === 'complete' || processingPhaseRef.current === 'error') return;
        setProcessingProgress((prev) => (prev < 94 ? 94 : prev));
      }, 13000),
    ];

    phaseTimersRef.current = timers;
    timelineScheduledRef.current = true;
    }, [clearPhaseTimers, updateProcessingPhase, setStatus]);

  const startUploadFallbackTimer = useCallback(() => {
    clearUploadFallbackTimer();
    uploadFallbackTimerRef.current = setTimeout(() => {
      if (processingPhaseRef.current === 'uploading') {
        updateProcessingPhase('analysing');
        setProcessingProgress((prev) => (prev < 55 ? 55 : prev));
        setStatus('Analysing scene intelligence with our vision stack...');
        scheduleProcessingPhases();
      }
    }, 2500);
    }, [clearUploadFallbackTimer, scheduleProcessingPhases, updateProcessingPhase, setStatus]);

  const phaseItems = useMemo(() => {
    const effectivePhase = processingPhase === 'error' && failurePhase ? failurePhase : processingPhase;
    const currentIndex = PROCESSING_PHASES.findIndex((phase) => phase.key === effectivePhase);
    const failureIndex = processingPhase === 'error' && failurePhase
      ? PROCESSING_PHASES.findIndex((phase) => phase.key === failurePhase)
      : -1;

    return PROCESSING_PHASES.map((phase, index) => {
      let status = 'pending';

      if (processingPhase === 'error') {
        if (failureIndex !== -1 && index < failureIndex) {
          status = 'complete';
        } else if (failureIndex !== -1 && index === failureIndex) {
          status = 'error';
        }
      } else if (currentIndex !== -1) {
        if (index < currentIndex) {
          status = 'complete';
        } else if (index === currentIndex) {
          status = processingPhase === 'complete' ? 'complete' : 'active';
        }
      }

      if (processingPhase === 'complete' && phase.key === 'complete') {
        status = 'complete';
      }

      return { ...phase, status };
    });
  }, [processingPhase, failurePhase]);

  const currentPhaseLabel = PHASE_LABEL_MAP[processingPhase] || PHASE_LABEL_MAP.idle;
  const overallProgressPercent = Math.max(0, Math.min(100, processingProgress || 0));
  const progressBarPercent = processingPhase === 'error'
    ? (overallProgressPercent > 0 ? overallProgressPercent : 100)
    : overallProgressPercent;
  const showProgressUI = processingPhase !== 'idle';
  const showUploadBar = uploadProgress !== null && processingPhase !== 'complete' && processingPhase !== 'error';

  const resolveUrl = useCallback((path) => {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) {
      return path;
    }
    if (!SCENE_API_BASE) {
      return path;
    }
    return `${SCENE_API_BASE}${path}`;
  }, [SCENE_API_BASE]);

  const resetState = useCallback(() => {
    clearPhaseTimers();
    clearUploadFallbackTimer();
    setStatus('');
    setError('');
    setResult(null);
    setUploadProgress(null);
    setProcessingProgress(0);
    setMediaDurationSec(null);
    updateProcessingPhase('idle');
  }, [clearPhaseTimers, clearUploadFallbackTimer, updateProcessingPhase]);

  const computeVideoDuration = useCallback((file, objectUrl) => {
    if (!file) {
      setMediaDurationSec(null);
      return;
    }

    let createdUrl = objectUrl;
    if (!createdUrl) {
      createdUrl = URL.createObjectURL(file);
    }
    const videoElement = document.createElement('video');
    videoElement.preload = 'metadata';
    videoElement.src = createdUrl;

    const cleanup = () => {
      videoElement.removeAttribute('src');
      videoElement.load();
      if (!objectUrl && createdUrl) {
        URL.revokeObjectURL(createdUrl);
      }
    };

    videoElement.onloadedmetadata = () => {
      const duration = Number.isFinite(videoElement.duration) ? videoElement.duration : null;
      setMediaDurationSec(duration && duration > 0 ? duration : null);
      cleanup();
    };

    videoElement.onerror = () => {
      setMediaDurationSec(null);
      cleanup();
    };
  }, []);

  const handleFileSelection = useCallback((file) => {
    if (!file) return;
    const isSupported = file.type.startsWith('image/') || file.type.startsWith('video/');
    if (!isSupported) {
      setError('Please choose an image or video file.');
      return;
    }

    if (file.size > 2 * 1024 * 1024 * 1024) {
      setError('File too large. Maximum supported size is 2GB.');
      return;
    }

    setSelectedFile(file);
    resetState();

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl('');
    }

    if (file.type.startsWith('image/')) {
      setPreviewUrl(URL.createObjectURL(file));
      setMediaDurationSec(null);
    } else if (file.type.startsWith('video/')) {
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
      computeVideoDuration(file, objectUrl);
    }
  }, [previewUrl, resetState, computeVideoDuration]);

  const onFileInputChange = (event) => {
    const file = event.target.files?.[0];
    handleFileSelection(file);
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    handleFileSelection(file);
  };

  const onDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const onDragLeave = (event) => {
    event.preventDefault();
    setDragOver(false);
  };

  useEffect(() => () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  }, [previewUrl]);

  const clearSelection = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl('');
    }
    resetState();
  };

  const analyseScene = async () => {
    if (!selectedFile) {
      setError('Select a video or image to begin.');
      return;
    }

    clearPhaseTimers();
    clearUploadFallbackTimer();
    updateProcessingPhase('uploading');
    timelineScheduledRef.current = false;
    setProcessingProgress(0);
    setProcessing(true);
    setStatus(PHASE_LABEL_MAP.uploading);
    setError('');
    setResult(null);
    setUploadProgress(0);
    startUploadFallbackTimer();

    const formData = new FormData();
    formData.append('media', selectedFile);
    formData.append('voice_id', voiceId);

    try {
      const endpoint = SCENE_API_BASE ? `${SCENE_API_BASE}/summarize` : '/summarize';
      const computedTimeout = derivedTimeoutMs || REQUEST_TIMEOUT;

      const { data } = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: computedTimeout,
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        onUploadProgress: (progressEvent) => {
          if (!progressEvent.total) {
            setUploadProgress(null);
            return;
          }
          const percent = Math.min(100, Math.round((progressEvent.loaded / progressEvent.total) * 100));
          setUploadProgress(percent);
          setProcessingProgress((prev) => {
            const uploadContribution = Math.round(percent * 0.4);
            return uploadContribution > prev ? uploadContribution : prev;
          });
          if (percent >= 100) {
            clearUploadFallbackTimer();
            updateProcessingPhase('analysing');
            setProcessingProgress((prev) => (prev < 55 ? 55 : prev));
            setStatus(PHASE_LABEL_MAP.analysing);
            scheduleProcessingPhases();
          }
        },
      });
      clearPhaseTimers();
      clearUploadFallbackTimer();
      updateProcessingPhase('complete');
      setProcessingProgress(100);
      setResult(data);
      setStatus('Scene summary generated successfully.');
    } catch (requestError) {
      clearPhaseTimers();
      clearUploadFallbackTimer();
      updateProcessingPhase('error');
      setProcessingProgress((prev) => (prev > 0 ? prev : 0));
      const backendMessage = requestError.response?.data?.error || requestError.message;
      setError(`Failed to summarise scene: ${backendMessage}`);
      setStatus(PHASE_LABEL_MAP.error);
    } finally {
      setProcessing(false);
      setUploadProgress(null);
    }
  };

  const metadata = result?.metadata || {};
  const objects = metadata.objects || [];
  const activities = metadata.activities || [];
  const scenes = metadata.scenes || [];
  const textDetections = metadata.textDetections || [];
  const people = metadata.people || [];
  const dominantEmotions = metadata.dominantEmotions || [];
  const celebrities = metadata.celebrities || [];
  const context = metadata.context || {};

  return (
    <Page>
      <Title>Scene Summarisation</Title>
      <Lead>
        Upload a frame or short clip and let our multimodal pipeline surface the people, objects, emotions, and text in the shot. We highlight the context, craft a natural-language recap, and render a polished narration in minutes.
      </Lead>

      <UploadCard
        htmlFor="scene-media-input"
        onDragOver={onDragOver}
        onDrop={onDrop}
        onDragLeave={onDragLeave}
        className={dragOver ? 'dragover' : ''}
      >
        <UploadIcon>📽️</UploadIcon>
        <UploadTitle>{selectedFile ? 'Replace media asset' : 'Drag & drop media or click to browse'}</UploadTitle>
        <UploadHint>MP4, MOV, MKV, or images (JPG, PNG, WEBP) — up to 2GB.</UploadHint>
        <HiddenInput
          id="scene-media-input"
          type="file"
          accept="video/*,image/*"
          onChange={onFileInputChange}
        />
      </UploadCard>

      {selectedFile && (
        <FileMeta>
          <span><strong>Selected:</strong> {selectedFile.name}</span>
          <span><strong>Size:</strong> {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</span>
          <span><strong>Type:</strong> {selectedFile.type || 'Unknown'}</span>
        </FileMeta>
      )}

      {previewUrl && (
        <PreviewWrap>
          {selectedFile?.type.startsWith('image/') ? (
            <PreviewImage src={previewUrl} alt="Scene preview" />
          ) : (
            <PreviewVideo src={previewUrl} controls preload="metadata" />
          )}
          <span>Preview generated from the uploaded asset.</span>
        </PreviewWrap>
      )}

      <VoiceRow>
        <VoiceLabel>Voiceover voice:</VoiceLabel>
        <VoiceSelect value={voiceId} onChange={(event) => setVoiceId(event.target.value)}>
          {voiceOptions.map((option) => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </VoiceSelect>
      </VoiceRow>

      <ActionsRow>
        <PrimaryButton type="button" onClick={analyseScene} disabled={processing || !selectedFile}>
          {processing ? 'Analysing scene…' : 'Analyse scene'}
        </PrimaryButton>
        <SecondaryButton type="button" onClick={clearSelection} disabled={processing || !selectedFile}>
          Clear selection
        </SecondaryButton>
      </ActionsRow>

      {status && !error && (
        <StatusBanner>{status}</StatusBanner>
      )}
      {error && (
        <StatusBanner $error>{error}</StatusBanner>
      )}

      {showProgressUI && (
        <>
          <ProgressContainer>
            <ProgressLabel>
              Scene workflow: {currentPhaseLabel}
              {processingPhase !== 'error' ? ` — ${Math.round(progressBarPercent)}%` : ''}
            </ProgressLabel>
            <ProgressTrack>
              <ProgressFill $percent={progressBarPercent} $error={processingPhase === 'error'} />
            </ProgressTrack>
            {showUploadBar && (
              <>
                <ProgressLabel>Upload progress: {uploadProgress}%</ProgressLabel>
                <ProgressTrack>
                  <ProgressFill $percent={uploadProgress} />
                </ProgressTrack>
              </>
            )}
          </ProgressContainer>
          <PhaseList>
            {phaseItems.map((phase) => (
              <PhaseItem key={phase.key} $status={phase.status}>
                <PhaseDot $status={phase.status} />
                {phase.label}
              </PhaseItem>
            ))}
          </PhaseList>
        </>
      )}

      {selectedFile && formattedTimeout && (
        <TimeoutNote>
          Request timeout set to {formattedTimeout}
          {usingDurationTimeout ? ' (5× detected duration)' : ' (fallback value)'}.
        </TimeoutNote>
      )}

      {result && (
        <>
          <SectionDivider />
          <ResultGrid>
            <Card>
              <CardTitle>Story Beat</CardTitle>
              <SummaryText>{result.summary}</SummaryText>
              {Array.isArray(result.highlights) && result.highlights.length > 0 && (
                <>
                  <SectionSubhead>Highlights</SectionSubhead>
                  <HighlightList>
                    {result.highlights.map((item, index) => (
                      <HighlightItem key={index}>{item}</HighlightItem>
                    ))}
                  </HighlightList>
                </>
              )}
              {result.audio_url && (
                <>
                  <SectionSubhead>Voiceover Preview</SectionSubhead>
                  <AudioPlayer controls src={resolveUrl(result.audio_url)} />
                  <AudioNote>
                    Narrated with neural voice {result.voice_id}. Use the download option in the player to save locally.
                  </AudioNote>
                </>
              )}
            </Card>

            <Card>
              <CardTitle>Scene Intelligence</CardTitle>
              <SectionSubhead>Environment</SectionSubhead>
              <MetadataList>
                <MetadataItem><strong>Setting:</strong> {context.environment || 'Unknown'}</MetadataItem>
                <MetadataItem><strong>Activity focus:</strong> {context.activityFocus || 'Unclear'}</MetadataItem>
                <MetadataItem><strong>Lighting hint:</strong> {context.lighting || 'Not detected'}</MetadataItem>
                <MetadataItem><strong>Crowd indicator:</strong> {context.crowdIndicator || 'Not detected'}</MetadataItem>
                <MetadataItem><strong>Frames analysed:</strong> {metadata.framesAnalysed || 1}</MetadataItem>
              </MetadataList>

              {result.source_video && (
                <>
                  <SectionSubhead>Source Video</SectionSubhead>
                  <MetadataList>
                    <MetadataItem><strong>Bucket:</strong> {result.source_video.bucket || '—'}</MetadataItem>
                    <MetadataItem><strong>Key:</strong> {result.source_video.key || '—'}</MetadataItem>
                    {result.source_video.uri && (
                      <MetadataItem><strong>URI:</strong> {result.source_video.uri}</MetadataItem>
                    )}
                  </MetadataList>
                </>
              )}

              {objects.length > 0 && (
                <>
                  <SectionSubhead>Key Objects</SectionSubhead>
                  <TagList>
                    {objects.slice(0, 10).map((entry) => (
                      <Tag key={`object-${entry.label}`}>{metadataLabel(entry)}</Tag>
                    ))}
                  </TagList>
                </>
              )}

              {activities.length > 0 && (
                <>
                  <SectionSubhead>Activities</SectionSubhead>
                  <TagList>
                    {activities.slice(0, 8).map((entry) => (
                      <Tag key={`activity-${entry.label}`}>{metadataLabel(entry)}</Tag>
                    ))}
                  </TagList>
                </>
              )}

              {scenes.length > 0 && (
                <>
                  <SectionSubhead>Scene Cues</SectionSubhead>
                  <TagList>
                    {scenes.slice(0, 8).map((entry) => (
                      <Tag key={`scene-${entry.label}`}>{metadataLabel(entry)}</Tag>
                    ))}
                  </TagList>
                </>
              )}

              {celebrities.length > 0 && (
                <>
                  <SectionSubhead>Recognised Talent</SectionSubhead>
                  <MetadataList>
                    {celebrities.slice(0, 5).map((celeb) => (
                      <MetadataItem key={`celeb-${celeb.name}`}>
                        {celeb.name} ({Math.round(celeb.confidence)}% match)
                      </MetadataItem>
                    ))}
                  </MetadataList>
                </>
              )}
            </Card>

            <Card>
              <CardTitle>People & Emotion</CardTitle>
              {people.length > 0 ? (
                <MetadataList>
                  {people.map((person, index) => (
                    <MetadataItem key={`person-${index}`}>
                      <strong>Subject {index + 1}:</strong> {person.gender || 'Unknown gender'}; age range {person.ageRange?.Low || '?'}-{person.ageRange?.High || '?'}; emotions {person.dominantEmotions?.length ? person.dominantEmotions.join(', ') : 'not detected'}
                    </MetadataItem>
                  ))}
                </MetadataList>
              ) : (
                <SummaryText>No faces detected in this scene.</SummaryText>
              )}

              {dominantEmotions.length > 0 && (
                <>
                  <SectionSubhead>Dominant Emotions</SectionSubhead>
                  <TagList>
                    {dominantEmotions.map((emotion) => (
                      <Tag key={`emotion-${emotion.emotion}`}>{`${emotion.emotion} (${Math.round(emotion.score)})`}</Tag>
                    ))}
                  </TagList>
                </>
              )}

              {textDetections.length > 0 && (
                <>
                  <SectionSubhead>Text On Screen</SectionSubhead>
                  <MetadataList>
                    {textDetections.slice(0, 6).map((textLine, index) => (
                      <MetadataItem key={`text-${index}`}>
                        “{textLine}”
                      </MetadataItem>
                    ))}
                  </MetadataList>
                </>
              )}
            </Card>

            <Card>
              <CardTitle>SSML Blueprint</CardTitle>
              <SsmlPre>{result.ssml}</SsmlPre>
            </Card>

            <Card>
              <CardTitle>Structured Metadata</CardTitle>
              <JsonPre>{JSON.stringify(metadata, null, 2)}</JsonPre>
            </Card>
          </ResultGrid>
        </>
      )}
    </Page>
  );
}
