import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled, { keyframes } from 'styled-components';

import { mediaSupplyChainTheme } from './theme/mediaSupplyChainTheme';

const API_BASE = '/media-supply-chain';
const theme = mediaSupplyChainTheme;

const StageGlyphs = {
  ingest: (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="5" y="5" width="22" height="22" rx="4" stroke="currentColor" strokeWidth="1.6" opacity="0.8" />
      <path d="M10 11h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M10 16h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity="0.8" />
      <path d="M10 21h7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity="0.6" />
      <circle cx="23" cy="21" r="1.3" fill="currentColor" />
    </svg>
  ),
  qc: (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M16 4l10 4v7c0 7-5 11-10 13-5-2-10-6-10-13V8l10-4z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M12.5 16l3 3 4.5-5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  personalize: (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="16" cy="16" r="9" stroke="currentColor" strokeWidth="1.6" opacity="0.7" />
      <path d="M16 7v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M16 22v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M9 16h3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M20 16h3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="16" cy="16" r="3" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  ),
  package: (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M6 10l10-6 10 6-10 6-10-6z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M6 10v12l10 6 10-6V10" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M16 4v12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  ),
};

StageGlyphs.default = (
  <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="16" cy="16" r="10" stroke="currentColor" strokeWidth="1.6" opacity="0.6" />
    <path d="M10 16h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    <path d="M16 10v12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

const STATUS_THEME = {
  completed: { color: theme.colors.success, bg: 'rgba(107, 255, 203, 0.12)', border: 'rgba(107, 255, 203, 0.35)' },
  running: { color: theme.colors.glowBlue, bg: 'rgba(74, 195, 255, 0.14)', border: 'rgba(74, 195, 255, 0.35)' },
  pending: { color: theme.colors.softLavender, bg: 'rgba(196, 198, 255, 0.12)', border: 'rgba(196, 198, 255, 0.3)' },
  failed: { color: theme.colors.danger, bg: 'rgba(255, 95, 126, 0.12)', border: 'rgba(255, 95, 126, 0.35)' },
  'not started': { color: theme.colors.glowAmber, bg: 'rgba(255, 180, 73, 0.14)', border: 'rgba(255, 180, 73, 0.45)' },
};

const DEFAULT_STAGE_LIST = [
  {
    id: 'ingest',
    name: 'Ingest',
    status: 'not started',
    notes: 'Waiting for S3 upload to kick off ingest.',
    metrics: {},
    artifacts: {},
    started_at: null,
    finished_at: null,
  },
  {
    id: 'qc',
    name: 'QC',
    status: 'not started',
    notes: 'QC will begin once Ingest wraps.',
    metrics: {},
    artifacts: { service_checks: [] },
    started_at: null,
    finished_at: null,
  },
  {
    id: 'personalize',
    name: 'Personalize',
    status: 'not started',
    notes: 'Personalization tracks will generate after QC.',
    metrics: {},
    artifacts: {},
    started_at: null,
    finished_at: null,
  },
  {
    id: 'package',
    name: 'Package',
    status: 'not started',
    notes: 'Package waits on Personalize + QC sign-off.',
    metrics: {},
    artifacts: {},
    started_at: null,
    finished_at: null,
  },
];

const metricFormatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });

function formatPercent(value) {
  if (typeof value !== 'number') {
    return '—';
  }
  return `${metricFormatter.format(value * 100)}%`;
}

function formatMetricValue(key, value) {
  if (typeof value === 'number') {
    if (key.toLowerCase().includes('coverage')) {
      return formatPercent(value);
    }
    return metricFormatter.format(value);
  }
  if (value === null || value === undefined) {
    return '—';
  }
  return String(value);
}

function formatMinutes(value) {
  if (typeof value !== 'number') {
    return '—';
  }
  if (value < 60) {
    return `${value} min`;
  }
  const hours = Math.floor(value / 60);
  const minutes = value % 60;
  if (!minutes) {
    return `${hours} hr`;
  }
  return `${hours} hr ${minutes} min`;
}

function formatTimestamp(value) {
  if (!value) {
    return '—';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return '—';
  }
  return parsed.toLocaleString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatBytes(bytes) {
  if (typeof bytes !== 'number' || Number.isNaN(bytes)) {
    return '—';
  }
  if (bytes === 0) {
    return '0 B';
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  const precision = index === 0 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[index]}`;
}

const deliverablePreviewFallbacks = {};

const PLAYABLE_FILE_PATTERN = /\.(mp4|m4v|webm)(\?|$)/i;

function isBrowserPlayableUrl(url) {
  if (typeof url !== 'string' || !url.trim()) {
    return false;
  }
  if (/\.m3u8(\?|$)/i.test(url)) {
    return true;
  }
  return PLAYABLE_FILE_PATTERN.test(url);
}

let hlsLibraryPromise = null;

function loadHlsLibrary() {
  if (typeof window === 'undefined') {
    return Promise.resolve(null);
  }
  if (window.Hls) {
    return Promise.resolve(window.Hls);
  }
  if (!hlsLibraryPromise) {
    hlsLibraryPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/hls.js@1.5.7/dist/hls.min.js';
      script.async = true;
      script.onload = () => resolve(window.Hls || null);
      script.onerror = () => reject(new Error('Failed to load hls.js'));
      document.body.appendChild(script);
    });
  }
  return hlsLibraryPromise
    .then((lib) => lib)
    .catch(() => null);
}

function convertS3PathToHttps(path) {
  const match = /^s3:\/\/([^/]+)\/(.+)$/i.exec(path || '');
  if (!match) {
    return null;
  }
  const [, bucket, key] = match;
  if (!bucket || !key) {
    return null;
  }
  const encodedKey = encodeURIComponent(key).replace(/%2F/g, '/');
  return `https://${bucket}.s3.amazonaws.com/${encodedKey}`;
}

function getDeliverablePreviewSource(deliverable) {
  if (!deliverable) {
    return null;
  }
  if (typeof deliverable.preview_url === 'string' && deliverable.preview_url.trim()) {
    return isBrowserPlayableUrl(deliverable.preview_url) ? deliverable.preview_url : null;
  }
  const path = typeof deliverable.path === 'string' ? deliverable.path.trim() : '';
  if (path) {
    if (/^https?:/i.test(path)) {
      return isBrowserPlayableUrl(path) ? path : null;
    }
    if (path.startsWith('s3://')) {
      const derived = convertS3PathToHttps(path);
      if (derived) {
        return isBrowserPlayableUrl(derived) ? derived : null;
      }
    }
    if (path.startsWith('/') && /\.(mp4|mov|m4v|mxf)$/i.test(path)) {
      return isBrowserPlayableUrl(path) ? path : null;
    }
  }
  if (deliverable.type && deliverablePreviewFallbacks[deliverable.type]) {
    return deliverablePreviewFallbacks[deliverable.type];
  }
  return null;
}

function canPlayHlsNatively(videoEl) {
  if (!videoEl || typeof videoEl.canPlayType !== 'function') {
    return false;
  }
  return videoEl.canPlayType('application/vnd.apple.mpegurl') === 'probably' || videoEl.canPlayType('application/vnd.apple.mpegurl') === 'maybe';
}

const DeliverablePreviewPlayer = ({ src, fallbackSrc, metadata }) => {
  const videoRef = useRef(null);
  const [activeSrc, setActiveSrc] = useState(() => src || fallbackSrc || null);
  const [usingFallback, setUsingFallback] = useState(false);
  const [playbackError, setPlaybackError] = useState(false);
  const isHls = useMemo(() => typeof activeSrc === 'string' && /\.m3u8(\?|$)/i.test(activeSrc), [activeSrc]);

  useEffect(() => {
    setUsingFallback(false);
    setPlaybackError(false);
    setActiveSrc(src || fallbackSrc || null);
  }, [src, fallbackSrc]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) {
      return undefined;
    }

    if (!activeSrc) {
      video.removeAttribute('src');
      video.load();
      return undefined;
    }

    if (!isHls) {
      video.src = activeSrc;
      return () => {
        video.removeAttribute('src');
        video.load();
      };
    }

    let isMounted = true;
    let hlsInstance = null;

    const setupHls = async () => {
      if (!video) {
        return;
      }
      if (canPlayHlsNatively(video)) {
        video.src = activeSrc;
        return;
      }
      const HlsClass = await loadHlsLibrary();
      if (!isMounted || !HlsClass) {
        video.src = activeSrc;
        return;
      }
      if (HlsClass.isSupported()) {
        hlsInstance = new HlsClass({ enableWorker: true, lowLatencyMode: true });
        hlsInstance.loadSource(activeSrc);
        hlsInstance.attachMedia(video);
      } else {
        video.src = activeSrc;
      }
    };

    setupHls();

    return () => {
      isMounted = false;
      if (hlsInstance) {
        hlsInstance.destroy();
        hlsInstance = null;
      }
    };
  }, [activeSrc, isHls]);

  const handleVideoError = useCallback(() => {
    if (fallbackSrc && activeSrc !== fallbackSrc) {
      setActiveSrc(fallbackSrc);
      setUsingFallback(true);
      setPlaybackError(false);
      return;
    }
    setPlaybackError(true);
  }, [fallbackSrc, activeSrc]);

  const formatLabel = metadata?.format || metadata?.profile || null;
  const resolutionLabel = metadata?.resolution || null;
  const bitrateLabel = metadata?.bitrate || null;
  const containerLabel = metadata?.container || metadata?.mode || null;

  return (
    <>
      <DeliverablePreviewFrame>
        <DeliverablePreview ref={videoRef} controls playsInline preload="metadata" onError={handleVideoError}>
          Your browser does not support video playback.
        </DeliverablePreview>
        {metadata && (
          <DeliverableMetaOverlay>
            {formatLabel && <DeliverableMetaTitle>{formatLabel}</DeliverableMetaTitle>}
            {resolutionLabel && <DeliverableMetaLine>{resolutionLabel}</DeliverableMetaLine>}
            {bitrateLabel && <DeliverableMetaLine>{bitrateLabel}</DeliverableMetaLine>}
            {containerLabel && <DeliverableMetaLine>{containerLabel}</DeliverableMetaLine>}
          </DeliverableMetaOverlay>
        )}
      </DeliverablePreviewFrame>
      {usingFallback && fallbackSrc && (
        <DeliverablePreviewNote>Remote preview unavailable, showing sample output.</DeliverablePreviewNote>
      )}
      {playbackError && (
        <DeliverablePreviewNote>Preview file is missing. Re-run the workflow to regenerate this deliverable.</DeliverablePreviewNote>
      )}
    </>
  );
};

function getStageTheme(status) {
  return STATUS_THEME[status] || STATUS_THEME.pending;
}

const panelFloat = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-6px); }
  100% { transform: translateY(0px); }
`;

const progressPulse = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!response.ok) {
    const message = body?.error || text || 'Request failed';
    throw new Error(message);
  }
  return body;
}

const MediaSupplyChain = () => {
  const uploadInputRef = useRef(null);
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [sessionRunId, setSessionRunId] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [isViewingHistory, setIsViewingHistory] = useState(false);
  const [creating, setCreating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [pendingInputs, setPendingInputs] = useState(null);
  const [pendingBlueprint, setPendingBlueprint] = useState(null);
  const [uploadedAsset, setUploadedAsset] = useState(null);
  const [toast, setToast] = useState(null);
  const [error, setError] = useState('');

  const loadRuns = useCallback(async (force = false) => {
    try {
      if (!runs.length) {
        setLoading(true);
      } else if (force) {
        setRefreshing(true);
      }
      const data = await fetchJSON(`${API_BASE}/workflows?limit=25`);
      const list = Array.isArray(data?.runs) ? data.runs : Array.isArray(data) ? data : [];
      setRuns(list);
      if (list.length) {
        const activeRun = list.find((run) => run.status === 'running') || null;
        setSessionRunId(activeRun ? activeRun.id : null);
        setSelectedRunId((current) => {
          if (!current) {
            return null;
          }
          return list.some((run) => run.id === current) ? current : null;
        });
      } else {
        setSessionRunId(null);
        setSelectedRunId(null);
      }
    } catch (err) {
      setError(err.message || 'Unable to load orchestration runs.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [runs.length]);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    if (!sessionRunId) {
      return undefined;
    }
    const intervalId = setInterval(() => {
      loadRuns();
    }, 8000);
    return () => clearInterval(intervalId);
  }, [sessionRunId, loadRuns]);

  const setActiveRun = useCallback((runId, { userInitiated = false } = {}) => {
    setSelectedRunId(runId);
    setIsViewingHistory(userInitiated);
  }, []);


  useEffect(() => {
    if (!toast) {
      return undefined;
    }
    const timer = setTimeout(() => setToast(null), 3600);
    return () => clearTimeout(timer);
  }, [toast]);

  const handleRefresh = useCallback(() => {
    loadRuns(true);
  }, [loadRuns]);

  const clearPendingUpload = useCallback(() => {
    setPendingInputs(null);
    setPendingBlueprint(null);
    setUploadedAsset(null);
  }, []);

  const triggerWorkflow = useCallback(async ({ blueprint, inputs } = {}) => {
    setCreating(true);
    try {
      const payload = {};
      if (blueprint) {
        payload.blueprint = blueprint;
      }
      if (inputs) {
        payload.inputs = inputs;
      }
      const run = await fetchJSON(`${API_BASE}/workflows/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      setToast(inputs ? 'Custom workflow kicked off from your upload.' : 'Workflow kicked off. Stages will update in seconds.');
      setSessionRunId(run.id);
      setActiveRun(run.id);
      if (inputs) {
        clearPendingUpload();
      }
      await loadRuns(true);
    } catch (err) {
      setError(err.message || 'Unable to trigger workflow.');
    } finally {
      setCreating(false);
    }
  }, [clearPendingUpload, loadRuns, setActiveRun]);

  const handleRunUploaded = useCallback(() => {
    if (!pendingInputs) {
      return;
    }
    triggerWorkflow({ blueprint: pendingBlueprint, inputs: pendingInputs });
  }, [pendingBlueprint, pendingInputs, triggerWorkflow]);

  const handleReturnToLive = useCallback(() => {
    setIsViewingHistory(false);
    if (sessionRunId) {
      setActiveRun(sessionRunId);
    } else {
      setActiveRun(null);
    }
  }, [sessionRunId, setActiveRun]);

  const handleRunCardClick = useCallback((runId) => {
    setActiveRun(runId, { userInitiated: true });
  }, [setActiveRun]);

  const uploadWithProgress = useCallback((file) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/uploads`);
    xhr.responseType = 'json';

    xhr.onload = () => {
      const body = xhr.response || (() => {
        try {
          return JSON.parse(xhr.responseText || '{}');
        } catch (err) {
          return {};
        }
      })();
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body);
      } else {
        reject(new Error(body?.error || `Upload failed with status ${xhr.status}`));
      }
    };

    xhr.onerror = () => {
      reject(new Error('Network error while uploading asset.'));
    };

    if (xhr.upload) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(percent);
        }
      };
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', file.name.replace(/\.[^.]+$/, ''));
    xhr.send(formData);
  }), []);

  const handleUpload = useCallback(async (event) => {
    const file = event.target?.files?.[0];
    if (!file) {
      return;
    }
    clearPendingUpload();
    setUploading(true);
    setUploadProgress(0);
    setError('');
    try {
      const data = await uploadWithProgress(file);
      setUploadedAsset(data?.asset || null);
      setPendingInputs(data?.inputs || null);
      setPendingBlueprint(data?.blueprint || null);
      setToast('Upload captured. Review the summary before starting the workflow.');
    } catch (err) {
      setError(err.message || 'Unable to upload asset.');
    } finally {
      if (uploadInputRef.current) {
        uploadInputRef.current.value = '';
      }
      setUploading(false);
      setUploadProgress(null);
    }
  }, [clearPendingUpload, uploadWithProgress]);

  const sessionRun = useMemo(() => {
    if (!sessionRunId) {
      return null;
    }
    return runs.find((run) => run.id === sessionRunId) || null;
  }, [runs, sessionRunId]);

  const selectedRun = useMemo(() => {
    if (sessionRun) {
      return sessionRun;
    }
    if (!selectedRunId) {
      return null;
    }
    return runs.find((run) => run.id === selectedRunId) || null;
  }, [runs, selectedRunId, sessionRun]);

  const historicalRun = useMemo(() => {
    if (!selectedRunId) {
      return null;
    }
    return runs.find((run) => run.id === selectedRunId) || null;
  }, [runs, selectedRunId]);

  useEffect(() => {
    if (isViewingHistory && selectedRunId && !historicalRun) {
      setIsViewingHistory(false);
    }
  }, [isViewingHistory, selectedRunId, historicalRun]);

  const hasPendingUpload = Boolean(uploadedAsset && pendingInputs);

  const stageMode = useMemo(() => {
    if (isViewingHistory && historicalRun && (!sessionRunId || historicalRun.id !== sessionRunId)) {
      return 'historical';
    }
    if (sessionRunId) {
      return 'active';
    }
    if (hasPendingUpload) {
      return 'pending';
    }
    return 'idle';
  }, [isViewingHistory, historicalRun, sessionRunId, hasPendingUpload]);

  const detailRun = useMemo(() => {
    if (stageMode === 'active') {
      return sessionRun;
    }
    if (stageMode === 'historical') {
      return historicalRun;
    }
    return null;
  }, [stageMode, sessionRun, historicalRun]);

  const statsRun = detailRun || selectedRun || runs[0] || null;

  const stageList = useMemo(() => {
    if (stageMode === 'active' && sessionRun?.stages?.length) {
      return sessionRun.stages;
    }
    if (stageMode === 'historical' && historicalRun?.stages?.length) {
      return historicalRun.stages;
    }

    const baseStages = DEFAULT_STAGE_LIST.map((stage) => ({ ...stage }));
    const ingestIndex = baseStages.findIndex((stage) => stage.id === 'ingest');
    if (ingestIndex === -1) {
      return baseStages;
    }

    if (uploading) {
      baseStages[ingestIndex].status = 'running';
      const percent = uploadProgress !== null ? `${uploadProgress}%` : '...';
      baseStages[ingestIndex].notes = `Uploading mezzanine to S3 (${percent})`;
    } else if (hasPendingUpload) {
      baseStages[ingestIndex].status = 'pending';
      baseStages[ingestIndex].notes = 'Upload captured. Kick off the run to continue ingest.';
    }

    return baseStages;
  }, [stageMode, sessionRun, historicalRun, uploading, uploadProgress, hasPendingUpload]);

  const insights = detailRun?.outputs?.insights || {};
  const packageManifest = detailRun?.outputs?.package_manifest;
  const qualityStage = stageList.find((stage) => stage.id === 'qc');
  const serviceChecks =
    qualityStage?.artifacts?.service_checks ||
    qualityStage?.artifacts?.qc_status?.service_checks ||
    [];
  const deliverables = packageManifest?.deliverables || [];
  const primaryProcessedDeliverable = useMemo(() => {
    if (!deliverables.length) {
      return null;
    }
    const ready = deliverables.filter((item) => item.status === 'ready');
    const pool = ready.length ? ready : deliverables;
    const priority = ['social_cut', 'ott_packaged', 'mezzanine'];
    for (const type of priority) {
      const match = pool.find((item) => item.type === type);
      if (match) {
        return match;
      }
    }
    return pool[0];
  }, [deliverables]);
  const processedPreviewSrc = primaryProcessedDeliverable ? getDeliverablePreviewSource(primaryProcessedDeliverable) : null;
  const processedPreviewFallbackSrc = primaryProcessedDeliverable
    ? primaryProcessedDeliverable.fallback_preview_url || (primaryProcessedDeliverable.type ? deliverablePreviewFallbacks[primaryProcessedDeliverable.type] : null) || null
    : null;
  const uploadLabel = uploading
    ? uploadProgress !== null
      ? `Uploading ${uploadProgress}%`
      : 'Uploading...'
    : 'Select mezzanine asset';
  const previewUrl = uploadedAsset?.download_url || uploadedAsset?.s3_uri;
  const downloadHref = previewUrl
    ? previewUrl.startsWith('/media-supply-chain/uploads')
      ? `${previewUrl}${previewUrl.includes('?') ? '&' : '?'}download=1`
      : previewUrl
    : null;

  const stats = [
    {
      label: 'Time to air',
      value: formatMinutes(insights.time_to_air_minutes),
      helper: 'From ingest to readiness',
    },
    {
      label: 'Automation coverage',
      value: formatPercent(insights.automation_coverage),
      helper: 'QC + packaging automation share',
    },
    {
      label: 'Latest status',
      value: statsRun ? statsRun.status : 'No runs yet',
      helper: statsRun ? formatTimestamp(statsRun.finished_at || statsRun.started_at) : 'Upload a mezzanine asset to start the workflow',
    },
  ];

  const highlightedRunId = detailRun?.id || (stageMode === 'active' ? sessionRunId : null);

  let stageHelperContent = null;
  if (stageMode === 'idle') {
    stageHelperContent = (
      <MutedText>
        Stage statuses stay in <strong>NOT STARTED</strong> until you upload a mezzanine asset and kick off a run.
      </MutedText>
    );
  } else if (stageMode === 'pending') {
    stageHelperContent = (
      <MutedText>
        Upload captured. Confirm the workflow run to move Ingest → Package into motion.
      </MutedText>
    );
  } else if (stageMode === 'active' && sessionRunId) {
    stageHelperContent = (
      <StageStatusPill>
        Tracking workflow {sessionRunId.slice(0, 8)} in real time.
      </StageStatusPill>
    );
  } else if (stageMode === 'historical' && detailRun) {
    stageHelperContent = (
      <HistoryBanner>
        <span>Viewing historical run {detailRun.id.slice(0, 8)}.</span>
        <GhostButton type="button" onClick={handleReturnToLive}>
          Return to live status
        </GhostButton>
      </HistoryBanner>
    );
  }

  return (
    <PageWrap>
      <Hero>
        <HeroBody>
          <Eyebrow>Media Supply Chain</Eyebrow>
          <HeroTitle>Agentic orchestration for ingest → delivery</HeroTitle>
          <HeroLead>
            Upload a mezzanine file below to drive Ingest → Package automatically. Each run now originates from your asset so you stay in control of the workflow and downstream transcoding/packaging.
          </HeroLead>
          <ActionRow>
            <GhostButton type="button" onClick={handleRefresh} disabled={refreshing || loading}>
              {refreshing ? 'Refreshing...' : 'Refresh status'}
            </GhostButton>
          </ActionRow>
        </HeroBody>
        <StatGrid>
          {stats.map((stat) => (
            <StatCard key={stat.label}>
              <StatLabel>{stat.label}</StatLabel>
              <StatValue>{stat.value}</StatValue>
              <StatHelper>{stat.helper}</StatHelper>
            </StatCard>
          ))}
        </StatGrid>
      </Hero>

      <UploadShell>
        <UploadCopy>
          <UploadEyebrow>Bring your own mezzanine</UploadEyebrow>
          <UploadTitle>Upload a video to prime ingest</UploadTitle>
          <UploadLead>
            The upload service stores your asset in the configured bucket (or local disk) and returns the ingest manifest expected by the workflow engine. We always pause here so you can confirm the run before downstream jobs pile up.
          </UploadLead>
          <UploadControls>
            <UploadButton type="button" onClick={() => uploadInputRef.current?.click()} disabled={uploading}>
              {uploadLabel}
            </UploadButton>
            <HiddenFileInput ref={uploadInputRef} type="file" accept="video/*" onChange={handleUpload} />
            <UploadHint>Uploads never auto-run. Confirming here prevents accidental AWS spend.</UploadHint>
            {uploading && (
              <ProgressWrapper>
                <ProgressBar>
                  <ProgressFill style={{ width: `${uploadProgress || 0}%` }} />
                </ProgressBar>
                <UploadProgressText>
                  {uploadProgress !== null ? `${uploadProgress}% complete` : 'Preparing upload...'}
                </UploadProgressText>
              </ProgressWrapper>
            )}
          </UploadControls>
        </UploadCopy>
        {hasPendingUpload ? (
          <UploadSummary>
            <SummaryGrid>
              <SummaryRow>
                <SummaryLabel>Filename</SummaryLabel>
                <SummaryValue>{uploadedAsset?.filename}</SummaryValue>
              </SummaryRow>
              <SummaryRow>
                <SummaryLabel>Size</SummaryLabel>
                <SummaryValue>{formatBytes(uploadedAsset?.size_bytes)}</SummaryValue>
              </SummaryRow>
              <SummaryRow>
                <SummaryLabel>Storage</SummaryLabel>
                <SummaryValue>{uploadedAsset?.s3_uri || uploadedAsset?.local_path}</SummaryValue>
              </SummaryRow>
            </SummaryGrid>
            {previewUrl && (
              <PreviewSection>
                <PreviewLabel>Preview</PreviewLabel>
                <VideoPreview controls src={previewUrl} preload="metadata">
                  Your browser does not support embedded previews. Use the download link below.
                </VideoPreview>
                {downloadHref && (
                  <DownloadLink href={downloadHref} download>
                    Download mezzanine
                  </DownloadLink>
                )}
              </PreviewSection>
            )}
            <UploadActions>
              <PrimaryButton type="button" onClick={handleRunUploaded} disabled={creating}>
                {creating ? 'Starting custom run...' : 'Run workflow with this asset'}
              </PrimaryButton>
              <GhostButton type="button" onClick={clearPendingUpload} disabled={creating || uploading}>
                Remove upload
              </GhostButton>
            </UploadActions>
            <UploadDecision>
              We require explicit confirmation before kicking off the workflow so large uploads don’t automatically trigger expensive downstream jobs.
            </UploadDecision>
          </UploadSummary>
        ) : (
          <UploadPlaceholder>
            <p>When you upload a file we’ll capture ingest-ready metadata and surface it here. Kick off the workflow once you’re satisfied.</p>
          </UploadPlaceholder>
        )}
      </UploadShell>

      {error && <ErrorBanner>{error}</ErrorBanner>}

      {loading ? (
        <LoadingState>Loading orchestration history...</LoadingState>
      ) : !runs.length ? (
        <EmptyState>
          <EmptyTitle>No orchestration runs yet</EmptyTitle>
          <EmptyText>Upload a mezzanine asset from the panel above to trigger Ingest → Package. We will surface the run here as soon as it starts.</EmptyText>
        </EmptyState>
      ) : (
        <ContentGrid>
          <PrimaryColumn>
            <Section>
              <SectionHeading>Stage timeline</SectionHeading>
              {stageHelperContent}
              <StageGrid>
                {stageList.map((stage) => {
                  const theme = getStageTheme(stage.status);
                  const glyph = StageGlyphs[stage.id] || StageGlyphs.default;
                  const metrics = Object.entries(stage.metrics || {}).filter(([, value]) => value !== undefined && value !== null);
                  const uploadStatus = stage.id === 'ingest' ? stage.artifacts?.upload_status : null;
                  const qcStatus = stage.id === 'qc' ? stage.artifacts?.qc_status : null;
                  const qcWarnings = stage.id === 'qc' && Array.isArray(qcStatus?.warnings) ? qcStatus.warnings : [];
                  const qcChecks = stage.id === 'qc' && Array.isArray(qcStatus?.service_checks) ? qcStatus.service_checks : [];
                  const warningsFound = typeof qcStatus?.warnings_found === 'number' ? qcStatus.warnings_found : qcWarnings.length;
                  const fallbackWarningTotal = qcWarnings.length > 0 ? qcWarnings.length : null;
                  const warningsTotal = typeof qcStatus?.warnings_total === 'number' ? qcStatus.warnings_total : fallbackWarningTotal;
                  return (
                    <StageCard key={stage.id} $theme={theme} $active={stage.status === 'running'}>
                      <StageHead>
                        <StageIcon>{glyph}</StageIcon>
                        <div>
                          <StageName>{stage.name}</StageName>
                          <StageStatus $theme={theme}>{stage.status}</StageStatus>
                        </div>
                      </StageHead>
                      <StageNote>{stage.notes || 'No notes captured for this stage.'}</StageNote>
                      {uploadStatus && (
                        <StageProgress>
                          <StageProgressHeader>
                            <span>{uploadStatus.state === 'completed' ? 'Upload completed' : 'Uploading to S3'}</span>
                            <strong>{uploadStatus.percent ? `${uploadStatus.percent}%` : '—'}</strong>
                          </StageProgressHeader>
                          <StageProgressBar>
                            <StageProgressFill style={{ width: `${uploadStatus.percent || 0}%` }} />
                          </StageProgressBar>
                          <StageProgressMeta>
                            {formatBytes(uploadStatus.bytes_uploaded)} / {formatBytes(uploadStatus.bytes_total)}
                          </StageProgressMeta>
                        </StageProgress>
                      )}
                      {qcStatus && (
                        <StageProgress>
                          <StageProgressHeader>
                            <span>{qcStatus.message || 'QC in progress'}</span>
                            <strong>{typeof qcStatus.percent === 'number' ? `${qcStatus.percent}%` : '—'}</strong>
                          </StageProgressHeader>
                          <StageProgressBar>
                            <StageProgressFill style={{ width: `${qcStatus.percent || 0}%` }} />
                          </StageProgressBar>
                          <StageProgressMeta>
                            {typeof warningsTotal === 'number'
                              ? `Warnings ${warningsFound}/${warningsTotal}`
                              : (qcStatus.state || 'QC running')}
                          </StageProgressMeta>
                        </StageProgress>
                      )}
                      {!!qcWarnings.length && (
                        <FindingList>
                          {qcWarnings.map((warning, index) => (
                            <FindingItem key={`${warning.type || 'warning'}-${index}`}>
                              <FindingBadge $severity={(warning.severity || '').toLowerCase()}>{warning.severity || 'info'}</FindingBadge>
                              <FindingCopy>
                                <strong>{warning.type || `Warning ${index + 1}`}</strong>
                                <span>{warning.description || 'No description provided.'}</span>
                              </FindingCopy>
                            </FindingItem>
                          ))}
                        </FindingList>
                      )}
                      {!!qcChecks.length && (
                        <InlineCheckList>
                          {qcChecks.map((check) => (
                            <InlineCheck key={check.name || check.url} $status={check.status}>
                              <span>{check.name || 'Service'}</span>
                              <strong>{check.status || 'unknown'}</strong>
                            </InlineCheck>
                          ))}
                        </InlineCheckList>
                      )}
                      <StageMeta>
                        <MetaItem>
                          <MetaLabel>Started</MetaLabel>
                          <MetaValue>{formatTimestamp(stage.started_at)}</MetaValue>
                        </MetaItem>
                        <MetaItem>
                          <MetaLabel>Finished</MetaLabel>
                          <MetaValue>{formatTimestamp(stage.finished_at)}</MetaValue>
                        </MetaItem>
                      </StageMeta>
                      {!!metrics.length && (
                        <MetricChips>
                          {metrics.map(([key, value]) => (
                            <MetricChip key={key}>
                              <span>{key.replace(/_/g, ' ')}</span>
                              <strong>{formatMetricValue(key, value)}</strong>
                            </MetricChip>
                          ))}
                        </MetricChips>
                      )}
                    </StageCard>
                  );
                })}
              </StageGrid>
            </Section>

            {processedPreviewSrc && primaryProcessedDeliverable && (
              <Section>
                <SectionHeading>Processed playback</SectionHeading>
                <ProcessedPreviewCard>
                  <DeliverablePreviewPlayer
                    src={processedPreviewSrc}
                    fallbackSrc={processedPreviewFallbackSrc}
                    metadata={primaryProcessedDeliverable.preview_metadata}
                  />
                  <ProcessedPreviewMeta>
                    <ProcessedPreviewTitle>{primaryProcessedDeliverable.label || 'Processed deliverable'}</ProcessedPreviewTitle>
                    <ProcessedPreviewDetails>
                      <span>Type: {primaryProcessedDeliverable.type || 'unknown'}</span>
                      <span>Status: {primaryProcessedDeliverable.status || 'pending'}</span>
                      {primaryProcessedDeliverable.path && <span className="path">{primaryProcessedDeliverable.path}</span>}
                    </ProcessedPreviewDetails>
                    <ProcessedPreviewActions>
                      {primaryProcessedDeliverable.path && (
                        <ProcessedPreviewLink href={primaryProcessedDeliverable.path} target="_blank" rel="noreferrer">
                          View path
                        </ProcessedPreviewLink>
                      )}
                      <ProcessedPreviewLink href={processedPreviewSrc} target="_blank" rel="noreferrer">
                        Open stream
                      </ProcessedPreviewLink>
                    </ProcessedPreviewActions>
                  </ProcessedPreviewMeta>
                </ProcessedPreviewCard>
              </Section>
            )}

            <Section>
              <SectionHeading>Service health checks</SectionHeading>
              <ServiceList>
                {!serviceChecks.length && <MutedText>No downstream service checks recorded.</MutedText>}
                {serviceChecks.map((check) => (
                  <ServiceItem key={check.name}>
                    <ServiceName>{check.name}</ServiceName>
                    <ServiceStatus $ok={check.status === 'ok'}>{check.status || 'unknown'}</ServiceStatus>
                    <ServiceMeta>{check.url}</ServiceMeta>
                  </ServiceItem>
                ))}
              </ServiceList>
            </Section>

            <Section>
              <SectionHeading>Deliverables & insights</SectionHeading>
              <InsightGrid>
                <InsightCard>
                  <InsightLabel>Primary asset</InsightLabel>
                  <InsightValue>{(packageManifest?.source_asset?.title) || 'Episode 5 Recap Trailer'}</InsightValue>
                  <InsightMeta>ID: {packageManifest?.source_asset?.asset_id || 'OTT-TRAILER-STARLIGHT-005'}</InsightMeta>
                </InsightCard>
                <InsightCard>
                  <InsightLabel>Distribution channels</InsightLabel>
                  <InsightValue>{(packageManifest?.distribution_channels || []).join(', ') || 'Prime OTT, YouTube Shorts, Partner Broadcast'}</InsightValue>
                  <InsightMeta>Editable inside the blueprint manifest.</InsightMeta>
                </InsightCard>
              </InsightGrid>
              <DeliverableList>
                {!deliverables.length && <MutedText>No deliverables recorded yet.</MutedText>}
                {deliverables.map((item) => {
                  const fallbackSrc = item.fallback_preview_url || (item.type ? deliverablePreviewFallbacks[item.type] : null) || null;
                  const previewSrc = getDeliverablePreviewSource(item) || fallbackSrc;
                  const typeLabel = item.type ? item.type.replace(/_/g, ' ') : null;
                  return (
                    <Deliverable key={`${item.label}-${item.path}`}>
                      <DeliverableHeader>
                        <div>
                          <DeliverableTitle>{item.label}</DeliverableTitle>
                          {typeLabel && <DeliverableTag>{typeLabel}</DeliverableTag>}
                        </div>
                        <DeliverableStatus $status={item.status}>{item.status || 'unknown'}</DeliverableStatus>
                      </DeliverableHeader>
                      <DeliverableMeta>{item.path}</DeliverableMeta>
                      {previewSrc ? (
                        <DeliverablePreviewPlayer src={previewSrc} fallbackSrc={fallbackSrc} metadata={item.preview_metadata} />
                      ) : (
                        <DeliverablePreviewNote>No preview available for this deliverable.</DeliverablePreviewNote>
                      )}
                    </Deliverable>
                  );
                })}
              </DeliverableList>
            </Section>
          </PrimaryColumn>

          <Sidebar>
            <SectionHeading>Recent runs</SectionHeading>
            <RunList>
              {runs.map((run) => {
                const progressValue = typeof run.progress === 'number' ? Math.round(run.progress * 100) : 0;
                return (
                  <RunCard
                    key={run.id}
                    $active={run.id === highlightedRunId}
                    type="button"
                    onClick={() => handleRunCardClick(run.id)}
                  >
                    <RunTopRow>
                      <RunStatus>{run.status}</RunStatus>
                      <RunTimestamp>{formatTimestamp(run.finished_at || run.started_at)}</RunTimestamp>
                    </RunTopRow>
                    <RunTitle>{run.name}</RunTitle>
                    <RunMeta>Progress: {progressValue}%</RunMeta>
                    <RunMeta>ID: {run.id.slice(0, 8)}...</RunMeta>
                  </RunCard>
                );
              })}
            </RunList>
          </Sidebar>
        </ContentGrid>
      )}

      {toast && <Toast>{toast}</Toast>}
    </PageWrap>
  );
};

const PageWrap = styled.section`
  position: relative;
  min-height: 100vh;
  padding: clamp(2.2rem, 4vw, 3rem) clamp(1.25rem, 4vw, 3.5rem) 4rem;
  color: ${theme.colors.textBright};
  font-family: ${theme.fonts.body};
  overflow: hidden;

  &::before,
  &::after {
    content: '';
    position: absolute;
    inset: 0;
    pointer-events: none;
  }

  &::before {
    background: none;
    filter: none;
    opacity: 0;
  }

  &::after {
    background-image: none;
    background-size: auto;
    opacity: 0;
  }

  > * {
    position: relative;
    z-index: 1;
  }
`;

const UploadShell = styled.div`
  position: relative;
  border: 1px solid ${theme.colors.panelBorder};
  border-radius: 30px;
  padding: clamp(1.7rem, 3vw, 2.7rem);
  background: ${theme.gradients.upload};
  display: grid;
  gap: 1.8rem;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  margin-bottom: 2.4rem;
  box-shadow: ${theme.shadows.glowSoft};
  backdrop-filter: blur(${theme.blur.panel});
  overflow: hidden;

  &::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(130deg, rgba(255, 255, 255, 0.05), transparent);
    opacity: 0.5;
  }

  > * {
    position: relative;
    z-index: 1;
  }
`;

const UploadCopy = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
`;
const UploadEyebrow = styled.span`
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.22em;
  color: ${theme.colors.softLavender};
`;

const UploadTitle = styled.h3`
  margin: 0;
  font-size: 1.4rem;
  font-family: ${theme.fonts.display};
  color: ${theme.colors.textBright};
`;

const UploadLead = styled.p`
  margin: 0;
  color: ${theme.colors.textMuted};
  line-height: 1.5;
`;

const UploadControls = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
`;


const UploadProgressText = styled.span`
  font-size: 0.8rem;
  color: ${theme.colors.softLavender};
`;

const ProgressWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
`;

const ProgressBar = styled.div`
  height: 6px;
  width: 100%;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
`;

const ProgressFill = styled.div`
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, ${theme.colors.glowTeal}, ${theme.colors.glowBlue});
  animation: ${progressPulse} 2.4s ease infinite;
`;

const UploadButton = styled.button`
  padding: 0.9rem 1.5rem;
  border-radius: 18px;
  border: 1px dashed rgba(54, 252, 224, 0.55);
  background: rgba(4, 14, 32, 0.7);
  color: ${theme.colors.glowTeal};
  font-weight: 600;
  cursor: pointer;
  transition: border-color ${theme.transitions.default}, transform ${theme.transitions.default};

  &:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  &:not(:disabled):hover {
    transform: translateY(-2px);
    border-color: ${theme.colors.glowBlue};
  }
`;

const UploadHint = styled.span`
  font-size: 0.85rem;
  color: rgba(128, 237, 195, 0.85);
`;

const HiddenFileInput = styled.input`
  display: none;
`;

const UploadSummary = styled.div`
  border-radius: 22px;
  border: 1px solid rgba(54, 252, 224, 0.35);
  padding: 1.3rem;
  background: rgba(4, 12, 30, 0.75);
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const SummaryGrid = styled.div`
  display: grid;
  gap: 0.8rem;
`;

const PreviewSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
`;

const PreviewLabel = styled.span`
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: ${theme.colors.softLavender};
`;

const VideoPreview = styled.video`
  width: 100%;
  border-radius: 18px;
  background: rgba(3, 5, 12, 0.8);
  border: 1px solid rgba(82, 228, 255, 0.35);
  box-shadow: ${theme.shadows.glowSoft};
`;

const DownloadLink = styled.a`
  align-self: flex-start;
  padding: 0.5rem 1rem;
  border-radius: 999px;
  border: 1px solid rgba(54, 252, 224, 0.45);
  color: ${theme.colors.glowTeal};
  font-weight: 600;
  text-decoration: none;
  transition: border-color ${theme.transitions.default}, color ${theme.transitions.default};

  &:hover {
    border-color: ${theme.colors.glowBlue};
    color: ${theme.colors.glowBlue};
  }
`;

const SummaryRow = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
`;

const SummaryLabel = styled.span`
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: ${theme.colors.softLavender};
`;

const SummaryValue = styled.span`
  font-weight: 600;
  color: ${theme.colors.textBright};
  word-break: break-all;
`;

const UploadPlaceholder = styled.div`
  border-radius: 22px;
  border: 1px dashed rgba(148, 163, 255, 0.35);
  padding: 1.2rem;
  color: ${theme.colors.textMuted};
  background: rgba(15, 23, 42, 0.45);

  p {
    margin: 0;
    line-height: 1.5;
  }
`;

const UploadActions = styled.div`
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
`;

const UploadDecision = styled.p`
  margin: 0;
  font-size: 0.85rem;
  color: ${theme.colors.softLavender};
`;

const Hero = styled.div`
  position: relative;
  background: ${theme.gradients.hero};
  border: 1px solid ${theme.colors.panelBorder};
  border-radius: 32px;
  padding: clamp(1.6rem, 4vw, 3.2rem);
  display: grid;
  gap: 2rem;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  margin-bottom: 2.6rem;
  box-shadow: ${theme.shadows.glowSoft};
  backdrop-filter: blur(${theme.blur.panel});
  overflow: hidden;

  &::after {
    content: '';
    position: absolute;
    inset: 0;
    background: ${theme.gradients.accentLine};
    opacity: 0.25;
    transform: translateY(60%);
  }

  > * {
    position: relative;
    z-index: 1;
  }
`;

const HeroBody = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  font-family: ${theme.fonts.display};
`;

const Eyebrow = styled.span`
  font-size: 0.8rem;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: ${theme.colors.softLavender};
  opacity: 0.85;
`;

const HeroTitle = styled.h1`
  margin: 0;
  font-size: clamp(2.1rem, 5vw, 3.3rem);
  color: ${theme.colors.textBright};
  text-transform: uppercase;
  letter-spacing: 0.08em;
`;

const HeroLead = styled.p`
  margin: 0;
  line-height: 1.75;
  color: ${theme.colors.textMuted};
`;

const ActionRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.85rem;
`;

const PrimaryButton = styled.button`
  padding: 0.85rem 1.6rem;
  border-radius: 999px;
  border: none;
  font-weight: 700;
  font-size: 0.95rem;
  background: linear-gradient(120deg, ${theme.colors.glowTeal}, ${theme.colors.glowBlue}, #6366f1);
  color: #041427;
  cursor: pointer;
  transition: transform ${theme.transitions.default}, box-shadow ${theme.transitions.default};
  box-shadow: 0 15px 35px rgba(54, 252, 224, 0.25);

  &:disabled {
    opacity: 0.65;
    cursor: not-allowed;
  }

  &:not(:disabled):hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 18px 48px rgba(74, 195, 255, 0.45);
  }
`;

const GhostButton = styled.button`
  padding: 0.82rem 1.4rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 255, 0.4);
  background: rgba(15, 23, 42, 0.4);
  color: ${theme.colors.textBright};
  font-weight: 600;
  cursor: pointer;
  transition: border-color ${theme.transitions.default}, transform ${theme.transitions.default}, color ${theme.transitions.default};

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  &:not(:disabled):hover {
    transform: translateY(-2px);
    border-color: ${theme.colors.glowBlue};
    color: ${theme.colors.glowBlue};
  }
`;

const StatGrid = styled.div`
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
`;

const StatCard = styled.div`
  position: relative;
  background: ${theme.gradients.holo};
  border: 1px solid rgba(74, 195, 255, 0.35);
  border-radius: 20px;
  padding: 1.2rem;
  box-shadow: ${theme.shadows.glowSoft};
  overflow: hidden;

  &::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(140deg, rgba(255, 255, 255, 0.08), transparent);
    opacity: 0.4;
  }

  > * {
    position: relative;
    z-index: 1;
  }
`;

const StatLabel = styled.span`
  font-size: 0.75rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: ${theme.colors.softLavender};
`;

const StatValue = styled.div`
  font-size: 2rem;
  font-weight: 700;
  margin: 0.4rem 0;
  color: ${theme.colors.textBright};
`;

const StatHelper = styled.p`
  margin: 0;
  color: ${theme.colors.textMuted};
  font-size: 0.9rem;
`;

const ErrorBanner = styled.div`
  margin-bottom: 1rem;
  padding: 0.85rem 1rem;
  border-radius: 16px;
  border: 1px solid rgba(248, 113, 113, 0.35);
  background: rgba(248, 113, 113, 0.12);
  color: #fecaca;
`;

const LoadingState = styled.div`
  text-align: center;
  padding: 3rem 1rem;
  color: #a5b4fc;
`;

const EmptyState = styled.div`
  border: 1px dashed rgba(148, 163, 255, 0.35);
  border-radius: 18px;
  padding: 3rem 1.5rem;
  text-align: center;
  color: #cbd5f5;
`;

const EmptyTitle = styled.h2`
  margin: 0 0 0.5rem;
  color: #f8fafc;
`;

const EmptyText = styled.p`
  margin: 0 auto 1.5rem;
  max-width: 560px;
`;

const ContentGrid = styled.div`
  display: grid;
  gap: 1.8rem;
  grid-template-columns: minmax(0, 1fr) 320px;

  @media (max-width: 1100px) {
    grid-template-columns: 1fr;
  }
`;

const PrimaryColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.75rem;
`;

const Sidebar = styled.aside`
  border: 1px solid ${theme.colors.panelBorder};
  border-radius: 24px;
  padding: 1.6rem;
  background: rgba(5, 8, 20, 0.9);
  backdrop-filter: blur(${theme.blur.panel});
`;

const Section = styled.section`
  border: 1px solid rgba(99, 102, 241, 0.25);
  border-radius: 24px;
  padding: 1.6rem;
  background: rgba(5, 9, 24, 0.92);
  box-shadow: ${theme.shadows.glowSoft};
  backdrop-filter: blur(${theme.blur.panel});
  animation: ${panelFloat} 16s ease-in-out infinite;
`;

const SectionHeading = styled.h2`
  margin: 0 0 1.1rem;
  font-size: 1.15rem;
  color: ${theme.colors.textBright};
  letter-spacing: 0.1em;
  text-transform: uppercase;
`;

const StageStatusPill = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.85rem;
  border-radius: 999px;
  border: 1px solid rgba(59, 130, 246, 0.35);
  background: rgba(37, 99, 235, 0.1);
  font-size: 0.85rem;
  color: #bfdbfe;
  font-weight: 600;
`;

const HistoryBanner = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
  padding: 0.65rem 0.85rem;
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 255, 0.3);
  background: rgba(15, 23, 42, 0.7);
  color: #cbd5f5;

  span {
    font-size: 0.88rem;
  }
`;

const StageGrid = styled.div`
  position: relative;
  display: grid;
  gap: 1rem;
  padding-left: 1.5rem;

  &::before {
    content: '';
    position: absolute;
    top: 0.5rem;
    bottom: 0.5rem;
    left: 1.1rem;
    width: 2px;
    background: linear-gradient(180deg, rgba(54, 252, 224, 0), rgba(54, 252, 224, 0.5), rgba(54, 252, 224, 0));
  }

  @media (max-width: 640px) {
    padding-left: 0;

    &::before {
      display: none;
    }
  }
`;

const StageCard = styled.div`
  position: relative;
  border-radius: 22px;
  padding: 1.2rem 1.4rem 1.2rem 3.4rem;
  border: 1px solid ${({ $theme }) => $theme.border};
  background: ${({ $theme }) => $theme.bg};
  box-shadow: ${({ $active }) => ($active ? theme.shadows.glowSoft : 'none')};
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    left: 0.9rem;
    top: 1.4rem;
    border: 2px solid ${({ $theme }) => $theme.color};
    box-shadow: 0 0 14px ${({ $theme }) => $theme.color};
    background: ${({ $theme }) => $theme.color};
  }

  @media (max-width: 640px) {
    padding: 1.2rem;

    &::before {
      display: none;
    }
  }
`;

const StageHead = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.75rem;
`;

const StageIcon = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 14px;
  border: 1px solid rgba(74, 195, 255, 0.45);
  display: grid;
  place-items: center;
  color: ${theme.colors.glowBlue};
  background: rgba(15, 23, 42, 0.65);
  box-shadow: ${theme.shadows.borderGlow};
`;

const StageName = styled.h3`
  margin: 0;
  color: ${theme.colors.textBright};
  font-size: 1.05rem;
`;

const StageStatus = styled.span`
  font-size: 0.75rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: ${({ $theme }) => $theme.color};
`;

const StageNote = styled.p`
  margin: 0 0 0.9rem;
  color: ${theme.colors.textMuted};
`;

const StageProgress = styled.div`
  margin-top: -0.3rem;
  margin-bottom: 0.9rem;
  padding: 0.65rem 0.85rem;
  border-radius: 14px;
  background: rgba(4, 14, 32, 0.35);
  border: 1px solid rgba(74, 195, 255, 0.22);
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
`;

const StageProgressHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: ${theme.colors.softLavender};

  strong {
    color: ${theme.colors.textBright};
    font-size: 0.85rem;
    letter-spacing: normal;
  }
`;

const StageProgressBar = styled.div`
  width: 100%;
  height: 6px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
`;

const StageProgressFill = styled.div`
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, ${theme.colors.glowTeal}, ${theme.colors.glowBlue});
  transition: width ${theme.transitions.slow};
`;

const StageProgressMeta = styled.span`
  font-size: 0.78rem;
  color: ${theme.colors.textMuted};
`;

const FindingList = styled.ul`
  list-style: none;
  margin: 0.8rem 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
`;

const FindingItem = styled.li`
  display: flex;
  gap: 0.85rem;
  padding: 0.65rem 0.85rem;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
`;

const FindingBadge = styled.span`
  align-self: flex-start;
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  background: ${(props) => {
    if (props.$severity === 'high') {
      return 'rgba(255, 95, 126, 0.18)';
    }
    if (props.$severity === 'medium') {
      return 'rgba(255, 180, 73, 0.18)';
    }
    return 'rgba(74, 195, 255, 0.18)';
  }};
  color: ${(props) => {
    if (props.$severity === 'high') {
      return theme.colors.danger;
    }
    if (props.$severity === 'medium') {
      return theme.colors.glowAmber;
    }
    return theme.colors.glowBlue;
  }};
`;

const FindingCopy = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.15rem;

  strong {
    font-size: 0.9rem;
    letter-spacing: 0.02em;
  }

  span {
    font-size: 0.82rem;
    color: ${theme.colors.slate};
  }
`;

const InlineCheckList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.75rem;
`;

const InlineCheck = styled.div`
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.8rem;
  border-radius: 999px;
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border: 1px solid rgba(255, 255, 255, 0.15);
  background: ${(props) => {
    if (props.$status === 'ok') {
      return 'rgba(107, 255, 203, 0.16)';
    }
    if (props.$status === 'offline') {
      return 'rgba(255, 95, 126, 0.18)';
    }
    return 'rgba(74, 195, 255, 0.15)';
  }};
  color: ${(props) => {
    if (props.$status === 'ok') {
      return theme.colors.success;
    }
    if (props.$status === 'offline') {
      return theme.colors.danger;
    }
    return theme.colors.softLavender;
  }};
`;

const StageMeta = styled.div`
  display: grid;
  gap: 0.95rem;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  margin-bottom: 0.75rem;
`;

const MetaItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  padding: 0.6rem 0.8rem;
  border-radius: 12px;
  background: rgba(148, 163, 255, 0.08);
`;

const MetaLabel = styled.span`
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: ${theme.colors.muted};
`;

const MetaValue = styled.span`
  font-weight: 600;
  color: ${theme.colors.textBright};
`;

const MetricChips = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
`;

const MetricChip = styled.span`
  padding: 0.3rem 0.85rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 255, 0.35);
  font-size: 0.78rem;
  display: inline-flex;
  gap: 0.35rem;
  align-items: baseline;
  color: ${theme.colors.textMuted};
  letter-spacing: 0.12em;
  text-transform: uppercase;

  strong {
    color: ${theme.colors.textBright};
  }
`;

const ServiceList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
`;

const ServiceItem = styled.div`
  position: relative;
  padding: 1rem 1.2rem;
  border-radius: 22px;
  background: ${theme.colors.panelBgStrong};
  border: 1px solid rgba(74, 195, 255, 0.28);
  box-shadow: ${theme.shadows.glowSoft};
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  overflow: hidden;
  transition: border-color ${theme.transitions.default}, box-shadow ${theme.transitions.default};

  &::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(54, 252, 224, 0.1), transparent 70%);
    opacity: 0.8;
  }

  > * {
    position: relative;
    z-index: 1;
  }

  &:hover {
    border-color: rgba(74, 195, 255, 0.55);
    box-shadow: ${theme.shadows.glowStrong};
  }
`;

const ServiceName = styled.div`
  font-weight: 600;
  color: ${theme.colors.textBright};
  font-size: 1rem;
`;

const ServiceStatus = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  text-transform: uppercase;
  font-size: 0.72rem;
  letter-spacing: 0.2em;
  color: ${({ $ok }) => ($ok ? theme.colors.success : theme.colors.warning)};
  background: ${({ $ok }) => ($ok ? 'rgba(107, 255, 203, 0.12)' : 'rgba(248, 198, 70, 0.12)')};
  padding: 0.25rem 0.65rem;
  border-radius: 999px;
  border: 1px solid ${({ $ok }) => ($ok ? 'rgba(107, 255, 203, 0.35)' : 'rgba(248, 198, 70, 0.35)')};
`;

const ServiceMeta = styled.div`
  font-size: 0.85rem;
  color: ${theme.colors.textMuted};
  word-break: break-all;
`;

const InsightGrid = styled.div`
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  margin-bottom: 1rem;
`;

const InsightCard = styled.div`
  position: relative;
  border-radius: 20px;
  border: 1px solid rgba(74, 195, 255, 0.28);
  padding: 1.15rem;
  background: ${theme.colors.panelBgStrong};
  box-shadow: ${theme.shadows.glowSoft};
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(140deg, rgba(54, 252, 224, 0.1), transparent 70%);
    opacity: 0.7;
  }

  > * {
    position: relative;
    z-index: 1;
  }
`;

const InsightLabel = styled.span`
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  color: ${theme.colors.softLavender};
`;

const InsightValue = styled.div`
  font-size: 1.15rem;
  font-weight: 600;
  color: ${theme.colors.textBright};
  margin: 0.4rem 0;
`;

const InsightMeta = styled.div`
  font-size: 0.85rem;
  color: ${theme.colors.textMuted};
`;

const DeliverableList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
`;

const Deliverable = styled.div`
  border-radius: 22px;
  border: 1px solid rgba(148, 163, 255, 0.3);
  padding: 1rem 1.2rem;
  background: ${theme.colors.panelBg};
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  transition: border-color ${theme.transitions.default}, box-shadow ${theme.transitions.default};

  &:hover {
    border-color: rgba(74, 195, 255, 0.55);
    box-shadow: ${theme.shadows.glowSoft};
  }
`;

const DeliverableHeader = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
`;

const DeliverableTitle = styled.div`
  font-weight: 600;
  color: ${theme.colors.textBright};
`;

const DeliverableTag = styled.span`
  display: inline-flex;
  margin-top: 0.2rem;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  font-size: 0.65rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: ${theme.colors.softLavender};
  border: 1px dashed rgba(148, 163, 255, 0.4);
`;

const DeliverableMeta = styled.div`
  font-size: 0.85rem;
  color: ${theme.colors.textMuted};
  word-break: break-all;
`;

const DeliverableStatus = styled.span`
  display: inline-flex;
  align-items: center;
  padding: 0.3rem 0.85rem;
  border-radius: 999px;
  font-size: 0.7rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: ${({ $status }) => ($status === 'ready' ? theme.colors.success : theme.colors.warning)};
  border: 1px solid ${({ $status }) => ($status === 'ready' ? 'rgba(107, 255, 203, 0.4)' : 'rgba(248, 198, 70, 0.45)')};
  background: ${({ $status }) => ($status === 'ready' ? 'rgba(107, 255, 203, 0.12)' : 'rgba(248, 198, 70, 0.12)')};
`;

const DeliverablePreview = styled.video`
  width: 100%;
  border-radius: 16px;
  margin-top: 0.75rem;
  border: 1px solid rgba(148, 163, 255, 0.35);
  background: rgba(3, 5, 12, 0.85);
  box-shadow: ${theme.shadows.glowSoft};
  position: relative;
  z-index: 1;
  pointer-events: auto;
`;

const DeliverablePreviewFrame = styled.div`
  position: relative;
`;

const DeliverablePreviewNote = styled.span`
  margin-top: 0.75rem;
  font-size: 0.8rem;
  color: ${theme.colors.textMuted};
  font-style: italic;
`;

const DeliverableMetaOverlay = styled.div`
  position: absolute;
  top: 0.6rem;
  right: 0.6rem;
  border-radius: 14px;
  padding: 0.55rem 0.85rem;
  background: rgba(5, 10, 30, 0.78);
  border: 1px solid rgba(148, 163, 255, 0.4);
  backdrop-filter: blur(8px);
  box-shadow: ${theme.shadows.borderGlow};
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 170px;
  pointer-events: none;
  z-index: 2;
`;

const DeliverableMetaTitle = styled.span`
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: ${theme.colors.textBright};
`;

const DeliverableMetaLine = styled.span`
  font-size: 0.72rem;
  color: ${theme.colors.softLavender};
  letter-spacing: 0.05em;
`;

const ProcessedPreviewCard = styled.div`
  display: grid;
  gap: 1.4rem;
  grid-template-columns: minmax(240px, 480px) 1fr;
  align-items: center;
  padding: 1.4rem;
  border-radius: 26px;
  border: 1px solid rgba(74, 195, 255, 0.35);
  background: ${theme.colors.panelBg};
  box-shadow: ${theme.shadows.glowSoft};

  @media (max-width: 900px) {
    grid-template-columns: 1fr;
  }
`;

const ProcessedPreviewMeta = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
`;

const ProcessedPreviewTitle = styled.h4`
  margin: 0;
  font-size: 1.2rem;
  color: ${theme.colors.textBright};
`;

const ProcessedPreviewDetails = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  color: ${theme.colors.textMuted};

  .path {
    font-size: 0.85rem;
    word-break: break-all;
  }
`;

const ProcessedPreviewActions = styled.div`
  display: flex;
  gap: 0.8rem;
  flex-wrap: wrap;
`;

const ProcessedPreviewLink = styled.a`
  padding: 0.45rem 0.95rem;
  border-radius: 999px;
  border: 1px solid rgba(74, 195, 255, 0.4);
  color: ${theme.colors.glowBlue};
  text-decoration: none;
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  transition: border-color ${theme.transitions.default}, color ${theme.transitions.default};

  &:hover {
    border-color: ${theme.colors.glowTeal};
    color: ${theme.colors.glowTeal};
  }
`;

const RunList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
`;

const RunCard = styled.button`
  width: 100%;
  text-align: left;
  border-radius: 16px;
  border: 1px solid ${({ $active }) => ($active ? 'rgba(94, 234, 212, 0.6)' : 'rgba(148, 163, 255, 0.25)')};
  background: ${({ $active }) => ($active ? 'rgba(16, 185, 129, 0.12)' : 'rgba(15, 23, 42, 0.45)')};
  color: inherit;
  padding: 0.95rem 1rem;
  cursor: pointer;
`;

const RunTopRow = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  color: #cbd5f5;
`;

const RunStatus = styled.span`
  text-transform: uppercase;
  letter-spacing: 0.08em;
`;

const RunTimestamp = styled.span`
  color: #94a3b8;
`;

const RunTitle = styled.div`
  font-weight: 700;
  color: #f8fafc;
  margin: 0.4rem 0;
`;

const RunMeta = styled.div`
  font-size: 0.85rem;
  color: #94a3b8;
`;

const Toast = styled.div`
  position: fixed;
  bottom: 24px;
  right: 24px;
  background: rgba(15, 23, 42, 0.9);
  border: 1px solid rgba(94, 234, 212, 0.4);
  color: #ccfbf1;
  padding: 0.85rem 1.1rem;
  border-radius: 14px;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.45);
`;

const MutedText = styled.p`
  margin: 0;
  color: #94a3b8;
  font-style: italic;
`;

export default MediaSupplyChain;
