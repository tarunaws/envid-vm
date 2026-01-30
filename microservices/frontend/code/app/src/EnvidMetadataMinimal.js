import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import styled from 'styled-components';

// Envid Metadata backend (Flask)
// Default to CRA dev proxy (see src/setupProxy.js). Can be overridden via REACT_APP_ENVID_METADATA_BACKEND_URL.
const BACKEND_URL = process.env.REACT_APP_ENVID_METADATA_BACKEND_URL || '/backend';
const POLL_INTERVAL_MS = 2000;
const WHISPER_LANGUAGE_OPTIONS = [
  { value: 'auto', label: 'Auto Detect' },
  { value: 'af', label: 'Afrikaans' },
  { value: 'am', label: 'Amharic' },
  { value: 'ar', label: 'Arabic' },
  { value: 'as', label: 'Assamese' },
  { value: 'az', label: 'Azerbaijani' },
  { value: 'ba', label: 'Bashkir' },
  { value: 'be', label: 'Belarusian' },
  { value: 'bg', label: 'Bulgarian' },
  { value: 'bn', label: 'Bengali' },
  { value: 'bo', label: 'Tibetan' },
  { value: 'br', label: 'Breton' },
  { value: 'bs', label: 'Bosnian' },
  { value: 'ca', label: 'Catalan' },
  { value: 'cs', label: 'Czech' },
  { value: 'cy', label: 'Welsh' },
  { value: 'da', label: 'Danish' },
  { value: 'de', label: 'German' },
  { value: 'el', label: 'Greek' },
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Spanish' },
  { value: 'et', label: 'Estonian' },
  { value: 'eu', label: 'Basque' },
  { value: 'fa', label: 'Persian' },
  { value: 'fi', label: 'Finnish' },
  { value: 'fo', label: 'Faroese' },
  { value: 'fr', label: 'French' },
  { value: 'gl', label: 'Galician' },
  { value: 'gu', label: 'Gujarati' },
  { value: 'ha', label: 'Hausa' },
  { value: 'haw', label: 'Hawaiian' },
  { value: 'he', label: 'Hebrew' },
  { value: 'hi', label: 'Hindi' },
  { value: 'hr', label: 'Croatian' },
  { value: 'ht', label: 'Haitian Creole' },
  { value: 'hu', label: 'Hungarian' },
  { value: 'hy', label: 'Armenian' },
  { value: 'id', label: 'Indonesian' },
  { value: 'is', label: 'Icelandic' },
  { value: 'it', label: 'Italian' },
  { value: 'ja', label: 'Japanese' },
  { value: 'jw', label: 'Javanese' },
  { value: 'ka', label: 'Georgian' },
  { value: 'kk', label: 'Kazakh' },
  { value: 'km', label: 'Khmer' },
  { value: 'kn', label: 'Kannada' },
  { value: 'ko', label: 'Korean' },
  { value: 'la', label: 'Latin' },
  { value: 'lb', label: 'Luxembourgish' },
  { value: 'ln', label: 'Lingala' },
  { value: 'lo', label: 'Lao' },
  { value: 'lt', label: 'Lithuanian' },
  { value: 'lv', label: 'Latvian' },
  { value: 'mg', label: 'Malagasy' },
  { value: 'mi', label: 'Maori' },
  { value: 'mk', label: 'Macedonian' },
  { value: 'ml', label: 'Malayalam' },
  { value: 'mn', label: 'Mongolian' },
  { value: 'mr', label: 'Marathi' },
  { value: 'ms', label: 'Malay' },
  { value: 'mt', label: 'Maltese' },
  { value: 'my', label: 'Burmese' },
  { value: 'ne', label: 'Nepali' },
  { value: 'nl', label: 'Dutch' },
  { value: 'nn', label: 'Norwegian Nynorsk' },
  { value: 'no', label: 'Norwegian' },
  { value: 'oc', label: 'Occitan' },
  { value: 'pa', label: 'Punjabi' },
  { value: 'pl', label: 'Polish' },
  { value: 'ps', label: 'Pashto' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ro', label: 'Romanian' },
  { value: 'ru', label: 'Russian' },
  { value: 'sa', label: 'Sanskrit' },
  { value: 'sd', label: 'Sindhi' },
  { value: 'si', label: 'Sinhala' },
  { value: 'sk', label: 'Slovak' },
  { value: 'sl', label: 'Slovenian' },
  { value: 'sn', label: 'Shona' },
  { value: 'so', label: 'Somali' },
  { value: 'sq', label: 'Albanian' },
  { value: 'sr', label: 'Serbian' },
  { value: 'su', label: 'Sundanese' },
  { value: 'sv', label: 'Swedish' },
  { value: 'sw', label: 'Swahili' },
  { value: 'ta', label: 'Tamil' },
  { value: 'te', label: 'Telugu' },
  { value: 'tg', label: 'Tajik' },
  { value: 'th', label: 'Thai' },
  { value: 'tk', label: 'Turkmen' },
  { value: 'tl', label: 'Tagalog' },
  { value: 'tr', label: 'Turkish' },
  { value: 'tt', label: 'Tatar' },
  { value: 'uk', label: 'Ukrainian' },
  { value: 'ur', label: 'Urdu' },
  { value: 'uz', label: 'Uzbek' },
  { value: 'vi', label: 'Vietnamese' },
  { value: 'yi', label: 'Yiddish' },
  { value: 'yo', label: 'Yoruba' },
  { value: 'zh', label: 'Chinese' }
];

const PageWrapper = styled.div`
  min-height: 100vh;
  font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: radial-gradient(1200px circle at 20% 0%, rgba(37, 99, 235, 0.12) 0%, rgba(0, 0, 0, 0) 55%),
    linear-gradient(135deg, #05070f 0%, #0b1222 60%, #0b1020 100%);
  color: #e6edff;
  padding: 30px 22px 64px;
  overflow-x: hidden;
  --panel: rgba(9, 14, 26, 0.82);
  --panel-border: rgba(59, 130, 246, 0.18);
  --muted: rgba(170, 181, 206, 0.82);
  --accent: #3b82f6;
  --accent-2: #2563eb;
  --danger: #ef4444;
  --success: #22c55e;
  --warning: #f59e0b;
  --chip: rgba(59, 130, 246, 0.14);
`;

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
  min-width: 0;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 20px;
`;

const Title = styled.h1`
  font-size: 2.1rem;
  font-weight: 800;
  margin: 0;
  color: #eef3ff;
  letter-spacing: -0.015em;
`;

const Subtitle = styled.p`
  font-size: 0.95rem;
  margin: 8px 0 0 0;
  color: var(--muted);
`;

const Section = styled.div`
  background: var(--panel);
  border-radius: 14px;
  padding: 20px;
  box-shadow: 0 14px 40px rgba(0, 0, 0, 0.35);
  border: 1px solid var(--panel-border);
  backdrop-filter: blur(12px);
  max-width: 100%;
  min-width: 0;
`;

const SectionTitle = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 800;
  color: #eaf1ff;
  margin-bottom: 12px;
`;

const Icon = styled.span`
  display: inline-flex;
  font-size: 1.1rem;
`;

const Message = styled.div`
  padding: 12px 14px;
  border-radius: 12px;
  margin: 10px 0 18px 0;
  font-weight: 700;
  color: ${(props) => (props.type === 'error' ? '#ffd1d1' : props.type === 'success' ? '#d7ffe7' : '#d8e8ff')};
  background: ${(props) =>
    props.type === 'error'
      ? 'rgba(239, 68, 68, 0.15)'
      : props.type === 'success'
        ? 'rgba(34, 197, 94, 0.16)'
        : 'rgba(59, 130, 246, 0.18)'};
  border: 1px solid
    ${(props) =>
      props.type === 'error'
        ? 'rgba(239, 68, 68, 0.25)'
        : props.type === 'success'
          ? 'rgba(34, 197, 94, 0.22)'
          : 'rgba(59, 130, 246, 0.28)'};
`;

const Row = styled.div`
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
`;

const Button = styled.button`
  background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
  color: white;
  border: none;
  padding: 10px 16px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 800;
  box-shadow: 0 10px 22px rgba(37, 99, 235, 0.3);
  transition: transform 0.15s ease, box-shadow 0.15s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 26px rgba(37, 99, 235, 0.35);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    box-shadow: none;
  }
`;
const SecondaryButton = styled.button`
  background: rgba(59, 130, 246, 0.08);
  color: #cfe0ff;
  border: 1px solid rgba(59, 130, 246, 0.3);
  padding: 10px 16px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 800;
  transition: border-color 0.15s ease, transform 0.15s ease;

  &:hover {
    transform: translateY(-1px);
    border-color: rgba(59, 130, 246, 0.5);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const IconButton = styled.button`
  width: 34px;
  height: 34px;
  border-radius: 10px;
  border: 1px solid rgba(59, 130, 246, 0.3);
  background: rgba(59, 130, 246, 0.12);
  color: #dbe7ff;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const DeleteButton = styled.button`
  background: rgba(239, 68, 68, 0.9);
  color: white;
  border: none;
  padding: 8px 12px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 900;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const ReprocessButton = styled.button`
  background: rgba(59, 130, 246, 0.9);
  color: white;
  border: none;
  padding: 8px 12px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 900;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const UploadArea = styled.div`
  border: 2px dashed ${(props) => (props.$dragging ? 'rgba(59, 130, 246, 0.95)' : 'rgba(255, 255, 255, 0.18)')};
  background: ${(props) => (props.$dragging ? 'rgba(59, 130, 246, 0.14)' : 'rgba(255, 255, 255, 0.04)')};
  border-radius: 16px;
  padding: 26px;
  text-align: center;
  cursor: pointer;
  max-width: 100%;
  box-sizing: border-box;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 10px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  margin-top: 12px;
  overflow: hidden;
`;

const ProgressFill = styled.div`
  height: 100%;
  width: ${(props) => props.$percent}%;
  background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
  transition: width 0.2s ease;
`;

const StatusPanel = styled.div`
  border: 1px solid rgba(59, 130, 246, 0.18);
  background: rgba(9, 14, 26, 0.6);
  border-radius: 12px;
  padding: 14px;
`;

const StatusTitleRow = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin-bottom: 10px;
`;

const StatusTitle = styled.div`
  font-weight: 900;
  letter-spacing: 0.2px;
  color: rgba(240, 242, 255, 0.95);
  text-align: center;
`;

const StatusTable = styled.div`
  display: grid;
  gap: 8px;
  margin-top: 10px;
`;

const StatusTableHeader = styled.div`
  display: grid;
  grid-template-columns: minmax(180px, 1fr) 160px;
  gap: 12px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  color: rgba(210, 214, 240, 0.7);
  padding: 0 6px;
`;

const StatusTableRow = styled.div`
  display: grid;
  grid-template-columns: minmax(180px, 1fr) 160px;
  gap: 12px;
  align-items: center;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(12, 18, 42, 0.55);
  border: 1px solid rgba(255, 255, 255, 0.08);
`;

const StatusTableCell = styled.div`
  font-size: 13px;
  color: rgba(240, 242, 255, 0.9);
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const StatusPill = styled.div`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: ${({ $tone }) => {
    if ($tone === 'completed') return 'rgba(56, 193, 114, 0.2)';
    if ($tone === 'in_progress') return 'rgba(245, 158, 11, 0.2)';
    return 'rgba(148, 163, 184, 0.2)';
  }};
  color: ${({ $tone }) => {
    if ($tone === 'completed') return '#4ade80';
    if ($tone === 'in_progress') return '#fbbf24';
    return '#cbd5f5';
  }};
`;

const StatusBucketsGrid = styled.div`
  display: grid;
  gap: 12px;
  margin-top: 10px;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
`;

const StatusBucket = styled.div`
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(12, 18, 42, 0.5);
  padding: 10px;
  min-height: 90px;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const StatusBucketTitle = styled.div`
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  color: rgba(210, 214, 240, 0.7);
`;

const StatusMeta = styled.div`
  color: var(--muted);
  font-size: 12px;
  text-align: center;
  word-break: break-all;
  line-height: 1.4;
`;

const StatusMetaRight = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
  width: 100%;
`;

const StepList = styled.div`
  display: grid;
  gap: 10px;
`;

const StepRow = styled.div`
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
  align-items: center;
`;

const StepLeft = styled.div`
  display: grid;
  gap: 6px;
`;

const StepLabelRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
`;

const StepLabel = styled.div`
  font-weight: 850;
  font-size: 13px;
  color: rgba(240, 242, 255, 0.95);
`;

const StepHint = styled.div`
  font-size: 12px;
  color: var(--muted);
`;

const StepBadge = styled.div`
  font-size: 12px;
  font-weight: 800;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: ${(props) => {
    const v = String(props.$variant || 'neutral');
    if (v === 'ok') return 'rgba(34, 197, 94, 0.16)';
    if (v === 'run') return 'rgba(59, 130, 246, 0.16)';
    if (v === 'warn') return 'rgba(245, 158, 11, 0.16)';
    if (v === 'bad') return 'rgba(239, 68, 68, 0.16)';
    return 'rgba(255, 255, 255, 0.05)';
  }};
  color: ${(props) => {
    const v = String(props.$variant || 'neutral');
    if (v === 'ok') return 'rgba(167, 243, 208, 0.95)';
    if (v === 'run') return 'rgba(199, 210, 254, 0.95)';
    if (v === 'warn') return 'rgba(253, 230, 138, 0.95)';
    if (v === 'bad') return 'rgba(254, 202, 202, 0.95)';
    return 'rgba(230, 232, 242, 0.85)';
  }};
`;

const SubtleNote = styled.div`
  color: var(--muted);
  font-size: 12px;
`;

const TabsRow = styled.div`
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 16px;
`;

const TabButton = styled.button`
  background: ${(props) => (props.$active ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255, 255, 255, 0.03)')};
  color: ${(props) => (props.$active ? '#eaf1ff' : 'rgba(210, 220, 238, 0.8)')};
  border: 1px solid ${(props) => (props.$active ? 'rgba(59, 130, 246, 0.45)' : 'rgba(255, 255, 255, 0.08)')};
  padding: 8px 14px;
  border-radius: 999px;
  font-weight: 800;
  cursor: pointer;
`;

const RunningJobsList = styled.div`
  display: grid;
  gap: 12px;
`;

const RunningJobRow = styled.div`
  border: 1px solid rgba(59, 130, 246, 0.18);
  background: rgba(10, 16, 32, 0.6);
  border-radius: 14px;
  padding: 14px;
`;

const RunningJobTitle = styled.div`
  font-weight: 900;
  color: #eaf1ff;
`;

const RunningJobStatus = styled.div`
  font-size: 12px;
  font-weight: 800;
  color: #9fb7ff;
`;

const RunningJobMeta = styled.div`
  margin-top: 6px;
  color: var(--muted);
  font-size: 12px;
`;

const RunningJobSteps = styled.div`
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
`;

const RunningJobId = styled.div`
  font-size: 12px;
  color: rgba(59, 130, 246, 0.9);
  font-weight: 800;
`;

const CompletedJobActions = styled.div`
  display: grid;
  gap: 6px;
  justify-items: end;
  align-content: center;
  justify-self: end;
`;

const CompletedDetailButton = styled.button`
  background: rgba(34, 197, 94, 0.92);
  color: white;
  border: none;
  padding: 5px 8px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 900;
  font-size: 11px;
`;

const CompletedDownloadTable = styled.div`
  display: grid;
  grid-template-columns: repeat(2, minmax(180px, 1fr));
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.02);
`;

const CompletedDownloadCell = styled.div`
  padding: 10px 12px 12px;
  border-right: 1px solid rgba(255, 255, 255, 0.06);
  display: grid;
  align-content: start;

  &:last-child {
    border-right: none;
  }
`;

const ExpandButton = styled.button`
  background: rgba(59, 130, 246, 0.18);
  color: #cfe0ff;
  border: 1px solid rgba(59, 130, 246, 0.3);
  padding: 6px 10px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 800;
`;

const Card = styled.div`
  border: 1px solid rgba(59, 130, 246, 0.16);
  background: rgba(9, 14, 26, 0.6);
  border-radius: 12px;
  padding: 14px;
`;

const CardHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
`;

const CardHeaderTitle = styled.div`
  font-weight: 900;
  color: #eaf1ff;
`;

const CardBody = styled.div`
  color: rgba(230, 232, 242, 0.9);
`;

const Grid2 = styled.div`
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
`;

const CountPill = styled.div`
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  background: rgba(59, 130, 246, 0.18);
  color: #cfe0ff;
  border: 1px solid rgba(59, 130, 246, 0.3);
`;

const DownloadGrid = styled.div`
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
`;

const DownloadGroup = styled.div`
  display: grid;
  gap: 10px;
`;

const DownloadGroupTitle = styled.div`
  font-weight: 900;
  color: #eaf1ff;
`;

const DownloadGroupHint = styled.div`
  color: var(--muted);
  font-size: 12px;
`;

const DownloadList = styled.div`
  display: grid;
  gap: 8px;
`;

const DownloadRow = styled.div`
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 8px;
  align-items: center;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid rgba(59, 130, 246, 0.14);
  background: rgba(9, 14, 26, 0.6);
`;

const DownloadLabel = styled.div`
  font-weight: 800;
  color: #eaf1ff;
`;

const LinkA = styled.a`
  color: #93c5fd;
  text-decoration: none;
  font-weight: 800;

  &:hover {
    color: #bfdbfe;
    text-decoration: underline;
  }
`;

const VideoFrame = styled.div`
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: rgba(0, 0, 0, 0.45);
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(59, 130, 246, 0.18);
`;

const PlayerFullscreenButton = styled.button`
  position: absolute;
  top: 10px;
  right: 10px;
  border: none;
  background: rgba(15, 23, 42, 0.8);
  color: #eaf1ff;
  border-radius: 10px;
  padding: 6px 8px;
  font-weight: 800;
  cursor: pointer;
`;

const VideoOverlayBar = styled.div`
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 8px 10px;
  background: linear-gradient(180deg, rgba(0, 0, 0, 0) 0%, rgba(7, 10, 20, 0.8) 100%);
`;

const OverlayScroller = styled.div`
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 4px;
`;

const OverlayChip = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.22);
  color: #eaf1ff;
  font-size: 12px;
  font-weight: 800;
`;

const OverlayChipThumb = styled.div`
  width: 22px;
  height: 22px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.1);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
`;

const OverlayChipName = styled.div``;

const VideoOverlaySidePanel = styled.div`
  position: absolute;
  right: 10px;
  top: 10px;
  bottom: 10px;
  width: 220px;
  background: rgba(10, 16, 32, 0.6);
  border-radius: 12px;
  border: 1px solid rgba(59, 130, 246, 0.2);
  padding: 8px;
`;

const OverlaySideList = styled.div`
  display: grid;
  gap: 8px;
  max-height: 100%;
  overflow-y: auto;
`;

const OverlaySideItem = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
`;

const OverlayThumb = styled.div`
  width: 34px;
  height: 34px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.08);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
`;

const OverlayName = styled.div`
  font-weight: 800;
  color: #eaf1ff;
  font-size: 12px;
`;

const SegmentChip = styled.button`
  background: rgba(59, 130, 246, 0.16);
  border: 1px solid rgba(59, 130, 246, 0.3);
  color: #cfe0ff;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 800;
  cursor: pointer;
  font-size: 12px;
`;

const StepChip = styled.button`
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 800;
  cursor: ${(props) => (props.$restartable ? 'pointer' : 'default')};
  border: 1px solid rgba(59, 130, 246, 0.25);
  background: rgba(59, 130, 246, 0.14);
  color: #cfe0ff;
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding-right: ${(props) => (props.$restartable ? '24px' : '10px')};

  ${(props) => {
    const s = String(props.$status || '').toLowerCase();
    if (s === 'completed') {
      return `
        background: rgba(34, 197, 94, 0.18);
        border-color: rgba(34, 197, 94, 0.35);
        color: #d1fae5;
      `;
    }
    if (s === 'running') {
      return `
        background: rgba(245, 158, 11, 0.2);
        border-color: rgba(245, 158, 11, 0.4);
        color: #fde68a;
      `;
    }
    if (s === 'failed') {
      return `
        background: rgba(239, 68, 68, 0.2);
        border-color: rgba(239, 68, 68, 0.4);
        color: #fecaca;
      `;
    }
    if (s === 'skipped') {
      return `
        background: rgba(148, 163, 184, 0.18);
        border-color: rgba(148, 163, 184, 0.35);
        color: rgba(226, 232, 240, 0.9);
      `;
    }
    return '';
  }}

  &::after {
    content: '⟲';
    position: absolute;
    right: 8px;
    opacity: 0;
    font-size: 18px;
    transition: opacity 0.15s ease;
  }

  ${(props) =>
    props.$restartable
      ? `
    &:hover::after {
      opacity: 0.85;
    }
  `
      : `
    &::after {
      display: none;
    }
  `}
`;

const StepDot = styled.span`
  display: none;
`;

const DetailPanel = styled.div`
  margin-top: 14px;
  border: 1px solid rgba(59, 130, 246, 0.18);
  border-radius: 14px;
  background: rgba(9, 14, 26, 0.7);
  padding: 16px;
`;

const DetailHeader = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 12px;
`;

const DetailTitle = styled.div`
  font-weight: 900;
  color: #eaf1ff;
`;

const DetailBody = styled.div`
  display: grid;
  gap: 16px;
`;

const ModalOverlay = styled.div`
  position: fixed;
  inset: 0;
  background: rgba(5, 7, 15, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  z-index: 50;
`;

const ModalCard = styled.div`
  background: rgba(9, 14, 26, 0.96);
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: 14px;
  width: min(720px, 95vw);
  padding: 16px;
  box-shadow: 0 18px 52px rgba(0, 0, 0, 0.45);
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const ModalHeader = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
`;

const ModalTitle = styled.div`
  font-weight: 900;
  color: #eaf1ff;
`;

const ModalSubtitle = styled.div`
  color: var(--muted);
  font-size: 12px;
`;

const ModalBody = styled.div`
  display: grid;
  gap: 12px;
  overflow-y: auto;
  padding-right: 4px;
`;

const ModalActions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 10px;
`;

const BrowserPath = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
  color: var(--muted);
  font-size: 12px;
`;

const BrowserList = styled.div`
  display: grid;
  gap: 6px;
  max-height: 55vh;
  overflow-y: auto;
  padding-right: 4px;
`;

const BrowserRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid rgba(59, 130, 246, 0.12);
  background: rgba(15, 23, 42, 0.6);
  cursor: pointer;
`;

const BrowserIcon = styled.span`
  font-size: 16px;
`;

const MultiColumnList = styled.div`
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
`;

const MultiColumnItem = styled.div`
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(59, 130, 246, 0.12);
  border-radius: 12px;
  padding: 10px 12px;
`;

const JobDetailPanel = styled.div`
  background: rgba(10, 16, 32, 0.7);
  border: 1px solid rgba(59, 130, 246, 0.18);
  border-radius: 12px;
  padding: 12px;
`;

const JobThumb = styled.div`
  width: 56px;
  height: 56px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(59, 130, 246, 0.2);
  display: inline-flex;
  align-items: center;
  justify-content: center;
`;

const TopStatsBar = styled.div`
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
`;

const TopStatsItem = styled.div`
  padding: 8px 12px;
  border-radius: 12px;
  border: 1px solid rgba(59, 130, 246, 0.25);
  background: rgba(59, 130, 246, 0.16);
  font-size: 12px;
  font-weight: 800;
  color: #cfe0ff;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
`;

const Th = styled.th`
  text-align: left;
  padding: 10px;
  background: rgba(59, 130, 246, 0.1);
  color: #eaf1ff;
  font-weight: 800;
`;

const Td = styled.td`
  padding: 10px;
  border-top: 1px solid rgba(59, 130, 246, 0.12);
  color: rgba(230, 232, 242, 0.85);
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 18px 0;
  color: rgba(230, 232, 242, 0.7);
`;

const EmptyIcon = styled.div`
  font-size: 2rem;
  margin-bottom: 6px;
`;

const Carousel = styled.div`
  display: flex;
  flex-wrap: nowrap;
  gap: 12px;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 4px 2px 10px 2px;
  scroll-snap-type: x mandatory;
  scroll-padding-left: 2px;

  &::-webkit-scrollbar {
    height: 10px;
  }
  &::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 999px;
  }
  &::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.16);
    border-radius: 999px;
  }
`;

const CarouselItem = styled.div`
  width: 260px;
  max-width: 260px;
  min-width: 220px;
  flex: 0 0 260px;
  scroll-snap-align: start;
  border: 1px solid ${(props) => (props.$active ? 'rgba(102, 126, 234, 0.7)' : 'rgba(255, 255, 255, 0.08)')};
  border-radius: 14px;
  background: ${(props) => (props.$active ? 'rgba(102, 126, 234, 0.10)' : 'rgba(255, 255, 255, 0.03)')};
  overflow: hidden;
  position: relative;
  cursor: pointer;
  transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease;

  box-shadow: ${(props) => (props.$active ? '0 0 0 3px rgba(102, 126, 234, 0.22), 0 16px 40px rgba(0, 0, 0, 0.35)' : 'none')};

  &:hover {
    transform: translateY(-2px);
    border-color: rgba(102, 126, 234, 0.35);
    background: rgba(255, 255, 255, 0.04);
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25), 0 16px 40px rgba(0, 0, 0, 0.35);
    border-color: rgba(56, 189, 248, 0.55);
  }
`;

const CarouselThumb = styled.div`
  width: 100%;
  height: 130px;
  background: radial-gradient(700px circle at 20% 20%, rgba(102, 126, 234, 0.35) 0%, rgba(0, 0, 0, 0) 55%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  align-items: center;
  justify-content: center;
`;

const CarouselThumbImg = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
`;

const CarouselBody = styled.div`
  padding: 10px 12px 12px 12px;
  min-width: 0;
`;

const CarouselTitle = styled.div`
  font-weight: 900;
  color: #e6e8f2;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const CarouselMeta = styled.div`
  margin-top: 6px;
  color: rgba(230, 232, 242, 0.7);
  font-size: 12px;
  line-height: 1.35;
  overflow-wrap: anywhere;
  word-break: break-word;
`;

const TinyButton = styled.button`
  border: 1px solid rgba(255, 255, 255, 0.12);
  width: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
`;

const BrowserMeta = styled.span`
  color: rgba(230, 232, 242, 0.65);
  font-size: 0.8rem;
`;

const BrowserEmpty = styled.div`
  padding: 12px;
  border-radius: 10px;
  border: 1px dashed rgba(255, 255, 255, 0.12);
  color: rgba(230, 232, 242, 0.6);
  text-align: center;
`;

const ConfirmText = styled.div`
  color: rgba(230, 232, 242, 0.8);
  font-size: 0.95rem;
  margin-bottom: 18px;
`;

const ConfirmActions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 10px;
`;

const DEFAULT_PIPELINE_STEPS = [
  { id: 'upload_to_cloud_storage', label: 'Upload to cloud storage', status: 'not_started', percent: 0, message: null },
  { id: 'technical_metadata', label: 'Technical Metadata', status: 'not_started', percent: 0, message: null },
  { id: 'transcode_normalize', label: 'Transcode to Normalize @ 1.5 mbps', status: 'not_started', percent: 0, message: null },
  { id: 'label_detection', label: 'Label detection', status: 'not_started', percent: 0, message: null },
  { id: 'moderation', label: 'Moderation', status: 'not_started', percent: 0, message: null },
  { id: 'text_on_screen', label: 'Text on Screen', status: 'not_started', percent: 0, message: null },
  { id: 'key_scene_detection', label: 'Key scene & high point detection', status: 'not_started', percent: 0, message: null },
  { id: 'transcribe', label: 'Audio Transcription', status: 'not_started', percent: 0, message: null },
  { id: 'synopsis_generation', label: 'Synopsis Generation', status: 'not_started', percent: 0, message: null },
  { id: 'scene_by_scene_metadata', label: 'Scene by scene metadata', status: 'not_started', percent: 0, message: null },
  { id: 'famous_location_detection', label: 'Famous location detection', status: 'not_started', percent: 0, message: null }, // disabled by default
  { id: 'translate_output', label: 'Translate output', status: 'not_started', percent: 0, message: null },
  { id: 'opening_closing_credit_detection', label: 'Opening/Closing credit detection', status: 'not_started', percent: 0, message: null }, // disabled by default
  { id: 'celebrity_detection', label: 'Celebrity detection', status: 'not_started', percent: 0, message: null }, // not implemented
  { id: 'celebrity_bio_image', label: 'Celebrity bio & Image', status: 'not_started', percent: 0, message: null }, // not implemented
];

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value)) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let out = value;
  let idx = 0;
  while (out >= 1024 && idx < units.length - 1) {
    out /= 1024;
    idx += 1;
  }
  const precision = out >= 10 || idx === 0 ? 0 : 1;
  return `${out.toFixed(precision)} ${units[idx]}`;
}

function formatTimestamp(value) {
  try {
    if (!value) return '—';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString();
  } catch {
    return String(value || '—');
  }
}

function formatDurationMs(startValue, endValue) {
  if (!startValue || !endValue) return '—';
  const start = new Date(startValue);
  const end = new Date(endValue);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return '—';
  const diffMs = Math.max(0, end.getTime() - start.getTime());
  const totalSeconds = Math.floor(diffMs / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(seconds)}` : `${minutes}:${pad(seconds)}`;
}

function asJpegDataUri(maybeBase64) {
  const v = (maybeBase64 || '').toString().trim();
  if (!v) return null;
  if (v.startsWith('data:image/')) return v;
  return `data:image/jpeg;base64,${v}`;
}

function formatSeconds(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s < 0) return '—';
  const total = Math.floor(s);
  const hh = Math.floor(total / 3600);
  const mm = Math.floor((total % 3600) / 60);
  const ss = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return hh > 0 ? `${hh}:${pad(mm)}:${pad(ss)}` : `${mm}:${pad(ss)}`;
}

function formatTaskDurationSummary(taskDurations) {
  if (!taskDurations || typeof taskDurations !== 'object') return '';
  const allowed = new Set(['label_detection', 'transcribe']);
  const entries = Object.entries(taskDurations)
    .map(([key, value]) => ({ key, value: Number(value) }))
    .filter((item) => allowed.has(item.key) && Number.isFinite(item.value) && item.value > 0);

  if (!entries.length) return '';

  entries.sort((a, b) => b.value - a.value);
  const parts = entries.map((item) => `${item.key} ${formatSeconds(item.value)}`);
  return parts.length ? `Task time: ${parts.join(' · ')}` : '';
}

function formatContainerFormat(raw) {
  const s = (raw || '').toString().trim();
  if (!s) return { label: '—', title: null };
  const parts = s
    .split(',')
    .map((p) => p.trim())
    .filter(Boolean);

  if (parts.length <= 1) return { label: parts[0] || s, title: null };

  // ffprobe often reports a single demuxer name with multiple aliases for the same container family.
  // Prefer the most recognizable alias for display.
  const preferred = ['mp4', 'mov', 'm4a', '3gp', '3g2', 'mj2'];
  let chosen = null;
  for (const p of preferred) {
    if (parts.includes(p)) {
      chosen = p;
      break;
    }
  }
  chosen = chosen || parts[0];
  return {
    label: chosen,
    title: `ffprobe format aliases: ${parts.join(', ')}`,
  };
}

function formatTimecode(seconds, fps) {
  const s = Number(seconds);
  const f = Number(fps);
  if (!Number.isFinite(s) || s < 0) return '—';
  const safeFps = Number.isFinite(f) && f > 0 ? f : 30;

  let totalSeconds = Math.floor(s);
  let frame = Math.floor((s - totalSeconds) * safeFps + 1e-9);
  // Guard against floating point rounding that can produce frame==fps
  if (frame >= safeFps) {
    frame = 0;
    totalSeconds += 1;
  }

  const hh = Math.floor(totalSeconds / 3600);
  const mm = Math.floor((totalSeconds % 3600) / 60);
  const ss = totalSeconds % 60;
  const pad2 = (n) => String(n).padStart(2, '0');
  const framePad = safeFps >= 100 ? 3 : 2;
  const padFrames = (n) => String(n).padStart(framePad, '0');
  return `${pad2(hh)}:${pad2(mm)}:${pad2(ss)}:${padFrames(frame)}`;
}

export default function EnvidMetadataMinimal({ initialTab = 'workflow' } = {}) {
  const fileInputRef = useRef(null);
  const pollRef = useRef(null);
  const playerRef = useRef(null);
  const playerContainerRef = useRef(null);

  const normalizeTab = (tab) => {
    const key = String(tab || '').toLowerCase();
    if (['running', 'completed', 'failed', 'workflow'].includes(key)) return key;
    return 'workflow';
  };

  const [message, setMessage] = useState(null);
  const [activeTab, setActiveTab] = useState(normalizeTab(initialTab));
  const [runningJobs, setRunningJobs] = useState([]);
  const [runningJobsLoading, setRunningJobsLoading] = useState(false);
  const [runningJobsError, setRunningJobsError] = useState('');
  const [completedJobs, setCompletedJobs] = useState([]);
  const [completedJobsLoading, setCompletedJobsLoading] = useState(false);
  const [completedJobsError, setCompletedJobsError] = useState('');
  const [failedJobs, setFailedJobs] = useState([]);
  const [failedJobsLoading, setFailedJobsLoading] = useState(false);
  const [failedJobsError, setFailedJobsError] = useState('');
  const [deletedJobIds, setDeletedJobIds] = useState(() => new Set());
  const [expandedCompletedJobId, setExpandedCompletedJobId] = useState('');
  const [jobSubmitModal, setJobSubmitModal] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [expandedJobId, setExpandedJobId] = useState('');
  const [expandedJobDetail, setExpandedJobDetail] = useState(null);
  const [expandedJobLoading, setExpandedJobLoading] = useState(false);

  // Multimodal task selection.
  // Defaults: core tasks ON, future/disabled tasks OFF.
  const [taskSelection, setTaskSelection] = useState(() => ({
    enable_label_detection: true,
    label_detection_model: 'gcp_video_intelligence',

    enable_text_on_screen: true,
    text_model: 'tesseract',

    enable_moderation: true,
    moderation_model: 'nudenet',

    enable_key_scene: true,
    key_scene_detection_model: 'transnetv2_clip_cluster',

    enable_scene_by_scene: true,
    enable_high_point: true,

    enable_transcribe: true,
    transcribe_model: 'openai-whisper',

    enable_synopsis_generation: true,
    enable_translate_output: true,

    enable_famous_locations: false,
    famous_location_detection_model: 'auto',

    enable_opening_closing: false,
    opening_closing_credit_detection_model: 'auto',

    enable_celebrity_detection: false,
    celebrity_detection_model: 'auto',

    enable_celebrity_bio_image: false,
    celebrity_bio_image_model: 'auto',
  }));

  const [targetTranslateLanguages, setTargetTranslateLanguages] = useState([]);
  const [transcribeLanguage, setTranscribeLanguage] = useState('auto');

  const taskSelectionPayload = useMemo(() => {
    const sel = taskSelection || {};
    return {
      enable_scene_by_scene_metadata: true,
      enable_scene_by_scene: true,

      enable_label_detection: Boolean(sel.enable_label_detection),
      label_detection_model: String(sel.label_detection_model || '').trim() || 'gcp_video_intelligence',

      enable_moderation: true,
      moderation_model: String(sel.moderation_model || '').trim() || 'nudenet',

      enable_text: true,
      enable_text_on_screen: true,
      text_model: String(sel.text_model || '').trim() || 'tesseract',

      enable_key_scene_detection: true,
      enable_key_scene: true,
      key_scene_detection_model: String(sel.key_scene_detection_model || '').trim() || 'transnetv2_clip_cluster',

      enable_high_point: true,

      enable_transcribe: true,
      transcribe_model: String(sel.transcribe_model || '').trim() || 'openai-whisper',
      transcribe_language: String(transcribeLanguage || '').trim() || 'auto',

      enable_synopsis_generation: true,
      synopsis_generation_model: 'auto',

      enable_translate_output: true,

      enable_famous_location_detection: Boolean(sel.enable_famous_locations),
      enable_famous_locations: Boolean(sel.enable_famous_locations),
      famous_location_detection_model: String(sel.famous_location_detection_model || '').trim() || 'auto',

      enable_opening_closing_credit_detection: Boolean(sel.enable_opening_closing),
      enable_opening_closing: Boolean(sel.enable_opening_closing),
      opening_closing_credit_detection_model: String(sel.opening_closing_credit_detection_model || '').trim() || 'auto',

      enable_celebrity_detection: Boolean(sel.enable_celebrity_detection),
      celebrity_detection_model: String(sel.celebrity_detection_model || '').trim() || 'auto',

      enable_celebrity_bio_image: Boolean(sel.enable_celebrity_bio_image),
      celebrity_bio_image_model: String(sel.celebrity_bio_image_model || '').trim() || 'auto',

      translate_targets: Array.isArray(targetTranslateLanguages) ? targetTranslateLanguages : [],
    };
  }, [taskSelection, targetTranslateLanguages, transcribeLanguage]);

  const [videoSource, setVideoSource] = useState('gcs'); // 'local' | 'gcs'
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const [internationalLanguageOptions, setInternationalLanguageOptions] = useState([]);
  const [indianLanguageOptions, setIndianLanguageOptions] = useState([]);
  const [translateLanguagesLoading, setTranslateLanguagesLoading] = useState(false);
  const [translateLanguagesError, setTranslateLanguagesError] = useState('');

  const [gcsRawVideoObject, setGcsRawVideoObject] = useState('');
  const [gcsRawVideoLoading, setGcsRawVideoLoading] = useState(false);
  const [gcsBrowserOpen, setGcsBrowserOpen] = useState(false);
  const [gcsBrowserPrefix, setGcsBrowserPrefix] = useState('');
  const [gcsBrowserQuery, setGcsBrowserQuery] = useState('');
  const [gcsBrowserObjects, setGcsBrowserObjects] = useState([]);
  const [gcsBrowserPrefixes, setGcsBrowserPrefixes] = useState([]);
  const [gcsBuckets, setGcsBuckets] = useState([]);
  const [gcsBucket, setGcsBucket] = useState('');
  const [gcsBucketLoading, setGcsBucketLoading] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [clientUploadProgress, setClientUploadProgress] = useState(0);
  const [uploadJob, setUploadJob] = useState(null);
  const [activeJob, setActiveJob] = useState(null); // { kind: 'upload'|'reprocess', jobId: string, videoId?: string }
  const [systemStats, setSystemStats] = useState(null);



  const [selectedVideoId, setSelectedVideoId] = useState(null);
  const [selectedVideoTitle, setSelectedVideoTitle] = useState(null);
  const [selectedMeta, setSelectedMeta] = useState(null);
  const [selectedMetaLoading, setSelectedMetaLoading] = useState(false);
  const [selectedMetaError, setSelectedMetaError] = useState(null);
  const [playerTime, setPlayerTime] = useState(0);
  const [playerIsPaused, setPlayerIsPaused] = useState(false);
  const [playerIsFullscreen, setPlayerIsFullscreen] = useState(false);
  const [isSmallScreen, setIsSmallScreen] = useState(false);

  const getJobFileName = (job) => {
    const uri = String(
      job?.gcs_video_uri || job?.gcs_working_uri || job?.result?.gcs_video_uri || job?.result?.gcs_uri || ''
    ).trim();
    if (uri) {
      const parts = uri.split('/');
      const name = parts[parts.length - 1] || '';
      if (name) return name;
    }
    return String(job?.title || job?.name || job?.id || job?.job_id || 'Unknown').trim();
  };

  const gcsBaseUriFromArtifacts = (artifacts) => {
    const bucket = String(artifacts?.bucket || '').trim();
    const basePrefix = String(artifacts?.base_prefix || '').trim();
    if (bucket && basePrefix) return `gs://${bucket}/${basePrefix}`;
    return '';
  };

  const gcsConsoleUrlFromUri = (gsUri) => {
    const raw = String(gsUri || '').trim();
    if (!raw.startsWith('gs://')) return '';
    const parts = raw.replace('gs://', '').split('/').filter(Boolean);
    if (!parts.length) return '';
    const bucket = parts.shift();
    const prefix = parts.join('/');
    const encodedPrefix = prefix
      .split('/')
      .map((segment) => encodeURIComponent(segment))
      .join('/');
    return `https://console.cloud.google.com/storage/browser/${bucket}/${encodedPrefix}`;
  };

  useEffect(() => {
    const onFsChange = () => {
      try {
        setPlayerIsFullscreen(Boolean(document.fullscreenElement || document.webkitFullscreenElement));
      } catch {
        setPlayerIsFullscreen(false);
      }
    };
    document.addEventListener('fullscreenchange', onFsChange);
    document.addEventListener('webkitfullscreenchange', onFsChange);
    onFsChange();
    return () => {
      document.removeEventListener('fullscreenchange', onFsChange);
      document.removeEventListener('webkitfullscreenchange', onFsChange);
    };
  }, []);

  useEffect(() => {
    setActiveTab(normalizeTab(initialTab));
  }, [initialTab]);

  useEffect(() => {
    if (!gcsBrowserOpen) return undefined;
    const { body } = document;
    const previous = body.style.overflow;
    body.style.overflow = 'hidden';
    return () => {
      body.style.overflow = previous || '';
    };
  }, [gcsBrowserOpen]);

  const manualRefreshRunningJobs = async () => {
    if (activeTab !== 'running') return;
    setRunningJobsLoading(true);
    setRunningJobsError('');
    try {
      const resp = await axios.get(`${BACKEND_URL}/jobs`, {
        params: {
          status: 'processing,preflight,queued,running,stopping,stopped',
          limit: 50,
        },
      });
      const list = Array.isArray(resp.data?.jobs) ? resp.data.jobs : [];
      const filtered = list.filter((j) => !deletedJobIds.has(String(j?.id || j?.job_id || '').trim()));
      setRunningJobs(filtered);
      setRunningJobsError('');
    } catch (err) {
      setRunningJobsError(err?.response?.data?.error || 'Failed to load running jobs');
    } finally {
      setRunningJobsLoading(false);
    }
  };


  const requestDeleteJob = (jobId) => {
    const id = String(jobId || '').trim();
    if (!id) return;
    setDeleteConfirm({ jobId: id });
  };

  const confirmDeleteJob = (jobId) => {
    const id = String(jobId || '').trim();
    if (!id) return;
    setDeleteConfirm(null);
    deleteJob(id);
  };

  const deleteJob = async (jobId) => {
    const id = String(jobId || '').trim();
    if (!id) return;
    const prevRunning = runningJobs;
    const prevCompleted = completedJobs;
    const prevFailed = failedJobs;

    setDeletedJobIds((prev) => new Set(prev).add(id));
    setRunningJobs((prev) => prev.filter((j) => String(j?.id || j?.job_id || '') !== id));
    setCompletedJobs((prev) => prev.filter((j) => String(j?.id || j?.job_id || '') !== id));
    setFailedJobs((prev) => prev.filter((j) => String(j?.id || j?.job_id || '') !== id));
    try {
      await axios.delete(`${BACKEND_URL}/jobs/${id}`);
      setMessage({ type: 'success', text: `Deleted job ${id}.` });
    } catch (err) {
      setDeletedJobIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setRunningJobs(prevRunning);
      setCompletedJobs(prevCompleted);
      setFailedJobs(prevFailed);
      setMessage({ type: 'error', text: err?.response?.data?.error || `Failed to delete job ${id}.` });
    } finally {
      setDeleteConfirm(null);
    }
  };

  const stopJob = async (jobId) => {
    const id = String(jobId || '').trim();
    if (!id) return;
    try {
      await axios.post(`${BACKEND_URL}/jobs/${id}/stop`);
      await manualRefreshRunningJobs();
      setMessage({ type: 'success', text: `Stop requested for job ${id}.` });
    } catch (err) {
      setMessage({ type: 'error', text: err?.response?.data?.error || `Failed to stop job ${id}.` });
    }
  };

  const restartJob = async (jobId) => {
    const id = String(jobId || '').trim();
    if (!id) return;
    try {
      await axios.post(`${BACKEND_URL}/jobs/${id}/restart`);
      setActiveTab('running');
      setMessage({ type: 'success', text: `Job ${id} restarted.` });
    } catch (err) {
      setMessage({ type: 'error', text: err?.response?.data?.error || `Failed to restart job ${id}.` });
    }
  };

  const restartJobStep = async (jobId, stepId) => {
    const id = String(jobId || '').trim();
    const step = String(stepId || '').trim();
    if (!id || !step) return;
    try {
      await axios.post(`${BACKEND_URL}/jobs/${id}/steps/${step}/restart`);
      setActiveTab('running');
      await manualRefreshRunningJobs();
      setMessage({ type: 'success', text: `Restarted step ${step} for job ${id}.` });
    } catch (err) {
      setMessage({ type: 'error', text: err?.response?.data?.error || `Failed to restart step ${step}.` });
    }
  };

  useEffect(() => {
    if (activeTab !== 'running') return undefined;

    let cancelled = false;
    const loadRunningJobs = async (isInitial) => {
      if (isInitial) {
        setRunningJobsLoading(true);
        setRunningJobsError('');
      }
      try {
        const resp = await axios.get(`${BACKEND_URL}/jobs`, {
          params: {
            status: 'processing,preflight,queued,running,stopping,stopped',
            limit: 50,
          },
        });
        const list = Array.isArray(resp.data?.jobs) ? resp.data.jobs : [];
        const filtered = list.filter((j) => !deletedJobIds.has(String(j?.id || j?.job_id || '').trim()));
        if (!cancelled) {
          setRunningJobs(filtered);
          setRunningJobsError('');
        }
      } catch (err) {
        if (!cancelled) {
          setRunningJobsError(err?.response?.data?.error || 'Failed to load running jobs');
        }
      } finally {
        if (!cancelled && isInitial) setRunningJobsLoading(false);
      }
    };

    loadRunningJobs(true);
    const intervalId = setInterval(() => loadRunningJobs(false), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [activeTab, deletedJobIds]);

  useEffect(() => {
    if (activeTab !== 'completed') return undefined;

    let cancelled = false;
    const loadCompletedJobs = async (isInitial) => {
      if (isInitial) {
        setCompletedJobsLoading(true);
        setCompletedJobsError('');
      }
      try {
        const resp = await axios.get(`${BACKEND_URL}/jobs`, {
          params: {
            status: 'completed',
            limit: 100,
          },
        });
        const list = Array.isArray(resp.data?.jobs) ? resp.data.jobs : [];
        const filtered = list.filter((j) => !deletedJobIds.has(String(j?.id || j?.job_id || '').trim()));
        if (!cancelled) {
          setCompletedJobs(filtered);
          setCompletedJobsError('');
        }
      } catch (err) {
        if (!cancelled) {
          setCompletedJobsError(err?.response?.data?.error || 'Failed to load completed jobs');
        }
      } finally {
        if (!cancelled && isInitial) setCompletedJobsLoading(false);
      }
    };

    loadCompletedJobs(true);
    const intervalId = setInterval(() => {
      loadCompletedJobs(false);
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [activeTab, deletedJobIds]);

  useEffect(() => {
    if (activeTab !== 'failed') return undefined;

    let cancelled = false;
    const loadFailedJobs = async (isInitial) => {
      if (isInitial) {
        setFailedJobsLoading(true);
        setFailedJobsError('');
      }
      try {
        const resp = await axios.get(`${BACKEND_URL}/jobs`, {
          params: {
            status: 'failed',
            limit: 100,
          },
        });
        const list = Array.isArray(resp.data?.jobs) ? resp.data.jobs : [];
        const filtered = list.filter((j) => !deletedJobIds.has(String(j?.id || j?.job_id || '').trim()));
        if (!cancelled) {
          setFailedJobs(filtered);
          setFailedJobsError('');
        }
      } catch (err) {
        if (!cancelled) {
          setFailedJobsError(err?.response?.data?.error || 'Failed to load failed jobs');
        }
      } finally {
        if (!cancelled && isInitial) setFailedJobsLoading(false);
      }
    };

    loadFailedJobs(true);
    const intervalId = setInterval(() => loadFailedJobs(false), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [activeTab, deletedJobIds]);

  useEffect(() => {
    const showStats =
      Boolean(activeJob?.jobId) ||
      uploading ||
      (videoSource === 'local'
        ? Boolean(selectedFile)
        : Boolean(String(gcsRawVideoObject || '').trim()));
    if (!showStats) {
      setSystemStats(null);
      return undefined;
    }

    let cancelled = false;
    const fetchStats = async () => {
      try {
        const sr = await axios.get(`${BACKEND_URL}/system/stats`);
        if (!cancelled && sr?.data?.status === 'ok') {
          setSystemStats(sr.data);
        }
      } catch (err) {
        // Ignore stats errors; keep last value.
      }
    };

    fetchStats();
    const intervalId = setInterval(fetchStats, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [activeJob?.jobId, uploading]);

  const togglePlayerFullscreen = async () => {
    const el = playerContainerRef.current;
    if (!el) return;
    const isFs = Boolean(document.fullscreenElement || document.webkitFullscreenElement);
    try {
      if (isFs) {
        if (document.exitFullscreen) await document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
      } else {
        if (el.requestFullscreen) await el.requestFullscreen();
        else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 520px)');
    const onChange = () => setIsSmallScreen(Boolean(mq.matches));
    onChange();
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    else mq.addListener(onChange);
    return () => {
      if (mq.removeEventListener) mq.removeEventListener('change', onChange);
      else mq.removeListener(onChange);
    };
  }, []);

  const inferredTitle = useMemo(() => {
    if (videoSource === 'gcs') {
      const raw = String(gcsRawVideoObject || '').trim();
      if (!raw) return '';
      const key = raw.toLowerCase().startsWith('gs://') ? raw.split('/').slice(3).join('/') : raw;
      const base = String(key).split('/').pop() || String(key);
      return base.replace(/\.[^/.]+$/, '') || String(key);
    }
    if (selectedFile?.name) {
      return String(selectedFile.name).replace(/\.[^/.]+$/, '');
    }
    return '';
  }, [videoSource, gcsRawVideoObject, selectedFile]);

  const uploadPhaseLabel = useMemo(() => {
    if (!uploading) return 'Upload & Analyze';
    if (uploadJob?.message) return 'Processing…';
    return 'Working…';
  }, [uploading, uploadJob?.message]);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => {
    return () => stopPolling();
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadLanguages = async () => {
      setTranslateLanguagesLoading(true);
      setTranslateLanguagesError('');
      try {
        const [intlResp, indicResp] = await Promise.all([
          axios.get(`${BACKEND_URL}/translate/languages`, { params: { provider: 'libretranslate' } }),
          axios.get(`${BACKEND_URL}/translate/languages`, { params: { provider: 'indictrans2' } }),
        ]);
        if (cancelled) return;
        const normalize = (payload) => {
          const langs = Array.isArray(payload?.languages) ? payload.languages : [];
          return langs
            .map((lang) => {
              const code = String(lang?.code || '').trim();
              const name = String(lang?.name || '').trim();
              if (!code) return null;
              return { code, name: name || code };
            })
            .filter(Boolean);
        };
        const intlList = normalize(intlResp?.data);
        const indicList = normalize(indicResp?.data);
        const indicCodes = new Set(indicList.map((lang) => String(lang.code).toLowerCase()));
        const indicNames = new Set(indicList.map((lang) => String(lang.name).toLowerCase()).filter(Boolean));
        const commonCodes = new Set(['en', 'hi', 'bn', 'ur']);
        const commonNames = new Set(['hindi']);
        const isIndicMatch = (code, name) => {
          if (indicCodes.has(code)) return true;
          for (const indicCode of indicCodes) {
            if (!indicCode) continue;
            if (code === indicCode || code.startsWith(`${indicCode}-`) || code.startsWith(`${indicCode}_`)) {
              return true;
            }
          }
          if (indicNames.has(name)) return true;
          return false;
        };
        const filteredIntl = intlList.filter((lang) => {
          const code = String(lang.code).toLowerCase();
          const name = String(lang.name || '').toLowerCase();
          if (commonCodes.has(code)) return false;
          for (const commonCode of commonCodes) {
            if (code.startsWith(`${commonCode}-`) || code.startsWith(`${commonCode}_`)) return false;
          }
          if (commonNames.has(name) || name.includes('hindi')) return false;
          return !isIndicMatch(code, name);
        });
        setInternationalLanguageOptions(filteredIntl);
        setIndianLanguageOptions(indicList);
      } catch (e) {
        if (!cancelled) {
          setTranslateLanguagesError('Failed to load translation languages.');
          setInternationalLanguageOptions([]);
          setIndianLanguageOptions([]);
        }
      } finally {
        if (!cancelled) setTranslateLanguagesLoading(false);
      }
    };
    loadLanguages();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!selectedVideoId) {
        setSelectedMeta(null);
        setSelectedMetaError(null);
        setSelectedMetaLoading(false);
        return;
      }
      setSelectedMetaLoading(true);
      setSelectedMetaError(null);
      try {
        const resp = await axios.get(`${BACKEND_URL}/video/${selectedVideoId}/metadata-json`);
        if (cancelled) return;
        setSelectedMeta(resp.data || {});
      } catch (e) {
        if (cancelled) return;
        setSelectedMeta(null);
        setSelectedMetaError(e.response?.data?.error || 'Failed to load metadata JSON');
      } finally {
        if (!cancelled) setSelectedMetaLoading(false);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [selectedVideoId]);

  const seekTo = (seconds) => {
    const v = playerRef.current;
    if (!v) return;
    const t = Number(seconds);
    if (!Number.isFinite(t) || t < 0) return;
    try {
      v.currentTime = t;
      v.play?.();
    } catch {
      // ignore
    }
  };

  const selectedCategories = useMemo(() => selectedMeta?.categories || {}, [selectedMeta]);
  const celebs = useMemo(
    () => selectedCategories?.celebrity_table?.celebrities || selectedCategories?.celebrity_detection?.celebrities || [],
    [selectedCategories]
  );
  const detectedContent = useMemo(() => selectedCategories?.detected_content || {}, [selectedCategories]);
  const locations = useMemo(() => selectedCategories?.famous_locations || {}, [selectedCategories]);
  const selectedJobForDownloads = useMemo(() => {
    if (!selectedVideoId) return null;
    const id = String(selectedVideoId || '').trim();
    if (!id) return null;
    return (
      completedJobs.find((j) => String(j?.id || j?.job_id || '').trim() === id) ||
      runningJobs.find((j) => String(j?.id || j?.job_id || '').trim() === id) ||
      failedJobs.find((j) => String(j?.id || j?.job_id || '').trim() === id) ||
      null
    );
  }, [selectedVideoId, completedJobs, runningJobs, failedJobs]);
  const selectedGcsBaseUri = useMemo(
    () => gcsBaseUriFromArtifacts(selectedJobForDownloads?.gcs_artifacts),
    [selectedJobForDownloads]
  );
  const selectedGcsConsoleUrl = useMemo(
    () => gcsConsoleUrlFromUri(selectedGcsBaseUri),
    [selectedGcsBaseUri]
  );
  const outputTaskSelection = useMemo(
    () => selectedMeta?.task_selection_effective || selectedMeta?.task_selection || selectedMeta?.task_selection_requested || null,
    [selectedMeta]
  );
  const isOutputTaskEnabled = (key) => {
    const selection = outputTaskSelection || null;
    if (selection && Object.prototype.hasOwnProperty.call(selection, key)) return Boolean(selection[key]);
    return true;
  };
  const technical = useMemo(() => {
    const rawTech = selectedCategories?.technical_metadata || selectedCategories?.technical || {};
    const raw = rawTech?.raw && typeof rawTech.raw === 'object' ? rawTech.raw : null;

    if (!raw || rawTech.container_format || rawTech.duration_seconds || rawTech.resolution) {
      return rawTech || {};
    }

    const format = raw?.format || {};
    const streams = Array.isArray(raw?.streams) ? raw.streams : [];
    const videoStream = streams.find((s) => s && s.codec_type === 'video') || {};
    const audioStream = streams.find((s) => s && s.codec_type === 'audio') || {};

    const parseFloatSafe = (v) => {
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    };

    const parseRatio = (v) => {
      if (!v || typeof v !== 'string') return null;
      if (!v.includes('/')) return parseFloatSafe(v);
      const [num, den] = v.split('/', 2).map((p) => Number(p));
      if (!Number.isFinite(num) || !Number.isFinite(den) || den === 0) return null;
      return num / den;
    };

    const derived = {
      ...rawTech,
      container_format: format?.format_name || rawTech.container_format,
      duration_seconds: parseFloatSafe(format?.duration) ?? rawTech.duration_seconds,
      file_size_bytes: parseFloatSafe(format?.size) ?? rawTech.file_size_bytes,
      resolution: {
        width: Number(videoStream?.width) || rawTech?.resolution?.width,
        height: Number(videoStream?.height) || rawTech?.resolution?.height,
      },
      video_codec: videoStream?.codec_name || rawTech.video_codec,
      audio_codec: audioStream?.codec_name || rawTech.audio_codec,
      frame_rate: parseRatio(videoStream?.avg_frame_rate) || parseRatio(videoStream?.r_frame_rate) || rawTech.frame_rate,
    };

    return derived;
  }, [selectedCategories]);
  const synopsisPayload = useMemo(() => (selectedCategories?.synopsis && typeof selectedCategories.synopsis === 'object' ? selectedCategories.synopsis : null), [selectedCategories]);
  const synopsisStep = useMemo(() => {
    const steps = Array.isArray(uploadJob?.steps) ? uploadJob.steps : [];
    return steps.find((s) => String(s?.id || '').toLowerCase() === 'synopsis_generation') || null;
  }, [uploadJob]);
  const sceneByScene = useMemo(() => {
    const raw = selectedCategories?.scene_by_scene_metadata;
    if (Array.isArray(raw)) return raw;
    if (raw && typeof raw === 'object' && Array.isArray(raw.scenes)) return raw.scenes;
    return [];
  }, [selectedCategories]);
  const keyScenes = useMemo(() => selectedCategories?.key_scenes || [], [selectedCategories]);
  const highPoints = useMemo(() => selectedCategories?.high_points || [], [selectedCategories]);

  const transcriptCategory = useMemo(() => (selectedCategories?.transcript && typeof selectedCategories.transcript === 'object' ? selectedCategories.transcript : {}), [selectedCategories]);
  const transcriptRawText = useMemo(() => {
    const raw = (transcriptCategory?.raw_text || '').toString();
    return raw;
  }, [transcriptCategory]);
  const transcriptRawSegments = useMemo(() => {
    const rawSegments = Array.isArray(transcriptCategory?.raw_segments) ? transcriptCategory.raw_segments : [];
    return rawSegments;
  }, [transcriptCategory]);
  const transcriptText = useMemo(() => {
    const fromCategory = (transcriptCategory?.text || '').toString();
    const fromLegacy = (selectedMeta?.transcript || '').toString();
    return (fromCategory || fromLegacy || '').toString();
  }, [transcriptCategory, selectedMeta]);
  const transcriptSegments = useMemo(() => {
    const fromCategory = Array.isArray(transcriptCategory?.segments) ? transcriptCategory.segments : [];
    const fromLegacy = Array.isArray(selectedMeta?.transcript_segments) ? selectedMeta.transcript_segments : [];
    return fromCategory.length ? fromCategory : fromLegacy;
  }, [transcriptCategory, selectedMeta]);
  const [locationSourceFilter, setLocationSourceFilter] = useState('all'); // all | transcript | landmarks

  const overlayCelebItems = useMemo(() => {
    if (!Array.isArray(celebs) || !celebs.length) return [];
    const t = Number(playerTime) || 0;
    const tMs = t * 1000;
    // Linger so the overlay doesn't flicker right at segment boundaries.
    const lingerMs = 1000;
    const lingerS = 1;
    const items = [];
    for (const c of celebs) {
      if (!c || typeof c !== 'object') continue;
      const nm = String(c.name || '').trim();
      if (!nm) continue;
      const segsMs = Array.isArray(c.segments_ms) ? c.segments_ms : [];
      const segs = Array.isArray(c.segments) ? c.segments : [];

      const hit = (segsMs.length ? segsMs : segs).some((s) => {
        // Prefer ms segments when present (supports sub-second/frame timecodes)
        if (segsMs.length) {
          const a = Number(s?.start_ms);
          const b = Number(s?.end_ms);
          if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
          return tMs >= a && tMs <= b + lingerMs;
        }

        const a = Number(s?.start_seconds);
        const b = Number(s?.end_seconds);
        if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
        return t >= a && t <= b + lingerS;
      });
      if (!hit) continue;

      const portraitUrl = String(c.portrait_url || '').trim();
      items.push({ name: nm, portrait_url: portraitUrl });
      if (items.length >= 20) break;
    }
    return items;
  }, [celebs, playerTime]);

  const showCelebOverlay = playerIsPaused && overlayCelebItems.length > 0;

  const overlayScrollThreshold = playerIsFullscreen ? 4 : isSmallScreen ? 2 : 6;
  const overlayIsScrollable = overlayCelebItems.length > overlayScrollThreshold;
  const fullscreenSideScrollable = overlayCelebItems.length > (isSmallScreen ? 2 : 4);

  const playerFps = useMemo(() => {
    const tech = technical || {};
    const candidates = [
      tech.frame_rate,
      tech?.ffprobe?.frame_rate,
      tech?.mediainfo?.frame_rate,
      tech?.verification?.frame_rate,
      tech?.verification?.frame_rate?.ffprobe,
      tech?.verification?.frame_rate?.mediainfo,
    ];

    let fps = null;
    for (const v of candidates) {
      if (v == null) continue;
      if (typeof v === 'number') {
        fps = v;
        break;
      }
      if (typeof v === 'string') {
        const n = Number(v);
        if (Number.isFinite(n)) {
          fps = n;
          break;
        }
      }
      if (typeof v === 'object') {
        const n = Number(v.ffprobe ?? v.mediainfo);
        if (Number.isFinite(n)) {
          fps = n;
          break;
        }
      }
    }

    const n = Number(fps);
    if (!Number.isFinite(n) || n <= 0) return 30;
    // avoid extreme/unrealistic values
    if (n > 240) return 240;
    return n;
  }, [technical]);


  const loadGcsBucketList = async () => {
    setGcsBucketLoading(true);
    try {
      const resp = await axios.get(`${BACKEND_URL}/gcs/buckets/list`);
      const buckets = Array.isArray(resp.data?.buckets) ? resp.data.buckets : [];
      setGcsBuckets(buckets);
      if (!gcsBucket) {
        const preferred = resp.data?.default_bucket || buckets?.[0]?.name || '';
        if (preferred) setGcsBucket(preferred);
      }
    } catch (e) {
      setGcsBuckets([]);
    } finally {
      setGcsBucketLoading(false);
    }
  };

  const loadRawVideoList = async (bucketOverride, prefixOverride) => {
    setGcsRawVideoLoading(true);
    try {
      const bucket = bucketOverride || gcsBucket;
      const prefix = typeof prefixOverride === 'string' ? prefixOverride : gcsBrowserPrefix;
      const resp = await axios.get(`${BACKEND_URL}/gcs/objects/list`, {
        params: { max_results: 500, ...(bucket ? { bucket } : {}), prefix: prefix ?? '' },
      });
      setGcsBrowserObjects(Array.isArray(resp.data?.objects) ? resp.data.objects : []);
      setGcsBrowserPrefixes(Array.isArray(resp.data?.prefixes) ? resp.data.prefixes : []);
      if (!gcsBucket && resp.data?.bucket) setGcsBucket(resp.data.bucket);
    } catch (e) {
      setGcsBrowserObjects([]);
      setGcsBrowserPrefixes([]);
    } finally {
      setGcsRawVideoLoading(false);
    }
  };

  const openGcsBrowser = async () => {
    if (!gcsBucket) {
      setMessage({ type: 'error', text: 'Select a bucket first.' });
      return;
    }
    setGcsBrowserOpen(true);
    setGcsBrowserQuery('');
    await loadRawVideoList(gcsBucket, gcsBrowserPrefix);
  };

  const toParentPrefix = (prefix) => {
    const trimmed = String(prefix || '').replace(/\/+$/, '');
    if (!trimmed) return '';
    const parts = trimmed.split('/').filter(Boolean);
    parts.pop();
    return parts.length ? `${parts.join('/')}/` : '';
  };

  const isVideoFile = (name) => /\.(mp4|mov|m4v|mkv|avi|webm|mxf)$/i.test(String(name || ''));

  useEffect(() => {
    if (videoSource === 'gcs') {
      loadGcsBucketList();
    }
  }, [videoSource]);

  useEffect(() => {
    if (videoSource === 'gcs' && gcsBucket && gcsBrowserOpen) {
      loadRawVideoList(gcsBucket, gcsBrowserPrefix);
    }
  }, [videoSource, gcsBucket, gcsBrowserPrefix, gcsBrowserOpen]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type?.startsWith('video/')) {
        setSelectedFile(file);
      } else {
        setMessage({ type: 'error', text: 'Please upload a video file' });
      }
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
  };

  const pollJob = async (jobId) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const jr = await axios.get(`${BACKEND_URL}/jobs/${jobId}`);
        const job = jr.data;
        setUploadJob(job);

        if (typeof job?.progress === 'number') {
          setUploadProgress(Math.max(0, Math.min(100, job.progress)));
        }

        if (job?.status === 'completed') {
          stopPolling();
          setUploadProgress(100);
          setMessage({ type: 'success', text: 'Video indexed successfully.' });

          setActiveJob(null);

          setSelectedFile(null);
          setGcsRawVideoObject('');
          if (fileInputRef.current) fileInputRef.current.value = '';

          setUploading(false);
          setClientUploadProgress(0);
          setUploadProgress(0);
          setUploadJob(null);
        }

        if (job?.status === 'failed') {
          stopPolling();
          setMessage({ type: 'error', text: job?.error || 'Failed to process video' });

          setActiveJob(null);
          setUploading(false);
          setClientUploadProgress(0);
          setUploadProgress(0);
        }
      } catch (e) {
        // Keep polling; backend may be briefly unavailable.
      }
    }, POLL_INTERVAL_MS);
  };

  const resetUploadForm = () => {
    setSelectedFile(null);
    setGcsRawVideoObject('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleUpload = async () => {
    setMessage(null);

    if (!Array.isArray(targetTranslateLanguages) || targetTranslateLanguages.length === 0) {
      setMessage({ type: 'error', text: 'Select at least one target translation language before analyzing.' });
      return;
    }

    if (videoSource === 'gcs') {
      const raw = String(gcsRawVideoObject || '').trim();
      if (!raw) {
        setMessage({ type: 'error', text: 'Select a Cloud Storage object from the bucket.' });
        return;
      }

      setUploading(true);
      setUploadProgress(0);
      setClientUploadProgress(0);
      setUploadJob(null);
      setMessage({ type: 'info', text: 'Starting Cloud Storage video analysis…' });

      let gcs_object = null;
      let gcs_uri = null;
      if (raw.toLowerCase().startsWith('gs://')) {
        gcs_uri = raw;
      } else {
        let key = raw.replace(/^\//, '');
        gcs_object = key;
      }

      const payload = {
        title: String(inferredTitle || '').trim() || undefined,
        ...(gcs_uri ? { gcs_uri } : { gcs_object }),
        task_selection: taskSelectionPayload,
      };

      try {
        const response = await axios.post(`${BACKEND_URL}/process-gcs-video-cloud`, payload);
        if (response.status === 202 && response.data?.job_id) {
          setUploadProgress(5);
          setJobSubmitModal({ jobId: String(response.data.job_id) });
          resetUploadForm();
          setActiveJob(null);
          setUploading(false);
          setClientUploadProgress(0);
          setUploadProgress(0);
          setUploadJob(null);
          setActiveTab('running');
          return;
        }
        setMessage({ type: 'error', text: response.data?.error || 'Failed to start Cloud Storage processing' });
      } catch (e) {
        setMessage({ type: 'error', text: e.response?.data?.error || 'Failed to start Cloud Storage processing' });
      } finally {
        setUploading(false);
        setActiveJob(null);
      }

      return;
    }

    if (!selectedFile) {
      fileInputRef.current?.click?.();
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setClientUploadProgress(0);
    setUploadJob(null);
    setMessage({ type: 'info', text: 'Uploading video…' });

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('title', inferredTitle || selectedFile.name);
    formData.append('task_selection', JSON.stringify(taskSelectionPayload));

    try {
      const response = await axios.post(`${BACKEND_URL}/upload-video`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const total = Number(progressEvent.total) || 0;
          const loaded = Number(progressEvent.loaded) || 0;
          if (total > 0) {
            const pct = Math.round((loaded * 100) / total);
            setClientUploadProgress(Math.max(0, Math.min(100, pct)));
            const percentCompleted = Math.round((loaded * 50) / total);
            setUploadProgress(Math.max(0, Math.min(100, percentCompleted)));
          }
        },
      });

      if (response.status === 202 && response.data?.job_id) {
        setClientUploadProgress(100);
        setUploadProgress((p) => Math.max(p, 55));
        setJobSubmitModal({ jobId: String(response.data.job_id) });
        resetUploadForm();
        setActiveJob(null);
        setUploading(false);
        setClientUploadProgress(0);
        setUploadProgress(0);
        setUploadJob(null);
        setActiveTab('running');
        return;
      }

      setMessage({ type: 'success', text: response.data?.message || 'Uploaded.' });
    } catch (e) {
      setMessage({ type: 'error', text: e.response?.data?.error || 'Failed to upload video' });
    } finally {
      setUploading(false);
      setClientUploadProgress(0);
      setUploadProgress(0);
      setUploadJob(null);
      setActiveJob(null);
    }
  };

  const taskSelectionOptions = useMemo(
    () => [
      {
        enableKey: 'enable_label_detection',
        label: 'Label detection',
        modelKey: 'label_detection_model',
        models: [
          { value: 'gcp_video_intelligence', label: 'Google Video Intelligence' },
        ],
      },
      {
        enableKey: 'enable_famous_locations',
        label: 'Famous location detection (future)',
        disabled: true,
        modelKey: 'famous_location_detection_model',
        models: [
          { value: 'auto', label: 'Auto (backend default)' },
          { value: 'gcp_language', label: 'Google Cloud Natural Language' },
        ],
      },
      {
        enableKey: 'enable_opening_closing',
        label: 'Opening and closing credit (future)',
        disabled: true,
        modelKey: 'opening_closing_credit_detection_model',
        models: [
          { value: 'auto', label: 'Auto (backend default)' },
          { value: 'ffmpeg_blackdetect', label: 'FFmpeg blackdetect' },
          { value: 'pyscenedetect', label: 'PySceneDetect' },
        ],
      },
      {
        enableKey: 'enable_celebrity_detection',
        label: 'Celebrity detection (future)',
        disabled: true,
        modelKey: 'celebrity_detection_model',
        models: [{ value: 'auto', label: 'Auto (backend default)' }],
      },
      {
        enableKey: 'enable_celebrity_bio_image',
        label: 'Celebrity bio & image (future)',
        disabled: true,
        modelKey: 'celebrity_bio_image_model',
        models: [{ value: 'auto', label: 'Auto (backend default)' }],
      },
    ],
    []
  );

  const stepSelection = useMemo(
    () => uploadJob?.task_selection_effective || uploadJob?.task_selection || uploadJob?.task_selection_requested || taskSelection || {},
    [uploadJob, taskSelection]
  );

  const isPipelineStepEnabled = (stepId) => {
    const id = String(stepId || '').toLowerCase();
    const selection = stepSelection || {};
    const allCoreSelected = taskSelectionOptions
      .filter((t) => !t.disabled)
      .every((t) => selection?.[t.enableKey] === true);
    if (id === 'label_detection') return Boolean(selection.enable_label_detection);
    if (id === 'moderation') return Boolean(selection.enable_moderation);
    if (id === 'text_on_screen') return Boolean(selection.enable_text_on_screen);
    if (id === 'key_scene_detection') return Boolean(selection.enable_key_scene);
    if (id === 'transcribe') return Boolean(selection.enable_transcribe);
    if (id === 'synopsis_generation') return Boolean(selection.enable_synopsis_generation);
    if (id === 'scene_by_scene_metadata') return Boolean(selection.enable_scene_by_scene);
    if (id === 'famous_location_detection') return Boolean(selection.enable_famous_locations);
    if (id === 'translate_output') return Boolean(selection.enable_translate_output);
    if (id === 'opening_closing_credit_detection') return Boolean(selection.enable_opening_closing);
    if (id === 'celebrity_detection') return Boolean(selection.enable_celebrity_detection);
    if (id === 'celebrity_bio_image') return Boolean(selection.enable_celebrity_bio_image);
    return true;
  };

  const pipelineSteps = useMemo(() => {
    const steps = Array.isArray(uploadJob?.steps) ? uploadJob.steps : [];
    if (steps.length) {
      return steps.filter((s) => {
        if (!s || typeof s !== 'object') return false;
        const id = String(s.id || '').toLowerCase();
        const label = String(s.label || '').toLowerCase();
        if (id === 'overall' || label === 'overall') return false;
        if (id === 'precheck_models' || label === 'precheck models') return false;
        if (id === 'preflight' || label.includes('preflight') || label.includes('precheck')) return false;
        return isPipelineStepEnabled(id);
      });
    }

    const hasSelectedInput =
      videoSource === 'local' ? Boolean(selectedFile) : Boolean(String(gcsRawVideoObject || '').trim());

    return hasSelectedInput ? DEFAULT_PIPELINE_STEPS.filter((step) => isPipelineStepEnabled(step?.id)) : [];
  }, [uploadJob, videoSource, selectedFile, gcsRawVideoObject, taskSelection, stepSelection]);

  const statusBadgeFor = (rawStatus) => {
    const s = String(rawStatus || '').toLowerCase();
    if (s === 'completed') return { text: 'Completed', variant: 'ok' };
    if (s === 'running' || s === 'processing') return { text: 'Started', variant: 'run' };
    if (s === 'failed') return { text: 'Failed', variant: 'bad' };
    if (s === 'skipped') return { text: 'Skipped', variant: 'warn' };
    return { text: 'Not started', variant: 'neutral' };
  };

  const tableStatusFor = (rawStatus) => {
    const s = String(rawStatus || '').toLowerCase();
    if (s === 'completed') return { text: 'Completed', tone: 'completed' };
    if (s === 'running' || s === 'processing') return { text: 'In progress', tone: 'in_progress' };
    return { text: 'Not started', tone: 'not_started' };
  };

  const buildStepMap = (steps, jobStatus) => {
    const map = new Map();
    if (!Array.isArray(steps)) return map;
    steps.forEach((step) => {
      if (!step || typeof step !== 'object') return;
      const id = String(step.id || step.step_id || '').trim();
      if (id) map.set(id, step);
    });
    return map;
  };

  const loadJobDetail = async (jobId) => {
    if (!jobId) return;
    setExpandedJobLoading(true);
    try {
      const jobResp = await axios.get(`${BACKEND_URL}/jobs/${jobId}`);
      let metadata = null;
      try {
        const metaResp = await axios.get(`${BACKEND_URL}/video/${jobId}`);
        metadata = metaResp?.data || null;
      } catch {
        metadata = null;
      }
      setExpandedJobDetail({ job: jobResp?.data || null, metadata });
    } catch (err) {
      setExpandedJobDetail({ error: err?.response?.data?.error || 'Failed to load job detail' });
    } finally {
      setExpandedJobLoading(false);
    }
  };

  const runningJobRow = (job) => {
    const jobId = String(job?.id || job?.job_id || '').trim();
    const title = String(job?.title || job?.name || '').trim();
    const status = String(job?.status || '').toLowerCase();
    const stopRequested = Boolean(job?.stop_requested);
    const stepsMap = buildStepMap(job?.steps, status);
    const displaySteps = DEFAULT_PIPELINE_STEPS;
    const isExpanded = expandedJobId === jobId;
    const reportedProgress = Number(job?.progress ?? job?.overall_progress ?? job?.percent);
    const derivedProgress = (() => {
      if (!displaySteps.length) return 0;
      const sum = displaySteps.reduce((acc, step) => {
        const id = String(step?.id || '');
        const stepObj = stepsMap.get(id);
        const stepStatus = String(stepObj?.status || '').toLowerCase();
        const computed = deriveStepPercent(stepObj);
        if (Number.isFinite(computed)) return acc + computed;
        if (['completed', 'failed', 'skipped'].includes(stepStatus)) return acc + 100;
        if (stepStatus === 'running') return acc + 5;
        return acc;
      }, 0);
      return clampPercent(Math.round(sum / displaySteps.length));
    })();
    const overallPercent = Number.isFinite(reportedProgress) ? clampPercent(reportedProgress) : derivedProgress;
    const currentStep = displaySteps.find((step) => {
      const stepObj = stepsMap.get(String(step?.id || ''));
      return String(stepObj?.status || '').toLowerCase() === 'running';
    });
    const currentStepLabel = currentStep?.label || '';

    return (
      <div key={jobId || title}>
        <RunningJobRow>
          <RunningJobMeta>
            <RunningJobId>{jobId || 'Unknown job'}</RunningJobId>
            <RunningJobTitle>{title || 'Untitled job'}</RunningJobTitle>
            <Row style={{ gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <RunningJobStatus>
                Overall Status: {status || 'running'}
                {stopRequested ? ' (stop requested)' : ''}
              </RunningJobStatus>
              <div
                style={{
                  padding: '4px 8px',
                  borderRadius: 999,
                  fontSize: 12,
                  fontWeight: 900,
                  background: 'rgba(102, 126, 234, 0.18)',
                  border: '1px solid rgba(102, 126, 234, 0.35)',
                  color: '#e6e8f2',
                }}
              >
                {overallPercent}%
              </div>
            </Row>
            <div style={{ display: 'grid', gap: 6, marginTop: 4 }}>
              <ProgressBar style={{ height: 6, marginTop: 0 }}>
                <ProgressFill $percent={overallPercent} />
              </ProgressBar>
              {currentStepLabel ? (
                <div style={{ fontSize: 11, color: 'rgba(230, 232, 242, 0.65)' }}>Current Task: {currentStepLabel}</div>
              ) : null}
            </div>
          </RunningJobMeta>
          {(() => {
            const buckets = {
              not_started: [],
              in_progress: [],
              completed: [],
              skipped: [],
            };
            displaySteps.forEach((step) => {
              const id = String(step?.id || '');
              const label = String(step?.label || id || 'Step');
              const stepObj = stepsMap.get(id);
              const statusRaw = String(stepObj?.status || '').toLowerCase();
              const statusInfo = tableStatusFor(stepObj?.status);
              const percentRaw = Number(stepObj?.percent);
              const percent = Number.isFinite(percentRaw) ? Math.max(0, Math.min(100, percentRaw)) : 0;
              if (statusRaw === 'failed' || statusRaw === 'skipped') buckets.skipped.push({ id, label, percent });
              else if (statusInfo.tone === 'completed') buckets.completed.push({ id, label, percent });
              else if (statusInfo.tone === 'in_progress') buckets.in_progress.push({ id, label, percent });
              else buckets.not_started.push({ id, label, percent });
            });

            return (
              <StatusBucketsGrid>
                <StatusBucket>
                  <StatusBucketTitle>Not Started</StatusBucketTitle>
                  <RunningJobSteps>
                    {buckets.not_started.length ? (
                      buckets.not_started.map((item) => (
                        <StepChip
                          key={`${jobId}-ns-${item.id}`}
                          $status="not_started"
                          $restartable
                          title={`Restart ${item.label}`}
                          onClick={() => restartJobStep(jobId, item.id)}
                        >
                          {item.label}
                          <span style={{ opacity: 0.85 }}>{item.percent}%</span>
                        </StepChip>
                      ))
                    ) : (
                      <SubtleNote>None</SubtleNote>
                    )}
                  </RunningJobSteps>
                </StatusBucket>

                <StatusBucket>
                  <StatusBucketTitle>In Progress</StatusBucketTitle>
                  <RunningJobSteps>
                    {buckets.in_progress.length ? (
                      buckets.in_progress.map((item) => (
                        <StepChip
                          key={`${jobId}-ip-${item.id}`}
                          $status="running"
                          $restartable
                          title={`Restart ${item.label}`}
                          onClick={() => restartJobStep(jobId, item.id)}
                        >
                          {item.label}
                          <span style={{ opacity: 0.9 }}>{item.percent}%</span>
                        </StepChip>
                      ))
                    ) : (
                      <SubtleNote>None</SubtleNote>
                    )}
                  </RunningJobSteps>
                </StatusBucket>

                <StatusBucket>
                  <StatusBucketTitle>Completed</StatusBucketTitle>
                  <RunningJobSteps>
                    {buckets.completed.length ? (
                      buckets.completed.map((item) => (
                        <StepChip
                          key={`${jobId}-c-${item.id}`}
                          $status="completed"
                          $restartable
                          title={`Restart ${item.label}`}
                          onClick={() => restartJobStep(jobId, item.id)}
                        >
                          {item.label}
                          <span style={{ opacity: 0.9 }}>{item.percent}%</span>
                        </StepChip>
                      ))
                    ) : (
                      <SubtleNote>None</SubtleNote>
                    )}
                  </RunningJobSteps>
                </StatusBucket>

                <StatusBucket>
                  <StatusBucketTitle>Skipped</StatusBucketTitle>
                  <RunningJobSteps>
                    {buckets.skipped.length ? (
                      buckets.skipped.map((item) => (
                        <StepChip
                          key={`${jobId}-s-${item.id}`}
                          $status="skipped"
                          $restartable
                          title={`Restart ${item.label}`}
                          onClick={() => restartJobStep(jobId, item.id)}
                        >
                          {item.label}
                          <span style={{ opacity: 0.8 }}>{item.percent}%</span>
                        </StepChip>
                      ))
                    ) : (
                      <SubtleNote>None</SubtleNote>
                    )}
                  </RunningJobSteps>
                </StatusBucket>
              </StatusBucketsGrid>
            );
          })()}
          <Row style={{ gap: 8, justifyContent: 'flex-end' }}>
            <SecondaryButton type="button" onClick={() => restartJob(jobId)} disabled={!jobId}>
              Restart
            </SecondaryButton>
            <SecondaryButton type="button" onClick={() => stopJob(jobId)} disabled={!jobId}>
              Stop
            </SecondaryButton>
            <DeleteButton type="button" onClick={() => requestDeleteJob(jobId)} disabled={!jobId}>
              Delete
            </DeleteButton>
            <ExpandButton
              type="button"
              onClick={async () => {
                if (isExpanded) {
                  setExpandedJobId('');
                  setExpandedJobDetail(null);
                  return;
                }
                setExpandedJobId(jobId);
                await loadJobDetail(jobId);
              }}
            >
              {isExpanded ? 'Hide details' : 'Show details'}
            </ExpandButton>
          </Row>
        </RunningJobRow>
        {isExpanded && (
          <JobDetailPanel>
            {expandedJobLoading ? 'Loading job details…' : null}
            {!expandedJobLoading && expandedJobDetail?.error ? expandedJobDetail.error : null}
            {!expandedJobLoading && !expandedJobDetail?.error ? (
              <>
                {expandedJobDetail?.job ? `Job:\n${JSON.stringify(expandedJobDetail.job, null, 2)}\n\n` : ''}
                {expandedJobDetail?.metadata
                  ? `Metadata:\n${JSON.stringify(expandedJobDetail.metadata, null, 2)}`
                  : 'Metadata: not available yet'}
              </>
            ) : null}
          </JobDetailPanel>
        )}
      </div>
    );
  };

  const completedJobRow = (job) => {
    const jobId = String(job?.id || job?.job_id || '').trim();
    const fileName = getJobFileName(job);
    const isExpanded = expandedCompletedJobId === jobId;
    const thumbSrc = asJpegDataUri(job?.thumbnail || job?.video?.thumbnail || job?.metadata?.thumbnail);
    const startedAt = job?.started_at || job?.created_at || job?.job_started_at || null;
    const completedAt = job?.completed_at || job?.job_completed_at || job?.updated_at || null;
    const durationLabel = formatDurationMs(startedAt, completedAt);
    const artifactsBaseUri = gcsBaseUriFromArtifacts(job?.gcs_artifacts);
    const artifactsConsoleUrl = gcsConsoleUrlFromUri(artifactsBaseUri);

    return (
      <div key={jobId || fileName}>
        <RunningJobRow>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(260px, 1.6fr) minmax(260px, 1fr) auto',
              gap: 16,
              alignItems: 'center',
            }}
          >
            <RunningJobMeta>
              <Row style={{ gap: 10, alignItems: 'center' }}>
                <JobThumb>{thumbSrc ? <img src={thumbSrc} alt={fileName || jobId || 'thumbnail'} /> : '🎞️'}</JobThumb>
                <div style={{ minWidth: 0 }}>
                  <RunningJobId>{jobId || 'Unknown job'}</RunningJobId>
                  <RunningJobTitle>{fileName || 'Untitled file'}</RunningJobTitle>
                  <div style={{ marginTop: 4, color: 'rgba(230, 232, 242, 0.65)', fontSize: 12 }}>
                    Completed: {formatTimestamp(completedAt)}
                  </div>
                  <div style={{ marginTop: 2, color: 'rgba(230, 232, 242, 0.65)', fontSize: 12 }}>
                    Total time: {durationLabel}
                  </div>
                </div>
              </Row>
            </RunningJobMeta>

            <div style={{ display: 'grid', justifyItems: 'center' }}>
              <CompletedDownloadTable>
                <CompletedDownloadCell>
                  <div style={{ fontSize: 11, fontWeight: 900, color: 'rgba(230, 232, 242, 0.7)', marginBottom: 6 }}>
                    Complete metadata
                  </div>
                  <RunningJobSteps>
                    <StepChip $status="completed" title="Download complete metadata zip">
                      <LinkA href={`${BACKEND_URL}/video/${jobId}/metadata-json.zip`} target="_blank" rel="noreferrer">
                        Download
                      </LinkA>
                    </StepChip>
                  </RunningJobSteps>
                </CompletedDownloadCell>
                <CompletedDownloadCell>
                  <div style={{ fontSize: 11, fontWeight: 900, color: 'rgba(230, 232, 242, 0.7)', marginBottom: 6 }}>
                    GCS location
                  </div>
                  {artifactsBaseUri ? (
                    artifactsConsoleUrl ? (
                      <LinkA href={artifactsConsoleUrl} target="_blank" rel="noreferrer">
                        {artifactsBaseUri}
                      </LinkA>
                    ) : (
                      <div style={{ color: 'rgba(230, 232, 242, 0.75)', fontSize: 12 }}>{artifactsBaseUri}</div>
                    )
                  ) : (
                    <div style={{ color: 'rgba(230, 232, 242, 0.6)', fontSize: 12 }}>Not available</div>
                  )}
                </CompletedDownloadCell>
              </CompletedDownloadTable>
            </div>

            <CompletedJobActions>
              <CompletedDetailButton
                type="button"
                onClick={() => {
                  if (!jobId) return;
                  if (isExpanded) {
                    setExpandedCompletedJobId('');
                    setSelectedVideoId(null);
                    setSelectedVideoTitle(null);
                    return;
                  }
                  setExpandedCompletedJobId(jobId);
                  setSelectedVideoId(jobId);
                  setSelectedVideoTitle(fileName || jobId);
                  setTimeout(() => {
                    document.getElementById('envid-metadata-details')?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
                  }, 0);
                }}
                disabled={!jobId}
                style={{ minWidth: 92, padding: '4px 8px', fontSize: 11 }}
              >
                {isExpanded ? 'Hide details' : 'Show details'}
              </CompletedDetailButton>
              <DeleteButton
                type="button"
                onClick={() => requestDeleteJob(jobId)}
                disabled={!jobId}
                style={{ minWidth: 92, padding: '4px 8px', fontSize: 11 }}
              >
                Delete
              </DeleteButton>
            </CompletedJobActions>
          </div>
        </RunningJobRow>
      </div>
    );
  };

  const failedJobRow = (job) => {
    const jobId = String(job?.id || job?.job_id || '').trim();
    const fileName = getJobFileName(job);
    const thumbSrc = asJpegDataUri(job?.thumbnail || job?.video?.thumbnail || job?.metadata?.thumbnail);
    const steps = Array.isArray(job?.steps) ? job.steps : [];
    const failedStep = steps.find((s) => String(s?.status || '').toLowerCase() === 'failed') || null;
    const failedStepId = String(failedStep?.id || '').toLowerCase();
    const failedStepLabel =
      DEFAULT_PIPELINE_STEPS.find((s) => String(s?.id || '').toLowerCase() === failedStepId)?.label ||
      failedStep?.label ||
      failedStepId ||
      '';
    const stepReason = String(failedStep?.message || failedStep?.error || '').trim();
    const remarks =
      String(
        job?.failure_reason ||
          job?.error_message ||
          job?.error ||
          job?.status_message ||
          job?.message ||
          job?.last_error ||
          ''
      ).trim() || 'Unknown failure';
    const remarksText = stepReason ? `${failedStepLabel ? `${failedStepLabel}: ` : ''}${stepReason}` : remarks;
    const failedAtText = failedStepLabel ? `Failed at ${failedStepLabel}` : 'Failed step unknown';

    return (
      <div key={jobId || fileName}>
        <RunningJobRow>
          <RunningJobMeta>
            <Row style={{ gap: 10, alignItems: 'center' }}>
              <JobThumb>{thumbSrc ? <img src={thumbSrc} alt={fileName || jobId || 'thumbnail'} /> : '⚠️'}</JobThumb>
              <div style={{ minWidth: 0 }}>
                <RunningJobId>{jobId || 'Unknown job'}</RunningJobId>
                <RunningJobTitle>{fileName || 'Untitled file'}</RunningJobTitle>
                <div style={{ marginTop: 6, fontSize: 12, color: 'rgba(255, 187, 187, 0.9)' }}>{failedAtText}</div>
                <div style={{ marginTop: 4, fontSize: 12, color: 'rgba(255, 187, 187, 0.9)' }}>
                  Remarks: {remarksText}
                </div>
              </div>
            </Row>
          </RunningJobMeta>
          <Row style={{ gap: 8, justifyContent: 'flex-end' }}>
            <SecondaryButton type="button" onClick={() => restartJob(jobId)} disabled={!jobId}>
              Restart
            </SecondaryButton>
            <DeleteButton type="button" onClick={() => requestDeleteJob(jobId)} disabled={!jobId}>
              Delete
            </DeleteButton>
          </Row>
        </RunningJobRow>
      </div>
    );
  };

  const clampPercent = (value) => Math.max(0, Math.min(100, Math.round(value)));

  const extractPercentFromMessage = (message) => {
    if (!message) return null;
    const match = String(message).match(/(\d{1,3})\s*%/);
    if (!match) return null;
    const parsed = Number(match[1]);
    if (!Number.isFinite(parsed)) return null;
    return clampPercent(parsed);
  };

  const deriveStepPercent = (step) => {
    if (!step || typeof step !== 'object') return null;
    const stepId = String(step.id || '').toLowerCase();
    const status = String(step.status || '').toLowerCase();
    const raw = typeof step.percent === 'number' ? clampPercent(step.percent) : null;
    if (raw !== null && raw > 0) return raw;
    const msgPercent = extractPercentFromMessage(step.message);
    if (msgPercent !== null) return msgPercent;
    if (stepId === 'upload_to_cloud_storage') {
      const msg = String(step.message || '').toLowerCase();
      if (msg.includes('uploaded') || msg.includes('downloaded')) return 100;
      return raw !== null ? raw : 0;
    }
    if (status === 'completed' || status === 'failed' || status === 'skipped') return 100;
    if (status !== 'running') return raw;
    const startedAt = step.started_at || step.startedAt;
    if (startedAt) {
      const startedMs = Date.parse(startedAt);
      if (Number.isFinite(startedMs)) {
        const elapsedSeconds = Math.max(0, (Date.now() - startedMs) / 1000);
        return clampPercent(Math.min(90, Math.max(2, elapsedSeconds * 1.2)));
      }
    }
    return raw !== null ? raw : 0;
  };

  const hasSelectedInput =
    videoSource === 'local' ? Boolean(selectedFile) : Boolean(String(gcsRawVideoObject || '').trim());

  const showStatusPanel = Boolean(activeJob?.jobId) || uploading || hasSelectedInput;

  const cpuPercentLabel = Number.isFinite(systemStats?.cpu_percent)
    ? `${Math.round(systemStats.cpu_percent)}%`
    : '—';
  const gpuPercentLabel = Number.isFinite(systemStats?.gpu_percent)
    ? `${Math.round(systemStats.gpu_percent)}%`
    : '—';
  const vramLabel = Number.isFinite(systemStats?.gpu_memory_percent)
    ? `${Math.round(systemStats.gpu_memory_percent)}%`
    : '—';
  const memPercentLabel = Number.isFinite(systemStats?.memory_percent)
    ? `${Math.round(systemStats.memory_percent)}%`
    : '—';
  const ramLabel = memPercentLabel;

  const combinedTranslateLanguageOptions = useMemo(() => {
    const map = new Map();
    internationalLanguageOptions.forEach((lang) => {
      if (lang?.code) map.set(String(lang.code).toLowerCase(), lang);
    });
    indianLanguageOptions.forEach((lang) => {
      if (lang?.code) map.set(String(lang.code).toLowerCase(), lang);
    });
    return Array.from(map.values());
  }, [internationalLanguageOptions, indianLanguageOptions]);

  const serverUploadStatus = (() => {
    if (videoSource !== 'local') return null;
    if (!hasSelectedInput) return null;
    if (!uploading) return { status: 'not_started', percent: 0 };
    if (clientUploadProgress >= 100) return { status: 'completed', percent: 100 };
    if (clientUploadProgress > 0) return { status: 'running', percent: clientUploadProgress };
    return { status: 'running', percent: 0 };
  })();

  const taskSelectionAllEnabled = useMemo(
    () => taskSelectionOptions.filter((t) => !t.disabled).every((t) => Boolean(taskSelection?.[t.enableKey])),
    [taskSelectionOptions, taskSelection]
  );

  const pipelineStepLabelMap = useMemo(() => {
    const map = {};
    DEFAULT_PIPELINE_STEPS.forEach((step) => {
      if (step?.id && step?.label) map[String(step.id).toLowerCase()] = String(step.label);
    });
    return map;
  }, []);

  const statusPanel = showStatusPanel ? (
    <div style={{ marginTop: 12 }}>
      <StatusPanel>
        <StatusTitleRow>
          <StatusTitle>Processing Status</StatusTitle>
          <StatusMetaRight>
            <StatusMeta>
              {(() => {
                const backendStatus = uploadJob?.status || uploadJob?.processing_status || uploadJob?.state;
                const label = backendStatus ? String(backendStatus) : uploading ? 'processing' : '—';
                return <div>Processing: {label}</div>;
              })()}
              <div>{activeJob?.kind ? `${activeJob.kind} job` : 'job'}</div>
              {activeJob?.jobId ? <div>Job ID: {String(activeJob.jobId)}</div> : null}
            </StatusMeta>
          </StatusMetaRight>
        </StatusTitleRow>

        {uploadJob?.message ? (
          <div style={{ color: 'rgba(230, 232, 242, 0.85)', fontSize: 12, marginBottom: 10 }}>{uploadJob.message}</div>
        ) : hasSelectedInput ? (
          <div style={{ color: 'rgba(230, 232, 242, 0.75)', fontSize: 12, marginBottom: 10 }}>
            Ready. Press “Upload & Analyze” to start.
          </div>
        ) : null}


        {serverUploadStatus && (
          <div style={{ marginBottom: 12 }}>
            <StepRow>
              <StepLeft>
                <StepLabelRow>
                  <StepLabel>Upload to server</StepLabel>
                  <StepHint>{Math.max(0, Math.min(100, Math.round(serverUploadStatus.percent)))}%</StepHint>
                </StepLabelRow>
                <ProgressBar>
                  <ProgressFill $percent={serverUploadStatus.percent} />
                </ProgressBar>
              </StepLeft>
              <StepBadge $variant={statusBadgeFor(serverUploadStatus.status).variant}>
                {statusBadgeFor(serverUploadStatus.status).text}
              </StepBadge>
            </StepRow>
          </div>
        )}

        {pipelineSteps.length > 0 && (
          <StepList>
            {pipelineSteps.length > 0 && (
              <StatusTable>
                <StatusTableHeader>
                  <div>Task</div>
                  <div>Status</div>
                </StatusTableHeader>
                {pipelineSteps.map((step) => {
                  const id = String(step?.id || '');
                  const label = String(step?.label || id || 'Step');
                  const pct = deriveStepPercent(step);
                  const hint = String(step?.message || '').trim();
                  const hintText = (() => {
                    if (pct !== null && pct > 0) return `${pct}%`;
                    if (hint) return hint;
                    if (pct !== null) return `${pct}%`;
                    return '';
                  })();
                  const statusInfo = tableStatusFor(step?.status);

                  return (
                    <StatusTableRow key={id || label}>
                      <StatusTableCell>
                        <strong style={{ fontWeight: 700 }}>{label}</strong>
                        {hintText ? <span style={{ color: 'rgba(180, 186, 220, 0.8)', fontSize: 12 }}>{hintText}</span> : null}
                      </StatusTableCell>
                      <StatusTableCell>
                        <StatusPill $tone={statusInfo.tone}>{statusInfo.text}</StatusPill>
                      </StatusTableCell>
                    </StatusTableRow>
                  );
                })}
              </StatusTable>
            )}
          </StepList>
        )}

        {typeof uploadProgress === 'number' && (
          <div style={{ marginTop: 12 }}>
            <StepRow>
              <StepLeft>
                <StepLabelRow>
                  <StepLabel>Overall</StepLabel>
                  <StepHint>{Math.max(0, Math.min(100, Math.round(uploadProgress)))}%</StepHint>
                </StepLabelRow>
                <ProgressBar>
                  <ProgressFill $percent={Math.max(0, Math.min(100, uploadProgress))} />
                </ProgressBar>
              </StepLeft>
              <StepBadge
                $variant={
                  String(uploadJob?.status || '').toLowerCase() === 'completed'
                    ? 'ok'
                    : String(uploadJob?.status || '').toLowerCase() === 'failed'
                      ? 'bad'
                      : 'run'
                }
              >
                {String(uploadJob?.status || 'processing')}
              </StepBadge>
            </StepRow>
          </div>
        )}
      </StatusPanel>
    </div>
  ) : null;

  const copyToClipboard = async (text) => {
    const value = String(text || '').trim();
    if (!value) return false;

    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
        return true;
      }
    } catch {
      // fall through to legacy copy
    }

    try {
      const ta = document.createElement('textarea');
      ta.value = value;
      ta.style.position = 'fixed';
      ta.style.top = '-1000px';
      ta.style.left = '-1000px';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return Boolean(ok);
    } catch {
      return false;
    }
  };

  const metadataDetailPanel =
    activeTab === 'completed' && selectedVideoId ? (
      <DetailPanel id="envid-metadata-details">
        <DetailHeader>
          <div style={{ minWidth: 0 }}>
            <DetailTitle>
              {selectedVideoTitle || 'Video'} · ID: {selectedVideoId}
            </DetailTitle>
            <div style={{ marginTop: 4, color: 'rgba(230, 232, 242, 0.7)', fontSize: 12 }}>
              Click any timestamp/segment to seek.
            </div>
          </div>
          <Row style={{ gap: 8 }}>
            <SecondaryButton
              type="button"
              onClick={() => {
                const el = document.getElementById('envid-script');
                if (el && typeof el.scrollIntoView === 'function') {
                  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
              }}
              disabled={selectedMetaLoading}
            >
              Jump to Script
            </SecondaryButton>
            <SecondaryButton
              type="button"
              onClick={() => {
                setSelectedVideoId(null);
                setSelectedVideoTitle(null);
                setSelectedMeta(null);
                setSelectedMetaError(null);
                setExpandedCompletedJobId('');
              }}
            >
              Close
            </SecondaryButton>
          </Row>
        </DetailHeader>

        <DetailBody>
          {selectedMetaLoading ? (
            <div style={{ color: 'rgba(230, 232, 242, 0.75)', fontWeight: 900 }}>Loading metadata…</div>
          ) : selectedMetaError ? (
            <div style={{ color: '#ffd1d1', fontWeight: 900 }}>{selectedMetaError}</div>
          ) : (
            <>
              <Grid2>
                {isOutputTaskEnabled('enable_celebrity_detection') ? (
                  <div>
                    <Card>
                      <CardHeader>
                        <CardHeaderTitle>Celebrity Detection on Player</CardHeaderTitle>
                        <div style={{ color: 'rgba(230, 232, 242, 0.65)', fontSize: 12 }}>
                          {formatTimecode(playerTime, playerFps)}
                        </div>
                      </CardHeader>
                      <CardBody>
                        <VideoFrame ref={playerContainerRef}>
                          <video
                            ref={playerRef}
                            src={`${BACKEND_URL}/video-file/${selectedVideoId}`}
                            controls
                            controlsList="nofullscreen"
                            style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block', position: 'relative', zIndex: 1 }}
                            onTimeUpdate={(e) => setPlayerTime(Number(e.currentTarget?.currentTime || 0))}
                            onPlay={() => {
                              setPlayerIsPaused(false);
                            }}
                            onPause={() => {
                              setPlayerIsPaused(true);
                            }}
                            onEnded={() => {
                              setPlayerIsPaused(true);
                            }}
                          />

                          <PlayerFullscreenButton
                            type="button"
                            onClick={togglePlayerFullscreen}
                            title={playerIsFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
                            aria-label={playerIsFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
                          >
                            {playerIsFullscreen ? 'X' : 'FS'}
                          </PlayerFullscreenButton>

                          {showCelebOverlay ? (
                            playerIsFullscreen ? (
                              <>
                                <VideoOverlaySidePanel>
                                  <OverlaySideList $scrollable={fullscreenSideScrollable}>
                                    {overlayCelebItems.map((c) => (
                                      <OverlaySideItem key={c.name}>
                                        <OverlayThumb>
                                          {c.portrait_url ? (
                                            <img src={c.portrait_url} alt={c.name} referrerPolicy="no-referrer" />
                                          ) : (
                                            <span>{String(c.name || '?').slice(0, 1).toUpperCase()}</span>
                                          )}
                                        </OverlayThumb>
                                        <OverlayName title={c.name}>{c.name}</OverlayName>
                                      </OverlaySideItem>
                                    ))}
                                  </OverlaySideList>
                                </VideoOverlaySidePanel>
                              </>
                            ) : (
                              <VideoOverlayBar>
                                <OverlayScroller $scrollable={overlayIsScrollable}>
                                  {overlayCelebItems.map((c) => (
                                    <OverlayChip key={c.name}>
                                      <OverlayChipThumb>
                                        {c.portrait_url ? (
                                          <img src={c.portrait_url} alt={c.name} referrerPolicy="no-referrer" />
                                        ) : (
                                          <span>{String(c.name || '?').slice(0, 1).toUpperCase()}</span>
                                        )}
                                      </OverlayChipThumb>
                                      <OverlayChipName>{c.name}</OverlayChipName>
                                    </OverlayChip>
                                  ))}
                                </OverlayScroller>
                              </VideoOverlayBar>
                            )
                          ) : null}
                        </VideoFrame>

                        <div style={{ marginTop: 10, color: 'rgba(230, 232, 242, 0.7)', fontSize: 12 }}>
                          Tip: use the Celebrity table segments below to jump.
                        </div>
                      </CardBody>
                    </Card>
                  </div>
                ) : null}

                <div>
                  <Card>
                    <CardHeader>
                      <CardHeaderTitle>Technical</CardHeaderTitle>
                    </CardHeader>
                    <CardBody>
                      <div style={{ display: 'grid', gap: 6, color: 'rgba(230, 232, 242, 0.85)', fontSize: 13 }}>
                        <div>Title: {technical?.title || selectedVideoTitle || '—'}</div>
                        <div>
                          Container:{' '}
                          {(() => {
                            const c = formatContainerFormat(technical?.container_format);
                            return (
                              <span title={c.title || undefined} style={{ fontWeight: 900 }}>
                                {c.label}
                              </span>
                            );
                          })()}
                        </div>
                        <div>
                          Size: {typeof technical?.file_size_bytes === 'number' ? formatBytes(technical.file_size_bytes) : '—'}
                        </div>
                        <div>Duration: {technical?.duration_seconds != null ? `${formatSeconds(technical.duration_seconds)} (${Number(technical.duration_seconds).toFixed(2)}s)` : '—'}</div>
                        <div>
                          Resolution:{' '}
                          {technical?.resolution?.width && technical?.resolution?.height
                            ? `${technical.resolution.width}×${technical.resolution.height}`
                            : '—'}
                        </div>
                        <div>Video codec: {technical?.video_codec || '—'}</div>
                        <div>Audio codec: {technical?.audio_codec || '—'}</div>
                      </div>
                    </CardBody>
                  </Card>
                </div>
              </Grid2>

              {isOutputTaskEnabled('enable_celebrity_detection') && isOutputTaskEnabled('enable_celebrity_bio_image') ? (
                <Card>
                  <CardHeader>
                    <CardHeaderTitle>Celebrity Table (timestamp + short bio)</CardHeaderTitle>
                  </CardHeader>
                  <CardBody>
                    {Array.isArray(celebs) && celebs.length ? (
                      <div style={{ overflowX: 'auto' }}>
                        <Table>
                          <thead>
                            <tr>
                              <Th>Celebrity</Th>
                              <Th>Occurrences</Th>
                              <Th>Bio</Th>
                              <Th>Src</Th>
                              <Th>Portrait License</Th>
                            </tr>
                          </thead>
                          <tbody>
                            {celebs.map((c) => {
                              const name = String(c?.name || '').trim() || '—';
                              const occ = Number(c?.occurrences);
                              const occurrences = Number.isFinite(occ) ? Math.max(0, Math.floor(occ)) : null;
                              const tsMs = Array.isArray(c?.timestamps_ms) ? c.timestamps_ms : [];
                              const tsSeconds = Array.isArray(c?.timestamps_seconds) ? c.timestamps_seconds : [];
                              const bio = (c?.bio_short || c?.bio || '').toString().trim();
                              const portrait = (c?.portrait_url || '').toString().trim();
                              const portraitLicense = (c?.portrait_license || '').toString().trim();
                              const portraitLicenseUrl = (c?.portrait_license_url || '').toString().trim();
                              const portraitSource = (c?.portrait_source || '').toString().trim();

                              const occurrenceTimesSeconds = (() => {
                                const raw = tsMs.length ? tsMs.map((ms) => Number(ms) / 1000) : tsSeconds.map((s) => Number(s));
                                const seen = new Set();
                                const out = [];
                                for (const t of raw) {
                                  if (!Number.isFinite(t) || t < 0) continue;
                                  const k = Math.round(t * 1000);
                                  if (seen.has(k)) continue;
                                  seen.add(k);
                                  out.push(t);
                                }
                                return out;
                              })();
                              return (
                                <tr key={name}>
                                  <Td>
                                    <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                                      {portrait ? (
                                        <img
                                          src={portrait}
                                          alt={name}
                                          style={{ width: 36, height: 36, borderRadius: 999, objectFit: 'cover', border: '1px solid rgba(255,255,255,0.12)' }}
                                        />
                                      ) : (
                                        <div
                                          style={{
                                            width: 36,
                                            height: 36,
                                            borderRadius: 999,
                                            background: 'rgba(255,255,255,0.06)',
                                            border: '1px solid rgba(255,255,255,0.12)',
                                          }}
                                        />
                                      )}
                                      <div style={{ minWidth: 0 }}>
                                        <div style={{ fontWeight: 900, color: '#e6e8f2' }}>{name}</div>
                                        {c?.max_confidence != null ? (
                                          <div style={{ marginTop: 2, color: 'rgba(230,232,242,0.7)', fontSize: 12 }}>
                                            Max conf: {Number(c.max_confidence).toFixed(2)}
                                          </div>
                                        ) : null}
                                      </div>
                                    </div>
                                  </Td>
                                  <Td>
                                    <div style={{ display: 'grid', gap: 8 }}>
                                      <div style={{ color: 'rgba(230, 232, 242, 0.85)', fontSize: 12, fontWeight: 900 }}>
                                        Occurrences: {occurrences != null ? occurrences : '—'}
                                      </div>
                                      {occurrenceTimesSeconds.length ? (
                                        <div style={{ display: 'grid', gap: 6 }}>
                                          <div style={{ color: 'rgba(230, 232, 242, 0.75)', fontSize: 12, fontWeight: 800 }}>
                                            Exact times:
                                          </div>
                                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                            {occurrenceTimesSeconds.slice(0, 20).map((t, idx) => (
                                              <SegmentChip key={`${name}-occ-${idx}`} type="button" onClick={() => seekTo(t)}>
                                                {formatTimecode(t, playerFps)}
                                              </SegmentChip>
                                            ))}
                                            {occurrenceTimesSeconds.length > 20 ? (
                                              <div style={{ color: 'rgba(230,232,242,0.6)', fontSize: 12, alignSelf: 'center' }}>
                                                +{occurrenceTimesSeconds.length - 20} more
                                              </div>
                                            ) : null}
                                          </div>
                                        </div>
                                      ) : null}
                                    </div>
                                  </Td>
                                  <Td>
                                    <div style={{ color: 'rgba(230, 232, 242, 0.85)' }}>{bio || '—'}</div>
                                  </Td>
                                  <Td>
                                    {portraitSource || portrait ? (
                                      <div style={{ display: 'grid', gap: 6, fontSize: 12 }}>
                                        {portraitSource ? (
                                          <div>
                                            <LinkA href={portraitSource} target="_blank" rel="noreferrer">
                                              Source
                                            </LinkA>
                                          </div>
                                        ) : null}
                                        {portrait ? (
                                          <div>
                                            <LinkA href={portrait} target="_blank" rel="noreferrer">
                                              Image
                                            </LinkA>
                                          </div>
                                        ) : null}
                                      </div>
                                    ) : (
                                      '—'
                                    )}
                                  </Td>
                                  <Td>
                                    {portraitLicense || portraitLicenseUrl ? (
                                      <div style={{ color: 'rgba(230,232,242,0.85)', fontWeight: 900, fontSize: 12 }}>
                                        {portraitLicense ? (
                                          portraitLicenseUrl ? (
                                            <LinkA href={portraitLicenseUrl} target="_blank" rel="noreferrer">
                                              {portraitLicense}
                                            </LinkA>
                                          ) : (
                                            portraitLicense
                                          )
                                        ) : portraitLicenseUrl ? (
                                          <LinkA href={portraitLicenseUrl} target="_blank" rel="noreferrer">
                                            License
                                          </LinkA>
                                        ) : (
                                          '—'
                                        )}
                                      </div>
                                    ) : (
                                      '—'
                                    )}
                                  </Td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </Table>
                      </div>
                    ) : (
                      <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>No celebrity detections.</div>
                    )}
                  </CardBody>
                </Card>
              ) : null}

              <div style={{ display: 'grid', gap: 16 }}>
                <div style={{ display: 'grid', gap: 16 }}>
                  {isOutputTaskEnabled('enable_text_on_screen') ? (
                    <Card>
                      <CardHeader>
                        <CardHeaderTitle>On-screen Text</CardHeaderTitle>
                        <CountPill>{Array.isArray(detectedContent?.on_screen_text) ? detectedContent.on_screen_text.length : 0}</CountPill>
                      </CardHeader>
                      <CardBody>
                        <div style={{ color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                          {Array.isArray(detectedContent?.on_screen_text) && detectedContent.on_screen_text.length ? (
                            <MultiColumnList>
                              {detectedContent.on_screen_text.slice(0, 50).map((t, idx) => {
                                const text = typeof t === 'string' ? t : t?.text;
                                const segs = Array.isArray(t?.segments) ? t.segments : [];
                                return (
                                  <MultiColumnItem key={`txt-${idx}`}>
                                    <div style={{ fontWeight: 900, color: '#e6e8f2' }}>{text || '—'}</div>
                                    {t?.first_seen_seconds != null && t?.last_seen_seconds != null ? (
                                      <div style={{ color: 'rgba(230,232,242,0.7)', fontSize: 12, marginTop: 2 }}>
                                        {formatSeconds(t.first_seen_seconds)} → {formatSeconds(t.last_seen_seconds)}
                                      </div>
                                    ) : null}
                                    {segs.length ? (
                                      <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                        {segs.slice(0, 10).map((s, j) => (
                                          <SegmentChip key={`txtseg-${idx}-${j}`} type="button" onClick={() => seekTo(s?.start_seconds)}>
                                            {formatSeconds(s?.start_seconds)}–{formatSeconds(s?.end_seconds)}
                                          </SegmentChip>
                                        ))}
                                      </div>
                                    ) : null}
                                  </MultiColumnItem>
                                );
                              })}
                            </MultiColumnList>
                          ) : (
                            '—'
                          )}
                        </div>
                        {Array.isArray(detectedContent?.on_screen_text) && detectedContent.on_screen_text.length > 50 ? (
                          <SubtleNote>Showing first 50 text items.</SubtleNote>
                        ) : null}
                      </CardBody>
                    </Card>
                  ) : null}

                  {isOutputTaskEnabled('enable_label_detection') ? (
                    <Card>
                      <CardHeader>
                        <CardHeaderTitle>Detected Content</CardHeaderTitle>
                        <CountPill>{Array.isArray(detectedContent?.labels) ? detectedContent.labels.length : 0}</CountPill>
                      </CardHeader>
                      <CardBody>
                        <div style={{ color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                          {Array.isArray(detectedContent?.labels) && detectedContent.labels.length ? (
                            <MultiColumnList>
                              {detectedContent.labels.map((l, idx) => {
                                const name = typeof l === 'string' ? l : l?.name;
                                const segs = Array.isArray(l?.segments) ? l.segments : [];
                                return (
                                  <MultiColumnItem key={`lbl-${idx}`}>
                                    <div style={{ fontWeight: 900, color: '#e6e8f2' }}>{name || '—'}</div>
                                    {l?.first_seen_seconds != null && l?.last_seen_seconds != null ? (
                                      <div style={{ color: 'rgba(230,232,242,0.7)', fontSize: 12, marginTop: 2 }}>
                                        {formatSeconds(l.first_seen_seconds)} → {formatSeconds(l.last_seen_seconds)}
                                      </div>
                                    ) : null}
                                    {segs.length ? (
                                      <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                        {segs.slice(0, 10).map((s, j) => (
                                          <SegmentChip key={`lblseg-${idx}-${j}`} type="button" onClick={() => seekTo(s?.start_seconds)}>
                                            {formatSeconds(s?.start_seconds)}–{formatSeconds(s?.end_seconds)}
                                          </SegmentChip>
                                        ))}
                                      </div>
                                    ) : null}
                                  </MultiColumnItem>
                                );
                              })}
                            </MultiColumnList>
                          ) : (
                            '—'
                          )}
                        </div>
                        {Array.isArray(detectedContent?.labels) && detectedContent.labels.length > 60 ? (
                          <SubtleNote>Showing first 60 label items.</SubtleNote>
                        ) : null}
                      </CardBody>
                    </Card>
                  ) : null}

                  {isOutputTaskEnabled('enable_moderation') ? (
                    <Card>
                      <CardHeader>
                        <CardHeaderTitle>Moderation</CardHeaderTitle>
                        <CountPill>{Array.isArray(detectedContent?.moderation) ? detectedContent.moderation.length : 0}</CountPill>
                      </CardHeader>
                      <CardBody>
                        <div style={{ color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                          {Array.isArray(detectedContent?.moderation) && detectedContent.moderation.length ? (
                            <MultiColumnList>
                              {detectedContent.moderation.slice(0, 60).map((m, idx) => {
                                const name = typeof m === 'string' ? m : m?.name;
                                const segs = Array.isArray(m?.segments) ? m.segments : [];
                                return (
                                  <MultiColumnItem key={`mod-${idx}`}>
                                    <div style={{ fontWeight: 900, color: '#e6e8f2' }}>{name || '—'}</div>
                                    {m?.first_seen_seconds != null && m?.last_seen_seconds != null ? (
                                      <div style={{ color: 'rgba(230,232,242,0.7)', fontSize: 12, marginTop: 2 }}>
                                        {formatSeconds(m.first_seen_seconds)} → {formatSeconds(m.last_seen_seconds)}
                                      </div>
                                    ) : null}
                                    {segs.length ? (
                                      <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                        {segs.slice(0, 10).map((s, j) => (
                                          <SegmentChip key={`modseg-${idx}-${j}`} type="button" onClick={() => seekTo(s?.start_seconds)}>
                                            {formatSeconds(s?.start_seconds)}–{formatSeconds(s?.end_seconds)}
                                          </SegmentChip>
                                        ))}
                                      </div>
                                    ) : null}
                                  </MultiColumnItem>
                                );
                              })}
                            </MultiColumnList>
                          ) : (
                            '—'
                          )}
                        </div>
                        {Array.isArray(detectedContent?.moderation) && detectedContent.moderation.length > 60 ? (
                          <SubtleNote>Showing first 60 moderation items.</SubtleNote>
                        ) : null}
                      </CardBody>
                    </Card>
                  ) : null}
                </div>

                {isOutputTaskEnabled('enable_synopsis_generation') ? (
                  <Card>
                    <CardHeader>
                      <CardHeaderTitle>Synopsis</CardHeaderTitle>
                    </CardHeader>
                    <CardBody>
                      {synopsisPayload ? (
                        <div style={{ display: 'grid', gap: 10 }}>
                          <div
                            style={{
                              border: '1px solid rgba(255, 255, 255, 0.08)',
                              background: 'rgba(255, 255, 255, 0.03)',
                              borderRadius: 12,
                              padding: 12,
                            }}
                          >
                            <div style={{ marginTop: 6, color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                              <div style={{ fontWeight: 900, color: 'rgba(230,232,242,0.75)', marginBottom: 4 }}>Short</div>
                              <div>{(synopsisPayload?.short || '').toString().trim() || '—'}</div>
                              <div style={{ fontWeight: 900, color: 'rgba(230,232,242,0.75)', margin: '10px 0 4px' }}>Long</div>
                              <div>{(synopsisPayload?.long || '').toString().trim() || '—'}</div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>
                          {synopsisStep?.status ? (
                            <>
                              {`Status: ${statusBadgeFor(synopsisStep.status).text}`}
                              {synopsisStep?.message ? ` • ${String(synopsisStep.message).trim()}` : ''}
                            </>
                          ) : (
                            '—'
                          )}
                        </div>
                      )}
                    </CardBody>
                  </Card>
                ) : null}

                {isOutputTaskEnabled('enable_scene_by_scene') ? (
                  <Card>
                    <CardHeader>
                      <CardHeaderTitle>Scene by Scene Metadata</CardHeaderTitle>
                      <CountPill>{Array.isArray(sceneByScene?.scenes) ? sceneByScene.scenes.length : 0}</CountPill>
                    </CardHeader>
                    <CardBody>
                      {Array.isArray(sceneByScene?.scenes) && sceneByScene.scenes.length ? (
                        <div style={{ display: 'grid', gap: 10 }}>
                          {sceneByScene.scenes.slice(0, 120).map((sc, idx) => {
                            const st = sc?.start_seconds ?? sc?.start;
                            const en = sc?.end_seconds ?? sc?.end;
                            const summary = sc?.summary_text || sc?.summary_text_llm || sc?.summary_text_transcript || sc?.summary_text_combined;
                            return (
                              <div key={`scene-${idx}`} style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: 12 }}>
                                <Row style={{ justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
                                  <div style={{ fontWeight: 900, color: '#e6e8f2' }}>
                                    Scene {sc?.scene_index ?? sc?.index ?? idx + 1}
                                  </div>
                                  <SegmentChip type="button" onClick={() => seekTo(st)}>
                                    {formatSeconds(st)}–{formatSeconds(en)}
                                  </SegmentChip>
                                </Row>
                                {summary ? (
                                  <div style={{ marginTop: 8, color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                                    {String(summary).trim()}
                                  </div>
                                ) : null}
                              </div>
                            );
                          })}
                          {sceneByScene.scenes.length > 120 ? <SubtleNote>Showing first 120 scenes.</SubtleNote> : null}
                        </div>
                      ) : (
                        <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>—</div>
                      )}
                    </CardBody>
                  </Card>
                ) : null}

                {isOutputTaskEnabled('enable_key_scene') ? (
                  <Card>
                    <CardHeader>
                      <CardHeaderTitle>Key Scenes</CardHeaderTitle>
                      <CountPill>{Array.isArray(keyScenes) ? keyScenes.length : 0}</CountPill>
                    </CardHeader>
                    <CardBody>
                      {Array.isArray(keyScenes) && keyScenes.length ? (
                        <div style={{ display: 'grid', gap: 10 }}>
                          {keyScenes.slice(0, 50).map((ks, idx) => {
                            const st = ks?.start_seconds ?? ks?.start;
                            const en = ks?.end_seconds ?? ks?.end;
                            const summary = ks?.summary || ks?.summary_text;
                            return (
                              <div key={`keyscene-${idx}`} style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: 12 }}>
                                <Row style={{ justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
                                  <div style={{ fontWeight: 900, color: '#e6e8f2' }}>Key scene {idx + 1}</div>
                                  <SegmentChip type="button" onClick={() => seekTo(st)}>
                                    {formatSeconds(st)}–{formatSeconds(en)}
                                  </SegmentChip>
                                </Row>
                                {summary ? (
                                  <div style={{ marginTop: 8, color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                                    {String(summary).trim()}
                                  </div>
                                ) : null}
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>—</div>
                      )}
                    </CardBody>
                  </Card>
                ) : null}

                {isOutputTaskEnabled('enable_high_point') ? (
                  <Card>
                    <CardHeader>
                      <CardHeaderTitle>High Points</CardHeaderTitle>
                      <CountPill>{Array.isArray(highPoints) ? highPoints.length : 0}</CountPill>
                    </CardHeader>
                    <CardBody>
                      {Array.isArray(highPoints) && highPoints.length ? (
                        <div style={{ display: 'grid', gap: 10 }}>
                          {highPoints.slice(0, 50).map((hp, idx) => {
                            const st = hp?.start_seconds ?? hp?.start;
                            const en = hp?.end_seconds ?? hp?.end;
                            const summary = hp?.summary || hp?.summary_text;
                            return (
                              <div key={`highpoint-${idx}`} style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: 12 }}>
                                <Row style={{ justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
                                  <div style={{ fontWeight: 900, color: '#e6e8f2' }}>High point {idx + 1}</div>
                                  <SegmentChip type="button" onClick={() => seekTo(st)}>
                                    {formatSeconds(st)}–{formatSeconds(en)}
                                  </SegmentChip>
                                </Row>
                                {summary ? (
                                  <div style={{ marginTop: 8, color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                                    {String(summary).trim()}
                                  </div>
                                ) : null}
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>—</div>
                      )}
                    </CardBody>
                  </Card>
                ) : null}

                <Card>
                  <CardHeader>
                    <CardHeaderTitle>Transcript</CardHeaderTitle>
                  </CardHeader>
                  <CardBody id="envid-script">
                    <div style={{ display: 'grid', gap: 18 }}>
                      <div>
                        <div style={{ fontWeight: 900, color: 'rgba(230,232,242,0.8)', marginBottom: 6 }}>
                          Raw transcript (before LLM correction)
                        </div>
                        {Array.isArray(transcriptRawSegments) && transcriptRawSegments.length ? (
                          <div style={{ display: 'grid', gap: 8 }}>
                            {transcriptRawSegments.slice(0, 2000).map((seg, idx) => (
                              <div key={`trs-${idx}`} style={{ color: 'rgba(230,232,242,0.88)', fontSize: 13, lineHeight: 1.45 }}>
                                <SegmentChip type="button" onClick={() => seekTo(seg?.start)}>
                                  {formatTimecode(seg?.start, playerFps)}–{formatTimecode(seg?.end, playerFps)}
                                </SegmentChip>{' '}
                                {(seg?.text || '').toString().trim() || '—'}
                              </div>
                            ))}
                            {transcriptRawSegments.length > 2000 ? (
                              <SubtleNote>Showing first 2000 raw transcript segments.</SubtleNote>
                            ) : null}
                          </div>
                        ) : transcriptRawText.trim() ? (
                          <div style={{ color: 'rgba(230,232,242,0.75)', fontSize: 13 }}>{transcriptRawText}</div>
                        ) : (
                          <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>—</div>
                        )}
                      </div>
                      <div>
                        <div style={{ fontWeight: 900, color: 'rgba(230,232,242,0.8)', marginBottom: 6 }}>
                          Time-banded audio script
                        </div>
                        {Array.isArray(transcriptSegments) && transcriptSegments.length ? (
                          <div style={{ display: 'grid', gap: 8 }}>
                            {transcriptSegments.slice(0, 2000).map((seg, idx) => (
                              <div key={`ts-${idx}`} style={{ color: 'rgba(230,232,242,0.88)', fontSize: 13, lineHeight: 1.45 }}>
                                <SegmentChip type="button" onClick={() => seekTo(seg?.start)}>
                                  {formatTimecode(seg?.start, playerFps)}–{formatTimecode(seg?.end, playerFps)}
                                </SegmentChip>{' '}
                                {(seg?.text || '').toString().trim() || '—'}
                              </div>
                            ))}
                            {transcriptSegments.length > 2000 ? (
                              <SubtleNote>Showing first 2000 transcript segments.</SubtleNote>
                            ) : null}
                          </div>
                        ) : transcriptText.trim() ? (
                          <div style={{ color: 'rgba(230,232,242,0.75)', fontSize: 13 }}>{transcriptText}</div>
                        ) : (
                          <div style={{ display: 'grid', gap: 8, color: 'rgba(230, 232, 242, 0.7)' }}>
                            <div>No transcript found for this video.</div>
                            <SubtleNote>
                              If this was processed before Audio Transcription existed (or transcription was disabled), click Reprocess and enable Audio Transcription.
                            </SubtleNote>
                          </div>
                        )}
                      </div>

                    </div>
                  </CardBody>
                </Card>
              </div>
            </>
          )}
        </DetailBody>
      </DetailPanel>
    ) : null;

  return (
    <PageWrapper>
      <Container>
        <Header>
          <Title>Envid Metadata (Multimodal)</Title>
          <Subtitle>Upload videos or analyze a GCS object (any folder)</Subtitle>
          {systemStats ? (
            <TopStatsBar>
              <TopStatsItem>CPU {cpuPercentLabel}</TopStatsItem>
              <TopStatsItem>RAM {ramLabel}</TopStatsItem>
              <TopStatsItem>GPU {gpuPercentLabel}</TopStatsItem>
              <TopStatsItem>VRAM {vramLabel}</TopStatsItem>
            </TopStatsBar>
          ) : null}
          <TabsRow>
            <TabButton
              type="button"
              $active={activeTab === 'workflow'}
              onClick={() => setActiveTab('workflow')}
              disabled={uploading}
            >
              Workflow
            </TabButton>
            <TabButton
              type="button"
              $active={activeTab === 'running'}
              onClick={() => setActiveTab('running')}
              disabled={uploading}
            >
              Running Jobs
            </TabButton>
            <TabButton
              type="button"
              $active={activeTab === 'completed'}
              onClick={() => setActiveTab('completed')}
              disabled={uploading}
            >
              Completed Jobs
            </TabButton>
            <TabButton
              type="button"
              $active={activeTab === 'failed'}
              onClick={() => setActiveTab('failed')}
              disabled={uploading}
            >
              Failed Jobs
            </TabButton>
          </TabsRow>
        </Header>

        {message && <Message type={message.type}>{message.text}</Message>}

        {jobSubmitModal?.jobId ? (
          <ModalOverlay>
            <ModalCard>
              <ModalTitle>Job Submitted</ModalTitle>
              <ModalBody>
                Specific job id <strong>{jobSubmitModal.jobId}</strong> has been submitted.
              </ModalBody>
              <ModalActions>
                <SecondaryButton type="button" onClick={() => setJobSubmitModal(null)}>
                  Close
                </SecondaryButton>
              </ModalActions>
            </ModalCard>
          </ModalOverlay>
        ) : null}

        {deleteConfirm?.jobId ? (
          <ModalOverlay
            onClick={(event) => {
              if (event.target === event.currentTarget) setDeleteConfirm(null);
            }}
          >
            <ModalCard>
              <ModalHeader>
                <div>
                  <ModalTitle>Delete job</ModalTitle>
                  <ModalSubtitle>Confirm removal</ModalSubtitle>
                </div>
                <SecondaryButton type="button" onClick={() => setDeleteConfirm(null)}>
                  Close
                </SecondaryButton>
              </ModalHeader>
              <ModalBody>
                <ConfirmText>
                  Are you sure you want to delete job <strong>{deleteConfirm.jobId}</strong>?
                </ConfirmText>
              </ModalBody>
              <ConfirmActions>
                <SecondaryButton type="button" onClick={() => setDeleteConfirm(null)}>
                  Cancel
                </SecondaryButton>
                <DeleteButton type="button" onClick={() => confirmDeleteJob(deleteConfirm.jobId)}>
                  Delete
                </DeleteButton>
              </ConfirmActions>
            </ModalCard>
          </ModalOverlay>
        ) : null}

        <div style={{ display: 'grid', gap: 18 }}>
          {activeTab === 'workflow' && (
          <Section>
            <SectionTitle>
              <Icon>📤</Icon>
              Upload
            </SectionTitle>

            <Row style={{ marginBottom: 12 }}>
              {videoSource === 'local' ? (
                <Button
                  type="button"
                  disabled={uploading}
                  onClick={() => {
                    setVideoSource('local');
                    setGcsRawVideoObject('');
                  }}
                >
                  Local device
                </Button>
              ) : (
                <SecondaryButton
                  type="button"
                  disabled={uploading}
                  onClick={() => {
                    setVideoSource('local');
                    setGcsRawVideoObject('');
                  }}
                >
                  Local device
                </SecondaryButton>
              )}

              {videoSource === 'gcs' ? (
                <Button
                  type="button"
                  disabled={uploading}
                  onClick={() => {
                    setVideoSource('gcs');
                    setSelectedFile(null);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                >
                  GCS (Cloud Storage)
                </Button>
              ) : (
                <SecondaryButton
                  type="button"
                  disabled={uploading}
                  onClick={() => {
                    setVideoSource('gcs');
                    setSelectedFile(null);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                >
                  GCS (Cloud Storage)
                </SecondaryButton>
              )}
            </Row>

            {videoSource === 'local' ? (
              <UploadArea
                $dragging={isDragging}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('videoInput')?.click?.()}
              >
                <input
                  id="videoInput"
                  type="file"
                  accept="video/*"
                  onChange={handleFileSelect}
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                />
                <EmptyIcon>🎥</EmptyIcon>
                <div style={{ fontWeight: 900, color: '#e6e8f2' }}>Drag & drop your video here</div>
                <div style={{ color: 'rgba(230, 232, 242, 0.7)', marginTop: 6 }}>or click to browse</div>
                {selectedFile && (
                  <div style={{ marginTop: 12, color: '#cbd5ff', fontWeight: 800 }}>
                    ✓ {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                  </div>
                )}
              </UploadArea>
            ) : (
              <div>
                <div style={{ marginTop: -2, marginBottom: 10, color: 'rgba(230, 232, 242, 0.7)', fontSize: 12 }}>
                  Uses an existing Cloud Storage object (no browser upload). Choose any bucket and folder prefix.
                </div>

                <Row style={{ justifyContent: 'space-between', marginBottom: 10 }}>
                  <SecondaryButton type="button" onClick={loadGcsBucketList} disabled={uploading || gcsBucketLoading}>
                    {gcsBucketLoading ? 'Loading buckets…' : 'Refresh buckets'}
                  </SecondaryButton>
                  <SecondaryButton type="button" onClick={openGcsBrowser} disabled={uploading || !gcsBucket}>
                    Browse bucket
                  </SecondaryButton>
                </Row>

                <div style={{ marginBottom: 12 }}>
                  <div style={{ color: 'rgba(230, 232, 242, 0.7)', fontSize: 12, marginBottom: 6 }}>GCS bucket</div>
                  <select
                    value={gcsBucket}
                    onChange={(e) => {
                      setGcsBucket(e.target.value);
                      setGcsRawVideoObject('');
                      setGcsBrowserPrefix('');
                    }}
                    style={{
                      padding: '12px 14px',
                      border: '1px solid rgba(255, 255, 255, 0.12)',
                      borderRadius: 10,
                      fontSize: '1rem',
                      width: '100%',
                      background: 'rgba(255, 255, 255, 0.04)',
                      color: '#e6e8f2',
                      marginBottom: 8,
                    }}
                  >
                    <option value="">Select a bucket…</option>
                    {gcsBuckets.map((bucket) => {
                      const name = bucket?.name;
                      if (!name) return null;
                      return (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      );
                    })}
                  </select>
                  <div style={{ color: 'rgba(230, 232, 242, 0.7)', fontSize: 12, marginBottom: 10 }}>
                    {gcsBucket ? `Browsing ${gcsBucket}` : 'Select a bucket to browse objects.'}
                  </div>
                </div>
                <div
                  style={{
                    padding: '12px 14px',
                    border: '1px solid rgba(255, 255, 255, 0.12)',
                    borderRadius: 10,
                    fontSize: '0.95rem',
                    width: '100%',
                    background: 'rgba(255, 255, 255, 0.04)',
                    color: '#e6e8f2',
                  }}
                >
                  {gcsRawVideoObject ? `Selected: ${gcsRawVideoObject}` : 'No file selected yet.'}
                </div>
              </div>
            )}

            <div
              style={{
                marginTop: 14,
                border: '1px solid rgba(255, 255, 255, 0.12)',
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: 14,
                padding: 14,
              }}
            >
              <div style={{ fontWeight: 900, color: 'rgba(240, 242, 255, 0.95)', marginBottom: 10 }}>Task Selection</div>

              <div style={{ display: 'grid', gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ fontWeight: 800, color: 'rgba(230, 232, 242, 0.7)', fontSize: 12 }}>Select tasks to run</div>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 10, fontWeight: 900, color: '#e6e8f2' }}>
                    <input
                      type="checkbox"
                      checked={taskSelectionAllEnabled}
                      onChange={(e) => {
                        const isChecked = Boolean(e.target.checked);
                        setTaskSelection((prev) => {
                          const next = { ...(prev || {}) };
                          taskSelectionOptions.forEach((t) => {
                            if (!t.disabled) next[t.enableKey] = isChecked;
                          });
                          return next;
                        });
                      }}
                    />
                    Select all
                  </label>
                </div>

                {taskSelectionOptions.map((t) => {
                  const checked = Boolean(taskSelection?.[t.enableKey]);

                  return (
                    <div
                      key={t.enableKey}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr',
                        gap: 10,
                        alignItems: 'center',
                      }}
                    >
                      <div style={{ display: 'grid', gap: 4 }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 10, fontWeight: 800, color: '#e6e8f2' }}>
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={Boolean(t.disabled)}
                            onChange={(e) => {
                              const isChecked = Boolean(e.target.checked);
                              setTaskSelection((prev) => {
                                const next = { ...(prev || {}), [t.enableKey]: isChecked };
                                return next;
                              });
                            }}
                          />
                          {t.label}
                        </label>
                        {t.hint ? <div style={{ fontSize: 12, color: 'rgba(230, 232, 242, 0.65)' }}>{t.hint}</div> : null}
                      </div>

                      {null}
                    </div>
                  );
                })}

                <div
                  style={{
                    marginTop: 10,
                    display: 'grid',
                    gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 2fr)',
                    gap: 12,
                  }}
                >
                  <div
                    style={{
                      padding: 10,
                      border: '1px solid rgba(255, 255, 255, 0.12)',
                      borderRadius: 10,
                      background: 'rgba(255, 255, 255, 0.04)',
                      minHeight: 120,
                    }}
                  >
                    <div style={{ fontWeight: 900, color: 'rgba(240, 242, 255, 0.95)', marginBottom: 8 }}>
                      Source Language
                    </div>
                    {Boolean(taskSelection?.enable_transcribe) ? (
                      <>
                        <select
                          value={transcribeLanguage}
                          onChange={(e) => setTranscribeLanguage(String(e.target.value || 'auto'))}
                          disabled={uploading}
                          style={{
                            padding: '10px 12px',
                            border: '1px solid rgba(255, 255, 255, 0.12)',
                            borderRadius: 8,
                            fontSize: '0.95rem',
                            width: '100%',
                            background: 'rgba(255, 255, 255, 0.04)',
                            color: '#e6e8f2',
                          }}
                        >
                          {WHISPER_LANGUAGE_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                        <div style={{ fontSize: 12, color: 'rgba(230, 232, 242, 0.65)', marginTop: 6 }}>
                          Choose auto-detect or a specific source language for transcription accuracy.
                        </div>
                      </>
                    ) : (
                      <div style={{ fontSize: 12, color: 'rgba(230, 232, 242, 0.7)' }}>
                        Enable transcription to select source language.
                      </div>
                    )}
                  </div>

                  <div
                    style={{
                      padding: 10,
                      border: '1px solid rgba(255, 255, 255, 0.12)',
                      borderRadius: 10,
                      background: 'rgba(255, 255, 255, 0.04)',
                    }}
                  >
                    <div style={{ fontWeight: 900, color: 'rgba(240, 242, 255, 0.95)', marginBottom: 8 }}>
                      Destination Languages
                    </div>
                    <div style={{ display: 'grid', gap: 8, marginBottom: 8 }}>
                      <details
                        style={{
                          border: '1px solid rgba(255, 255, 255, 0.12)',
                          borderRadius: 10,
                          padding: 10,
                          background: 'rgba(255, 255, 255, 0.04)',
                        }}
                      >
                        <summary
                          style={{
                            cursor: 'pointer',
                            fontWeight: 700,
                            color: '#e6e8f2',
                            listStyle: 'none',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '10px 12px',
                            border: '1px solid rgba(255, 255, 255, 0.12)',
                            borderRadius: 10,
                            fontSize: 14,
                            width: '100%',
                            background: 'rgba(255, 255, 255, 0.04)',
                          }}
                        >
                          <span>Indian Languages</span>
                          <span style={{ opacity: 0.7 }}>▾</span>
                        </summary>
                        <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
                          <Row style={{ gap: 8 }}>
                            <SecondaryButton
                              type="button"
                              onClick={() => {
                                const all = indianLanguageOptions.map((lang) => String(lang.code).toLowerCase());
                                setTargetTranslateLanguages((prev) => {
                                  const current = Array.isArray(prev) ? prev : [];
                                  const merged = new Set(current);
                                  all.forEach((c) => merged.add(c));
                                  return Array.from(merged);
                                });
                              }}
                              disabled={uploading || translateLanguagesLoading}
                            >
                              Select All
                            </SecondaryButton>
                            <SecondaryButton
                              type="button"
                              onClick={() =>
                                setTargetTranslateLanguages((prev) => {
                                  const current = Array.isArray(prev) ? prev : [];
                                  const remove = new Set(
                                    indianLanguageOptions.map((lang) => String(lang.code).toLowerCase())
                                  );
                                  return current.filter((c) => !remove.has(c));
                                })
                              }
                              disabled={uploading || translateLanguagesLoading}
                            >
                              Clear
                            </SecondaryButton>
                          </Row>
                          <div style={{ display: 'grid', gap: 6, maxHeight: 260, overflowY: 'auto' }}>
                            {indianLanguageOptions.map((lang) => {
                              const code = String(lang.code || '').toLowerCase();
                              const checked = targetTranslateLanguages.includes(code);
                              return (
                                <label key={code} style={{ display: 'flex', gap: 10, alignItems: 'center', color: '#e6e8f2' }}>
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    disabled={uploading || translateLanguagesLoading}
                                    onChange={() => {
                                      setTargetTranslateLanguages((prev) => {
                                        const current = Array.isArray(prev) ? prev : [];
                                        if (current.includes(code)) return current.filter((c) => c !== code);
                                        return [...current, code];
                                      });
                                    }}
                                  />
                                  <span>
                                    {lang.name} ({code})
                                  </span>
                                </label>
                              );
                            })}
                          </div>
                        </div>
                      </details>
                      <details
                        style={{
                          border: '1px solid rgba(255, 255, 255, 0.12)',
                          borderRadius: 10,
                          padding: 10,
                          background: 'rgba(255, 255, 255, 0.04)',
                        }}
                      >
                        <summary
                          style={{
                            cursor: 'pointer',
                            fontWeight: 700,
                            color: '#e6e8f2',
                            listStyle: 'none',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '10px 12px',
                            border: '1px solid rgba(255, 255, 255, 0.12)',
                            borderRadius: 10,
                            fontSize: 14,
                            width: '100%',
                            background: 'rgba(255, 255, 255, 0.04)',
                          }}
                        >
                          <span>International Languages</span>
                          <span style={{ opacity: 0.7 }}>▾</span>
                        </summary>
                        <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
                          <Row style={{ gap: 8 }}>
                            <SecondaryButton
                              type="button"
                              onClick={() => {
                                const all = internationalLanguageOptions.map((lang) => String(lang.code).toLowerCase());
                                setTargetTranslateLanguages((prev) => {
                                  const current = Array.isArray(prev) ? prev : [];
                                  const merged = new Set(current);
                                  all.forEach((c) => merged.add(c));
                                  return Array.from(merged);
                                });
                              }}
                              disabled={uploading || translateLanguagesLoading}
                            >
                              Select All
                            </SecondaryButton>
                            <SecondaryButton
                              type="button"
                              onClick={() =>
                                setTargetTranslateLanguages((prev) => {
                                  const current = Array.isArray(prev) ? prev : [];
                                  const remove = new Set(
                                    internationalLanguageOptions.map((lang) => String(lang.code).toLowerCase())
                                  );
                                  return current.filter((c) => !remove.has(c));
                                })
                              }
                              disabled={uploading || translateLanguagesLoading}
                            >
                              Clear
                            </SecondaryButton>
                          </Row>
                          <div style={{ display: 'grid', gap: 6, maxHeight: 260, overflowY: 'auto' }}>
                            {internationalLanguageOptions.map((lang) => {
                              const code = String(lang.code || '').toLowerCase();
                              const checked = targetTranslateLanguages.includes(code);
                              return (
                                <label key={code} style={{ display: 'flex', gap: 10, alignItems: 'center', color: '#e6e8f2' }}>
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    disabled={uploading || translateLanguagesLoading}
                                    onChange={() => {
                                      setTargetTranslateLanguages((prev) => {
                                        const current = Array.isArray(prev) ? prev : [];
                                        if (current.includes(code)) return current.filter((c) => c !== code);
                                        return [...current, code];
                                      });
                                    }}
                                  />
                                  <span>
                                    {lang.name} ({code})
                                  </span>
                                </label>
                              );
                            })}
                          </div>
                        </div>
                      </details>
                    </div>
                    <div style={{ fontSize: 12, color: 'rgba(230, 232, 242, 0.65)' }}>
                      Original is included by default. Choose one or more target languages for translation.
                    </div>
                    {translateLanguagesError ? (
                      <div style={{ fontSize: 12, color: '#ff6b6b' }}>{translateLanguagesError}</div>
                    ) : null}
                  </div>
                </div>
              </div>
            </div>

            <div style={{ marginTop: 14, display: 'grid', gap: 10 }}>
              <Button type="button" onClick={handleUpload} disabled={uploading}>
                {uploadPhaseLabel}
              </Button>

              {(uploading || (activeJob?.kind === 'reprocess' && activeJob?.jobId)) && (
                <>
                  <ProgressBar>
                    <ProgressFill $percent={uploadProgress} />
                  </ProgressBar>

                  {statusPanel}
                </>
              )}
            </div>
          </Section>
          )}

          {activeTab === 'running' && (
          <Section>
            <SectionTitle>
              <Icon>⏳</Icon>
              <span>Running Jobs</span>
              <div style={{ flex: 1 }} />
              <IconButton
                type="button"
                title="Refresh running jobs"
                onClick={manualRefreshRunningJobs}
                disabled={runningJobsLoading}
              >
                🔄
              </IconButton>
            </SectionTitle>

            {runningJobsLoading ? (
              <Message type="info">Loading running jobs…</Message>
            ) : null}

            {runningJobsError ? (
              <Message type="error">{runningJobsError}</Message>
            ) : null}

            {!runningJobsLoading && !runningJobsError && runningJobs.length === 0 ? (
              <EmptyState>
                <EmptyIcon>✅</EmptyIcon>
                No running jobs right now.
              </EmptyState>
            ) : (
              <RunningJobsList>
                {runningJobs.map((job) => runningJobRow(job))}
              </RunningJobsList>
            )}
          </Section>
          )}

          {activeTab === 'completed' && (
          <Section>
            <SectionTitle>
              <Icon>✅</Icon>
              Completed Jobs
            </SectionTitle>

            {completedJobsLoading ? (
              <Message type="info">Loading completed jobs…</Message>
            ) : null}

            {completedJobsError ? (
              <Message type="error">{completedJobsError}</Message>
            ) : null}

            {!completedJobsLoading && !completedJobsError && completedJobs.length === 0 ? (
              <EmptyState>
                <EmptyIcon>✅</EmptyIcon>
                No completed jobs yet.
              </EmptyState>
            ) : (
              <RunningJobsList>
                {completedJobs.map((job) => completedJobRow(job))}
              </RunningJobsList>
            )}

            {metadataDetailPanel}
          </Section>
          )}

          {activeTab === 'failed' && (
          <Section>
            <SectionTitle>
              <Icon>⚠️</Icon>
              Failed Jobs
            </SectionTitle>

            {failedJobsLoading ? (
              <Message type="info">Loading failed jobs…</Message>
            ) : null}

            {failedJobsError ? (
              <Message type="error">{failedJobsError}</Message>
            ) : null}

            {!failedJobsLoading && !failedJobsError && failedJobs.length === 0 ? (
              <EmptyState>
                <EmptyIcon>✅</EmptyIcon>
                No failed jobs.
              </EmptyState>
            ) : (
              <RunningJobsList>
                {failedJobs.map((job) => failedJobRow(job))}
              </RunningJobsList>
            )}
          </Section>
          )}
        </div>

        {gcsBrowserOpen && (
          <ModalOverlay
            onClick={(event) => {
              if (event.target === event.currentTarget) setGcsBrowserOpen(false);
            }}
          >
            <ModalCard>
              <ModalHeader>
                <div>
                  <ModalTitle>Browse bucket</ModalTitle>
                  <ModalSubtitle>{gcsBucket || 'No bucket selected'}</ModalSubtitle>
                </div>
                <SecondaryButton type="button" onClick={() => setGcsBrowserOpen(false)}>
                  Close
                </SecondaryButton>
              </ModalHeader>

              <ModalBody>
                <BrowserPath>
                  <span>Path:</span>
                  <strong>{gcsBrowserPrefix || '/'}</strong>
                </BrowserPath>

                <input
                  value={gcsBrowserQuery}
                  onChange={(event) => setGcsBrowserQuery(event.target.value)}
                  placeholder="Search folders or files"
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 10,
                    border: '1px solid rgba(255, 255, 255, 0.12)',
                    background: 'rgba(255, 255, 255, 0.04)',
                    color: '#e6e8f2',
                    marginBottom: 12,
                  }}
                />

                <Row style={{ marginBottom: 12, justifyContent: 'space-between' }}>
                  <SecondaryButton
                    type="button"
                    onClick={() => {
                      const parent = toParentPrefix(gcsBrowserPrefix);
                      setGcsBrowserPrefix(parent);
                      loadRawVideoList(gcsBucket, parent);
                    }}
                    disabled={!gcsBrowserPrefix}
                  >
                    Up
                  </SecondaryButton>
                  <SecondaryButton type="button" onClick={() => loadRawVideoList(gcsBucket, gcsBrowserPrefix)}>
                    {gcsRawVideoLoading ? 'Loading…' : 'Refresh'}
                  </SecondaryButton>
                </Row>

                <BrowserList>
                  {gcsBrowserPrefixes
                    .filter((prefix) => {
                      if (!gcsBrowserQuery) return true;
                      const name = String(prefix || '').replace(gcsBrowserPrefix, '').replace(/\/+$/, '').toLowerCase();
                      return name.includes(gcsBrowserQuery.toLowerCase());
                    })
                    .map((prefix) => {
                    const name = String(prefix || '').replace(gcsBrowserPrefix, '').replace(/\/+$/, '');
                    if (!name) return null;
                    return (
                      <BrowserRow
                        key={prefix}
                        onClick={() => {
                          setGcsBrowserPrefix(prefix);
                          loadRawVideoList(gcsBucket, prefix);
                        }}
                      >
                        <BrowserIcon>📁</BrowserIcon>
                        <div>{name}</div>
                      </BrowserRow>
                    );
                  })}

                  {gcsBrowserObjects
                    .filter((obj) => isVideoFile(obj?.name))
                    .filter((obj) => {
                      if (!gcsBrowserQuery) return true;
                      const name = String(obj?.name || '').replace(gcsBrowserPrefix, '').toLowerCase();
                      return name.includes(gcsBrowserQuery.toLowerCase());
                    })
                    .map((obj) => {
                    const name = String(obj?.name || '').replace(gcsBrowserPrefix, '');
                    const size = typeof obj?.size === 'number' ? formatBytes(obj.size) : null;
                    return (
                      <BrowserRow
                        key={obj?.uri || obj?.name}
                        onClick={() => {
                          const value = obj?.bucket ? `gs://${obj.bucket}/${obj.name}` : obj?.uri || obj?.name;
                          setGcsRawVideoObject(value || '');
                          setGcsBrowserOpen(false);
                        }}
                      >
                        <BrowserIcon>🎞️</BrowserIcon>
                        <div>{name}</div>
                        {size ? <BrowserMeta>{size}</BrowserMeta> : null}
                      </BrowserRow>
                    );
                  })}

                  {!gcsBrowserPrefixes.length && !gcsBrowserObjects.some((obj) => isVideoFile(obj?.name)) && (
                    <BrowserEmpty>No video files found in this folder.</BrowserEmpty>
                  )}
                </BrowserList>
              </ModalBody>
            </ModalCard>
          </ModalOverlay>
        )}

      </Container>
    </PageWrapper>
  );
}
