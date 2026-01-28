import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import Hls from 'hls.js';
import dashjs from 'dashjs';

const resolveSubtitleApiBase = () => {
  const envValue = process.env.REACT_APP_SUBTITLE_API_BASE;
  if (envValue) {
    return envValue.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined') {
    const { protocol, hostname } = window.location;
    const localHosts = new Set(['localhost', '127.0.0.1', '0.0.0.0']);
    const isLanHost = /^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1]))/.test(hostname) || hostname.endsWith('.local');
    if (localHosts.has(hostname) || isLanHost) {
      return `${protocol}//${hostname}:5001`;
    }
    return `${protocol}//${hostname}`;
  }
  return '';
};

const TRANSCRIBE_LANGUAGE_OPTIONS = [
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

const TRANSLATE_LANGUAGE_OPTIONS = [
  { value: 'af', label: 'Afrikaans' },
  { value: 'am', label: 'Amharic' },
  { value: 'ar', label: 'Arabic' },
  { value: 'az', label: 'Azerbaijani' },
  { value: 'bg', label: 'Bulgarian' },
  { value: 'bn', label: 'Bengali' },
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
  { value: 'fa', label: 'Farsi' },
  { value: 'fi', label: 'Finnish' },
  { value: 'fr', label: 'French' },
  { value: 'ga', label: 'Irish' },
  { value: 'gu', label: 'Gujarati' },
  { value: 'ha', label: 'Hausa' },
  { value: 'he', label: 'Hebrew' },
  { value: 'hi', label: 'Hindi' },
  { value: 'hr', label: 'Croatian' },
  { value: 'hu', label: 'Hungarian' },
  { value: 'hy', label: 'Armenian' },
  { value: 'id', label: 'Indonesian' },
  { value: 'is', label: 'Icelandic' },
  { value: 'it', label: 'Italian' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ka', label: 'Georgian' },
  { value: 'kk', label: 'Kazakh' },
  { value: 'km', label: 'Khmer' },
  { value: 'ko', label: 'Korean' },
  { value: 'ku', label: 'Kurdish (Kurmanji)' },
  { value: 'ky', label: 'Kyrgyz' },
  { value: 'lt', label: 'Lithuanian' },
  { value: 'lv', label: 'Latvian' },
  { value: 'mk', label: 'Macedonian' },
  { value: 'ml', label: 'Malayalam' },
  { value: 'mn', label: 'Mongolian' },
  { value: 'mr', label: 'Marathi' },
  { value: 'ms', label: 'Malay' },
  { value: 'mt', label: 'Maltese' },
  { value: 'my', label: 'Burmese' },
  { value: 'ne', label: 'Nepali' },
  { value: 'nl', label: 'Dutch' },
  { value: 'no', label: 'Norwegian' },
  { value: 'pa', label: 'Punjabi' },
  { value: 'pl', label: 'Polish' },
  { value: 'ps', label: 'Pashto' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ro', label: 'Romanian' },
  { value: 'ru', label: 'Russian' },
  { value: 'sd', label: 'Sindhi' },
  { value: 'si', label: 'Sinhala' },
  { value: 'sk', label: 'Slovak' },
  { value: 'sl', label: 'Slovenian' },
  { value: 'so', label: 'Somali' },
  { value: 'sq', label: 'Albanian' },
  { value: 'sr', label: 'Serbian' },
  { value: 'sv', label: 'Swedish' },
  { value: 'sw', label: 'Swahili' },
  { value: 'ta', label: 'Tamil' },
  { value: 'te', label: 'Telugu' },
  { value: 'th', label: 'Thai' },
  { value: 'tl', label: 'Tagalog' },
  { value: 'tr', label: 'Turkish' },
  { value: 'uk', label: 'Ukrainian' },
  { value: 'ur', label: 'Urdu' },
  { value: 'uz', label: 'Uzbek' },
  { value: 'vi', label: 'Vietnamese' },
  { value: 'xh', label: 'Xhosa' },
  { value: 'yi', label: 'Yiddish' },
  { value: 'yo', label: 'Yoruba' },
  { value: 'zh', label: 'Chinese (Simplified)' },
  { value: 'zh-TW', label: 'Chinese (Traditional)' },
  { value: 'zu', label: 'Zulu' }
];

const languageLabelMap = new Map(
  TRANSLATE_LANGUAGE_OPTIONS.map((option) => [option.value.toLowerCase(), option.label])
);

const STAGE_FALLBACK_MESSAGES = {
  upload: 'Uploading video and extracting audio…',
  transcribe: 'Transcription job in progress…',
  complete: 'Finalizing subtitles…'
};

const computeStageBaseline = ({
  stage,
  readyForTranscription,
  subtitlesInProgress,
  subtitlesAvailable
}) => {
  if (!stage) {
    return 0;
  }

  if (stage === 'upload') {
    return readyForTranscription ? 55 : 12;
  }

  if (stage === 'transcribe') {
    if (subtitlesAvailable) {
      return 90;
    }
    return subtitlesInProgress ? 78 : 64;
  }

  if (stage === 'complete') {
    return 100;
  }

  return 0;
};

const Page = styled.section`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 3rem;
  color: #dce7ff;
`;

const Title = styled.h1`
  color: #ffffff;
  font-weight: 800;
  margin: 0 0 0.85rem 0;
  font-size: clamp(1.65rem, 3.2vw, 2.2rem);
  text-align: center;
`;

const Description = styled.p`
  text-align: center;
  margin: 0 auto 2.25rem;
  max-width: 760px;
  line-height: 1.8;
  color: #b8c9f5;
`;

const UploadContainer = styled.div`
  background: linear-gradient(160deg, rgba(14, 26, 48, 0.92), rgba(22, 36, 63, 0.88));
  border: 2px dashed rgba(99, 102, 241, 0.28);
  border-radius: 16px;
  padding: 3.2rem 2.4rem;
  text-align: center;
  margin-bottom: 2.25rem;
  transition: border-color 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
  cursor: pointer;
  box-shadow: 0 24px 54px rgba(7, 15, 30, 0.55);

  &:hover,
  &.dragover {
    border-color: rgba(56, 189, 248, 0.7);
    background: linear-gradient(160deg, rgba(18, 34, 61, 0.94), rgba(30, 48, 82, 0.9));
    box-shadow: 0 28px 62px rgba(15, 36, 68, 0.6);
  }
`;

const UploadIcon = styled.div`
  font-size: 3rem;
  color: #60a5fa;
  margin-bottom: 1rem;
`;

const UploadText = styled.div`
  color: #f8fafc;
  font-size: 1.1rem;
  margin-bottom: 0.6rem;
`;

const UploadSubtext = styled.div`
  color: #a8b4da;
  font-size: 0.92rem;
`;

const HiddenInput = styled.input`
  display: none;
`;

const LanguageControls = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin: 1rem 0;
`;

const ControlGroup = styled.div`
  display: flex;
  flex-direction: column;
  min-width: 200px;
  flex: 1;
`;

const Label = styled.label`
  font-size: 0.82rem;
  color: #9fb3dd;
  margin-bottom: 0.35rem;
  letter-spacing: 0.25px;
`;

const Select = styled.select`
  background: rgba(10, 20, 38, 0.85);
  color: #e2e9f5;
  border: 1px solid rgba(99, 102, 241, 0.28);
  border-radius: 8px;
  padding: 0.65rem;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;

  &:focus {
    outline: none;
    border-color: rgba(var(--accent-rgb), 0.6);
    box-shadow: 0 0 0 3px rgba(var(--accent-rgb), 0.15);
  }
`;

const HelperText = styled.span`
  font-size: 0.75rem;
  color: #7888aa;
  margin-top: 0.45rem;
`;

const Button = styled.button`
  padding: 0.95rem 1.35rem;
  background: linear-gradient(135deg, var(--button-accent), var(--button-accent-2));
  color: var(--on-accent);
  font-weight: 800;
  border: none;
  border-radius: 999px;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
  margin-top: 1.1rem;
  letter-spacing: 0.4px;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 36px rgba(var(--button-accent-rgb), 0.32);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    box-shadow: none;
  }
`;

const SecondaryButton = styled(Button)`
  background: rgba(45, 55, 95, 0.6);
  color: #dce7ff;
  border: 1px solid rgba(148, 163, 255, 0.4);
  margin-top: 0;

  &:hover {
    box-shadow: 0 14px 28px rgba(79, 70, 229, 0.35);
  }
`;

const VideoPreview = styled.div`
  background: linear-gradient(160deg, rgba(11, 22, 42, 0.92), rgba(22, 38, 65, 0.88));
  border: 1px solid rgba(99, 102, 241, 0.22);
  border-radius: 16px;
  padding: 1.65rem;
  margin-bottom: 2.25rem;
  box-shadow: 0 20px 48px rgba(7, 15, 30, 0.52);
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
  color: #94a3e6;
  font-size: 0.9rem;
`;

const StreamingPlayerContainer = styled.div`
  background: linear-gradient(166deg, rgba(12, 24, 46, 0.95), rgba(25, 42, 74, 0.88));
  border: 1px solid rgba(99, 102, 241, 0.22);
  border-radius: 18px;
  padding: 1.75rem;
  margin-bottom: 2.5rem;
  box-shadow: 0 24px 52px rgba(6, 15, 30, 0.55);
`;

const FileMeta = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1rem;
`;

const FileMetaRow = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: #b8c9f5;
  font-size: 0.92rem;
`;

const FileLabel = styled.span`
  color: #ffffff;
  font-weight: 600;
`;

const StreamingVideoWrapper = styled.div`
  position: relative;
  display: inline-block;
  width: 100%;
  max-width: 720px;
`;

const StreamingVideo = styled.video`
  width: 100%;
  max-width: 720px;
  height: auto;
  border-radius: 10px;
  background: #050e1a;
  box-shadow: 0 14px 32px rgba(5, 15, 32, 0.6);
`;

const CaptionToggleButton = styled.button`
  position: absolute;
  top: 12px;
  right: 12px;
  padding: 0.45rem 0.75rem;
  border-radius: 999px;
  font-size: 0.8rem;
  border: 1px solid ${props => (props.$active ? 'rgba(var(--accent-rgb), 0.8)' : 'rgba(148, 163, 255, 0.35)')};
  color: ${props => (props.$active ? '#0f172a' : '#e2e9f5')};
  background: ${props => (props.$active ? 'linear-gradient(135deg, var(--button-accent), var(--button-accent-2))' : 'rgba(10, 18, 32, 0.75)')};
  backdrop-filter: blur(6px);
  cursor: pointer;
  transition: transform 0.2s ease, border-color 0.2s ease, color 0.2s ease, background 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    border-color: rgba(var(--accent-rgb), 0.85);
  }

  &:focus-visible {
    outline: 2px solid rgba(99, 102, 241, 0.75);
    outline-offset: 2px;
  }
`;

const LanguageCheckboxGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.55rem 0.85rem;
  background: rgba(9, 18, 32, 0.92);
  border: 1px solid rgba(79, 70, 229, 0.28);
  border-radius: 12px;
  padding: 0.85rem;
  max-height: 240px;
  overflow-y: auto;
  box-shadow: inset 0 0 0 1px rgba(var(--accent-rgb), 0.05);
`;

const LanguageCheckboxLabel = styled.label`
  display: flex;
  align-items: center;
  gap: 0.55rem;
  font-size: 0.85rem;
  color: #dce7ff;
`;

const LanguageCheckbox = styled.input`
  accent-color: var(--text);
  width: 16px;
  height: 16px;
`;

const LanguageActions = styled.div`
  display: flex;
  gap: 0.75rem;
  margin-top: 0.5rem;
  flex-wrap: wrap;
`;

const LanguageLink = styled.button`
  background: none;
  border: none;
  color: #7dd3fc;
  cursor: pointer;
  font-size: 0.82rem;
  padding: 0;
  transition: color 0.2s ease;

  &:hover {
    color: #bae6fd;
    text-decoration: underline;
  }

  &:disabled {
    opacity: 0.45;
    cursor: not-allowed;
    text-decoration: none;
  }
`;

const PlayerControls = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1rem;
  align-items: center;
`;

const ControlRow = styled.div`
  display: flex;
  gap: 0.75rem;
  align-items: center;
`;

const ToggleButton = styled.button`
  padding: 0.6rem 1rem;
  border-radius: 999px;
  border: 1px solid ${props => (props.$active ? 'rgba(var(--accent-rgb), 0.65)' : 'rgba(99, 102, 241, 0.25)')};
  background: ${props => (props.$active ? 'linear-gradient(135deg, var(--button-accent), var(--button-accent-2))' : 'rgba(19, 32, 58, 0.75)')};
  color: ${props => (props.$active ? 'var(--on-accent)' : '#dce7ff')};
  cursor: pointer;
  font-weight: 600;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    border-color: rgba(var(--accent-rgb), 0.75);
    box-shadow: 0 12px 28px rgba(var(--accent-rgb), 0.22);
  }
`;

const ProcessingContainer = styled.div`
  background: linear-gradient(162deg, rgba(9, 20, 38, 0.92), rgba(19, 34, 62, 0.88));
  border: 1px solid rgba(79, 70, 229, 0.28);
  border-radius: 18px;
  padding: 2.2rem;
  text-align: center;
  margin-bottom: 2.25rem;
  box-shadow: 0 24px 52px rgba(6, 15, 30, 0.5);
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 10px;
  background: rgba(15, 28, 50, 0.9);
  border-radius: 999px;
  overflow: hidden;
  margin: 1.1rem 0;
  border: 1px solid rgba(var(--accent-rgb), 0.18);
`;

const ProgressFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, var(--button-accent), var(--button-accent-2) 70%, var(--button-accent));
  width: ${props => props.$progress}%;
  transition: width 0.3s ease;
`;

const LanguageStatus = styled.div`
  margin-top: 0.75rem;
  color: #b8c9f5;
  font-size: 0.85rem;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  justify-content: center;
`;

const DownloadLink = styled.a`
  color: #7dd3fc;
  font-size: 0.85rem;
  text-decoration: none;
  transition: color 0.2s ease;

  &:hover {
    color: #bae6fd;
    text-decoration: underline;
  }
`;

const CompactActions = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 0.75rem;
  flex-wrap: wrap;
`;

const CompactButton = styled.button`
  background: rgba(21, 32, 54, 0.85);
  border: 1px solid rgba(79, 70, 229, 0.35);
  color: #dce7ff;
  padding: 0.55rem 0.9rem;
  border-radius: 999px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    border-color: rgba(56, 189, 248, 0.6);
    box-shadow: 0 12px 26px rgba(56, 189, 248, 0.28);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    box-shadow: none;
  }
`;

const ErrorMessage = styled.div`
  background: rgba(60, 17, 34, 0.65);
  border: 1px solid rgba(248, 113, 113, 0.55);
  border-radius: 12px;
  padding: 1.1rem;
  color: #fecaca;
  margin: 1rem 0;
`;

const SuccessMessage = styled.div`
  background: rgba(15, 48, 35, 0.6);
  border: 1px solid rgba(74, 222, 128, 0.45);
  border-radius: 12px;
  padding: 1.1rem;
  color: #bbf7d0;
  margin: 1rem 0;
`;

export default function AISubtitling() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [currentFileId, setCurrentFileId] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState('');
  const [phase, setPhase] = useState('');
  const [hasRequestedTranscription, setHasRequestedTranscription] = useState(false);
  const [hasAttemptedSubtitleFetch, setHasAttemptedSubtitleFetch] = useState(false);
  const [sourceLanguage, setSourceLanguage] = useState('auto');
  const [languageDetectionError, setLanguageDetectionError] = useState('');
  const [detectedLanguage, setDetectedLanguage] = useState('');
  const [availableSubtitles, setAvailableSubtitles] = useState([]);
  const [selectedSubtitle, setSelectedSubtitle] = useState('');
  const [captionsEnabled, setCaptionsEnabled] = useState(true);
  const [streams, setStreams] = useState({});
  const [selectedProtocol, setSelectedProtocol] = useState('');
  const [requestedTargetLanguages, setRequestedTargetLanguages] = useState([]);
  const [selectedTargetLanguages, setSelectedTargetLanguages] = useState([]);
  const [translationApplied, setTranslationApplied] = useState(false);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const hlsInstanceRef = useRef(null);
  const dashPlayerRef = useRef(null);

  const SUBTITLE_API_BASE = useMemo(() => resolveSubtitleApiBase(), []);

  const resolveUrl = useCallback((path) => {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) {
      return path;
    }
    return `${SUBTITLE_API_BASE}${path}`;
  }, [SUBTITLE_API_BASE]);

  const cleanupVideoPlayers = () => {
    if (hlsInstanceRef.current) {
      hlsInstanceRef.current.destroy();
      hlsInstanceRef.current = null;
    }
    if (dashPlayerRef.current) {
      dashPlayerRef.current.reset();
      dashPlayerRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.removeAttribute('src');
      videoRef.current.load();
    }
  };

  useEffect(() => () => cleanupVideoPlayers(), []);

  useEffect(() => {
    if (!availableSubtitles.length) {
      setSelectedSubtitle('');
      return;
    }

    setSelectedSubtitle((previous) => {
      if (previous) {
        const exists = availableSubtitles.some((track) => track.code === previous);
        if (exists) return previous;
      }
      const original = availableSubtitles.find((track) => track.isOriginal);
      return original ? original.code : availableSubtitles[0].code;
    });
  }, [availableSubtitles]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const tracks = video.textTracks;
    for (let index = 0; index < tracks.length; index += 1) {
      const track = tracks[index];
      const trackCode = track.language || track.label || '';
      if (!captionsEnabled || !selectedSubtitle) {
        track.mode = 'disabled';
        continue;
      }
      if (trackCode.toLowerCase() === selectedSubtitle.toLowerCase()) {
        track.mode = 'showing';
      } else {
        track.mode = 'disabled';
      }
    }
  }, [captionsEnabled, selectedSubtitle, availableSubtitles]);

  useEffect(() => {
    if (!Object.keys(streams).length) {
      setSelectedProtocol('');
      cleanupVideoPlayers();
      return;
    }
    setSelectedProtocol((previous) => {
      if (previous && streams[previous]) {
        return previous;
      }
      return Object.keys(streams)[0];
    });
  }, [streams]);

  useEffect(() => {
    if (!selectedProtocol || !streams[selectedProtocol]) {
      cleanupVideoPlayers();
      return;
    }

    const video = videoRef.current;
    if (!video) return;

    const manifestPath = streams[selectedProtocol]?.manifest;
    if (!manifestPath) return;

    const manifestUrl = resolveUrl(manifestPath);

    cleanupVideoPlayers();

    if (selectedProtocol === 'hls') {
      if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = manifestUrl;
        video.load();
      } else if (Hls.isSupported()) {
        const hls = new Hls({ enableWorker: true });
        hls.loadSource(manifestUrl);
        hls.attachMedia(video);
        hlsInstanceRef.current = hls;
      } else {
        video.src = manifestUrl;
        video.load();
      }
    } else if (selectedProtocol === 'dash') {
      const player = dashjs.MediaPlayer().create();
      player.initialize(video, manifestUrl, false);
      dashPlayerRef.current = player;
    }
  }, [selectedProtocol, streams, resolveUrl]);

  const toggleTargetLanguage = (code) => {
    setSelectedTargetLanguages((previous) =>
      previous.includes(code)
        ? previous.filter((item) => item !== code)
        : [...previous, code]
    );
  };

  const selectAllTargetLanguages = () => {
    setSelectedTargetLanguages(TRANSLATE_LANGUAGE_OPTIONS.map((option) => option.value));
  };

  const clearTargetLanguages = () => {
    setSelectedTargetLanguages([]);
  };

  const handleFileSelect = (file) => {
    if (!file || !file.type.startsWith('video/')) {
      setError('Please select a valid video file.');
      return;
    }

    const maxSize = 5 * 1024 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File too large. Maximum size allowed is 5GB.');
      return;
    }

    setSelectedFile(file);
    setCurrentFileId('');
    setProgress(0);
    setProgressMessage('');
    setError('');
    setPhase('');
    setHasRequestedTranscription(false);
    setHasAttemptedSubtitleFetch(false);
    setDetectedLanguage('');
    setAvailableSubtitles([]);
    setSelectedSubtitle('');
  setStreams({});
    setRequestedTargetLanguages([]);
    setSelectedTargetLanguages([]);
    setTranslationApplied(false);
    setCaptionsEnabled(true);
    setLanguageDetectionError('');
    cleanupVideoPlayers();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragOver(false);
  };

  const handleFileInputChange = (event) => {
    const file = event.target.files[0];
    handleFileSelect(file);
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 Bytes';
    const units = ['Bytes', 'KB', 'MB', 'GB'];
    const index = Math.floor(Math.log(bytes) / Math.log(1024));
    const size = bytes / Math.pow(1024, index);
    return `${size.toFixed(2)} ${units[index]}`;
  };

  const checkBackendHealth = async (retries = 2) => {
    for (let attempt = 0; attempt < retries; attempt += 1) {
      try {
        const response = await axios.get(`${SUBTITLE_API_BASE}/health`, { timeout: 5000 });
        if (response.status === 200) {
          return true;
        }
      } catch (healthError) {
        if (attempt < retries - 1) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      }
    }
    return false;
  };

  const pollProgress = async (fileId) => {
    try {
      const { data } = await axios.get(`${SUBTITLE_API_BASE}/progress/${fileId}`);

      const serverStage = data.stage;
      const subtitlesAvailable = Boolean(data.readyForFetch);
      const readyForTranscription = Boolean(data.readyForTranscription);
      const subtitlesInProgress = Boolean(data.subtitlesInProgress);

      if (serverStage && serverStage !== phase) {
        setPhase(serverStage);
      }

      if (data.languageDetectionError) {
        setLanguageDetectionError(data.languageDetectionError);
      }

      if (typeof data.detectedLanguage === 'string' && data.detectedLanguage.length) {
        setDetectedLanguage(data.detectedLanguage);
      }

      if (Array.isArray(data.targetLanguageRequested)) {
        setRequestedTargetLanguages(data.targetLanguageRequested);
      }

      if (Array.isArray(data.availableSubtitles)) {
        setAvailableSubtitles((previous) => (previous.length ? previous : data.availableSubtitles));
      }

      if (data.streamsReady) {
        setStreams((previous) => ({ ...previous, ...data.streamsReady }));
      }

      if (typeof data.translationApplied === 'boolean') {
        setTranslationApplied(data.translationApplied);
      }

      let rawProgressValue = 0;
      if (typeof data.progress === 'number') {
        rawProgressValue = data.progress;
      } else if (typeof data.progress === 'string') {
        const parsed = Number.parseFloat(data.progress);
        rawProgressValue = Number.isFinite(parsed) ? parsed : 0;
      }

      if (rawProgressValue === -1) {
        const fallbackMessage = STAGE_FALLBACK_MESSAGES[serverStage] || '';
        setProgress((previous) => Math.max(previous, computeStageBaseline({
          stage: serverStage,
          readyForTranscription,
          subtitlesInProgress,
          subtitlesAvailable
        })));
        setProgressMessage(data.message || fallbackMessage);
        setError(data.message || 'Processing failed on the server.');
        setProcessing(false);
        return;
      }

      const baseline = computeStageBaseline({
        stage: serverStage,
        readyForTranscription,
        subtitlesInProgress,
        subtitlesAvailable
      });

      let visualProgress = Number.isFinite(rawProgressValue) ? rawProgressValue : baseline;
      visualProgress = Math.max(baseline, visualProgress);
      visualProgress = Math.max(0, Math.min(100, Math.round(visualProgress)));

      setProgress((previous) => Math.max(previous, visualProgress));
      const fallbackMessage = STAGE_FALLBACK_MESSAGES[serverStage] || '';
      setProgressMessage(data.message || fallbackMessage);

      if (!subtitlesAvailable && readyForTranscription && !hasRequestedTranscription) {
        setHasRequestedTranscription(true);
        setPhase('transcribe');
        generateSubtitles(fileId);
        return;
      }

      if (subtitlesAvailable && !hasAttemptedSubtitleFetch) {
        setHasAttemptedSubtitleFetch(true);
        fetchSubtitles(fileId);
        return;
      }

      if (!subtitlesAvailable) {
        setTimeout(() => pollProgress(fileId), 2000);
      }
    } catch (pollError) {
      setError(`Failed to track progress: ${pollError.message}`);
      setProcessing(false);
    }
  };

  const generateSubtitles = async (fileId) => {
    try {
      const payload = {
        file_id: fileId,
        source_language: sourceLanguage,
        target_languages: selectedTargetLanguages
      };

      setTranslationApplied(false);

      await axios.post(`${SUBTITLE_API_BASE}/generate-subtitles`, payload, {
        headers: { 'Content-Type': 'application/json' }
      });

      const transcriptionBaseline = computeStageBaseline({
        stage: 'transcribe',
        readyForTranscription: true,
        subtitlesInProgress: true,
        subtitlesAvailable: false
      });
      setProgress((previous) => Math.max(previous, transcriptionBaseline));
      setProgressMessage('Starting subtitle generation...');
      setTimeout(() => pollProgress(fileId), 2000);
    } catch (generationError) {
      const backendMessage = generationError.response?.data?.error || generationError.message;

      if (generationError.response?.status === 404) {
        setHasRequestedTranscription(false);
        setProgressMessage('Waiting for audio extraction to finish...');
        setTimeout(() => pollProgress(fileId), 2000);
        return;
      }

      setError(`Failed to generate subtitles: ${backendMessage}`);
      setProcessing(false);
    }
  };

  const fetchStreams = useCallback(async (fileId, attempt = 0) => {
    try {
      const { data } = await axios.get(`${SUBTITLE_API_BASE}/streams/${fileId}`);
      if (data?.protocols) {
        setStreams(data.protocols);
      }
    } catch (streamError) {
      const status = streamError.response?.status;
      if (status === 404 && attempt < 8) {
        setTimeout(() => fetchStreams(fileId, attempt + 1), 2000);
        return;
      }
      console.warn('Stream manifest lookup failed:', streamError.message);
    }
  }, [SUBTITLE_API_BASE]);

  const fetchSubtitles = async (fileId) => {
    try {
      const { data } = await axios.get(`${SUBTITLE_API_BASE}/subtitles/${fileId}`, { timeout: 10000 });

      const tracks = Array.isArray(data.tracks) ? data.tracks : [];
      const withUrls = tracks.map((track) => ({
        code: track.code,
        label: track.label,
        isOriginal: track.isOriginal,
        srt: resolveUrl(track.srt),
        vtt: resolveUrl(track.vtt)
      }));

      const deduped = withUrls.reduce((accumulator, track) => {
        if (!track.code) {
          accumulator.push(track);
          return accumulator;
        }
        const existingIndex = accumulator.findIndex((entry) => entry.code === track.code);
        if (existingIndex === -1) {
          accumulator.push(track);
        } else if (track.isOriginal && !accumulator[existingIndex].isOriginal) {
          accumulator[existingIndex] = track;
        }
        return accumulator;
      }, []);

      setAvailableSubtitles(deduped);
      setProcessing(false);
      setPhase('complete');
      setProgress(100);
      setProgressMessage('✅ Subtitles generated successfully!');
      setCaptionsEnabled(true);

  await fetchStreams(fileId);
    } catch (fetchError) {
      setHasAttemptedSubtitleFetch(false);
      setTimeout(() => pollProgress(fileId), 3000);
    }
  };

  const startProcessing = async () => {
    if (!selectedFile || processing) return;

    const reuseExisting = Boolean(currentFileId && phase === 'complete');

    setProcessing(true);
    setProgress(
      computeStageBaseline({
        stage: 'upload',
        readyForTranscription: false,
        subtitlesInProgress: false,
        subtitlesAvailable: false
      })
    );
    setProgressMessage('Checking backend availability...');
    setError('');
    setPhase(reuseExisting ? 'transcribe' : 'upload');
    setHasRequestedTranscription(false);
    setHasAttemptedSubtitleFetch(false);
    setDetectedLanguage('');
    setTranslationApplied(false);
    setLanguageDetectionError('');
    setAvailableSubtitles([]);
    setSelectedSubtitle('');
  setStreams({});

    try {
      setProgressMessage('Connecting to subtitle service...');
      const isHealthy = await checkBackendHealth(3);

      if (!isHealthy) {
        throw new Error('Cannot connect to AI Subtitle service. Please ensure the backend is running.');
      }

      if (reuseExisting && currentFileId) {
        setProgressMessage('Reusing existing audio for subtitle generation...');
        setRequestedTargetLanguages(selectedTargetLanguages);
        setHasRequestedTranscription(true);
        await generateSubtitles(currentFileId);
        return;
      }

      const formData = new FormData();
      formData.append('video', selectedFile);

      setProgressMessage('Starting upload...');

      const uploadResponse = await axios.post(`${SUBTITLE_API_BASE}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 30 * 60 * 1000
      });

      const uploadResult = uploadResponse.data;
      if (!uploadResult.file_id) {
        throw new Error('No file ID returned from upload');
      }

      setCurrentFileId(uploadResult.file_id);
      setRequestedTargetLanguages(selectedTargetLanguages);
      setProgressMessage('Upload completed, processing video...');
      setTimeout(() => pollProgress(uploadResult.file_id), 1000);
    } catch (startError) {
      setError(`Upload failed: ${startError.response?.data?.error || startError.message}`);
      setProcessing(false);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setCurrentFileId('');
    setProgress(0);
    setProgressMessage('');
    setError('');
    setProcessing(false);
    setPhase('');
    setHasRequestedTranscription(false);
  setHasAttemptedSubtitleFetch(false);
  setDetectedLanguage('');
    setAvailableSubtitles([]);
    setSelectedSubtitle('');
  setStreams({});
    setRequestedTargetLanguages([]);
    setSelectedTargetLanguages([]);
    setTranslationApplied(false);
    setLanguageDetectionError('');
    cleanupVideoPlayers();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const toggleCaptions = () => {
    setCaptionsEnabled((previous) => !previous);
  };

  const handleProtocolSwitch = (protocol) => {
    if (!streams[protocol]) return;
    setSelectedProtocol(protocol);
  };

  const downloadTrack = (track, format = 'srt') => {
    if (!track) return;
    const href = format === 'vtt' ? track.vtt : track.srt;
    if (!href) return;

    const link = document.createElement('a');
    link.href = href;
    link.download = `${currentFileId || 'subtitles'}_${track.code}.${format}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handlePreviewClick = () => {
    if (!selectedSubtitle && availableSubtitles.length) {
      setSelectedSubtitle(availableSubtitles[0].code);
    }
    setCaptionsEnabled(true);
  };

  const handleDownload = (format) => {
    const track = selectedSubtitle
      ? availableSubtitles.find((item) => item.code === selectedSubtitle)
      : availableSubtitles[0];
    if (!track) return;
    downloadTrack(track, format);
  };

  return (
    <Page>
      <Title>AI Subtitle Generation</Title>
      <Description>
        Upload a video to generate multi-language subtitles with adaptive streaming delivery. Select
        HLS or MPEG-DASH playback, toggle captions on the fly, and download the subtitle files you
        need without re-uploading your media.
      </Description>

      {!selectedFile && (
        <UploadContainer
          className={dragOver ? 'dragover' : ''}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          <UploadIcon>📁</UploadIcon>
          <UploadText>Drop your video file here or click to browse</UploadText>
          <UploadSubtext>Supports MP4, AVI, MOV, MKV, WEBM, FLV (up to 5GB)</UploadSubtext>
          <HiddenInput
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileInputChange}
          />
        </UploadContainer>
      )}

      {selectedFile && (
        <VideoPreview>
          <VideoInfo>
            <VideoName>{selectedFile.name}</VideoName>
            <VideoSize>{formatFileSize(selectedFile.size)}</VideoSize>
          </VideoInfo>
          <FileMeta>
            <FileMetaRow>
              <FileLabel>Type</FileLabel>
              <span>{selectedFile.type || 'Unknown'}</span>
            </FileMetaRow>
            <FileMetaRow>
              <FileLabel>Last modified</FileLabel>
              <span>
                {selectedFile.lastModified
                  ? new Date(selectedFile.lastModified).toLocaleString()
                  : '—'}
              </span>
            </FileMetaRow>
          </FileMeta>
          <LanguageControls>
            <ControlGroup>
              <Label htmlFor="source-language">Source language</Label>
              <Select
                id="source-language"
                value={sourceLanguage}
                onChange={(event) => setSourceLanguage(event.target.value)}
                disabled={processing}
              >
                {TRANSCRIBE_LANGUAGE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </Select>
              <HelperText>
                {sourceLanguage === 'auto'
                  ? 'Let auto-detect choose the spoken language automatically.'
                  : 'For best accuracy, choose the exact dialect that matches your audio.'}
              </HelperText>
            </ControlGroup>

            <ControlGroup>
              <Label>Subtitle languages</Label>
              <LanguageActions>
                <LanguageLink
                  type="button"
                  onClick={selectAllTargetLanguages}
                  disabled={processing}
                >
                  Select all
                </LanguageLink>
                <LanguageLink
                  type="button"
                  onClick={clearTargetLanguages}
                  disabled={processing || selectedTargetLanguages.length === 0}
                >
                  Clear selection
                </LanguageLink>
              </LanguageActions>
              <LanguageCheckboxGrid>
                {TRANSLATE_LANGUAGE_OPTIONS.map((option) => (
                  <LanguageCheckboxLabel key={option.value}>
                    <LanguageCheckbox
                      type="checkbox"
                      checked={selectedTargetLanguages.includes(option.value)}
                      onChange={() => toggleTargetLanguage(option.value)}
                      disabled={processing}
                    />
                    <span>{option.label}</span>
                  </LanguageCheckboxLabel>
                ))}
              </LanguageCheckboxGrid>
              <HelperText>
                Choose the subtitle languages you need. Leave empty to keep the original language
                only.
              </HelperText>
            </ControlGroup>
          </LanguageControls>
          <div style={{ marginTop: '1rem' }}>
            <Button onClick={startProcessing} disabled={processing}>
              {processing ? 'Processing…' : 'Generate Subtitles'}
            </Button>
            <SecondaryButton
              onClick={removeFile}
              disabled={processing}
              style={{ marginLeft: '0.75rem' }}
            >
              Remove File
            </SecondaryButton>
          </div>
        </VideoPreview>
      )}

      {error && <ErrorMessage>{error}</ErrorMessage>}

      {processing && (
        <ProcessingContainer>
          <h3 style={{ margin: '0 0 1rem 0', color: '#ffffff' }}>Processing Video</h3>
          <ProgressBar>
            <ProgressFill $progress={progress} />
          </ProgressBar>
          <div style={{ color: '#e5e5e5', marginBottom: '0.5rem' }}>{progress}% Complete</div>
          <div style={{ color: '#b3b3b3', fontSize: '0.9rem' }}>{progressMessage}</div>

          {(detectedLanguage || requestedTargetLanguages.length) && (
            <LanguageStatus>
              {detectedLanguage && <span>Detected source: {detectedLanguage}</span>}
              {requestedTargetLanguages.length > 0 && (
                <span>
                  Translation requested: {requestedTargetLanguages.join(', ')}
                  {translationApplied ? ' (applied)' : ' (pending)'}
                </span>
              )}
            </LanguageStatus>
          )}

          {!detectedLanguage && !languageDetectionError && sourceLanguage === 'auto' && (
            <HelperText style={{ color: '#ffc107' }}>Detecting spoken language automatically…</HelperText>
          )}

          {languageDetectionError && (
            <HelperText style={{ color: '#ff6b6b' }}>{languageDetectionError}</HelperText>
          )}

          {progress >= 100 && phase !== 'complete' && currentFileId && (
            <div style={{ marginTop: '1rem' }}>
              <div style={{ color: '#b3b3b3', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
                Current File ID: {currentFileId}
              </div>
              <Button onClick={() => fetchSubtitles(currentFileId)} style={{ background: '#28a745' }}>
                Fetch Results
              </Button>
            </div>
          )}
        </ProcessingContainer>
      )}

      {!processing && phase === 'complete' && availableSubtitles.length > 0 && (
        <>
          <SuccessMessage>
            ✅ Subtitles and streams are ready. Use the player below to preview captions in your
            preferred language.
          </SuccessMessage>

          <StreamingPlayerContainer>
            <h3 style={{ margin: '0 0 1rem 0', color: '#ffffff' }}>Streaming Preview</h3>
            <StreamingVideoWrapper>
              <StreamingVideo ref={videoRef} controls crossOrigin="anonymous" poster="">
                {availableSubtitles.map((track) => (
                  <track
                    key={track.code}
                    label={track.label}
                    kind="subtitles"
                    srcLang={track.code}
                    src={track.vtt}
                    default={captionsEnabled && selectedSubtitle === track.code}
                  />
                ))}
              </StreamingVideo>
              <CaptionToggleButton type="button" onClick={toggleCaptions} $active={captionsEnabled}>
                CC {captionsEnabled ? 'On' : 'Off'}
              </CaptionToggleButton>
            </StreamingVideoWrapper>

            <PlayerControls>
              <ControlRow>
                <Label style={{ marginBottom: 0 }}>Protocol</Label>
                {['hls', 'dash']
                  .filter((protocol) => streams[protocol])
                  .map((protocol) => (
                    <ToggleButton
                      key={protocol}
                      onClick={() => handleProtocolSwitch(protocol)}
                      $active={selectedProtocol === protocol}
                    >
                      {protocol.toUpperCase()}
                    </ToggleButton>
                  ))}
              </ControlRow>

              <ControlRow>
                <Label style={{ marginBottom: 0 }}>Subtitle track</Label>
                <Select value={selectedSubtitle} onChange={(event) => setSelectedSubtitle(event.target.value)}>
                  {availableSubtitles.map((track) => (
                    <option key={track.code} value={track.code}>
                      {languageLabelMap.get(track.code.toLowerCase()) || track.label || track.code}
                      {track.isOriginal ? ' (original)' : ''}
                    </option>
                  ))}
                </Select>
              </ControlRow>
            </PlayerControls>
            <CompactActions>
              <CompactButton type="button" onClick={handlePreviewClick} disabled={!availableSubtitles.length}>
                Preview captions
              </CompactButton>
              <CompactButton
                type="button"
                onClick={() => handleDownload('srt')}
                disabled={!availableSubtitles.length}
              >
                Download SRT
              </CompactButton>
              <CompactButton
                type="button"
                onClick={() => handleDownload('vtt')}
                disabled={!availableSubtitles.length}
              >
                Download VTT
              </CompactButton>
            </CompactActions>
          </StreamingPlayerContainer>

          <HelperText style={{ display: 'block', marginTop: '1rem', color: '#b3b3b3' }}>
            Need more languages? Adjust the selections above and run “Generate Subtitles” again—your
            uploaded media is reused, so only new translations are added.
          </HelperText>

          {currentFileId && (
            <HelperText style={{ display: 'block', marginTop: '0.5rem' }}>
              Stream manifests: {streams.hls && (
                <span>
                  HLS <DownloadLink href={resolveUrl(streams.hls.manifest)} target="_blank" rel="noreferrer">master.m3u8</DownloadLink>
                </span>
              )}
              {streams.hls && streams.dash && ' · '}
              {streams.dash && (
                <span>
                  DASH <DownloadLink href={resolveUrl(streams.dash.manifest)} target="_blank" rel="noreferrer">manifest.mpd</DownloadLink>
                </span>
              )}
            </HelperText>
          )}
        </>
      )}
    </Page>
  );
}