import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import styled from 'styled-components';

// Envid Metadata backend (Flask)
// Default to CRA dev proxy (see src/setupProxy.js). Can be overridden via REACT_APP_ENVID_METADATA_BACKEND_URL.
const BACKEND_URL = process.env.REACT_APP_ENVID_METADATA_BACKEND_URL || '/envid-multimodal';
const POLL_INTERVAL_MS = 2000;

const PageWrapper = styled.div`
  min-height: 100vh;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: radial-gradient(1200px circle at 20% 10%, rgba(102, 126, 234, 0.18) 0%, rgba(0, 0, 0, 0) 55%),
    linear-gradient(135deg, #0b1020 0%, #05070f 100%);
  padding: 24px;
  overflow-x: hidden;
`;

const Container = styled.div`
  max-width: 980px;
  margin: 0 auto;
  width: 100%;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 18px;
`;

const Title = styled.h1`
  font-size: 2.1rem;
  font-weight: 900;
  margin: 0;
  color: #e6e8f2;
`;

const Subtitle = styled.p`
  font-size: 1rem;
  margin: 8px 0 0 0;
  color: rgba(230, 232, 242, 0.7);
`;

const Section = styled.div`
  background: rgba(16, 20, 34, 0.72);
  border-radius: 14px;
  padding: 20px;
  box-shadow: 0 14px 44px rgba(0, 0, 0, 0.45);
  border: 1px solid rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(12px);
  max-width: 100%;
`;

const SectionTitle = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 900;
  color: #e6e8f2;
  margin-bottom: 14px;
`;

const Icon = styled.span`
  display: inline-flex;
`;

const Message = styled.div`
  padding: 12px 14px;
  border-radius: 10px;
  margin: 10px 0 18px 0;
  font-weight: 700;
  color: ${(props) => (props.type === 'error' ? '#ffd1d1' : props.type === 'success' ? '#d7ffe7' : '#d8e8ff')};
  background: ${(props) =>
    props.type === 'error'
      ? 'rgba(229, 62, 62, 0.18)'
      : props.type === 'success'
        ? 'rgba(34, 197, 94, 0.16)'
        : 'rgba(56, 189, 248, 0.14)'};
  border: 1px solid
    ${(props) =>
      props.type === 'error'
        ? 'rgba(229, 62, 62, 0.25)'
        : props.type === 'success'
          ? 'rgba(34, 197, 94, 0.22)'
          : 'rgba(56, 189, 248, 0.2)'};
`;

const Row = styled.div`
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
`;

const Button = styled.button`
  background: linear-gradient(90deg, #667eea 0%, #7c3aed 100%);
  color: white;
  border: none;
  padding: 10px 16px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 800;
  box-shadow: 0 10px 24px rgba(102, 126, 234, 0.28);

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const SecondaryButton = styled.button`
  background: rgba(255, 255, 255, 0.06);
  color: #cbd5ff;
  border: 1px solid rgba(102, 126, 234, 0.35);
  padding: 10px 16px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 800;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const DeleteButton = styled.button`
  background: rgba(229, 62, 62, 0.9);
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
  background: rgba(102, 126, 234, 0.92);
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
  border: 2px dashed ${(props) => (props.$dragging ? 'rgba(102, 126, 234, 0.95)' : 'rgba(255, 255, 255, 0.18)')};
  background: ${(props) => (props.$dragging ? 'rgba(102, 126, 234, 0.14)' : 'rgba(255, 255, 255, 0.04)')};
  border-radius: 14px;
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
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  transition: width 0.2s ease;
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
  { id: 'famous_location_detection', label: 'Famous location detection', status: 'not_started', percent: 0, message: null },
  { id: 'translate_output', label: 'Translate everything in English, Arabic,Indonesian', status: 'not_started', percent: 0, message: null },
  { id: 'opening_closing_credit_detection', label: 'Opening/Closing credit detection', status: 'not_started', percent: 0, message: null },
  { id: 'celebrity_detection', label: 'Celebrity detection', status: 'not_started', percent: 0, message: null },
  { id: 'celebrity_bio_image', label: 'Celebrity bio & Image', status: 'not_started', percent: 0, message: null },
  { id: 'save_as_json', label: 'Save as Json', status: 'not_started', percent: 0, message: null },
];

const StatusPanel = styled.div`
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.03);
  border-radius: 14px;
  padding: 14px;
`;

const StatusTitleRow = styled.div`
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
`;

const StatusTitle = styled.div`
  font-weight: 900;
  letter-spacing: 0.2px;
  color: rgba(240, 242, 255, 0.95);
`;

const StatusMeta = styled.div`
  color: rgba(230, 232, 242, 0.7);
  font-size: 12px;
`;

const StatusMetaRight = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
`;

const SystemStatsRow = styled.div`
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin: 8px 0 12px 0;
`;

const SystemStatPill = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  color: rgba(230, 232, 242, 0.9);
  background: rgba(102, 126, 234, 0.16);
  border: 1px solid rgba(102, 126, 234, 0.3);
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
  color: rgba(230, 232, 242, 0.65);
`;

const StepBadge = styled.div`
  font-size: 12px;
  font-weight: 900;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  background: ${(props) => {
    const v = String(props.$variant || 'neutral');
    if (v === 'ok') return 'rgba(16, 185, 129, 0.18)';
    if (v === 'run') return 'rgba(102, 126, 234, 0.18)';
    if (v === 'warn') return 'rgba(245, 158, 11, 0.18)';
    if (v === 'bad') return 'rgba(229, 62, 62, 0.18)';
    return 'rgba(255, 255, 255, 0.06)';
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
  gap: 12px;
  width: 100%;
  max-width: 100%;
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
  width: calc((100% - 36px) / 4);
  max-width: calc((100% - 36px) / 4);
  flex: 0 0 calc((100% - 36px) / 4);
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
`;

const TinyButton = styled.button`
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.06);
  color: #e6e8f2;
  padding: 4px 8px;
  border-radius: 999px;
  cursor: pointer;
  font-weight: 900;
  font-size: 11px;
  line-height: 1;

  &:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
`;

const DetailPanel = styled.div`
  margin-top: 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
  overflow: hidden;
`;

const DetailHeader = styled.div`
  padding: 14px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
`;

const DetailTitle = styled.div`
  color: #e6e8f2;
  font-weight: 900;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const DetailBody = styled.div`
  padding: 16px;
  display: grid;
  gap: 16px;
`;

const Grid2 = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;

  @media (min-width: 980px) {
    grid-template-columns: 1.15fr 0.85fr;
  }
`;

const Card = styled.div`
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  background: rgba(16, 20, 34, 0.55);
  overflow: hidden;
`;

const CardHeader = styled.div`
  padding: 12px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
`;

const CardHeaderTitle = styled.div`
  color: #e6e8f2;
  font-weight: 900;
`;

const CountPill = styled.div`
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: rgba(230, 232, 242, 0.9);
  padding: 4px 8px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 12px;
  line-height: 1;
`;

const MultiColumnList = styled.div`
  column-count: 1;
  column-gap: 14px;

  @media (min-width: 980px) {
    column-count: 2;
  }

  @media (min-width: 1280px) {
    column-count: 3;
  }
`;

const MultiColumnItem = styled.div`
  break-inside: avoid;
  -webkit-column-break-inside: avoid;
  page-break-inside: avoid;
  display: block;
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
`;

const SubtleNote = styled.div`
  margin-top: 10px;
  color: rgba(230, 232, 242, 0.6);
  font-size: 12px;
`;

const CardBody = styled.div`
  padding: 12px 14px;
`;

const VideoFrame = styled.div`
  width: 100%;
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(0, 0, 0, 0.35);
  position: relative;

  &:fullscreen {
    border-radius: 0;
    border: none;
    background: #000;
  }

  &:-webkit-full-screen {
    border-radius: 0;
    border: none;
    background: #000;
  }

  &:fullscreen video,
  &:-webkit-full-screen video {
    width: 100vw;
    height: 100vh;
    max-width: 100vw;
    max-height: 100vh;
    object-fit: contain;
    background: #000;
  }
`;

const VideoOverlayBar = styled.div`
  position: absolute;
  left: 10px;
  right: 10px;
  top: 10px;
  z-index: 5;
  display: flex;
  flex-wrap: nowrap;
  gap: 8px;
  align-items: center;
  pointer-events: none;
`;

const PlayerFullscreenButton = styled.button`
  position: absolute;
  right: 10px;
  top: 10px;
  z-index: 8;
  width: 34px;
  height: 34px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: rgba(0, 0, 0, 0.45);
  color: #e6e8f2;
  cursor: pointer;
  font-weight: 900;
  pointer-events: auto;

  &:hover {
    background: rgba(0, 0, 0, 0.6);
  }
`;

const VideoOverlaySidePanel = styled.div`
  position: absolute;
  /* Exact left edge; start ~30% down from top */
  left: 0;
  top: 30%;
  /* Content-sized panel; cap height so it never covers the control bar */
  max-height: min(420px, calc(70% - 88px));
  width: 240px;
  z-index: 6;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 0;
  border-radius: 0;
  background: transparent;
  border: none;
  pointer-events: none;

  @media (max-width: 520px) {
    width: 200px;
    max-height: min(360px, calc(70% - 104px));
  }
`;

const OverlaySideList = styled.div`
  display: grid;
  gap: 8px;
  overflow-y: ${(props) => (props.$scrollable ? 'auto' : 'hidden')};
  overflow-x: hidden;
  padding: 10px;
  padding-right: 12px;
  -webkit-overflow-scrolling: touch;
  flex: 0 0 auto;
  max-height: ${(props) => (props.$scrollable ? '100%' : 'none')};
  pointer-events: auto;

  &::-webkit-scrollbar {
    width: 8px;
  }
  &::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 999px;
  }
  &::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.18);
    border-radius: 999px;
  }
`;

const OverlaySideItem = styled.div`
  display: grid;
  grid-template-columns: 68px 1fr;
  gap: 10px;
  align-items: center;
  padding: 8px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.10);
`;

const OverlayThumb = styled.div`
  width: 68px;
  height: 68px;
  border-radius: 14px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.14);
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(230, 232, 242, 0.9);
  font-weight: 900;
  font-size: 12px;

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
`;

const OverlayName = styled.div`
  color: #e6e8f2;
  font-weight: 900;
  font-size: 13px;
  line-height: 1.2;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const OverlayChip = styled.div`
  pointer-events: none;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.55);
  border: 1px solid rgba(255, 255, 255, 0.16);
  color: #e6e8f2;
  font-weight: 900;
  font-size: 12px;
  min-width: 0;
`;

const OverlayChipThumb = styled.div`
  width: 20px;
  height: 20px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.35);
  border: 1px solid rgba(255, 255, 255, 0.14);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  color: rgba(230, 232, 242, 0.95);
  font-weight: 900;
  font-size: 10px;

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
`;

const OverlayChipName = styled.div`
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const OverlayScroller = styled.div`
  flex: 1;
  min-width: 0;
  display: flex;
  gap: 8px;
  align-items: center;
  overflow-x: ${(props) => (props.$scrollable ? 'auto' : 'hidden')};
  overflow-y: hidden;
  white-space: nowrap;
  pointer-events: ${(props) => (props.$scrollable ? 'auto' : 'none')};
  -webkit-overflow-scrolling: touch;

  &::-webkit-scrollbar {
    height: ${(props) => (props.$scrollable ? '8px' : '0px')};
  }
  &::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 999px;
  }
  &::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.18);
    border-radius: 999px;
  }
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
`;

const Th = styled.th`
  text-align: left;
  color: rgba(230, 232, 242, 0.75);
  font-weight: 900;
  padding: 10px 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
`;

const Td = styled.td`
  vertical-align: top;
  color: rgba(230, 232, 242, 0.85);
  padding: 10px 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
`;

const LinkA = styled.a`
  color: #cbd5ff;
  font-weight: 900;
  text-decoration: none;

  &:hover {
    text-decoration: underline;
  }
`;

const SegmentChip = styled.button`
  border: 1px solid rgba(102, 126, 234, 0.35);
  background: rgba(102, 126, 234, 0.12);
  color: #e6e8f2;
  padding: 4px 8px;
  border-radius: 999px;
  cursor: pointer;
  font-weight: 900;
  font-size: 11px;
  line-height: 1;
`;

const IconButton = styled.button`
  width: 34px;
  height: 34px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.06);
  color: #e6e8f2;
  cursor: pointer;
  font-weight: 900;

  &:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
`;

function formatBytes(bytes) {
  const b = Number(bytes);
  if (!Number.isFinite(b) || b < 0) return '—';
  if (b < 1024) return `${b} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = b;
  let i = -1;
  do {
    value /= 1024;
    i++;
  } while (value >= 1024 && i < units.length - 1);
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[i]}`;
}

const ModalOverlay = styled.div`
  position: fixed;
  inset: 0;
  background: rgba(2, 4, 12, 0.7);
  backdrop-filter: blur(6px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  z-index: 40;
`;

const ModalCard = styled.div`
  width: min(760px, 100%);
  max-height: 80vh;
  background: rgba(12, 16, 32, 0.95);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const ModalHeader = styled.div`
  padding: 18px 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
`;

const ModalTitle = styled.div`
  font-weight: 900;
  font-size: 1.1rem;
  color: #e6e8f2;
`;

const ModalSubtitle = styled.div`
  color: rgba(230, 232, 242, 0.7);
  font-size: 0.85rem;
`;

const ModalBody = styled.div`
  padding: 18px 20px 22px;
  overflow: auto;
`;

const BrowserPath = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
  color: rgba(230, 232, 242, 0.7);
  font-size: 0.85rem;
  margin-bottom: 12px;

  strong {
    color: #e6e8f2;
  }
`;

const BrowserList = styled.div`
  display: grid;
  gap: 8px;
`;

const BrowserRow = styled.button`
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: center;
  padding: 10px 12px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #e6e8f2;
  cursor: pointer;
  text-align: left;

  &:hover {
    background: rgba(255, 255, 255, 0.06);
  }
`;

const BrowserIcon = styled.span`
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

function buildParagraphsFromTranscriptSegments(segments) {
  const segs = Array.isArray(segments) ? segments : [];
  const out = [];

  let cur = null;
  const flush = () => {
    if (!cur) return;
    const text = String(cur.text || '').replace(/\s+/g, ' ').trim();
    if (text) out.push({ start: cur.start, end: cur.end, text });
    cur = null;
  };

  for (const raw of segs) {
    if (!raw || typeof raw !== 'object') continue;
    const st = Number(raw.start);
    const en = Number(raw.end);
    const txt = String(raw.text || '').replace(/\s+/g, ' ').trim();
    if (!txt) continue;

    const safeSt = Number.isFinite(st) && st >= 0 ? st : null;
    const safeEn = Number.isFinite(en) && en >= 0 ? en : safeSt;

    if (!cur) {
      cur = {
        start: safeSt ?? 0,
        end: safeEn ?? (safeSt ?? 0),
        text: txt,
      };
      continue;
    }

    const gap = safeSt != null ? Math.max(0, safeSt - Number(cur.end || 0)) : 0;
    const nextText = `${cur.text} ${txt}`.trim();

    // Start a new paragraph on larger pauses or if the paragraph becomes too long.
    const tooLong = nextText.length > 700;
    const bigGap = gap >= 1.25;
    if (bigGap || tooLong) {
      flush();
      cur = {
        start: safeSt ?? 0,
        end: safeEn ?? (safeSt ?? 0),
        text: txt,
      };
      continue;
    }

    cur.text = nextText;
    if (safeEn != null) cur.end = Math.max(Number(cur.end || 0), safeEn);
  }
  flush();
  return out;
}


export default function EnvidMetadataMinimal() {
  const fileInputRef = useRef(null);
  const pollRef = useRef(null);
  const historyCarouselRef = useRef(null);
  const playerRef = useRef(null);
  const playerContainerRef = useRef(null);

  const [message, setMessage] = useState(null);

  // Multimodal task selection.
  // Defaults: everything is OFF until the user explicitly enables it.
  const [taskSelection, setTaskSelection] = useState(() => ({
    enable_famous_location_detection: false,
    famous_location_detection_model: 'auto',

    enable_opening_closing_credit_detection: false,
    opening_closing_credit_detection_model: 'auto',

    enable_celebrity_detection: false,
    celebrity_detection_model: 'auto',

    enable_celebrity_bio_image: false,
    celebrity_bio_image_model: 'auto',
  }));

  const [targetTranslateLanguages, setTargetTranslateLanguages] = useState([]);

  const taskSelectionPayload = useMemo(() => {
    const sel = taskSelection || {};
    return {
      // Backend-controlled (scene detection + transcript + Meta Llama).
      enable_scene_by_scene_metadata: true,

      enable_label_detection: true,
      label_detection_model: String(sel.label_detection_model || '').trim() || 'gcp_video_intelligence',

      enable_moderation: true,
      moderation_model: String(sel.moderation_model || '').trim() || 'nsfwjs',

      enable_text: true,
      text_model: String(sel.text_model || '').trim() || 'tesseract',

      enable_key_scene_detection: true,
      key_scene_detection_model: String(sel.key_scene_detection_model || '').trim() || 'pyscenedetect_clip_cluster',

      enable_transcribe: true,
      transcribe_model: String(sel.transcribe_model || '').trim() || 'whisper',

      enable_famous_location_detection: Boolean(sel.enable_famous_location_detection),
      famous_location_detection_model: String(sel.famous_location_detection_model || '').trim() || 'auto',

      // Backend-controlled (Meta Llama).
      enable_synopsis_generation: true,
      synopsis_generation_model: 'auto',

      enable_opening_closing_credit_detection: Boolean(sel.enable_opening_closing_credit_detection),
      opening_closing_credit_detection_model: String(sel.opening_closing_credit_detection_model || '').trim() || 'auto',

      enable_celebrity_detection: Boolean(sel.enable_celebrity_detection),
      celebrity_detection_model: String(sel.celebrity_detection_model || '').trim() || 'auto',

      enable_celebrity_bio_image: Boolean(sel.enable_celebrity_bio_image),
      celebrity_bio_image_model: String(sel.celebrity_bio_image_model || '').trim() || 'auto',

      translate_targets: Array.isArray(targetTranslateLanguages) ? targetTranslateLanguages : [],
    };
  }, [taskSelection, targetTranslateLanguages]);

  const [videoSource, setVideoSource] = useState('gcs'); // 'local' | 'gcs'
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const [translateLanguageOptions, setTranslateLanguageOptions] = useState([]);
  const [translateLanguagesLoading, setTranslateLanguagesLoading] = useState(false);
  const [translateLanguagesError, setTranslateLanguagesError] = useState('');

  const [gcsRawVideoObject, setGcsRawVideoObject] = useState('');
  const [gcsRawVideoLoading, setGcsRawVideoLoading] = useState(false);
  const [gcsBrowserOpen, setGcsBrowserOpen] = useState(false);
  const [gcsBrowserPrefix, setGcsBrowserPrefix] = useState('');
  const [gcsBrowserQuery, setGcsBrowserQuery] = useState('');
  const [gcsBrowserObjects, setGcsBrowserObjects] = useState([]);
  const [gcsBrowserPrefixes, setGcsBrowserPrefixes] = useState([]);
  const [deletePending, setDeletePending] = useState(null);
  const [reprocessPending, setReprocessPending] = useState(null);
  const [gcsBuckets, setGcsBuckets] = useState([]);
  const [gcsBucket, setGcsBucket] = useState('');
  const [gcsBucketLoading, setGcsBucketLoading] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [clientUploadProgress, setClientUploadProgress] = useState(0);
  const [uploadJob, setUploadJob] = useState(null);
  const [activeJob, setActiveJob] = useState(null); // { kind: 'upload'|'reprocess', jobId: string, videoId?: string }
  const [systemStats, setSystemStats] = useState(null);

  const [allVideos, setAllVideos] = useState([]);

  const [selectedVideoId, setSelectedVideoId] = useState(null);
  const [selectedVideoTitle, setSelectedVideoTitle] = useState(null);
  const [selectedMeta, setSelectedMeta] = useState(null);
  const [selectedMetaLoading, setSelectedMetaLoading] = useState(false);
  const [selectedMetaError, setSelectedMetaError] = useState(null);
  const [playerTime, setPlayerTime] = useState(0);
  const [playerIsPaused, setPlayerIsPaused] = useState(false);
  const [playerIsFullscreen, setPlayerIsFullscreen] = useState(false);
  const [isSmallScreen, setIsSmallScreen] = useState(false);

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
        const resp = await axios.get(`${BACKEND_URL}/translate/languages`);
        if (cancelled) return;
        const langs = Array.isArray(resp.data?.languages) ? resp.data.languages : [];
        const normalized = langs
          .map((lang) => {
            const code = String(lang?.code || '').trim();
            const name = String(lang?.name || '').trim();
            if (!code) return null;
            return { code, name: name || code };
          })
          .filter(Boolean);
        setTranslateLanguageOptions(normalized);
      } catch (e) {
        if (!cancelled) {
          setTranslateLanguagesError('Failed to load LibreTranslate languages.');
          setTranslateLanguageOptions([]);
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
  const synopsesByAge = useMemo(() => selectedCategories?.synopses_by_age_group || {}, [selectedCategories]);
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
  const transcriptParagraphs = useMemo(() => buildParagraphsFromTranscriptSegments(transcriptSegments), [transcriptSegments]);
  const rawTranscriptText = useMemo(() => {
    const t = (transcriptText || '').toString();
    if (t.trim()) return t;
    if (Array.isArray(transcriptSegments) && transcriptSegments.length) {
      return transcriptSegments
        .map((s) => (s?.text || '').toString().trim())
        .filter(Boolean)
        .join(' ');
    }
    return '';
  }, [transcriptText, transcriptSegments]);

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

  const loadAllVideos = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/videos`);
      const videos = Array.isArray(response.data?.videos) ? response.data.videos : [];
      setAllVideos(videos);

      if (!videos.length) {
        if (selectedVideoId) {
          setSelectedVideoId(null);
          setSelectedVideoTitle(null);
        }
        return;
      }

      const hasSelected = videos.some((video) => {
        const videoId = video?.video_id || video?.id;
        return videoId && selectedVideoId && String(videoId) === String(selectedVideoId);
      });

      if (!hasSelected) {
        const latest = [...videos].sort((a, b) => {
          const aTime = a?.uploaded_at ? Date.parse(a.uploaded_at) : 0;
          const bTime = b?.uploaded_at ? Date.parse(b.uploaded_at) : 0;
          return (Number.isFinite(bTime) ? bTime : 0) - (Number.isFinite(aTime) ? aTime : 0);
        })[0];
        const latestId = latest?.video_id || latest?.id;
        if (latestId) {
          setSelectedVideoId(String(latestId));
          setSelectedVideoTitle(latest?.title || latest?.filename || latest?.name || null);
        }
      }
    } catch (error) {
      // Non-fatal.
    }
  };

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
    loadAllVideos();
  }, []);

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

          await loadAllVideos();

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

  const handleUpload = async () => {
    setMessage(null);

    if (!Array.isArray(targetTranslateLanguages) || targetTranslateLanguages.length === 0) {
      setMessage({ type: 'error', text: 'Select at least one target translation language before analyzing.' });
      return;
    }

    let handedOffToPoller = false;

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
          if (response.data?.job) setUploadJob(response.data.job);
          setActiveJob({ kind: 'upload', jobId: response.data.job_id });
          handedOffToPoller = true;
          await pollJob(response.data.job_id);
          return;
        }
        setMessage({ type: 'error', text: response.data?.error || 'Failed to start Cloud Storage processing' });
      } catch (e) {
        setMessage({ type: 'error', text: e.response?.data?.error || 'Failed to start Cloud Storage processing' });
      } finally {
        if (!handedOffToPoller) {
          setUploading(false);
          setActiveJob(null);
        }
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
        if (response.data?.job) setUploadJob(response.data.job);
        setActiveJob({ kind: 'upload', jobId: response.data.job_id });
        handedOffToPoller = true;
        await pollJob(response.data.job_id);
        return;
      }

      setMessage({ type: 'success', text: response.data?.message || 'Uploaded.' });
      await loadAllVideos();
    } catch (e) {
      setMessage({ type: 'error', text: e.response?.data?.error || 'Failed to upload video' });
    } finally {
      if (!handedOffToPoller) {
        setUploading(false);
        setClientUploadProgress(0);
        setUploadProgress(0);
        setUploadJob(null);
        setActiveJob(null);
      }
    }
  };

  const pipelineSteps = useMemo(() => {
    const steps = Array.isArray(uploadJob?.steps) ? uploadJob.steps : [];
    if (steps.length) {
      return steps.filter((s) => {
        if (!s || typeof s !== 'object') return false;
        const id = String(s.id || '').toLowerCase();
        const label = String(s.label || '').toLowerCase();
        return id !== 'overall' && label !== 'overall';
      });
    }

    const hasSelectedInput =
      videoSource === 'local' ? Boolean(selectedFile) : Boolean(String(gcsRawVideoObject || '').trim());

    return hasSelectedInput ? DEFAULT_PIPELINE_STEPS : [];
  }, [uploadJob, videoSource, selectedFile, gcsRawVideoObject]);

  const statusBadgeFor = (rawStatus) => {
    const s = String(rawStatus || '').toLowerCase();
    if (s === 'completed') return { text: 'Completed', variant: 'ok' };
    if (s === 'running' || s === 'processing') return { text: 'Started', variant: 'run' };
    if (s === 'failed') return { text: 'Failed', variant: 'bad' };
    if (s === 'skipped') return { text: 'Skipped', variant: 'warn' };
    return { text: 'Not started', variant: 'neutral' };
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

  const subtitleLanguageLabels = useMemo(() => {
    const map = new Map();
    translateLanguageOptions.forEach((lang) => {
      if (lang?.code) map.set(String(lang.code).toLowerCase(), lang.name || lang.code);
    });
    map.set('orig', 'Original');
    map.set('en', 'English');
    return map;
  }, [translateLanguageOptions]);

  const subtitleLanguages = useMemo(() => {
    const langs = new Set();
    langs.add('orig');
    const translationBlock =
      selectedMeta?.combined?.translations || selectedMeta?.translations || selectedMeta?.categories?.translations || null;
    if (translationBlock?.languages && Array.isArray(translationBlock.languages)) {
      translationBlock.languages.forEach((lang) => {
        if (lang) langs.add(String(lang).toLowerCase());
      });
    }
    const lc = String(selectedMeta?.combined?.transcript?.language_code || selectedMeta?.language_code || '').toLowerCase();
    if (lc) {
      langs.add(lc);
    }
    langs.add('en');
    return Array.from(langs);
  }, [selectedMeta]);

  const metadataLanguages = useMemo(() => {
    const ordered = [];
    const add = (lang) => {
      const code = String(lang || '').trim().toLowerCase();
      if (!code) return;
      const normalized = code === 'original' ? 'orig' : code;
      if (!ordered.includes(normalized)) ordered.push(normalized);
    };

    add('orig');
    add('en');

    const taskTargets =
      selectedMeta?.task_selection_effective?.translate_targets ||
      selectedMeta?.task_selection?.translate_targets ||
      selectedMeta?.task_selection_requested?.translate_targets ||
      [];
    if (Array.isArray(taskTargets)) {
      taskTargets.forEach((lang) => add(lang));
    }

    const translationBlock =
      selectedMeta?.combined?.translations || selectedMeta?.translations || selectedMeta?.categories?.translations || null;
    if (translationBlock?.languages && Array.isArray(translationBlock.languages)) {
      translationBlock.languages.forEach((lang) => add(lang));
    }

    const lc = String(selectedMeta?.combined?.transcript?.language_code || selectedMeta?.language_code || '').toLowerCase();
    if (lc) add(lc);

    return ordered;
  }, [selectedMeta]);

  const serverUploadStatus = (() => {
    if (videoSource !== 'local') return null;
    if (!hasSelectedInput) return null;
    if (!uploading) return { status: 'not_started', percent: 0 };
    if (clientUploadProgress >= 100) return { status: 'completed', percent: 100 };
    if (clientUploadProgress > 0) return { status: 'running', percent: clientUploadProgress };
    return { status: 'running', percent: 0 };
  })();

  const statusPanel = showStatusPanel ? (
    <div style={{ marginTop: 12 }}>
      <StatusPanel>
        <StatusTitleRow>
          <StatusTitle>Processing Status</StatusTitle>
          <StatusMetaRight>
            <StatusMeta>
              {activeJob?.kind ? `${activeJob.kind} job` : 'job'}
              {activeJob?.jobId ? ` • ${String(activeJob.jobId).slice(0, 8)}…` : ''}
            </StatusMeta>
            {(activeJob?.jobId || uploading) && (
              <SystemStatsRow>
                <SystemStatPill>VM CPU: {cpuPercentLabel}</SystemStatPill>
                <SystemStatPill>VM GPU: {gpuPercentLabel}</SystemStatPill>
              </SystemStatsRow>
            )}
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
            {pipelineSteps.map((step) => {
              const id = String(step?.id || '');
              const label = String(step?.label || id || 'Step');
              const pct = typeof step?.percent === 'number' ? Math.max(0, Math.min(100, step.percent)) : null;
              const badge = statusBadgeFor(step?.status);
              const hint = String(step?.message || '').trim();
              const hintText = (() => {
                if (pct !== null && pct > 0) return `${pct}%`;
                if (hint) return hint;
                if (pct !== null) return `${pct}%`;
                return '';
              })();

              return (
                <StepRow key={id || label}>
                  <StepLeft>
                    <StepLabelRow>
                      <StepLabel>{label}</StepLabel>
                      <StepHint>{hintText}</StepHint>
                    </StepLabelRow>
                    <ProgressBar>
                      <ProgressFill
                        $percent={
                          pct !== null
                            ? pct
                            : String(step?.status || '').toLowerCase() === 'completed'
                              ? 100
                              : 0
                        }
                      />
                    </ProgressBar>
                  </StepLeft>
                  <StepBadge $variant={badge.variant}>{badge.text}</StepBadge>
                </StepRow>
              );
            })}
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

  const handleDeleteVideo = async (videoId, videoTitle) => {
    if (!videoId) return;
    setDeletePending({ videoId, videoTitle });
  };

  const confirmDeleteVideo = async () => {
    if (!deletePending) return;
    const { videoId, videoTitle } = deletePending;
    setDeletePending(null);
    setMessage(null);
    try {
      const resp = await axios.delete(`${BACKEND_URL}/video/${videoId}`);
      const baseMsg = resp.data?.message || `Deleted "${videoTitle}".`;
      const warningsRaw =
        resp.data?.storage_warnings ?? resp.data?.gcs_warnings ?? resp.data?.s3_warnings ?? resp.data?.warnings ?? [];
      const warnings = Array.isArray(warningsRaw) ? warningsRaw.filter(Boolean) : [];
      if (warnings.length) {
        setMessage({
          type: 'info',
          text: `${baseMsg} (Storage warnings: ${warnings.slice(0, 4).join(' | ')}${warnings.length > 4 ? ' | …' : ''})`,
        });
      } else {
        setMessage({ type: 'success', text: baseMsg });
      }
      await loadAllVideos();
    } catch (e) {
      setMessage({ type: 'error', text: e.response?.data?.error || 'Failed to delete video' });
    }
  };

  const handleReprocessVideo = async (videoId, videoTitle) => {
    if (!videoId) return;
    setReprocessPending({ videoId, videoTitle });
  };

  const confirmReprocessVideo = async () => {
    if (!reprocessPending) return;
    const { videoId, videoTitle } = reprocessPending;
    setReprocessPending(null);
    setMessage(null);
    setUploading(true);
    setUploadProgress(0);
    setClientUploadProgress(0);
    setUploadJob(null);
    setMessage({ type: 'info', text: 'Reprocessing started…' });

    let handedOffToPoller = false;
    try {
      const resp = await axios.post(`${BACKEND_URL}/video/${videoId}/reprocess`, { task_selection: taskSelectionPayload });
      if (resp.status === 202 && resp.data?.job_id) {
        setUploadProgress(5);
        if (resp.data?.job) setUploadJob(resp.data.job);
        setActiveJob({ kind: 'reprocess', videoId: String(videoId), jobId: resp.data.job_id });
        handedOffToPoller = true;
        await pollJob(resp.data.job_id);
        return;
      }
      setMessage({ type: 'success', text: resp.data?.message || `Reprocess requested for "${videoTitle}".` });
      await loadAllVideos();
    } catch (e) {
      setMessage({ type: 'error', text: e.response?.data?.error || `Failed to start reprocess for "${videoTitle}"` });
    } finally {
      if (!handedOffToPoller) {
        setUploading(false);
        setUploadProgress(0);
        setUploadJob(null);
        setActiveJob(null);
      }
    }
  };

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

  return (
    <PageWrapper>
      <Container>
        <Header>
          <Title>Envid Metadata (Multimodal)</Title>
          <Subtitle>Upload videos or analyze a GCS object (any folder)</Subtitle>
        </Header>

        {message && <Message type={message.type}>{message.text}</Message>}

        <div style={{ display: 'grid', gap: 18 }}>
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
              <div style={{ fontWeight: 900, color: 'rgba(240, 242, 255, 0.95)', marginBottom: 10 }}>Target Translation</div>
              <div style={{ display: 'grid', gap: 8, marginBottom: 12 }}>
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
                    <span>Select target languages ({targetTranslateLanguages.length} selected)</span>
                    <span style={{ opacity: 0.7 }}>▾</span>
                  </summary>
                  <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
                    <Row style={{ gap: 8 }}>
                      <SecondaryButton
                        type="button"
                        onClick={() => {
                          const all = translateLanguageOptions.map((lang) => String(lang.code).toLowerCase());
                          setTargetTranslateLanguages(all);
                        }}
                        disabled={uploading || translateLanguagesLoading}
                      >
                        Select All
                      </SecondaryButton>
                      <SecondaryButton
                        type="button"
                        onClick={() => setTargetTranslateLanguages([])}
                        disabled={uploading || translateLanguagesLoading}
                      >
                        Clear
                      </SecondaryButton>
                    </Row>
                    <div style={{ display: 'grid', gap: 6, maxHeight: 320, overflowY: 'auto' }}>
                      {translateLanguageOptions.map((lang) => {
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
                <div style={{ fontSize: 12, color: 'rgba(230, 232, 242, 0.65)' }}>
                  Original and English are included by default. Choose one or more target languages for translation.
                </div>
                {translateLanguagesError ? (
                  <div style={{ fontSize: 12, color: '#ff6b6b' }}>{translateLanguagesError}</div>
                ) : null}
              </div>
            </div>

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
                {[
                  {
                    enableKey: 'enable_famous_location_detection',
                    label: 'Famous location detection',
                    modelKey: 'famous_location_detection_model',
                    models: [
                      { value: 'auto', label: 'Auto (backend default)' },
                      { value: 'gcp_language', label: 'Google Cloud Natural Language' },
                    ],
                  },
                  {
                    enableKey: 'enable_opening_closing_credit_detection',
                    label: 'Opening and closing credit',
                    modelKey: 'opening_closing_credit_detection_model',
                    models: [
                      { value: 'auto', label: 'Auto (backend default)' },
                      { value: 'ffmpeg_blackdetect', label: 'FFmpeg blackdetect' },
                      { value: 'pyscenedetect', label: 'PySceneDetect' },
                    ],
                  },
                  {
                    enableKey: 'enable_celebrity_detection',
                    label: 'Celebrity detection',
                    modelKey: 'celebrity_detection_model',
                    models: [{ value: 'auto', label: 'Auto (backend default)' }],
                  },
                  {
                    enableKey: 'enable_celebrity_bio_image',
                    label: 'Celebrity bio & image',
                    modelKey: 'celebrity_bio_image_model',
                    models: [{ value: 'auto', label: 'Auto (backend default)' }],
                  },
                ].map((t) => {
                  const defaultModelsByKey = {
                    famous_location_detection_model: 'auto',
                    opening_closing_credit_detection_model: 'auto',
                    celebrity_detection_model: 'auto',
                    celebrity_bio_image_model: 'auto',
                  };

                  const checked = Boolean(taskSelection?.[t.enableKey]);
                  const showModel = Boolean(t.modelKey) && Array.isArray(t.models);
                  const modelValue = showModel ? String(taskSelection?.[t.modelKey] || '') : '';

                  return (
                    <div
                      key={t.enableKey}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: showModel ? 'minmax(220px, 1.2fr) minmax(220px, 1fr)' : '1fr',
                        gap: 10,
                        alignItems: 'center',
                      }}
                    >
                      <div style={{ display: 'grid', gap: 4 }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 10, fontWeight: 800, color: '#e6e8f2' }}>
                          <input
                            type="checkbox"
                            checked={checked}
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

                      {showModel ? (
                        <select
                          value={modelValue}
                          onChange={(e) => {
                            const v = e.target.value;
                            setTaskSelection((prev) => ({ ...(prev || {}), [t.modelKey]: v }));
                          }}
                          disabled={!checked}
                          style={{
                            padding: '10px 12px',
                            border: '1px solid rgba(255, 255, 255, 0.12)',
                            borderRadius: 10,
                            fontSize: 14,
                            width: '100%',
                            background: 'rgba(255, 255, 255, 0.04)',
                            color: '#e6e8f2',
                          }}
                        >
                          {t.models.map((m) => (
                            <option key={m.value} value={m.value}>
                              {String(m.label)}
                              {t.modelKey && t.modelKey !== 'transcribe_model' && defaultModelsByKey[t.modelKey] === m.value
                                ? ' (Default)'
                                : ''}
                            </option>
                          ))}
                        </select>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{ marginTop: 14, display: 'grid', gap: 10 }}>
              <Button type="button" onClick={handleUpload} disabled={uploading}>
                {uploadPhaseLabel}
              </Button>

              {uploading && (
                <>
                  <ProgressBar>
                    <ProgressFill $percent={uploadProgress} />
                  </ProgressBar>

                  {statusPanel}
                </>
              )}
            </div>
          </Section>

          <Section>
            <SectionTitle>
              <Icon>🕘</Icon>
              History
            </SectionTitle>

            {activeJob?.jobId && activeJob?.kind === 'reprocess' && statusPanel}

            <Row style={{ justifyContent: 'space-between', marginBottom: 12 }}>
              <div style={{ color: 'rgba(230, 232, 242, 0.7)', fontSize: 12 }}>
                {allVideos.length ? `${allVideos.length} video(s)` : 'No videos yet'}
              </div>
              <Row style={{ gap: 8 }}>
                <IconButton
                  type="button"
                  aria-label="Scroll history left"
                  onClick={() => {
                    const width = historyCarouselRef.current?.clientWidth || 520;
                    historyCarouselRef.current?.scrollBy?.({ left: -width, behavior: 'smooth' });
                  }}
                  disabled={!allVideos.length}
                >
                  ‹
                </IconButton>
                <IconButton
                  type="button"
                  aria-label="Scroll history right"
                  onClick={() => {
                    const width = historyCarouselRef.current?.clientWidth || 520;
                    historyCarouselRef.current?.scrollBy?.({ left: width, behavior: 'smooth' });
                  }}
                  disabled={!allVideos.length}
                >
                  ›
                </IconButton>
                <SecondaryButton type="button" onClick={loadAllVideos} disabled={uploading}>
                  Refresh
                </SecondaryButton>
              </Row>
            </Row>

            {allVideos.length > 0 ? (
              <Carousel ref={historyCarouselRef}>
                {allVideos
                  .slice()
                  .sort((a, b) => new Date(b?.uploaded_at || 0).getTime() - new Date(a?.uploaded_at || 0).getTime())
                  .map((video) => {
                    const videoId = video?.video_id || video?.id;
                    const title = video?.title || videoId || 'Untitled';
                    const status = video?.status || video?.processing_status || video?.state;
                    const thumbSrc = asJpegDataUri(video?.thumbnail);
                    const frameCount = Number(video?.frame_count ?? video?.frameCount ?? 0) || 0;
                    const taskSummary = formatTaskDurationSummary(video?.task_durations);

                    return (
                      <CarouselItem
                        key={videoId || title}
                        $active={Boolean(videoId && selectedVideoId && String(videoId) === String(selectedVideoId))}
                        role="button"
                        tabIndex={0}
                        aria-pressed={Boolean(videoId && selectedVideoId && String(videoId) === String(selectedVideoId))}
                        onClick={() => {
                          if (!videoId) return;
                          setSelectedVideoId(String(videoId));
                          setSelectedVideoTitle(title);
                          // bring detail panel into view
                          setTimeout(() => {
                            document.getElementById('envid-metadata-details')?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
                          }, 0);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            if (!videoId) return;
                            setSelectedVideoId(String(videoId));
                            setSelectedVideoTitle(title);
                            setTimeout(() => {
                              document.getElementById('envid-metadata-details')?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
                            }, 0);
                          }
                        }}
                      >
                        <CarouselThumb>
                          {thumbSrc ? (
                            <CarouselThumbImg src={thumbSrc} alt={title} />
                          ) : (
                            <div style={{ color: 'rgba(230, 232, 242, 0.65)', fontWeight: 800 }}>No thumbnail</div>
                          )}
                        </CarouselThumb>
                        <CarouselBody>
                          <CarouselTitle title={title}>{title}</CarouselTitle>
                          <CarouselMeta>
                            <Row style={{ gap: 8, alignItems: 'center' }}>
                              <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>{videoId ? `ID: ${videoId}` : 'ID: —'}</div>
                              {videoId ? (
                                <TinyButton
                                  type="button"
                                  aria-label="Copy video id"
                                  disabled={uploading}
                                  onClick={async () => {
                                    const ok = await copyToClipboard(videoId);
                                    setMessage({
                                      type: ok ? 'success' : 'error',
                                      text: ok ? 'Copied video ID to clipboard.' : 'Failed to copy video ID.',
                                    });
                                  }}
                                >
                                  Copy
                                </TinyButton>
                              ) : null}
                            </Row>
                            {taskSummary ? (
                              <>
                                <br />
                                <span style={{ color: 'rgba(230, 232, 242, 0.7)' }}>{taskSummary}</span>
                              </>
                            ) : null}
                            <br />
                            {status ? `Status: ${status}` : 'Status: —'}
                            {activeJob?.kind === 'reprocess' && String(activeJob?.videoId) === String(videoId) ? (
                              <>
                                <br />
                                <span style={{ color: 'rgba(230, 232, 242, 0.75)', fontWeight: 900 }}>
                                  Reprocessing: {typeof uploadJob?.progress === 'number' ? `${Math.round(uploadJob.progress)}%` : '…'}
                                  {uploadJob?.message ? ` · ${uploadJob.message}` : ''}
                                </span>
                              </>
                            ) : null}
                            <br />
                            Uploaded: {formatTimestamp(video?.uploaded_at)}
                            {frameCount ? ` · Frames: ${frameCount}` : ''}
                          </CarouselMeta>
                        </CarouselBody>

                        {videoId ? (
                          <div style={{ position: 'absolute', top: 10, right: 10 }}>
                            <div style={{ display: 'flex', gap: 8 }}>
                              <ReprocessButton
                                type="button"
                                disabled={uploading}
                                onClick={(e) => {
                                  e.preventDefault();
                                  e.stopPropagation();
                                  handleReprocessVideo(videoId, title);
                                }}
                              >
                                Reprocess
                              </ReprocessButton>
                              <DeleteButton
                                type="button"
                                disabled={uploading}
                                onClick={(e) => {
                                  e.preventDefault();
                                  e.stopPropagation();
                                  handleDeleteVideo(videoId, title);
                                }}
                              >
                                Delete
                              </DeleteButton>
                            </div>
                          </div>
                        ) : null}
                      </CarouselItem>
                    );
                  })}
              </Carousel>
            ) : (
              <EmptyState>
                <EmptyIcon>🎬</EmptyIcon>
                <div style={{ fontWeight: 800, color: '#e6e8f2' }}>No videos indexed yet</div>
                <div style={{ marginTop: 6 }}>Upload a video to start extracting metadata.</div>
              </EmptyState>
            )}

            {selectedVideoId ? (
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
                      onClick={() =>
                        window.open(
                          `${BACKEND_URL}/video/${selectedVideoId}/metadata-json?lang=orig&download=1`,
                          '_blank',
                          'noopener,noreferrer'
                        )
                      }
                      disabled={selectedMetaLoading}
                    >
                      Download Full Metadata (Original)
                    </SecondaryButton>
                    <SecondaryButton
                      type="button"
                      onClick={() => {
                        setSelectedVideoId(null);
                        setSelectedVideoTitle(null);
                        setSelectedMeta(null);
                        setSelectedMetaError(null);
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
                                  style={{ width: '100%', display: 'block', position: 'relative', zIndex: 1 }}
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

                        <div>
                          <Card>
                            <CardHeader>
                              <CardHeaderTitle>Downloads</CardHeaderTitle>
                            </CardHeader>
                            <CardBody>
                              <div style={{ display: 'grid', gap: 10 }}>
                                <div>
                                  <div style={{ color: 'rgba(230, 232, 242, 0.75)', fontWeight: 900, marginBottom: 6 }}>Subtitles</div>
                                  <div style={{ display: 'grid', gap: 8 }}>
                                    {subtitleLanguages.map((lang) => {
                                      const label = subtitleLanguageLabels.get(lang) || lang.toUpperCase();
                                      const isOrig = lang === 'orig';
                                      const srtLabel = isOrig ? '.srt' : `.${lang}.srt`;
                                      const vttLabel = isOrig ? '.vtt' : `.${lang}.vtt`;
                                      return (
                                        <Row key={lang} style={{ gap: 10, flexWrap: 'wrap' }}>
                                          <div style={{ color: 'rgba(230, 232, 242, 0.7)', fontWeight: 800 }}>{label}</div>
                                          <LinkA
                                            href={`${BACKEND_URL}/video/${selectedVideoId}/subtitles/${lang}.srt`}
                                            target="_blank"
                                            rel="noreferrer"
                                          >
                                            Download {srtLabel}
                                          </LinkA>
                                          <LinkA
                                            href={`${BACKEND_URL}/video/${selectedVideoId}/subtitles/${lang}.vtt`}
                                            target="_blank"
                                            rel="noreferrer"
                                          >
                                            Download {vttLabel}
                                          </LinkA>
                                        </Row>
                                      );
                                    })}
                                  </div>
                                </div>

                                <div>
                                  <div style={{ color: 'rgba(230, 232, 242, 0.75)', fontWeight: 900, marginBottom: 6 }}>Metadata</div>
                                  <div style={{ display: 'grid', gap: 8 }}>
                                    {metadataLanguages.map((lang) => {
                                      const label = subtitleLanguageLabels.get(lang) || lang.toUpperCase();
                                      const suffix = lang === 'orig' ? '' : ` (${lang.toUpperCase()})`;
                                      return (
                                        <Row key={lang} style={{ gap: 10, flexWrap: 'wrap' }}>
                                          <div style={{ color: 'rgba(230, 232, 242, 0.7)', fontWeight: 800 }}>{label}</div>
                                          <LinkA
                                            href={`${BACKEND_URL}/video/${selectedVideoId}/metadata-json?lang=${lang}&download=1`}
                                            target="_blank"
                                            rel="noreferrer"
                                          >
                                            Download Full Metadata JSON{suffix}
                                          </LinkA>
                                        </Row>
                                      );
                                    })}
                                  </div>
                                </div>
                              </div>
                            </CardBody>
                          </Card>

                          <Card style={{ marginTop: 16 }}>
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

                      <div style={{ display: 'grid', gap: 16 }}>
                        <div style={{ display: 'grid', gap: 16 }}>
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

                          <Card>
                            <CardHeader>
                              <CardHeaderTitle>Detected Content</CardHeaderTitle>
                              <CountPill>{Array.isArray(detectedContent?.labels) ? detectedContent.labels.length : 0}</CountPill>
                            </CardHeader>
                            <CardBody>
                              <div style={{ color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                                {Array.isArray(detectedContent?.labels) && detectedContent.labels.length ? (
                                  <MultiColumnList>
                                    {detectedContent.labels.slice(0, 60).map((l, idx) => {
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
                                <SubtleNote>Showing first 60 labels.</SubtleNote>
                              ) : null}
                            </CardBody>
                          </Card>

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
                        </div>
                      </div>

                      <Card>
                        <CardHeader>
                          <CardHeaderTitle>Synopses (by age group)</CardHeaderTitle>
                        </CardHeader>
                        <CardBody>
                          {synopsesByAge && typeof synopsesByAge === 'object' && Object.keys(synopsesByAge).length ? (
                            <div style={{ display: 'grid', gap: 10 }}>
                              {Object.entries(synopsesByAge).map(([group, s]) => (
                                <div
                                  key={group}
                                  style={{
                                    border: '1px solid rgba(255, 255, 255, 0.08)',
                                    background: 'rgba(255, 255, 255, 0.03)',
                                    borderRadius: 12,
                                    padding: 12,
                                  }}
                                >
                                  <div style={{ fontWeight: 900, color: '#e6e8f2' }}>{group}</div>
                                  <div style={{ marginTop: 6, color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                                    <div style={{ fontWeight: 900, color: 'rgba(230,232,242,0.75)', marginBottom: 4 }}>Short</div>
                                    <div>{(s?.short || '').toString().trim() || '—'}</div>
                                    <div style={{ fontWeight: 900, color: 'rgba(230,232,242,0.75)', margin: '10px 0 4px' }}>Long</div>
                                    <div>{(s?.long || '').toString().trim() || '—'}</div>
                                  </div>
                                </div>
                              ))}
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

                      <Grid2>
                        <Card>
                          <CardHeader>
                            <CardHeaderTitle>High Points</CardHeaderTitle>
                          </CardHeader>
                          <CardBody>
                            {Array.isArray(highPoints) && highPoints.length ? (
                              <div style={{ display: 'grid', gap: 8 }}>
                                {highPoints.slice(0, 100).map((hp, idx) => (
                                  <div key={`hp-${idx}`} style={{ color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                                    {hp?.start_seconds != null || hp?.end_seconds != null ? (
                                      <SegmentChip type="button" onClick={() => seekTo(hp?.start_seconds)}>
                                        {formatSeconds(hp?.start_seconds)}–{formatSeconds(hp?.end_seconds)}
                                      </SegmentChip>
                                    ) : null}{' '}
                                    {hp?.text || hp?.summary || hp?.reason || (typeof hp === 'string' ? hp : '')}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>—</div>
                            )}
                          </CardBody>
                        </Card>

                        <Card>
                          <CardHeader>
                            <CardHeaderTitle>Key Scenes</CardHeaderTitle>
                          </CardHeader>
                          <CardBody>
                            {Array.isArray(keyScenes) && keyScenes.length ? (
                              <div style={{ display: 'grid', gap: 8 }}>
                                {keyScenes.slice(0, 100).map((ks, idx) => (
                                  <div key={`ks-${idx}`} style={{ color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>
                                    {ks?.start_seconds != null || ks?.end_seconds != null ? (
                                      <SegmentChip type="button" onClick={() => seekTo(ks?.start_seconds)}>
                                        {formatSeconds(ks?.start_seconds)}–{formatSeconds(ks?.end_seconds)}
                                      </SegmentChip>
                                    ) : ks?.scene_index != null ? (
                                      <SegmentChip type="button" onClick={() => {
                                        const match = Array.isArray(sceneByScene)
                                          ? sceneByScene.find((s) => String(s?.scene_index) === String(ks?.scene_index))
                                          : null;
                                        if (match?.start_seconds != null) seekTo(match.start_seconds);
                                      }}>
                                        Scene {ks.scene_index}
                                      </SegmentChip>
                                    ) : null}{' '}
                                    {ks?.key_reason || ks?.reason || ks?.text || (typeof ks === 'string' ? ks : '')}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>—</div>
                            )}
                          </CardBody>
                        </Card>
                      </Grid2>

                      <Card>
                        <CardHeader>
                          <CardHeaderTitle>Scene-by-scene metadata (timestamp mapped)</CardHeaderTitle>
                        </CardHeader>
                        <CardBody>
                          {Array.isArray(sceneByScene) && sceneByScene.length ? (
                            <div style={{ display: 'grid', gap: 10 }}>
                              {sceneByScene.slice(0, 200).map((s, idx) => (
                                <div
                                  key={`sc-${idx}`}
                                  style={{
                                    border: '1px solid rgba(255, 255, 255, 0.08)',
                                    background: 'rgba(255, 255, 255, 0.03)',
                                    borderRadius: 12,
                                    padding: 12,
                                  }}
                                >
                                  <Row style={{ justifyContent: 'space-between', gap: 10 }}>
                                    <div style={{ fontWeight: 900, color: '#e6e8f2' }}>Scene {s?.scene_index ?? idx + 1}</div>
                                    <SegmentChip type="button" onClick={() => seekTo(s?.start_seconds)}>
                                      {formatSeconds(s?.start_seconds)}–{formatSeconds(s?.end_seconds)}
                                    </SegmentChip>
                                  </Row>
                                  {s?.summary_text ? (
                                    <div style={{ marginTop: 8, color: 'rgba(230,232,242,0.85)', fontSize: 13 }}>{s.summary_text}</div>
                                  ) : null}
                                  {Array.isArray(s?.celebrities) && s.celebrities.length ? (
                                    <div style={{ marginTop: 8, color: 'rgba(230,232,242,0.7)', fontSize: 12 }}>
                                      Celebrities: {s.celebrities.map((c) => (typeof c === 'string' ? c : c?.name || '')).filter(Boolean).join(', ')}
                                    </div>
                                  ) : null}
                                  {Array.isArray(s?.labels) && s.labels.length ? (
                                    <div style={{ marginTop: 6, color: 'rgba(230,232,242,0.7)', fontSize: 12 }}>
                                      Content: {s.labels.map((l) => (typeof l === 'string' ? l : l?.name || '')).filter(Boolean).slice(0, 15).join(', ')}
                                    </div>
                                  ) : null}
                                  {Array.isArray(s?.transcript_segments) && s.transcript_segments.length ? (
                                    <div style={{ marginTop: 6, color: 'rgba(230,232,242,0.7)', fontSize: 12 }}>
                                      Transcript: {(s.transcript_segments[0]?.text || '').toString().slice(0, 180)}
                                      {s.transcript_segments.length > 1 ? '…' : ''}
                                    </div>
                                  ) : null}
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div style={{ color: 'rgba(230, 232, 242, 0.7)' }}>—</div>
                          )}
                        </CardBody>
                      </Card>

                      <Card>
                        <CardHeader>
                          <CardHeaderTitle>Famous Locations</CardHeaderTitle>
                        </CardHeader>
                        <CardBody>
                          <Row style={{ gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
                            <SegmentChip type="button" onClick={() => setLocationSourceFilter('all')} style={{ opacity: locationSourceFilter === 'all' ? 1 : 0.65 }}>
                              All
                            </SegmentChip>
                            <SegmentChip
                              type="button"
                              onClick={() => setLocationSourceFilter('transcript')}
                              style={{ opacity: locationSourceFilter === 'transcript' ? 1 : 0.65 }}
                            >
                              Transcript
                            </SegmentChip>
                            <SegmentChip
                              type="button"
                              onClick={() => setLocationSourceFilter('landmarks')}
                              style={{ opacity: locationSourceFilter === 'landmarks' ? 1 : 0.65 }}
                            >
                              Landmarks
                            </SegmentChip>
                          </Row>
                          <div style={{ display: 'grid', gap: 8, color: 'rgba(230, 232, 242, 0.85)', fontSize: 13 }}>
                            {Array.isArray(locations?.time_mapped) && locations.time_mapped.length
                              ? locations.time_mapped
                                  .filter((loc) => {
                                    if (!loc || typeof loc !== 'object') return false;
                                    const srcs = Array.isArray(loc.sources) ? loc.sources : [];
                                    const hasTranscript = srcs.some((s) => /transcript|speech|asr|whisper/i.test(String(s || '')));
                                    const hasLandmarks = srcs.some((s) => /landmark/i.test(String(s || '')));
                                    if (locationSourceFilter === 'transcript') return hasTranscript;
                                    if (locationSourceFilter === 'landmarks') return hasLandmarks;
                                    return true;
                                  })
                                  .slice(0, 60)
                                  .map((loc, idx) => {
                                  const name = String(loc?.name || loc?.location || loc?.label || '').trim() || 'Location';
                                  const segs = Array.isArray(loc?.segments) ? loc.segments : [];
                                  const geocode = loc?.geocode && typeof loc.geocode === 'object' ? loc.geocode : null;
                                  const lat = geocode ? Number(geocode.lat) : null;
                                  const lng = geocode ? Number(geocode.lng) : null;
                                  const hasCoords = Number.isFinite(lat) && Number.isFinite(lng);
                                  const mapsHref = hasCoords ? `https://www.google.com/maps?q=${lat},${lng}` : null;
                                  const sources = Array.isArray(loc?.sources) ? loc.sources.filter(Boolean).slice(0, 3) : [];

                                  return (
                                    <div key={`loc-${idx}`}>
                                      <Row style={{ justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
                                        <div style={{ fontWeight: 900, color: '#e6e8f2' }}>{name}</div>
                                        {mapsHref ? (
                                          <LinkA href={mapsHref} target="_blank" rel="noreferrer">
                                            Map
                                          </LinkA>
                                        ) : null}
                                      </Row>

                                      {sources.length ? (
                                        <div style={{ color: 'rgba(230,232,242,0.6)', marginTop: 2, fontSize: 12 }}>
                                          Source: {sources.join(', ')}
                                        </div>
                                      ) : null}

                                      {segs.length ? (
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                                          {segs.slice(0, 10).map((s, j) => (
                                            <SegmentChip key={`locseg-${idx}-${j}`} type="button" onClick={() => seekTo(s?.start_seconds)}>
                                              {formatSeconds(s?.start_seconds)}–{formatSeconds(s?.end_seconds)}
                                            </SegmentChip>
                                          ))}
                                        </div>
                                      ) : null}
                                    </div>
                                  );
                                })
                              : Array.isArray(locations?.locations) && locations.locations.length
                                ? locations.locations.slice(0, 60).map((loc, idx) => (
                                    <div key={`loc2-${idx}`}>{typeof loc === 'string' ? loc : loc?.name || JSON.stringify(loc)}</div>
                                  ))
                                : '—'}
                          </div>
                        </CardBody>
                      </Card>

                      <Card id="envid-script">
                        <CardHeader>
                          <CardHeaderTitle>Script</CardHeaderTitle>
                          <CountPill>
                            {Array.isArray(transcriptSegments) && transcriptSegments.length
                              ? `${transcriptSegments.length} segment(s)`
                              : transcriptText.trim()
                                ? '1 transcript'
                                : '0'}
                          </CountPill>
                        </CardHeader>
                        <CardBody>
                          <div style={{ display: 'grid', gap: 14 }}>
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
                                <div style={{ color: 'rgba(230,232,242,0.75)', fontSize: 13 }}>
                                  {transcriptText}
                                </div>
                              ) : (
                                <div style={{ display: 'grid', gap: 8, color: 'rgba(230, 232, 242, 0.7)' }}>
                                  <div>No transcript found for this video.</div>
                                  <SubtleNote>
                                    If this was processed before Audio Transcription existed (or transcription was disabled), click Reprocess and enable Audio Transcription.
                                  </SubtleNote>
                                </div>
                              )}
                            </div>

                            <div>
                              <div style={{ fontWeight: 900, color: 'rgba(230,232,242,0.8)', marginBottom: 6 }}>
                                Paragraph-wise script
                              </div>
                              {transcriptParagraphs.length ? (
                                <div style={{ display: 'grid', gap: 10 }}>
                                  {transcriptParagraphs.slice(0, 400).map((p, idx) => (
                                    <div
                                      key={`tp-${idx}`}
                                      style={{
                                        border: '1px solid rgba(255, 255, 255, 0.08)',
                                        background: 'rgba(255, 255, 255, 0.03)',
                                        borderRadius: 12,
                                        padding: 12,
                                      }}
                                    >
                                      <Row style={{ justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
                                        <div style={{ color: 'rgba(230,232,242,0.65)', fontSize: 12, fontWeight: 900 }}>
                                          {formatTimecode(p.start, playerFps)}–{formatTimecode(p.end, playerFps)}
                                        </div>
                                        <SecondaryButton type="button" onClick={() => seekTo(p.start)}>
                                          Seek
                                        </SecondaryButton>
                                      </Row>
                                      <div style={{ marginTop: 8, color: 'rgba(230,232,242,0.9)', fontSize: 13, lineHeight: 1.5 }}>
                                        {p.text}
                                      </div>
                                    </div>
                                  ))}
                                  {transcriptParagraphs.length > 400 ? (
                                    <SubtleNote>Showing first 400 paragraphs.</SubtleNote>
                                  ) : null}
                                </div>
                              ) : transcriptText.trim() ? (
                                <div style={{ whiteSpace: 'pre-wrap', color: 'rgba(230,232,242,0.9)', fontSize: 13, lineHeight: 1.5 }}>
                                  {transcriptText}
                                </div>
                              ) : (
                                <div style={{ display: 'grid', gap: 8, color: 'rgba(230, 232, 242, 0.7)' }}>
                                  <div>No transcript found for this video.</div>
                                  <SubtleNote>
                                    If this was processed before Audio Transcription existed (or transcription was disabled), click Reprocess and enable Audio Transcription.
                                  </SubtleNote>
                                </div>
                              )}
                            </div>

                            <div>
                              <div style={{ fontWeight: 900, color: 'rgba(230,232,242,0.8)', marginBottom: 6 }}>
                                Raw transcript (no timestamps)
                              </div>
                              {rawTranscriptText.trim() ? (
                                <div style={{ whiteSpace: 'pre-wrap', color: 'rgba(230,232,242,0.9)', fontSize: 13, lineHeight: 1.55 }}>
                                  {rawTranscriptText}
                                </div>
                              ) : (
                                <div style={{ display: 'grid', gap: 8, color: 'rgba(230, 232, 242, 0.7)' }}>
                                  <div>No transcript found for this video.</div>
                                  <SubtleNote>Click Reprocess and enable Audio Transcription to generate a transcript.</SubtleNote>
                                </div>
                              )}
                            </div>
                          </div>
                        </CardBody>
                      </Card>
                    </>
                  )}
                </DetailBody>
              </DetailPanel>
            ) : null}
          </Section>
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

        {deletePending && (
          <ModalOverlay
            onClick={(event) => {
              if (event.target === event.currentTarget) setDeletePending(null);
            }}
          >
            <ModalCard>
              <ModalHeader>
                <div>
                  <ModalTitle>Delete video</ModalTitle>
                  <ModalSubtitle>{deletePending.videoTitle || 'Selected video'}</ModalSubtitle>
                </div>
              </ModalHeader>
              <ModalBody>
                <ConfirmText>
                  Are you sure you want to delete this video? This action cannot be undone.
                </ConfirmText>
                <ConfirmActions>
                  <SecondaryButton type="button" onClick={() => setDeletePending(null)}>
                    Cancel
                  </SecondaryButton>
                  <DeleteButton type="button" onClick={confirmDeleteVideo}>
                    Delete
                  </DeleteButton>
                </ConfirmActions>
              </ModalBody>
            </ModalCard>
          </ModalOverlay>
        )}

        {reprocessPending && (
          <ModalOverlay
            onClick={(event) => {
              if (event.target === event.currentTarget) setReprocessPending(null);
            }}
          >
            <ModalCard>
              <ModalHeader>
                <div>
                  <ModalTitle>Reprocess video</ModalTitle>
                  <ModalSubtitle>{reprocessPending.videoTitle || 'Selected video'}</ModalSubtitle>
                </div>
              </ModalHeader>
              <ModalBody>
                <ConfirmText>
                  Reprocessing will rerun the selected pipeline steps for this video. Proceed?
                </ConfirmText>
                <ConfirmActions>
                  <SecondaryButton type="button" onClick={() => setReprocessPending(null)}>
                    Cancel
                  </SecondaryButton>
                  <ReprocessButton type="button" onClick={confirmReprocessVideo}>
                    Reprocess
                  </ReprocessButton>
                </ConfirmActions>
              </ModalBody>
            </ModalCard>
          </ModalOverlay>
        )}
      </Container>
    </PageWrapper>
  );
}
