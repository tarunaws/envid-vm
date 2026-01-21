import React, { useMemo } from 'react';
import styled from 'styled-components';

const Page = styled.section`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 3.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
`;

const Header = styled.header`
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
`;

const Title = styled.h1`
  margin: 0;
  font-size: clamp(1.9rem, 4vw, 2.6rem);
  font-weight: 800;
  color: #f8fafc;
`;

const Subtitle = styled.p`
  margin: 0;
  color: var(--text-muted);
  line-height: 1.7;
  max-width: 820px;
`;

const FrameWrap = styled.div`
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(var(--accent-2-rgb), 0.25);
  background: rgba(10, 7, 18, 0.75);
  box-shadow: 0 22px 48px rgba(6, 15, 30, 0.55);
`;

const Frame = styled.iframe`
  width: 100%;
  height: min(78vh, 820px);
  border: none;
  display: block;
`;

const HelperRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  color: var(--text-faint);
  font-size: 0.95rem;
`;

const Link = styled.a`
  color: var(--text-subtle);
  font-weight: 700;
  text-decoration: none;

  &:hover {
    text-decoration: underline;
  }
`;

function resolveImageApiBase() {
  const envValue = process.env.REACT_APP_API_BASE;
  if (envValue) {
    return envValue.replace(/\/+$/, '');
  }

  const currentHost = window.location.hostname;
  if (currentHost && currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
    return `http://${currentHost}:5002`;
  }

  return 'http://localhost:5002';
}

export default function MoviePosterGeneration() {
  const apiBase = useMemo(() => resolveImageApiBase(), []);

  return (
    <Page>
      <Header>
        <Title>Movie Poster Generation</Title>
      </Header>

      <FrameWrap>
        <Frame title="Movie Poster Generation" src={apiBase} />
      </FrameWrap>
    </Page>
  );
}
