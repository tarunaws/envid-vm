import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, NavLink } from 'react-router-dom';
import styled, { createGlobalStyle } from 'styled-components';

import About from './About';
import AISubtitling from './AISubtitling';
import AskMe from './AskMe';
import DynamicAdInsertion from './DynamicAdInsertion';
import HighlightTrailer from './HighlightTrailer';
import Home from './Home';
import InteractiveShoppable from './InteractiveShoppable';
import EnvidMetadataMinimal from './EnvidMetadataMinimal';
import MoviePosterGeneration from './MoviePosterGeneration';
import MovieScriptCreation from './MovieScriptCreation';
import SceneSummarization from './SceneSummarization';
import SyntheticVoiceover from './SyntheticVoiceover';
import AIBasedTrailer from './PersonalizedTrailer';
import UseCaseDetail from './UseCaseDetail';
import UseCases from './UseCases';
import VideoGeneration from './VideoGeneration';
import { VisibleUseCasesProvider } from './VisibleUseCasesContext';

function App() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 12);
    handleScroll();
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <VisibleUseCasesProvider>
      <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <GlobalStyle />
        <AppContainer>
          <NeonNav $scrolled={scrolled}>
            <NavInner>
              <Brand to="/">
                <BrandLogo src="/envid.png?v=1" alt="Envid AI Studio" />
                <BrandCopy>
                  <BrandText>Envid AI Studio</BrandText>
                  <BrandTagline>AI studio for media workflows</BrandTagline>
                </BrandCopy>
              </Brand>
              <NavLinks>
                <NeonLink to="/">Home</NeonLink>
                <NeonLink to="/use-cases">Live Demo</NeonLink>
                <NeonLink to="/about">About</NeonLink>
              </NavLinks>
              <Spacer />
            </NavInner>
          </NeonNav>

          <MainContent>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/use-cases" element={<UseCases />} />
              <Route path="/use-cases/:slug" element={<UseCaseDetail />} />
              <Route path="/movie-script-creation" element={<MovieScriptCreation />} />
              <Route path="/movie-poster-generation" element={<MoviePosterGeneration />} />
              {/* AWS/S3 demo routes removed (GCS-only frontend). */}
              <Route path="/content-moderation" element={<Navigate to="/envid-metadata" replace />} />
              <Route path="/personalized-trailers" element={<AIBasedTrailer />} />
              <Route path="/envid-metadata" element={<EnvidMetadataMinimal />} />
              <Route path="/envid-metadata/multimodal" element={<EnvidMetadataMinimal />} />
              <Route path="/video-generation" element={<VideoGeneration />} />
              <Route path="/dynamic-ad-insertion" element={<DynamicAdInsertion />} />
              <Route path="/highlight-trailer" element={<HighlightTrailer />} />
              <Route path="/synthetic-voiceover" element={<SyntheticVoiceover />} />
              <Route path="/scene-summarization" element={<SceneSummarization />} />
              <Route path="/ai-subtitling" element={<AISubtitling />} />
              <Route path="/media-supply-chain" element={<Navigate to="/envid-metadata" replace />} />
              <Route path="/about" element={<About />} />
                <Route path="/interactive-video" element={<InteractiveShoppable />} />
              <Route path="/tech-stack" element={<Navigate to="/" replace />} />
              {/* Admin use-case toggles disabled (GCS-only frontend). */}
              <Route path="/admin-usecases" element={<Navigate to="/envid-metadata" replace />} />
              <Route path="/solutions" element={<Navigate to="/use-cases" replace />} />
            </Routes>
          </MainContent>

          <AskMe />
        </AppContainer>
      </Router>
    </VisibleUseCasesProvider>
  );
}

export default App;

const GlobalStyle = createGlobalStyle`
  :root {
    color-scheme: dark;

    --bg: #040306;
    --bg-2: #040306;
    --bg-rgb: 4, 3, 6;
    --surface: rgba(var(--bg-rgb), 0.78);
    --surface-2: rgba(var(--bg-rgb), 0.9);

    --text: #f5f3ff;
    --text-muted: rgba(221, 214, 254, 0.84);
    --text-subtle: rgba(196, 181, 253, 0.82);
    --text-faint: rgba(167, 139, 250, 0.75);

    --accent: #e5e7eb;
    --accent-rgb: 229, 231, 235;
    --accent-2: #9ca3af;
    --accent-2-rgb: 156, 163, 175;
    --on-accent: #000000;

    --button-accent: #a855f7;
    --button-accent-rgb: 168, 85, 247;
    --button-accent-2: #7c3aed;
    --button-accent-2-rgb: 124, 58, 237;
    --on-button-accent: #ffffff;
  }

  * {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  ::selection {
    background: rgba(255, 255, 255, 0.22);
    color: var(--text);
  }

  a {
    color: inherit;
  }
`;

const AppContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
`;

const MainContent = styled.main`
  flex: 1;
  padding: 120px 4vw 48px;
`;

const NeonNav = styled.nav`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 10;
  transition: background 0.2s ease, box-shadow 0.2s ease;
  background: ${props => (props.$scrolled ? 'rgba(7, 12, 22, 0.9)' : 'transparent')};
  box-shadow: ${props => (props.$scrolled ? '0 10px 40px rgba(0, 0, 0, 0.35)' : 'none')};
  border-bottom: 1px solid rgba(var(--accent-rgb), ${props => (props.$scrolled ? 0.18 : 0)});
`;

const NavInner = styled.div`
  display: flex;
  align-items: center;
  gap: 32px;
  padding: 20px clamp(16px, 4vw, 48px);
`;

const Brand = styled(NavLink)`
  display: inline-flex;
  align-items: center;
  gap: 0.65rem;
  padding-bottom: 4px;
  text-decoration: none;
  flex-shrink: 0;
`;

const BrandLogo = styled.img`
  height: clamp(34px, 4vw, 44px);
  filter: drop-shadow(0 10px 26px rgba(var(--accent-rgb), 0.28));
`;

const BrandCopy = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
`;

const BrandText = styled.span`
  font-weight: 700;
  font-size: 1.05rem;
  letter-spacing: 0.01em;
`;

const BrandTagline = styled.span`
  font-size: 0.78rem;
  color: var(--text-subtle);
`;

const NavLinks = styled.div`
  display: flex;
  align-items: center;
  gap: clamp(0.5rem, 1.6vw, 1.5rem);
`;

const NeonLink = styled(NavLink)`
  text-decoration: none;
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--text-muted);
  padding-bottom: 4px;
  border-bottom: 2px solid transparent;
  transition: color 0.15s ease, border-color 0.15s ease;

  &.active {
    color: var(--text);
    border-color: var(--accent);
  }

  &:hover {
    color: var(--text);
  }
`;

const Spacer = styled.div`
  flex: 1;
`;

