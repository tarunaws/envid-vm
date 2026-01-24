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

const BACKEND_URL = process.env.REACT_APP_ENVID_METADATA_BACKEND_URL || '/envid-multimodal';
const STATS_POLL_MS = 5000;

function App() {
  const [scrolled, setScrolled] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    try {
      return window.localStorage.getItem('envidAuth') === 'true';
    } catch (err) {
      return false;
    }
  });
  const buildId = process.env.REACT_APP_BUILD_ID || 'dev';
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [systemStats, setSystemStats] = useState(null);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 12);
    handleScroll();
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    try {
      const lastBuild = window.localStorage.getItem('envidBuildId');
      if (lastBuild && lastBuild !== buildId) {
        setIsAuthenticated(false);
        window.localStorage.removeItem('envidAuth');
      }
      window.localStorage.setItem('envidBuildId', buildId);
    } catch (err) {
      // ignore storage errors
    }
  }, [buildId]);

  useEffect(() => {
    if (!isAuthenticated) return undefined;
    let active = true;
    const fetchStats = async () => {
      try {
        const resp = await fetch(`${BACKEND_URL}/system/stats`, { cache: 'no-store' });
        if (!resp.ok) throw new Error('Failed to fetch system stats');
        const data = await resp.json();
        if (active) setSystemStats(data);
      } catch (err) {
        if (active) setSystemStats(null);
      }
    };
    fetchStats();
    const id = setInterval(fetchStats, STATS_POLL_MS);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [isAuthenticated]);

  const handleLogin = (event) => {
    event.preventDefault();
    const user = username.trim();
    const pass = password.trim();
    if (user === 'meta' && pass === 'meta') {
      setIsAuthenticated(true);
      setAuthError('');
      setPassword('');
      try {
        window.localStorage.setItem('envidAuth', 'true');
      } catch (err) {
        // ignore storage errors
      }
    } else {
      setAuthError('Invalid credentials.');
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUsername('');
    setPassword('');
    setAuthError('');
    try {
      window.localStorage.removeItem('envidAuth');
    } catch (err) {
      // ignore storage errors
    }
  };

  const formatPercent = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '—';
    return `${Math.round(value)}%`;
  };

  const formatGb = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '—';
    return value.toFixed(1);
  };

  const cpuLabel = formatPercent(systemStats?.cpu_percent);
  const gpuLabel = formatPercent(systemStats?.gpu_percent);
  const memPercent = formatPercent(systemStats?.memory_percent);

  if (!isAuthenticated) {
    return (
      <>
        <GlobalStyle />
        <LoginShell>
          <LoginCard>
            <LoginBrand>Development environment</LoginBrand>
            <LoginTitle>Sign in to continue</LoginTitle>
            <LoginSubtitle>Use the provided credentials.</LoginSubtitle>
            <LoginForm onSubmit={handleLogin}>
              <LoginField>
                <LoginLabel htmlFor="envid-username">Username</LoginLabel>
                <LoginInput
                  id="envid-username"
                  type="text"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  placeholder="Username"
                  autoComplete="username"
                  required
                />
              </LoginField>
              <LoginField>
                <LoginLabel htmlFor="envid-password">Password</LoginLabel>
                <LoginInput
                  id="envid-password"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Password"
                  autoComplete="current-password"
                  required
                />
              </LoginField>
              {authError ? <LoginError>{authError}</LoginError> : null}
              <LoginButton type="submit">Enter</LoginButton>
            </LoginForm>
          </LoginCard>
        </LoginShell>
      </>
    );
  }

  return (
    <VisibleUseCasesProvider>
      <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <GlobalStyle />
        <AppContainer>
          <NeonNav $scrolled={scrolled}>
            <NavInner>
              <Brand to="/">
                <BrandLogo src="/envid.png?v=1" alt="Envid AI Studio - Dev" />
                <BrandCopy>
                  <BrandText>Envid AI Studio - Dev</BrandText>
                  <BrandTagline>AI studio for media workflows</BrandTagline>
                </BrandCopy>
              </Brand>
              <NavLinks>
                <NeonLink to="/">Home</NeonLink>
                <NeonLink to="/use-cases">Live Demo</NeonLink>
                <NeonLink to="/about">About</NeonLink>
              </NavLinks>
              <NavCenter>
                <StatsGroup>
                  <StatsPill>CPU: {cpuLabel}</StatsPill>
                  <StatsPill>RAM: {memPercent}</StatsPill>
                  <StatsPill>GPU: {gpuLabel}</StatsPill>
                </StatsGroup>
              </NavCenter>
              <Spacer />
              <LogoutButton type="button" onClick={handleLogout}>
                Logout
              </LogoutButton>
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
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
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

const NavCenter = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  min-width: 0;
`;

const StatsGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
`;

const StatsPill = styled.div`
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.08);
  color: var(--text);
  font-size: 0.8rem;
  font-weight: 700;
  white-space: nowrap;
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

const LogoutButton = styled.button`
  background: rgba(255, 255, 255, 0.08);
  color: #e6e8f2;
  border: 1px solid rgba(255, 255, 255, 0.14);
  padding: 8px 14px;
  border-radius: 999px;
  font-weight: 700;
  cursor: pointer;

  &:hover {
    background: rgba(255, 255, 255, 0.14);
  }
`;

const LoginShell = styled.div`
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: radial-gradient(circle at top, rgba(124, 58, 237, 0.18), transparent 55%),
    linear-gradient(180deg, rgba(5, 4, 8, 0.94), rgba(4, 3, 6, 0.98));
`;

const LoginCard = styled.div`
  width: min(420px, 100%);
  border-radius: 20px;
  padding: 32px;
  background: rgba(10, 8, 16, 0.92);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
  border: 1px solid rgba(255, 255, 255, 0.08);
`;

const LoginBrand = styled.span`
  display: inline-block;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.78rem;
  color: var(--text-faint);
  margin-bottom: 12px;
`;

const LoginTitle = styled.h1`
  margin: 0 0 6px;
  font-size: 1.6rem;
`;

const LoginSubtitle = styled.p`
  margin: 0 0 20px;
  color: var(--text-muted);
  font-size: 0.95rem;
`;

const LoginForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const LoginField = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const LoginLabel = styled.label`
  font-size: 0.85rem;
  color: var(--text-subtle);
`;

const LoginInput = styled.input`
  padding: 12px 14px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(6, 5, 10, 0.9);
  color: var(--text);
  font-size: 0.95rem;
  outline: none;
  transition: border 0.2s ease, box-shadow 0.2s ease;

  &:focus {
    border-color: rgba(168, 85, 247, 0.7);
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2);
  }
`;

const LoginError = styled.div`
  color: #fda4af;
  font-size: 0.85rem;
`;

const LoginButton = styled.button`
  padding: 12px 16px;
  border-radius: 10px;
  border: none;
  font-weight: 600;
  background: linear-gradient(135deg, var(--button-accent), var(--button-accent-2));
  color: var(--on-button-accent);
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 30px rgba(124, 58, 237, 0.3);
  }
`;

