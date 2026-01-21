import React from 'react';
import styled from 'styled-components';
import { NavLink } from 'react-router-dom';

const Hero = styled.section`
  position: relative;
  min-height: 56vh;
  display: flex;
  justify-content: center;
  text-align: left;
  padding: 5rem 1.25rem 3.25rem;
  background: var(--bg);
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  @media (max-width: 720px) {
    text-align: center;
    padding: 4rem 1.1rem 2.75rem;
  }
`;

const HeroInner = styled.div`
  width: min(1120px, 100%);
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 1rem;
  @media (max-width: 720px) {
    align-items: center;
  }
`;

const HeroBrand = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: clamp(0.6rem, 2vw, 0.9rem);
  margin-bottom: clamp(1.6rem, 3vw, 2.6rem);
  @media (max-width: 720px) {
    align-items: center;
  }
`;

const HeroLogo = styled.img`
  height: clamp(38px, 6vw, 60px);
  filter: drop-shadow(0 12px 26px rgba(0, 0, 0, 0.65));
`;

const HeroTagline = styled.span`
  font-size: clamp(1rem, 2.5vw, 1.35rem);
  font-weight: 600;
  color: #e2e9f5;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  @media (max-width: 720px) {
    letter-spacing: 0.08em;
  }
`;

const HighlightTag = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 999px;
  font-size: 0.85rem;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  background: rgba(255, 255, 255, 0.08);
  color: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(255, 255, 255, 0.14);
  margin-bottom: 1.25rem;
`;

const Title = styled.h1`
  font-size: clamp(2.25rem, 4.6vw, 3.6rem);
  color: #ffffff;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0 0 0.75rem 0;
`;

const Subtitle = styled.p`
  font-size: clamp(1rem, 2.1vw, 1.2rem);
  color: var(--text-muted);
  margin: 0;
  line-height: 1.8;
  max-width: 840px;
  @media (max-width: 720px) {
    text-align: center;
  }
`;

const CTAGroup = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  justify-content: flex-start;
  margin-top: 1.75rem;
  @media (max-width: 720px) {
    justify-content: center;
  }
`;

const CTA = styled(NavLink)`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  padding: 0.95rem 1.45rem;
  background: linear-gradient(135deg, var(--button-accent), var(--button-accent-2));
  color: var(--on-button-accent);
  border-radius: 999px;
  font-weight: 800;
  text-decoration: none;
  letter-spacing: 0.3px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  box-shadow: 0 18px 40px rgba(var(--button-accent-rgb), 0.32);
  &:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 24px 46px rgba(var(--button-accent-2-rgb), 0.38);
  }
`;

export default function Home() {
  return (
    <>
      <Hero>
        <HeroInner>
          <HeroBrand>
            <HeroLogo src="/envid.png?v=1" alt="Envid AI Studio" />
            <HeroTagline>Envid AI Studio</HeroTagline>
          </HeroBrand>
          <HighlightTag>AI for media workflows</HighlightTag>
            <Title>Build next-gen content ecosystems with Envid AI Studio</Title>
          <Subtitle>
              Adaptive subtitling, synthetic voice, intelligent editing, and semantic discovery—one workspace built for modern media pipelines.
          </Subtitle>
          <CTAGroup>
              <CTA to="/use-cases">Explore solutions</CTA>
          </CTAGroup>
        </HeroInner>
      </Hero>
    </>
  );
}
