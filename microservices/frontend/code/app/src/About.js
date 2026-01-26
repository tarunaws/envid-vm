import React from 'react';
import styled from 'styled-components';

const Hero = styled.section`
  position: relative;
  min-height: 32vh;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 2.5rem 1.25rem 2rem;
  background: var(--bg);
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
`;

const HeroTitle = styled.h1`
  color: #ffffff;
  font-weight: 800;
  margin: 0;
  font-size: clamp(1.85rem, 3.8vw, 2.6rem);
`;

const AboutContainer = styled.div`
  padding: 2rem 1.5rem 3rem;
  max-width: 980px;
  margin: 0 auto;
  color: #ced9f8;
`;

const Title = styled.h1`
  color: #ffffff;
  font-weight: 800;
  margin: 0 0 1rem 0;
  font-size: clamp(1.6rem, 3.2vw, 2.2rem);
`;

const SectionHeading = styled.h3`
  color: #f8fafc;
  font-weight: 800;
  margin: 2rem 0 0.75rem 0;
  font-size: 1.2rem;
`;

const Paragraph = styled.p`
  margin: 0.35rem 0 1rem;
  line-height: 1.85;
`;

const Divider = styled.div`
  height: 1px;
  width: 100%;
  background: rgba(var(--accent-rgb), 0.2);
  margin: 1.5rem 0;
`;

const BulletList = styled.ul`
  margin: 0.5rem 0 1.25rem 1.15rem;
  padding: 0;
  list-style: disc;
  color: #d5e1ff;
  line-height: 1.7;
`;

const BulletItem = styled.li`
  margin: 0.45rem 0;
`;

const Accent = styled.span`
  color: var(--accent);
  font-weight: 700;
`;

const Pill = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.55rem 1.1rem;
  border-radius: 999px;
  background: rgba(var(--accent-rgb), 0.14);
  color: var(--text-muted);
  border: 1px solid rgba(var(--accent-rgb), 0.32);
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.45px;
  margin-bottom: 1.2rem;
`;

export default function About() {
  return (
    <>
      <Hero>
        <HeroTitle>GenAI Media & Entertainment Showcase</HeroTitle>
      </Hero>
      <AboutContainer>
        <Pill>MediaGenAI Studio</Pill>
        <Title>We build production-ready GenAI pipelines for modern storytellers.</Title>
        <Paragraph>
          MediaGenAI Studio is a proving ground for the media industry’s most impactful generative experiences. We orchestrate adaptive streaming, speech intelligence, and creative automation into workflows that scale from pilots to primetime.
        </Paragraph>

        <Divider />

        <SectionHeading>What you can explore</SectionHeading>
        <BulletList>
          <BulletItem>
            <Accent>Composable media services</Accent> — Mix subtitling, dubbing, voice synthesis, and creative tooling into a unified control plane.
          </BulletItem>
          <BulletItem>
            <Accent>Policy-aware automation</Accent> — Apply guardrails, review queues, and localization standards so every deliverable ships on brand and on time.
          </BulletItem>
          <BulletItem>
            <Accent>Audience intelligence</Accent> — Activate metadata-rich analytics, semantic discovery, and campaign insights from day one.
          </BulletItem>
          <BulletItem>
            <Accent>Studio-grade infrastructure</Accent> — Deploy adaptive HLS/DASH delivery, high-availability APIs, and secure data workflows without re-platforming.
          </BulletItem>
        </BulletList>

        <Divider />

        <SectionHeading>Why GenAI transforms media</SectionHeading>
        <BulletList>
          <BulletItem>Reduce localization cycles from weeks to hours.</BulletItem>
          <BulletItem>Empower creative teams with AI copilots that respect brand and compliance.</BulletItem>
          <BulletItem>Deliver personalized, accessible experiences to every audience segment.</BulletItem>
          <BulletItem>Extend the lifetime value of content catalogs through automation and insights.</BulletItem>
        </BulletList>
      </AboutContainer>
    </>
  );
}
