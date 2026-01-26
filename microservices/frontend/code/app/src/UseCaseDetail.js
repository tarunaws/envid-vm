import React, { useEffect, useMemo } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import styled from 'styled-components';
import useCases from './data/useCases';

const Page = styled.section`
  padding: 3rem 1.5rem 4rem;
  max-width: 920px;
  margin: 0 auto;
  color: var(--text);
`;

const BackButton = styled.button`
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.45rem 0.9rem;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  background: rgba(255, 255, 255, 0.06);
  color: rgba(255, 255, 255, 0.86);
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  margin-bottom: 1.5rem;
  transition: transform 0.2s ease, border-color 0.2s ease;

  &:hover {
    transform: translateX(-2px);
    border-color: rgba(255, 255, 255, 0.22);
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.12);
  }
`;

const Title = styled.h1`
  margin: 0 0 0.75rem;
  font-size: clamp(2rem, 5vw, 2.85rem);
  font-weight: 800;
  color: #ffffff;
`;

const StatusTag = styled.span`
  display: inline-flex;
  align-items: center;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.85);
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.14);
  margin-right: 0.65rem;
`;

const Summary = styled.p`
  font-size: 1.08rem;
  line-height: 1.8;
  color: var(--text-muted);
  margin: 0 0 1.75rem;
`;

const HighlightsHeading = styled.h2`
  margin: 2.5rem 0 1rem;
  color: #f8fafc;
  font-size: 1.35rem;
  font-weight: 700;
`;

const HighlightsList = styled.ul`
  margin: 0;
  padding-left: 1.2rem;
  display: grid;
  gap: 0.85rem;
`;

const HighlightItem = styled.li`
  color: var(--text-muted);
  line-height: 1.7;
`;

const ActionRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.85rem;
  margin-top: 2.5rem;
`;

const PrimaryAction = styled.button`
  padding: 0.75rem 1.35rem;
  border-radius: 999px;
  border: none;
  background: linear-gradient(135deg, var(--button-accent), var(--button-accent-2));
  color: var(--on-button-accent);
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 36px rgba(var(--button-accent-rgb), 0.32);
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 3px rgba(var(--button-accent-rgb), 0.45);
  }
`;

const SecondaryAction = styled.button`
  padding: 0.75rem 1.3rem;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  background: rgba(255, 255, 255, 0.06);
  color: rgba(255, 255, 255, 0.86);
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  transition: transform 0.2s ease, border-color 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.22);
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.12);
  }
`;

export default function UseCaseDetail() {
  const { slug } = useParams();
  const navigate = useNavigate();

  const useCase = useMemo(() => useCases.find((item) => item.id === slug), [slug]);

  useEffect(() => {
    if (!useCase || useCase.hidden) {
      navigate('/use-cases', { replace: true });
    }
  }, [useCase, navigate]);

  if (!useCase || useCase.hidden) {
    return null;
  }

  const isComingSoon = useCase.status === 'coming-soon';

  return (
    <Page>
      <BackButton type="button" onClick={() => navigate('/use-cases')}>
        ← Back to solutions
      </BackButton>
      <div>
        {isComingSoon && <StatusTag>Coming soon</StatusTag>}
        <Title>{useCase.title}</Title>
      </div>
      <Summary>{useCase.detailDescription}</Summary>
      {useCase.highlights?.length > 0 && (
        <>
          <HighlightsHeading>What you can expect</HighlightsHeading>
          <HighlightsList>
            {useCase.highlights.map((point, index) => (
              <HighlightItem key={index}>{point}</HighlightItem>
            ))}
          </HighlightsList>
        </>
      )}
      <ActionRow>
        {!isComingSoon && (
            <PrimaryAction
              type="button"
              onClick={() => navigate(useCase.workspacePath || useCase.path)}
            >
            Launch workspace
          </PrimaryAction>
        )}
        <SecondaryAction type="button" onClick={() => navigate('/use-cases')}>
          Back to solutions
        </SecondaryAction>
      </ActionRow>
    </Page>
  );
}
