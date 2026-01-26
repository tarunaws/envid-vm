import React, { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import useCases from './data/useCases';
import { useVisibleUseCases } from './VisibleUseCasesContext';

const PageWrap = styled.section`
  padding: 2.5rem 1.5rem 3.5rem;
  max-width: 1200px;
  margin: 0 auto;
`;

const Header = styled.div`
  text-align: center;
  max-width: 760px;
  margin: 0 auto 2rem;
`;

const Title = styled.h1`
  margin: 0 0 0.75rem;
  font-size: clamp(1.9rem, 4vw, 2.6rem);
  font-weight: 800;
  color: #f8fafc;
`;

const Lead = styled.p`
  margin: 0;
  font-size: 1.05rem;
  line-height: 1.8;
  color: var(--text-muted);
`;

const Grid = styled.div`
  display: grid;
  gap: 1.5rem;
  grid-template-columns: repeat(4, minmax(0, 1fr));

  @media (max-width: 1280px) {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  @media (max-width: 960px) {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  @media (max-width: 640px) {
    grid-template-columns: 1fr;
  }
`;

const Card = styled.article`
  background: var(--surface);
  border-radius: 16px;
  border: 1px solid rgba(var(--accent-2-rgb), 0.18);
  padding: 0;
  text-align: left;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  cursor: pointer;
  color: #e5efff;
  overflow: hidden;
  box-shadow: 0 22px 48px rgba(6, 15, 30, 0.55);
  min-height: 100%;
  display: flex;
  flex-direction: column;
  &:hover {
    transform: translateY(-6px);
    border-color: rgba(255, 255, 255, 0.22);
    box-shadow: 0 26px 56px rgba(0, 0, 0, 0.65);
  }
  &:focus-within,
  &:focus-visible {
    outline: none;
    border-color: rgba(255, 255, 255, 0.28);
    box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.12), 0 26px 56px rgba(0, 0, 0, 0.65);
  }
  @media (max-width: 640px) {
    min-height: auto;
  }
`;

const ThumbWrap = styled.div`
  height: 150px;
  background: rgba(var(--bg-rgb), 0.95);
  display: flex;
  align-items: center;
  justify-content: center;
  border-bottom: 1px solid rgba(var(--accent-2-rgb), 0.25);
`;

const Thumb = styled.img`
  height: 72px;
  width: 72px;
  object-fit: contain;
  filter: drop-shadow(0 10px 24px rgba(0, 0, 0, 0.65));
`;

const Body = styled.div`
  padding: 1.15rem 1.15rem 1.35rem;
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  gap: 0.6rem;
`;

const CardTitle = styled.h2`
  color: #f8fafc;
  margin: 0 0 0.6rem 0;
  font-size: 1.18rem;
  font-weight: 800;
`;

const CardDesc = styled.p`
  color: var(--text-muted);
  margin: 0;
  font-size: 0.98rem;
  line-height: 1.65;
  flex: 1 1 auto;
`;

const StatusTag = styled.span`
  display: inline-flex;
  align-items: center;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
  color: rgba(255, 255, 255, 0.85);
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.14);
`;

const LaunchButton = styled.button`
  margin-top: auto;
  align-self: flex-start;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.45rem;
  padding: 0.55rem 1.1rem;
  border-radius: 999px;
  border: none;
  background: linear-gradient(135deg, var(--button-accent), var(--button-accent-2));
  color: var(--on-button-accent);
  font-weight: 700;
  font-size: 0.9rem;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 32px rgba(var(--button-accent-rgb), 0.32);
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 3px rgba(var(--button-accent-rgb), 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
  }
`;

const FooterNote = styled.p`
  margin: 2.5rem auto 0;
  max-width: 640px;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.95rem;
  line-height: 1.7;
`;

const LoadingState = styled.div`
  text-align: center;
  padding: 3rem 1rem 4rem;
  color: var(--text-muted);
  font-size: 1.05rem;
`;

function UseCases() {
  const navigate = useNavigate();
  const { visible, loading } = useVisibleUseCases();
  const catalog = useCases.filter((useCase) => !useCase.hidden);
  const solutions = Array.isArray(visible)
    ? catalog.filter(u => visible.includes(u.id))
    : catalog;

  const hasVisibilityOverride = Array.isArray(visible);
  const showEmptyState = hasVisibilityOverride && solutions.length === 0;

  const handleNavigate = useCallback((solution) => {
    if (solution?.id) {
      navigate(`/use-cases/${solution.id}`);
    }
  }, [navigate]);

  const handleCardKeyDown = useCallback((event, solution) => {
    if ((event.key === 'Enter' || event.key === ' ') && solution?.id) {
      event.preventDefault();
      handleNavigate(solution);
    }
  }, [handleNavigate]);

  const onLaunch = useCallback((event, solution) => {
    event.stopPropagation();
    if (!solution) return;

    const isComingSoon = solution.status === 'coming-soon';
    if (isComingSoon && solution.id) {
      navigate(`/use-cases/${solution.id}`);
      return;
    }

    const destination = solution.workspacePath || solution.path;
    if (destination) navigate(destination);
  }, [navigate]);

  return (
    <PageWrap>
      <Header>
        <Title>Explore solutions</Title>
        <Lead>
          Click a solution to view details, then launch the workspace.
        </Lead>
      </Header>
      {loading ? (
        <LoadingState>Loading available solutions...</LoadingState>
      ) : showEmptyState ? (
        <LoadingState>
          All solutions are currently turned off in Admin.
        </LoadingState>
      ) : (
        <Grid>
          {solutions.map((solution) => {
            const isComingSoon = solution.status === 'coming-soon';
            return (
              <Card
                key={solution.id || solution.title}
                id={solution.id}
                onClick={() => handleNavigate(solution)}
                onKeyDown={(event) => handleCardKeyDown(event, solution)}
                tabIndex={0}
                role="link"
                aria-label={`${solution.title} solution`}
              >
                <ThumbWrap>
                  <Thumb src={solution.image} alt={solution.title} onError={(e) => { e.currentTarget.src = '/usecases/placeholder.svg'; }} />
                </ThumbWrap>
                <Body>
                  <CardTitle>{solution.title}</CardTitle>
                  {solution.status === 'coming-soon' && (
                    <StatusTag $variant="coming-soon">Coming soon</StatusTag>
                  )}
                  <CardDesc>{solution.cardDescription}</CardDesc>
                  <LaunchButton
                    type="button"
                    onClick={(event) => onLaunch(event, solution)}
                    disabled={!isComingSoon && !solution.path && !solution.workspacePath}
                  >
                    {isComingSoon ? 'View details' : 'Launch'}
                  </LaunchButton>
                </Body>
              </Card>
            );
          })}
        </Grid>
      )}
      <FooterNote>
        Looking for something bespoke? We can co-create guardrailed workflows tailored to your content supply chain.
      </FooterNote>
    </PageWrap>
  );
}

export default UseCases;
