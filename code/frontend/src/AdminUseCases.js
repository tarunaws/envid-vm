import React, { useState, useEffect, useMemo, useCallback } from 'react';
import styled from 'styled-components';
import useCases from './data/useCases';
import { useVisibleUseCases } from './VisibleUseCasesContext';

const Page = styled.section`
  max-width: 1100px;
  margin: 0 auto;
  padding: 72px clamp(20px, 5vw, 60px) 96px;
`;

const Header = styled.header`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 20px;
`;

const Title = styled.h1`
  margin: 0;
  font-size: clamp(2rem, 4vw, 2.8rem);
  font-weight: 800;
  color: #f8fafc;
`;

const Subtitle = styled.p`
  margin: 0;
  color: var(--text-subtle);
  font-size: 1.05rem;
  max-width: 720px;
`;

const AutoSaveNote = styled.p`
  margin: 4px 0 0;
  color: rgba(148, 163, 184, 0.85);
  font-size: 0.92rem;
`;

const TabsWrapper = styled.div`
  margin: 32px 0 24px;
`;

const TabList = styled.div`
  display: inline-flex;
  gap: 0.5rem;
  padding: 0.35rem;
  border-radius: 999px;
  background: rgba(10, 7, 18, 0.65);
  border: 1px solid rgba(var(--accent-2-rgb), 0.25);
`;

const TabButton = styled.button`
  border: none;
  padding: 0.5rem 1.3rem;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  color: ${({ $active }) => ($active ? 'var(--on-accent)' : 'var(--text-muted)')};
  background: ${({ $active }) => ($active ? 'linear-gradient(135deg, var(--button-accent), var(--button-accent-2))' : 'transparent')};
  transition: background 0.2s ease, color 0.2s ease;
`;

const TabActionButton = styled.button`
  border: none;
  padding: 0.5rem 1.3rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.95rem;
  cursor: ${({ disabled }) => (disabled ? 'not-allowed' : 'pointer')};
  color: var(--text-muted);
  background: rgba(10, 7, 18, 0.25);
  border: 1px solid rgba(var(--accent-2-rgb), 0.35);
  opacity: ${({ disabled }) => (disabled ? 0.55 : 1)};
  transition: background 0.2s ease, color 0.2s ease;

  &:hover {
    background: rgba(10, 7, 18, 0.45);
  }
`;

const StatsRow = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 40px;
`;

const StatCard = styled.div`
  background: linear-gradient(145deg, rgba(10, 7, 18, 0.96), rgba(20, 12, 36, 0.92));
  border-radius: 16px;
  padding: 1.25rem 1.5rem;
  border: 1px solid rgba(var(--accent-2-rgb), 0.25);
  box-shadow: 0 18px 40px rgba(8, 15, 40, 0.55);
`;

const StatLabel = styled.p`
  margin: 0;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #94a3b8;
`;

const StatValue = styled.p`
  margin: 0.35rem 0 0;
  font-size: 2.1rem;
  font-weight: 700;
  color: var(--text);
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 18px;
`;

const Card = styled.article`
  background: rgba(10, 7, 18, 0.92);
  border-radius: 18px;
  border: 1px solid rgba(var(--accent-2-rgb), 0.22);
  padding: 1.2rem;
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
  box-shadow: 0 22px 48px rgba(7, 15, 32, 0.65);
`;

const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
`;

const CaseTitle = styled.h2`
  margin: 0;
  font-size: 1.2rem;
  color: #f8fafc;
`;

const CaseTag = styled.span`
  font-size: 0.75rem;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  background: rgba(34, 197, 94, 0.12);
  color: #86efac;
  border: 1px solid rgba(34, 197, 94, 0.35);
`;

const ComingSoonTag = styled(CaseTag)`
  background: rgba(251, 191, 36, 0.12);
  color: #fde68a;
  border-color: rgba(251, 191, 36, 0.4);
`;

const SystemHiddenTag = styled(CaseTag)`
  background: rgba(148, 163, 184, 0.18);
  color: #e2e8f0;
  border-color: rgba(148, 163, 184, 0.35);
`;

const CaseDesc = styled.p`
  margin: 0;
  color: var(--text-muted);
  font-size: 0.95rem;
  line-height: 1.6;
`;

const ToggleRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
`;

const ToggleLabel = styled.span`
  font-size: 0.9rem;
  color: rgba(226, 232, 240, 0.8);
`;

const ToggleButton = styled.button`
  position: relative;
  width: 52px;
  height: 28px;
  border-radius: 999px;
  border: none;
  cursor: ${({ disabled }) => (disabled ? 'not-allowed' : 'pointer')};
  background: ${({ $active }) => ($active ? 'linear-gradient(135deg, var(--button-accent), var(--button-accent-2))' : 'rgba(71, 85, 105, 0.8)')};
  opacity: ${({ disabled }) => (disabled ? 0.5 : 1)};
  padding: 0;
  transition: background 0.2s ease;

  &:after {
    content: '';
    position: absolute;
    top: 3px;
    left: ${({ $active }) => ($active ? '26px' : '3px')};
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: rgba(12, 7, 18, 0.95);
    box-shadow: ${({ $active }) => ($active ? '0 8px 16px rgba(var(--accent-2-rgb), 0.35)' : 'none')};
    transition: left 0.2s ease;
  }
`;

const STORAGE_KEY = 'enabledUseCases';

export default function AdminUseCases() {
  const { visible, setVisible, loading, syncing } = useVisibleUseCases();
  const allUseCaseIds = useMemo(() => useCases.filter(u => !u.hidden).map(u => u.id), []);
  const [activeTab, setActiveTab] = useState('usecases');
  const [enabled, setEnabled] = useState(() => {
    // Important: an empty array is a valid state (meaning: everything disabled).
    if (Array.isArray(visible)) {
      return visible;
    }
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          return parsed;
        }
      }
    } catch (error) {
      console.warn('Failed to parse enabled use cases from storage', error);
    }
    return allUseCaseIds;
  });

  useEffect(() => {
    if (visible === null) {
      setEnabled(allUseCaseIds);
      return;
    }
    if (Array.isArray(visible)) {
      setEnabled(visible);
    }
  }, [visible, allUseCaseIds]);

  const toggle = useCallback((id) => {
    if (loading || !allUseCaseIds.includes(id)) {
      return;
    }
    setEnabled(prev => {
      const next = prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id];
      const deduped = Array.from(new Set(next));
      const nextValue = deduped.length === allUseCaseIds.length ? null : deduped;
      setVisible(nextValue);
      return next;
    });
  }, [allUseCaseIds, loading, setVisible]);

  const turnOffAllUseCases = useCallback(() => {
    if (loading) {
      return;
    }
    setEnabled([]);
    // Empty visible list means: hide everything.
    setVisible([]);
  }, [loading, setVisible]);

  const total = useCases.length;
  const productionReady = useCases.filter(u => u.status !== 'coming-soon' && !u.hidden);
  const productionReadyIds = productionReady.map(u => u.id);
  const productionReadyCount = productionReady.length;
  const comingSoonCount = useCases.filter(u => u.status === 'coming-soon').length;
  const hiddenCount = productionReadyIds.filter(id => !enabled.includes(id)).length;

  const renderUseCasesTab = () => (
    <>
      <StatsRow>
        <StatCard>
          <StatLabel>Total Use Cases</StatLabel>
          <StatValue>{total}</StatValue>
        </StatCard>
        <StatCard>
          <StatLabel>Production Ready</StatLabel>
          <StatValue>{productionReadyCount}</StatValue>
        </StatCard>
        <StatCard>
          <StatLabel>Hidden</StatLabel>
          <StatValue>{hiddenCount}</StatValue>
        </StatCard>
        <StatCard>
          <StatLabel>Coming Soon</StatLabel>
          <StatValue>{comingSoonCount}</StatValue>
        </StatCard>
      </StatsRow>

      <Grid>
        {useCases.map((u) => {
          const isSystemHidden = Boolean(u.hidden);
          const isEnabled = !isSystemHidden && enabled.includes(u.id);
          const TagComponent = u.status === 'coming-soon' ? ComingSoonTag : CaseTag;
          return (
            <Card key={u.id}>
              <CardHeader>
                <CaseTitle>{u.title}</CaseTitle>
                <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                  <TagComponent>{u.status === 'coming-soon' ? 'Coming soon' : 'Live'}</TagComponent>
                  {isSystemHidden && <SystemHiddenTag>Hidden</SystemHiddenTag>}
                </div>
              </CardHeader>
              <CaseDesc>{u.cardDescription}</CaseDesc>
              <ToggleRow>
                <ToggleLabel>
                  {isSystemHidden
                    ? 'Hidden in code (not end-user visible)'
                    : isEnabled
                      ? 'Visible in catalog'
                      : 'Hidden from catalog'}
                </ToggleLabel>
                <ToggleButton
                  type="button"
                  $active={isEnabled}
                  onClick={() => toggle(u.id)}
                  aria-pressed={isEnabled}
                  aria-label={`Toggle visibility for ${u.title}`}
                  disabled={isSystemHidden || loading}
                />
              </ToggleRow>
            </Card>
          );
        })}
      </Grid>
    </>
  );

  return (
    <Page>
      <Header>
        <Title>Use Case Visibility</Title>
        <Subtitle>Control which accelerators appear in the Live Demo tab. Disable items that are still in R&D or that require approvals before demoing.</Subtitle>
        <AutoSaveNote>
          {loading
            ? 'Loading saved visibility settings...'
            : syncing
              ? 'Syncing changes to the server...'
              : 'Changes are applied instantly — no save button needed.'}
        </AutoSaveNote>
      </Header>

      <TabsWrapper>
        <TabList>
          <TabButton
            type="button"
            $active={activeTab === 'usecases'}
            onClick={() => setActiveTab('usecases')}
          >
            Use Cases
          </TabButton>
          <TabActionButton
            type="button"
            onClick={turnOffAllUseCases}
            disabled={loading || enabled.length === 0}
          >
            turnoff use case
          </TabActionButton>
        </TabList>
      </TabsWrapper>

      {activeTab === 'usecases' && renderUseCasesTab()}
    </Page>
  );
}
