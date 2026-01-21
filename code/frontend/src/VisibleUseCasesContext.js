import React, { createContext, useContext, useState, useEffect, useMemo, useCallback, useRef } from 'react';
import useCases from './data/useCases';

const STORAGE_KEY = 'enabledUseCases';
const DEFAULT_VALUE = { visible: null, setVisible: () => {}, loading: false, syncing: false };
const VisibleUseCasesContext = createContext(DEFAULT_VALUE);
const ALL_USE_CASE_IDS = useCases
  .filter((useCase) => !useCase.hidden)
  .map(useCase => useCase.id);

const isBrowser = typeof window !== 'undefined';
const VISIBILITY_ENDPOINT = '/usecase-visibility/visibility';

function sanitizeIds(ids) {
  if (!Array.isArray(ids)) {
    return [];
  }
  return Array.from(new Set(ids.filter(id => ALL_USE_CASE_IDS.includes(id))));
}

function readHiddenFromStorage() {
  if (!isBrowser) {
    return [];
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return [];
    }
    const parsed = JSON.parse(stored);
    if (Array.isArray(parsed)) {
      // Backwards compatibility: legacy format stored *enabled* IDs as an array.
      // An empty array should mean "no enabled IDs" (i.e., hide everything).
      if (parsed.length === 0) {
        return [...ALL_USE_CASE_IDS];
      }
      const visibleIds = sanitizeIds(parsed);
      // If the stored list contained no valid IDs (stale data), fall back to "no hidden".
      if (!visibleIds.length) {
        return [];
      }
      return ALL_USE_CASE_IDS.filter(id => !visibleIds.includes(id));
    }
    if (parsed && Array.isArray(parsed.hidden)) {
      return sanitizeIds(parsed.hidden);
    }
  } catch (error) {
    console.warn('Failed to parse use case visibility from storage', error);
  }
  return [];
}

function persistHidden(hidden) {
  if (!isBrowser) {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ hidden }));
  } catch (error) {
    console.warn('Failed to persist use case visibility', error);
  }
}

export function VisibleUseCasesProvider({ children }) {
  const [hidden, setHidden] = useState(() => readHiddenFromStorage());
  const [remoteLoaded, setRemoteLoaded] = useState(false);
  const [loadingRemote, setLoadingRemote] = useState(true);
  const [isSyncing, setIsSyncing] = useState(false);
  const lastSyncedRef = useRef(JSON.stringify(hidden));

  const visible = useMemo(() => {
    if (!hidden.length) {
      return null;
    }
    return ALL_USE_CASE_IDS.filter(id => !hidden.includes(id));
  }, [hidden]);

  const setVisible = useCallback((nextVisible) => {
    if (nextVisible === null || !Array.isArray(nextVisible)) {
      setHidden([]);
      return;
    }
    // Important: allow turning OFF everything.
    // An empty visible array means: hide all use cases.
    if (nextVisible.length === 0) {
      setHidden([...ALL_USE_CASE_IDS]);
      return;
    }

    const sanitized = sanitizeIds(nextVisible);
    // If the caller provided only invalid IDs, treat it as "no override" (all visible).
    if (!sanitized.length) {
      setHidden([]);
      return;
    }

    const nextHidden = ALL_USE_CASE_IDS.filter(id => !sanitized.includes(id));
    setHidden(nextHidden);
  }, []);

  useEffect(() => {
    persistHidden(hidden);
  }, [hidden]);

  useEffect(() => {
    if (!isBrowser) {
      setLoadingRemote(false);
      setRemoteLoaded(true);
      return undefined;
    }

    let active = true;

    async function loadRemoteState() {
      try {
        const response = await fetch(VISIBILITY_ENDPOINT, {
          headers: { Accept: 'application/json' },
        });
        if (!response.ok) {
          throw new Error(`Failed to load visibility state: ${response.status}`);
        }
        const data = await response.json();
        if (!active) {
          return;
        }
        const remoteHidden = sanitizeIds(data.hidden || []);
        lastSyncedRef.current = JSON.stringify(remoteHidden);
        setHidden(remoteHidden);
        persistHidden(remoteHidden);
      } catch (error) {
        if (active) {
          console.warn('Falling back to local visibility state', error);
        }
      } finally {
        if (active) {
          setLoadingRemote(false);
          setRemoteLoaded(true);
        }
      }
    }

    loadRemoteState();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!remoteLoaded) {
      return undefined;
    }

    const serialized = JSON.stringify(hidden);
    if (serialized === lastSyncedRef.current) {
      return undefined;
    }
    lastSyncedRef.current = serialized;

    let cancelled = false;
    setIsSyncing(true);

    fetch(VISIBILITY_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({ hidden }),
    }).catch((error) => {
      if (!cancelled) {
        console.warn('Failed to persist use case visibility to server', error);
      }
    }).finally(() => {
      if (!cancelled) {
        setIsSyncing(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [hidden, remoteLoaded]);

  useEffect(() => {
    if (!isBrowser) {
      return undefined;
    }
    const handler = () => {
      setHidden(readHiddenFromStorage());
    };
    window.addEventListener('storage', handler);
    return () => window.removeEventListener('storage', handler);
  }, []);

  const value = useMemo(() => ({
    visible,
    setVisible,
    loading: loadingRemote,
    syncing: isSyncing,
  }), [visible, setVisible, loadingRemote, isSyncing]);

  return (
    <VisibleUseCasesContext.Provider value={value}>
      {children}
    </VisibleUseCasesContext.Provider>
  );
}

export function useVisibleUseCases() {
  return useContext(VisibleUseCasesContext);
}
