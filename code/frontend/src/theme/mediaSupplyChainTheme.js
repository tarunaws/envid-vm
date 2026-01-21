export const mediaSupplyChainTheme = {
  fonts: {
    display: "'Sora', 'Orbitron', 'Space Grotesk', system-ui, sans-serif",
    body: "'Space Grotesk', 'IBM Plex Sans', system-ui, sans-serif",
    mono: "'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace",
  },
  colors: {
    pageBg: 'var(--bg)',
    pageBgAccent: 'var(--bg-2)',
    glowTeal: 'var(--accent)',
    glowBlue: 'var(--accent-2)',
    glowAmber: '#ffb449',
    softLavender: '#c4c6ff',
    slate: '#94a3b8',
    panelBorder: 'rgba(var(--accent-rgb), 0.28)',
    panelBg: 'rgba(10, 7, 18, 0.78)',
    panelBgStrong: 'rgba(10, 7, 18, 0.92)',
    danger: '#ff5f7e',
    warning: '#f8c646',
    success: '#6bffcb',
    muted: '#6b7280',
    textBright: 'var(--text)',
    textMuted: 'var(--text-muted)',
  },
  gradients: {
    hero: 'linear-gradient(135deg, rgba(15, 52, 96, 0.75), rgba(0, 0, 0, 0.65))',
    holo: 'linear-gradient(140deg, rgba(var(--accent-rgb), 0.14), rgba(var(--accent-2-rgb), 0.09))',
    upload: 'linear-gradient(125deg, rgba(var(--accent-rgb), 0.12), rgba(var(--accent-2-rgb), 0.14))',
    accentLine: 'linear-gradient(90deg, rgba(var(--accent-rgb), 0), rgba(var(--accent-rgb), 0.8), rgba(var(--accent-rgb), 0))',
  },
  shadows: {
    glowStrong: '0 20px 55px rgba(var(--accent-rgb), 0.22)',
    glowSoft: '0 12px 30px rgba(var(--accent-2-rgb), 0.2)',
    borderGlow: '0 0 22px rgba(var(--accent-rgb), 0.32)',
  },
  blur: {
    panel: '24px',
  },
  layout: {
    maxWidth: '1200px',
    sectionGap: '2rem',
  },
  transitions: {
    default: '180ms ease',
    slow: '360ms ease',
  },
};
