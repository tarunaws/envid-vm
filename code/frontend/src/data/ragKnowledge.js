const ragKnowledgeBase = [
  {
    id: 'synthetic-voiceover-personas',
    title: 'Synthetic Voiceover personas and output controls',
    summary:
      'The Synthetic Voiceover workspace provides preset personas, SSML editing, persona chips, and output format controls (MP3, OGG Vorbis, PCM).',
    detail:
      'In `SyntheticVoiceover.js`, users can pick from persona presets like “Warm guide” or “Energetic launch,” tweak SSML directly, and choose MP3, OGG Vorbis, or PCM outputs. The UI keeps recordings, persona badges, and synthesis previews organized in responsive panels.',
    source: 'frontend/src/SyntheticVoiceover.js',
    tags: ['voiceover', 'ssml', 'personas', 'audio'],
  },
  {
    id: 'synthetic-voiceover-api-fallbacks',
    title: 'Voice API base resolution',
    summary:
      'Voice synthesis automatically resolves the API base URL, preferring `REACT_APP_VOICE_API_BASE` and falling back to localhost:5003 for LAN setups.',
    detail:
      'The helper `resolveVoiceApiBase` checks `REACT_APP_VOICE_API_BASE`, normalises trailing slashes, and defaults to the client origin. For local testing it switches to the current hostname on port 5003, covering LAN `.local` host names.',
    source: 'frontend/src/SyntheticVoiceover.js',
    tags: ['voiceover', 'configuration', 'api'],
  },
  {
    id: 'ai-subtitling-streaming',
    title: 'Subtitle Lab streaming preview',
    summary:
      'AI Subtitling ships with an HLS/DASH adaptive streaming preview and caption toggle baked into the UI.',
    detail:
      'In `AISubtitling.js`, the `StreamingPlayerContainer` wires up `Hls.js` and `dashjs` depending on the manifest type, exposes a caption toggle button, and renders multi-track outputs directly inside the player.',
    source: 'frontend/src/AISubtitling.js',
    tags: ['subtitles', 'streaming', 'player', 'hls', 'dash'],
  },
  {
    id: 'ai-subtitling-language-options',
    title: 'Language packs for transcription and translation',
    summary:
      'Subtitle Lab exposes extensive transcription (Transcribe) and translation language dropdowns including auto-detect.',
    detail:
      'Two arrays – `TRANSCRIBE_LANGUAGE_OPTIONS` and `TRANSLATE_LANGUAGE_OPTIONS` – enumerate 40+ locales and feed the `<select>` inputs in the upload workflow, letting reviewers mix capture and translation languages.',
    source: 'frontend/src/AISubtitling.js',
    tags: ['subtitles', 'languages', 'translation'],
  },
  {
    id: 'scene-summarization-api',
    title: 'Scene Summarization API resolution and timeout',
    summary:
      'Scene Summarization resolves API base URLs dynamically and guards long-running jobs with configurable timeouts.',
    detail:
      '`resolveSceneApiBase` gently corrects protocol mismatches for same-host HTTPS deployments, while `resolveRequestTimeout` enforces a default 3 hour timeout via `REACT_APP_SCENE_TIMEOUT_MS`.',
    source: 'frontend/src/SceneSummarization.js',
    tags: ['scene', 'configuration', 'timeout'],
  },
  {
    id: 'scene-summarization-results',
    title: 'Summaries, highlights, and tagged metadata',
    summary:
      'Scene Summarization outputs recap paragraphs, highlight bullet lists, tagged entities, and audio previews when available.',
    detail:
      'Result cards render summarised narratives, highlight bullets, tag chips, metadata lists, and optional audio previews directly in the grid layout so editors can scan context without leaving the page.',
    source: 'frontend/src/SceneSummarization.js',
    tags: ['scene', 'summaries', 'highlights', 'metadata'],
  },
  {
    id: 'use-cases-routing',
    title: 'Use-case routing and availability states',
    summary:
      'The use-case gallery relies on a shared data module that encodes status, routes, icons, and highlight copy for every workflow.',
    detail:
      '`useCases.js` exports cards with `status` flags (available vs. coming-soon) and `path` values so navigation buttons can disable themselves when routes are pending.',
    source: 'frontend/src/data/useCases.js',
    tags: ['routing', 'use-cases', 'availability'],
  },
];

export default ragKnowledgeBase;
