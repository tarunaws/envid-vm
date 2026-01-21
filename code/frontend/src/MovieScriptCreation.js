import React, { useCallback, useMemo, useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const GENRES = [
  'Action',
  'Adventure',
  'Animation',
  'Biographical',
  'Comedy',
  'Crime Thriller',
  'Documentary',
  'Drama',
  'Epic Fantasy',
  'Family',
  'Historical',
  'Horror',
  'Musical',
  'Mystery',
  'Romance',
  'Science Fiction',
  'Sports Drama',
  'Superhero',
  'Techno Thriller',
  'War',
  'Western',
];

const MOODS = [
  'Adrenaline-pumping',
  'Bittersweet',
  'Bleak and gritty',
  'Darkly comic',
  'Epic and awe-inspiring',
  'Feel-good and uplifting',
  'Heart-wrenching',
  'High-tension suspense',
  'Hopeful and redemptive',
  'Romantic and nostalgic',
  'Satirical',
  'Spine-chilling',
  'Thrilling mystery',
  'Whimsical',
];

const AUDIENCES = [
  {
    value: 'Kids (6-11)',
    label: 'Kids (6-11)',
    description: 'Wholesome adventure with clear morals and age-appropriate stakes.',
  },
  {
    value: 'Tweens (10-13)',
    label: 'Tweens (10-13)',
    description: 'Coming-of-age plots with relatable school and friendship drama.',
  },
  {
    value: 'Teens (13-17)',
    label: 'Teens (13-17)',
    description: 'High-energy stories balancing romance, rebellion, and social dynamics.',
  },
  {
    value: 'Young adults (18-24)',
    label: 'Young adults (18-24)',
    description: 'Culturally current narratives with sharp dialogue and rapid pacing.',
  },
  {
    value: 'Adults (25-44)',
    label: 'Adults (25-44)',
    description: 'Character-driven arcs blending career, relationship, and family tensions.',
  },
  {
    value: 'Mature adults (45+)',
    label: 'Mature adults (45+)',
    description: 'Reflective storytelling with legacy themes and emotionally grounded stakes.',
  },
  {
    value: 'Family four-quadrant',
    label: 'Family four-quadrant',
    description: 'All-ages entertainment mixing humor, heart, and spectacle for shared viewing.',
  },
  {
    value: 'Festival cinephiles',
    label: 'Festival cinephiles',
    description: 'Auteur-style storytelling with layered themes and daring structure.',
  },
  {
    value: 'Streaming binge audiences',
    label: 'Streaming binge audiences',
    description: 'Hooky serialized beats that deliver cliffhangers and rapid payoffs.',
  },
];

const REGIONS = [
  'Global (All Regions)',
  'North America',
  'Latin America',
  'Western Europe',
  'Central & Eastern Europe',
  'Middle East & North Africa',
  'Sub-Saharan Africa',
  'India & South Asia',
  'China',
  'Japan',
  'Korea',
  'Southeast Asia',
  'Australia & New Zealand',
  'Caribbean',
];

const TRANSLATE_LANGUAGES_BASE = {
  en: 'English (default)',
  af: 'Afrikaans',
  sq: 'Albanian',
  am: 'Amharic',
  ar: 'Arabic',
  hy: 'Armenian',
  az: 'Azerbaijani',
  bn: 'Bengali',
  bs: 'Bosnian',
  bg: 'Bulgarian',
  ca: 'Catalan',
  zh: 'Chinese (Simplified)',
  'zh-TW': 'Chinese (Traditional)',
  hr: 'Croatian',
  cs: 'Czech',
  da: 'Danish',
  'fa-AF': 'Dari',
  nl: 'Dutch',
  et: 'Estonian',
  fa: 'Farsi (Persian)',
  fi: 'Finnish',
  fr: 'French',
  'fr-CA': 'French (Canada)',
  ka: 'Georgian',
  de: 'German',
  el: 'Greek',
  gu: 'Gujarati',
  ht: 'Haitian Creole',
  ha: 'Hausa',
  he: 'Hebrew',
  hi: 'Hindi',
  hu: 'Hungarian',
  is: 'Icelandic',
  id: 'Indonesian',
  it: 'Italian',
  ja: 'Japanese',
  kn: 'Kannada',
  kk: 'Kazakh',
  ko: 'Korean',
  lv: 'Latvian',
  lt: 'Lithuanian',
  mk: 'Macedonian',
  ms: 'Malay',
  ml: 'Malayalam',
  mr: 'Marathi',
  mn: 'Mongolian',
  no: 'Norwegian',
  ps: 'Pashto',
  pl: 'Polish',
  pt: 'Portuguese (Brazil)',
  'pt-PT': 'Portuguese (Portugal)',
  pa: 'Punjabi',
  ro: 'Romanian',
  ru: 'Russian',
  sr: 'Serbian',
  si: 'Sinhala',
  sk: 'Slovak',
  sl: 'Slovenian',
  so: 'Somali',
  es: 'Spanish',
  'es-MX': 'Spanish (Mexico)',
  sw: 'Swahili',
  sv: 'Swedish',
  tl: 'Tagalog',
  ta: 'Tamil',
  te: 'Telugu',
  th: 'Thai',
  tr: 'Turkish',
  uk: 'Ukrainian',
  ur: 'Urdu',
  uz: 'Uzbek',
  vi: 'Vietnamese',
  cy: 'Welsh',
};

const LANGUAGES = Object.entries(TRANSLATE_LANGUAGES_BASE)
  .sort(([valueA, labelA], [valueB, labelB]) => {
    if (valueA === 'en') return -1;
    if (valueB === 'en') return 1;
    return labelA.localeCompare(labelB, undefined, { sensitivity: 'base' });
  })
  .map(([value, label]) => ({ value, label }));

const RATINGS = [
  { value: 'G', label: 'G — General Audiences (all ages admitted)' },
  { value: 'PG', label: 'PG — Parental guidance suggested (some material may not suit children)' },
  { value: 'PG-13', label: 'PG-13 — Parents strongly cautioned (may be inappropriate for under 13)' },
  { value: 'R', label: 'R — Restricted (under 17 requires accompanying adult)' },
  { value: 'NC-17', label: 'NC-17 — Adults only (no one 17 and under admitted)' },
  { value: 'Unrated / Festival', label: 'Unrated / Festival — No rating restraints; suitable for internal or festival review' },
];

const RUNTIMES = ['90', '100', '110', '120', '130', '140', '150', '160', '180'];

const STEPS = [
  {
    title: '',
    description: '',
  },
  {
    title: 'Creative palette',
    description: 'Pick the genres and tonal palette that define the script.',
  },
  {
    title: 'Audience & localisation',
    description: 'Dial in who the story is for and where it should resonate.',
  },
  {
    title: 'Runtime & compliance',
    description: 'Lock duration, rating, and review the creative DNA summary.',
  },
];

const LAST_STEP_INDEX = STEPS.length - 1;

const resolveScriptApiBase = () => {
  const envValue = process.env.REACT_APP_SCRIPT_API_BASE;
  if (envValue) {
    return envValue.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined') {
    const { protocol, hostname } = window.location;
    const localHosts = new Set(['localhost', '127.0.0.1', '0.0.0.0']);
    const isLanHost = /^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1]))/.test(hostname) || hostname.endsWith('.local');
    if (localHosts.has(hostname) || isLanHost) {
      return `${protocol}//${hostname}:5005`;
    }
  }
  return '';
};

const SCRIPT_API_BASE = resolveScriptApiBase();

const Page = styled.section`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.75rem 1.5rem 3.5rem;
  color: #dce7ff;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 2.5rem;
`;

const Title = styled.h1`
  color: #ffffff;
  font-weight: 800;
  margin: 0 0 0.75rem 0;
  font-size: clamp(1.95rem, 4.2vw, 2.8rem);
`;

const Layout = styled.div`
  display: grid;
  gap: 2rem;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
`;

const Panel = styled.div`
  background: linear-gradient(160deg, rgba(14, 26, 48, 0.92), rgba(22, 36, 63, 0.88));
  border-radius: 18px;
  border: 1px solid rgba(99, 102, 241, 0.18);
  padding: 1.75rem;
  box-shadow: 0 24px 54px rgba(7, 15, 30, 0.55);
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const SectionTitle = styled.h2`
  margin: 0;
  color: #f8fafc;
  font-size: 1.35rem;
  font-weight: 800;
`;

const StepIntro = styled.p`
  margin: 0 0 1.4rem;
  color: rgba(203, 213, 225, 0.78);
  font-size: 0.95rem;
  line-height: 1.6;
`;

const FieldGrid = styled.div`
  display: grid;
  gap: 1.15rem;
`;

const FinalStepGrid = styled(FieldGrid)`
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  align-items: end;
`;

const Field = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
`;

const Label = styled.label`
  font-weight: 600;
  font-size: 0.95rem;
  letter-spacing: 0.3px;
  color: var(--text-muted);
`;

const TextArea = styled.textarea`
  border-radius: 14px;
  border: 1px solid rgba(99, 102, 241, 0.25);
  background: rgba(8, 18, 34, 0.88);
  color: #f8fafc;
  padding: 1rem 1.05rem;
  font-size: 1rem;
  line-height: 1.65;
  min-height: 140px;
  resize: vertical;
  transition: border 0.2s ease, box-shadow 0.2s ease;

  &:focus {
    outline: none;
    border-color: rgba(var(--accent-rgb), 0.6);
    box-shadow: 0 0 0 3px rgba(var(--accent-rgb), 0.15);
  }
`;

const Select = styled.select`
  border-radius: 14px;
  border: 1px solid rgba(99, 102, 241, 0.25);
  background: rgba(8, 18, 34, 0.88);
  color: #f8fafc;
  padding: 0.85rem 1rem;
  font-size: 1rem;
  min-height: 54px;
  transition: border 0.2s ease, box-shadow 0.2s ease;

  &:focus {
    outline: none;
    border-color: rgba(var(--accent-rgb), 0.6);
    box-shadow: 0 0 0 3px rgba(var(--accent-rgb), 0.15);
  }
`;

const MultiSelect = styled(Select)`
  min-height: 160px;
`;

const HelperText = styled.span`
  font-size: 0.8rem;
  color: rgba(148, 163, 184, 0.75);
`;

const Blueprint = styled.div`
  background: rgba(11, 22, 40, 0.78);
  border: 1px solid rgba(96, 165, 250, 0.25);
  border-radius: 16px;
  padding: 1.2rem 1.3rem;
  display: grid;
  gap: 0.65rem;
`;

const BlueprintTitle = styled.h3`
  margin: 0;
  font-size: 1rem;
  font-weight: 700;
  color: var(--text);
`;

const BlueprintList = styled.ul`
  margin: 0;
  padding-left: 1.1rem;
  color: var(--text-muted);
  font-size: 0.92rem;
  line-height: 1.6;
`;

const ButtonRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
`;

const PrimaryButton = styled.button`
  padding: 0.9rem 1.6rem;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--button-accent), var(--button-accent-2));
  border: none;
  color: var(--on-accent);
  font-weight: 700;
  font-size: 1rem;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 36px rgba(var(--button-accent-rgb), 0.32);
  }

  &:disabled {
    opacity: 0.55;
    cursor: not-allowed;
    box-shadow: none;
  }
`;

const SecondaryButton = styled.button`
  padding: 0.9rem 1.4rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 255, 0.35);
  background: rgba(18, 30, 52, 0.9);
  color: #dce7ff;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  transition: transform 0.2s ease, border 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    border-color: rgba(var(--accent-rgb), 0.45);
  }
`;

const TertiaryButton = styled.button`
  padding: 0.75rem 1.2rem;
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.28);
  background: rgba(15, 26, 44, 0.8);
  color: #cbd5f5;
  font-weight: 600;
  font-size: 0.88rem;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  transition: background 0.2s ease, transform 0.2s ease, border 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    background: rgba(25, 40, 62, 0.9);
    border-color: rgba(148, 163, 184, 0.45);
  }
`;

const ErrorBanner = styled.div`
  border-radius: 14px;
  border: 1px solid rgba(248, 113, 113, 0.5);
  background: rgba(239, 68, 68, 0.12);
  color: #fecaca;
  padding: 0.85rem 1rem;
  font-size: 0.95rem;
`;

const EmptyState = styled.div`
  text-align: center;
  color: #94a3e6;
  font-size: 0.95rem;
  line-height: 1.8;
`;

const ResultHeader = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
`;

const ResultTitle = styled.h3`
  margin: 0;
  font-size: 1.2rem;
  font-weight: 700;
  color: #f8fafc;
`;

const ScriptBlock = styled.pre`
  background: rgba(7, 16, 32, 0.78);
  border: 1px solid rgba(96, 165, 250, 0.18);
  border-radius: 16px;
  padding: 1.4rem 1.6rem;
  color: #e5edff;
  line-height: 1.8;
  overflow-y: auto;
  max-height: 70vh;
  white-space: pre-wrap;
  word-break: break-word;
`;

const FooterButtons = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
`;

const CopyStatus = styled.span`
  font-size: 0.82rem;
  color: #a5f3fc;
`;

const HelperNotice = styled.div`
  font-size: 0.85rem;
  color: rgba(148, 163, 184, 0.8);
`;

const slugify = (value = '') =>
  value
    .toString()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'movie-script';

const runtimeToInt = (value) => {
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const normaliseErrorMessage = (message = '') => {
  const raw = String(message || '').trim();
  if (!raw) {
    return 'Unable to generate a script right now. Please try again in a moment.';
  }
  if (/language model invocation failed/i.test(raw)) {
    return 'Language model invocation failed. Confirm the Movie Script service has valid Bedrock access and try again.';
  }
  if (/^network error/i.test(raw)) {
    return 'Network error while contacting the Movie Script service. Check your connection and retry.';
  }
  if (/status code 5\d{2}/i.test(raw)) {
    return 'Our servers are unavailable right now (5xx). Please retry shortly.';
  }
  return raw;
};

export default function MovieScriptCreation() {
  const [logline, setLogline] = useState('');
  const [notes, setNotes] = useState('');
  const [genres, setGenres] = useState(['Epic Fantasy', 'Adventure']);
  const [moods, setMoods] = useState(['Epic and awe-inspiring']);
  const [audience, setAudience] = useState(['Young adults (18-24)']);
  const [regions, setRegions] = useState(['Global (All Regions)']);
  const [runtime, setRuntime] = useState('130');
  const [rating, setRating] = useState('PG-13');
  const [language, setLanguage] = useState('en');
  const [step, setStep] = useState(0);
  const [hasTriggeredGeneration, setHasTriggeredGeneration] = useState(false);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copyStatus, setCopyStatus] = useState('');

  const handleMultiSelect = useCallback(
    (setter) => (event) => {
      const values = Array.from(event.target.selectedOptions).map((option) => option.value);
      setter(values);
    },
    []
  );

  const handleNext = useCallback(() => {
    setStep((prev) => Math.min(prev + 1, LAST_STEP_INDEX));
  }, []);

  const handlePrev = useCallback(() => {
    setStep((prev) => Math.max(prev - 1, 0));
  }, []);

  const creativePreview = useMemo(() => {
    const ratingLabel = RATINGS.find((option) => option.value === rating)?.label;
    const audienceLabelList = audience.length
      ? audience
          .map((value) => AUDIENCES.find((option) => option.value === value)?.label || value)
          .join(', ')
      : '';
    const languageLabel = LANGUAGES.find((option) => option.value === language)?.label || 'English (default)';
    return [
      { label: 'Genres', value: genres.length ? genres.join(', ') : 'Blend genres for a fresh cinematic voice.' },
      { label: 'Mood palette', value: moods.length ? moods.join(', ') : 'Balance drama, humour, suspense, and awe.' },
      {
        label: 'Audience focus',
        value: audienceLabelList || 'Design for a global, four-quadrant audience.',
      },
      { label: 'Regional emphasis', value: regions.length ? regions.join(', ') : 'Global relevance with local authenticity.' },
      { label: 'Era & setting', value: 'Present day (default)' },
      { label: 'Runtime target', value: runtime ? `${runtime} minutes` : 'Approx. 130 minutes (feature length).' },
      {
        label: 'Rating / compliance',
        value: ratingLabel || 'Tailor tone to the chosen audience rating.',
      },
      {
        label: 'Dialogue language',
        value: languageLabel,
      },
      {
        label: 'Additional notes',
        value: notes.trim() || 'Invite the model to propose bold twists and cultural specificity.',
      },
    ];
  }, [genres, moods, audience, regions, runtime, rating, notes, language]);

  const currentStepMeta = STEPS[step];
  const isOnLastStep = step === LAST_STEP_INDEX;
  const showOutputPanel = hasTriggeredGeneration || Boolean(result) || loading;

  const scriptAvailable = Boolean(result?.script);

  const handleGenerate = useCallback(
    async (event) => {
      event.preventDefault();
      if (!isOnLastStep) {
        setStep(LAST_STEP_INDEX);
        return;
      }

      setHasTriggeredGeneration(true);

      setLoading(true);
      setError('');
      setResult(null);
      setCopyStatus('');

      const payload = {
        logline: logline.trim() || undefined,
        additionalGuidance: notes.trim() || undefined,
        genres,
        moods,
        audience,
        regions,
        era: 'Present day',
        targetRuntimeMinutes: runtimeToInt(runtime),
        targetRating: rating || undefined,
        language,
      };

      try {
        const url = SCRIPT_API_BASE ? `${SCRIPT_API_BASE}/generate-script` : '/generate-script';
        const response = await axios.post(url, payload, {
          headers: { 'Content-Type': 'application/json' },
        });
        setResult(response.data);
      } catch (err) {
        const message = err?.response?.data?.error || err?.message || 'Failed to generate screenplay.';
        setError(normaliseErrorMessage(message));
      } finally {
        setLoading(false);
      }
    },
    [audience, genres, isOnLastStep, language, logline, moods, notes, rating, regions, runtime]
  );

  const handleReset = useCallback(() => {
    setLogline('');
    setNotes('');
    setGenres(['Epic Fantasy', 'Adventure']);
    setMoods(['Epic and awe-inspiring']);
    setAudience(['Young adults (18-24)']);
    setRegions(['Global (All Regions)']);
    setRuntime('130');
    setRating('PG-13');
  setLanguage('en');
    setStep(0);
    setHasTriggeredGeneration(false);
    setResult(null);
    setError('');
    setCopyStatus('');
  }, []);

  const handleCopy = useCallback(async (text, label) => {
    if (!text) {
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus(`Copied ${label}.`);
    } catch (copyError) {
      setCopyStatus('Clipboard access is unavailable.');
    }
    setTimeout(() => setCopyStatus(''), 2500);
  }, []);

  const handleDownload = useCallback(() => {
    if (!result?.script) {
      return;
    }
    const filename = `${slugify(result.title || 'movie-script')}.md`;
    const content = `# ${result.title || 'Movie Script'}\n\n${result.script}`;
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setCopyStatus(`Downloaded ${filename}`);
    setTimeout(() => setCopyStatus(''), 2500);
  }, [result]);

  return (
    <Page>
      <Header>
        <Title>Movie Script Generation</Title>
      </Header>
      <Layout>
        <Panel as="form" onSubmit={handleGenerate}>
          {currentStepMeta.title && <SectionTitle>{currentStepMeta.title}</SectionTitle>}
          {currentStepMeta.description && <StepIntro>{currentStepMeta.description}</StepIntro>}

          {step === 0 && (
            <FieldGrid>
              <Field>
                <Label htmlFor="script-logline">Context for Movie Script</Label>
                <TextArea
                  id="script-logline"
                  placeholder="Give the model your premise, hook, and key stakes."
                  value={logline}
                  onChange={(event) => setLogline(event.target.value)}
                />
              </Field>
              <Field>
                <Label htmlFor="script-notes">Additional guidance</Label>
                <TextArea
                  id="script-notes"
                  placeholder="e.g. Must-have scenes, visual motifs, production budget, character arcs, or localisation cues."
                  value={notes}
                  onChange={(event) => setNotes(event.target.value)}
                />
              </Field>
              <HelperNotice>
                Tip: The more specific you are here, the smarter the screenplay structure and localisation choices will be.
              </HelperNotice>
            </FieldGrid>
          )}

          {step === 1 && (
            <FieldGrid>
              <Field>
                <Label htmlFor="script-genres">Primary genres</Label>
                <MultiSelect
                  id="script-genres"
                  value={genres}
                  onChange={handleMultiSelect(setGenres)}
                  multiple
                >
                  {GENRES.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </MultiSelect>
                <HelperText>Hold Cmd/Ctrl to choose multiple genres.</HelperText>
              </Field>
              <Field>
                <Label htmlFor="script-moods">Mood and tone</Label>
                <MultiSelect
                  id="script-moods"
                  value={moods}
                  onChange={handleMultiSelect(setMoods)}
                  multiple
                >
                  {MOODS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </MultiSelect>
              </Field>
            </FieldGrid>
          )}

          {step === 2 && (
            <FieldGrid>
              <Field>
                <Label htmlFor="script-audience">Target audience</Label>
                <MultiSelect
                  id="script-audience"
                  value={audience}
                  onChange={handleMultiSelect(setAudience)}
                  multiple
                >
                  {AUDIENCES.map((option) => (
                    <option key={option.value} value={option.value}>
                      {`${option.label} — ${option.description}`}
                    </option>
                  ))}
                </MultiSelect>
                <HelperText>
                  Choose the viewer cohorts you want the screenplay to delight—this steers tone, references, and pacing toward
                  their expectations.
                </HelperText>
                  Dialogue and on-screen text are translated into the selected language with stage directions remaining in
                  English for clarity.
                <Label htmlFor="script-regions">Priority regions</Label>
                <MultiSelect
                  id="script-regions"
                  value={regions}
                  onChange={handleMultiSelect(setRegions)}
                  multiple
                >
                  {REGIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </MultiSelect>
              </Field>
              <Field>
                <Label htmlFor="script-language">Dialogue language</Label>
                <Select
                  id="script-language"
                  value={language}
                  onChange={(event) => setLanguage(event.target.value)}
                >
                  {LANGUAGES.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
                <HelperText>
                  Hindi selection delivers dialogue in Hinglish (Latin script), while other options use the chosen language for
                  spoken lines.
                </HelperText>
              </Field>
            </FieldGrid>
          )}

          {step === 3 && (
            <FinalStepGrid>
              <Field>
                <Label htmlFor="script-runtime">Target runtime (minutes)</Label>
                <Select id="script-runtime" value={runtime} onChange={(event) => setRuntime(event.target.value)}>
                  <option value="">130 (default)</option>
                  {RUNTIMES.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field>
                <Label htmlFor="script-rating">Audience rating</Label>
                <Select id="script-rating" value={rating} onChange={(event) => setRating(event.target.value)}>
                  <option value="">Let tone dictate</option>
                  {RATINGS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </Field>
            </FinalStepGrid>
          )}

          {isOnLastStep && (
            <Blueprint>
              <BlueprintTitle>Creative DNA preview</BlueprintTitle>
              <BlueprintList>
                {creativePreview.map((item) => (
                  <li key={item.label}>
                    <strong>{item.label}:</strong> {item.value}
                  </li>
                ))}
              </BlueprintList>
            </Blueprint>
          )}

          <ButtonRow>
            {step > 0 && (
              <SecondaryButton type="button" onClick={handlePrev} disabled={loading}>
                Previous
              </SecondaryButton>
            )}
            {!isOnLastStep && (
              <PrimaryButton type="button" onClick={handleNext}>
                Next
              </PrimaryButton>
            )}
            {isOnLastStep && (
              <>
                <PrimaryButton type="submit" disabled={loading}>
                  {loading ? 'Generating screenplay…' : 'Generate script'}
                </PrimaryButton>
                <SecondaryButton type="button" onClick={handleReset} disabled={loading}>
                  Reset brief
                </SecondaryButton>
              </>
            )}
          </ButtonRow>

          {error && !hasTriggeredGeneration && <ErrorBanner role="alert">{error}</ErrorBanner>}
        </Panel>

        {showOutputPanel && (
          <Panel>
            <SectionTitle>Script</SectionTitle>
            {error && !loading && !result && <ErrorBanner role="alert">{error}</ErrorBanner>}
            {!result && !loading && !error && (
              <EmptyState>
                Configure the creative palette and generate a screenplay. The complete script will appear here when ready.
              </EmptyState>
            )}
            {loading && (
              <EmptyState>
                The story engine is crafting your screenplay. This can take a little while—stay tuned.
              </EmptyState>
            )}
            {result && (
              <>
                <ResultHeader>
                  <ResultTitle>{result.title || 'Full screenplay'}</ResultTitle>
                </ResultHeader>

                {scriptAvailable ? (
                  <ScriptBlock>{result.script}</ScriptBlock>
                ) : (
                  <HelperText>No screenplay text returned. Try generating again with more guidance.</HelperText>
                )}

                <FooterButtons>
                  {scriptAvailable && (
                    <TertiaryButton type="button" onClick={() => handleCopy(result.script, 'screenplay')}>
                      Copy screenplay
                    </TertiaryButton>
                  )}
                  {scriptAvailable && (
                    <TertiaryButton type="button" onClick={handleDownload}>
                      Download as Markdown
                    </TertiaryButton>
                  )}
                </FooterButtons>
                {copyStatus && <CopyStatus>{copyStatus}</CopyStatus>}
              </>
            )}
          </Panel>
        )}
      </Layout>
    </Page>
  );
}
