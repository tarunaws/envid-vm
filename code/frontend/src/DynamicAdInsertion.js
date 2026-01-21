import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const getBackendUrl = () => {
  const configured = process.env.REACT_APP_DAI_BACKEND_URL?.trim();
  if (configured) {
    return configured;
  }
  if (typeof window !== 'undefined' && window.location?.hostname) {
    return `${window.location.protocol}//${window.location.hostname}:5010`;
  }
  return 'http://localhost:5010';
};

const BACKEND_URL = getBackendUrl();
const CONTENT_VIDEO_URL = '/dai-content.mp4';

// Styled Components
const PageWrapper = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 40px 20px;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 40px;
`;

const Title = styled.h1`
  color: white;
  font-size: 2.5rem;
  margin-bottom: 10px;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
`;

const Container = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const Card = styled.div`
  background: white;
  border-radius: 15px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
`;

const DiagramImage = styled.img`
  width: 100%;
  max-width: 960px;
  display: block;
  margin: 10px auto 0;
  border-radius: 14px;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.25);
  border: 1px solid rgba(102, 126, 234, 0.25);
`;

const PlaybackStatus = styled.div`
  margin-top: 15px;
  text-align: center;
  font-size: 0.95rem;
  color: #4c51bf;
  font-weight: 600;
`;

const SectionTitle = styled.h2`
  color: #667eea;
  font-size: 1.5rem;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
`;


const Icon = styled.span`
  font-size: 1.8rem;
`;

const Label = styled.label`
  display: block;
  color: #555;
  font-weight: 600;
  margin-bottom: 10px;
`;

const Select = styled.select`
  width: 100%;
  padding: 12px 15px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 1rem;
  background: white;
  cursor: pointer;
  transition: all 0.3s;
  
  &:hover {
    border-color: #667eea;
  }
  
  &:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }
`;

const Button = styled.button`
  width: 100%;
  padding: 15px 30px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  margin-top: 20px;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
  }
  
  &:active {
    transform: translateY(0);
  }
  
  &:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
  }
`;

const VideoContainer = styled.div`
  width: 100%;
  background: #000;
  border-radius: 10px;
  overflow: hidden;
  margin-top: 20px;
  aspect-ratio: 16/9;
`;

const VideoPlayer = styled.video`
  width: 100%;
  height: 100%;
  display: block;
`;

const AdInfoPanel = styled.div`
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 10px;
  padding: 20px;
  margin-top: 20px;
`;

const AdInfoRow = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  
  &:last-child {
    border-bottom: none;
  }
`;

const AdLabel = styled.span`
  font-weight: 600;
  color: #555;
`;

const AdValue = styled.span`
  color: #333;
`;

const ProfileInfo = styled.div`
  background: #f8f9fa;
  border-radius: 10px;
  padding: 15px;
  margin-top: 15px;
  color: #000;
  
  strong {
    color: #000;
  }
  
  div {
    color: #000;
  }
`;

const ProfileRow = styled.div`
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 10px;
`;

const Tag = styled.span`
  background: #667eea;
  color: white;
  padding: 5px 12px;
  border-radius: 15px;
  font-size: 0.85rem;
  font-weight: 500;
`;

const LogsContainer = styled.div`
  max-height: 400px;
  overflow-y: auto;
  margin-top: 20px;
  background: #f8f9fa;
  border-radius: 10px;
  padding: 15px;
`;

const LogEntry = styled.div`
  background: white;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 10px;
  border-left: 4px solid #667eea;
  font-size: 0.9rem;
`;

const LogTime = styled.div`
  color: #888;
  font-size: 0.8rem;
  margin-bottom: 5px;
`;

const LogDetails = styled.div`
  color: #333;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-top: 20px;
`;

const StatCard = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  border-radius: 10px;
  text-align: center;
`;

const StatValue = styled.div`
  font-size: 2rem;
  font-weight: bold;
  margin-bottom: 5px;
`;

const StatLabel = styled.div`
  font-size: 0.9rem;
  opacity: 0.9;
`;

const Message = styled.div`
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
  background: ${props => props.type === 'error' ? '#fee' : props.type === 'success' ? '#efe' : '#eef'};
  color: ${props => props.type === 'error' ? '#c33' : props.type === 'success' ? '#3c3' : '#33c'};
  border-left: 4px solid ${props => props.type === 'error' ? '#c33' : props.type === 'success' ? '#3c3' : '#33c'};
`;

const VastContainer = styled.div`
  margin-top: 20px;
  background: #1e1e1e;
  border-radius: 10px;
  padding: 20px;
  max-height: 400px;
  overflow-y: auto;
`;

const VastXml = styled.pre`
  color: #a9b7c6;
  font-family: 'Courier New', monospace;
  font-size: 0.85rem;
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  
  .xml-tag { color: #e8bf6a; }
  .xml-attr { color: #9876aa; }
  .xml-value { color: #6a8759; }
`;

const CopyButton = styled.button`
  padding: 8px 16px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.9rem;
  margin-top: 10px;
  transition: background 0.2s;
  
  &:hover {
    background: #5568d3;
  }
`;

function DynamicAdInsertion() {
  const [profiles, setProfiles] = useState([]);
  const [selectedProfile, setSelectedProfile] = useState('');
  const [profileDetails, setProfileDetails] = useState(null);
  const [currentAd, setCurrentAd] = useState(null);
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(null);
  const [message, setMessage] = useState(null);
  const [vastXml, setVastXml] = useState('');
  const personalizedVideoRef = useRef(null);
  const [playbackPhase, setPlaybackPhase] = useState('idle'); // idle | ready | ad | content
  const [personalizedSource, setPersonalizedSource] = useState('');
  const [playbackKey, setPlaybackKey] = useState(0);

  useEffect(() => {
    loadProfiles();
    loadLogs();
    loadStats();
    
    // Refresh logs and stats every 5 seconds
    const interval = setInterval(() => {
      loadLogs();
      loadStats();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const loadProfiles = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/profiles`);
      setProfiles(response.data.profiles);
      setMessage((prev) => (prev?.type === 'error' ? null : prev));
    } catch (error) {
      console.error('Failed to load user profiles:', error);
      const reason = error?.response?.data?.error || error.message || 'Unknown error';
      setMessage({ type: 'error', text: `Failed to load user profiles: ${reason}` });
    }
  };

  const loadLogs = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/logs?limit=10`);
      setLogs(response.data.logs.reverse());
    } catch (error) {
      console.error('Failed to load logs:', error);
    }
  };

  const loadStats = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const playbackStatusText = (() => {
    switch (playbackPhase) {
      case 'ad':
        return currentAd ? `Playing pre-roll ad: ${currentAd.ad_name}` : 'Playing pre-roll ad';
      case 'content':
        return 'Playing main content segment';
      case 'ready':
        return currentAd ? `Ad ready: ${currentAd.ad_name}. Press play to watch with a personalized pre-roll.` : 'Personalized ad ready for playback.';
      default:
        return 'Request a personalized ad to enable the pre-roll preview.';
    }
  })();

  const generateVastXml = (ad) => {
    if (!ad) return '';
    
    const timestamp = new Date().toISOString();
    const adId = ad.ad_id || 'unknown';
    const adTitle = ad.ad_name || 'Advertisement';
    const duration = ad.duration || 10;
    const videoUrl = ad.video_url ? ad.video_url.split('?')[0] : ''; // Remove cache buster for VAST
    
    return `<?xml version="1.0" encoding="UTF-8"?>
<VAST version="4.0" xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <Ad id="${adId}" sequence="1">
    <InLine>
      <AdSystem version="1.0">MediaGenAI Dynamic Ad Insertion</AdSystem>
      <AdTitle>${adTitle}</AdTitle>
      <Description>Personalized advertisement powered by MediaGenAI</Description>
      <Advertiser>MediaGenAI Platform</Advertiser>
      <Pricing model="CPM" currency="USD">5.00</Pricing>
      <Survey type="generic">http://localhost:5010/survey/${adId}</Survey>
      <Error>http://localhost:5010/error/${adId}</Error>
      <Impression id="impression-1">
        <![CDATA[http://localhost:5010/impression/${adId}?timestamp=${timestamp}]]>
      </Impression>
      <Creatives>
        <Creative id="creative-${adId}" sequence="1" adId="${adId}">
          <Linear>
            <Duration>00:00:${duration.toString().padStart(2, '0')}</Duration>
            <TrackingEvents>
              <Tracking event="start">
                <![CDATA[http://localhost:5010/tracking/start/${adId}]]>
              </Tracking>
              <Tracking event="firstQuartile">
                <![CDATA[http://localhost:5010/tracking/firstQuartile/${adId}]]>
              </Tracking>
              <Tracking event="midpoint">
                <![CDATA[http://localhost:5010/tracking/midpoint/${adId}]]>
              </Tracking>
              <Tracking event="thirdQuartile">
                <![CDATA[http://localhost:5010/tracking/thirdQuartile/${adId}]]>
              </Tracking>
              <Tracking event="complete">
                <![CDATA[http://localhost:5010/tracking/complete/${adId}]]>
              </Tracking>
              <Tracking event="mute">
                <![CDATA[http://localhost:5010/tracking/mute/${adId}]]>
              </Tracking>
              <Tracking event="unmute">
                <![CDATA[http://localhost:5010/tracking/unmute/${adId}]]>
              </Tracking>
              <Tracking event="pause">
                <![CDATA[http://localhost:5010/tracking/pause/${adId}]]>
              </Tracking>
              <Tracking event="resume">
                <![CDATA[http://localhost:5010/tracking/resume/${adId}]]>
              </Tracking>
              <Tracking event="fullscreen">
                <![CDATA[http://localhost:5010/tracking/fullscreen/${adId}]]>
              </Tracking>
              <Tracking event="exitFullscreen">
                <![CDATA[http://localhost:5010/tracking/exitFullscreen/${adId}]]>
              </Tracking>
            </TrackingEvents>
            <VideoClicks>
              <ClickThrough id="clickthrough">
                <![CDATA[http://localhost:5010/clickthrough/${adId}]]>
              </ClickThrough>
              <ClickTracking id="clicktracking">
                <![CDATA[http://localhost:5010/click/${adId}]]>
              </ClickTracking>
            </VideoClicks>
            <MediaFiles>
              <MediaFile id="media-${adId}" delivery="progressive" type="video/mp4" 
                         width="1280" height="720" codec="H.264" bitrate="2500">
                <![CDATA[${videoUrl}]]>
              </MediaFile>
            </MediaFiles>
          </Linear>
        </Creative>
      </Creatives>
      <Extensions>
        <Extension type="GenAI">
          <Category>${ad.category || 'General'}</Category>
          <TargetingReason>${ad.targeting_reason || 'Profile-based targeting'}</TargetingReason>
          <GenerationTimestamp>${timestamp}</GenerationTimestamp>
        </Extension>
      </Extensions>
    </InLine>
  </Ad>
</VAST>`;
  };

  const copyVastToClipboard = () => {
    navigator.clipboard.writeText(vastXml).then(() => {
      setMessage({ type: 'success', text: 'VAST XML copied to clipboard!' });
      setTimeout(() => setMessage(null), 3000);
    }).catch(() => {
      setMessage({ type: 'error', text: 'Failed to copy to clipboard' });
    });
  };

  const triggerPersonalizedPlayback = () => {
    const videoElement = personalizedVideoRef.current;
    if (!videoElement) return;
    try {
      videoElement.load();
    } catch (err) {
      console.warn('Personalized player load warning:', err);
    }
    const playPromise = videoElement.play();
    if (playPromise && playPromise.catch) {
      playPromise.catch(() => {
        // Autoplay might be blocked; ask the user to press play manually
        setMessage({ type: 'info', text: 'Press play on the personalized stream to continue playback.' });
      });
    }
  };

  const resetPersonalizedStream = () => {
    setPlaybackPhase('idle');
    setPersonalizedSource('');
    setPlaybackKey(prev => prev + 1);
    if (personalizedVideoRef.current) {
      try {
        personalizedVideoRef.current.pause();
        personalizedVideoRef.current.currentTime = 0;
      } catch (err) {
        console.warn('Personalized player reset warning:', err);
      }
    }
  };

  const startPersonalizedPlayback = (source, phase, autoPlay = true) => {
    setPersonalizedSource(source);
    setPlaybackPhase(phase);
    setPlaybackKey(prev => prev + 1);
    if (autoPlay) {
      setTimeout(() => {
        triggerPersonalizedPlayback();
      }, 150);
    }
  };

  const handlePlayPersonalizedStream = () => {
    if (!currentAd) {
      setMessage({ type: 'error', text: 'Request a personalized ad before previewing the stream.' });
      return;
    }
    startPersonalizedPlayback(currentAd.video_url, 'ad');
  };

  const handlePersonalizedPlaybackStart = () => {
    if (playbackPhase === 'ready') {
      setPlaybackPhase('ad');
    }
  };

  const handlePersonalizedPlaybackEnded = () => {
    if (playbackPhase === 'ad') {
      startPersonalizedPlayback(CONTENT_VIDEO_URL, 'content');
    } else if (playbackPhase === 'content') {
      if (currentAd) {
        startPersonalizedPlayback(currentAd.video_url, 'ready', false);
      } else {
        resetPersonalizedStream();
      }
    }
  };

  const handleProfileChange = async (profileId) => {
    setSelectedProfile(profileId);
    
    // Clear current ad and reset video player first
    setCurrentAd(null);
    setProfileDetails(null);
    setVastXml('');
    resetPersonalizedStream();
    
    // If empty profile selected, just clear everything
    if (!profileId) {
      setMessage({ type: 'info', text: 'Please select a viewer profile' });
      return;
    }
    
    try {
      // Only load profile details, do NOT load ad automatically
      const response = await axios.get(`${BACKEND_URL}/profiles/${profileId}`);
      setProfileDetails(response.data);
      setMessage({ type: 'info', text: `Profile selected: ${response.data.name}. Click "Request Personalized Ad" to load advertisement.` });
      
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to load profile details' });
    }
  };

  const handleRequestAd = async () => {
    if (!selectedProfile) {
      setMessage({ type: 'error', text: 'Please select a user profile' });
      return;
    }

    // Clear current ad and reset video player first
    setCurrentAd(null);
    setVastXml('');
    resetPersonalizedStream();

    try {
      setMessage({ type: 'info', text: 'Requesting personalized ad...' });
      
      const response = await axios.get(`${BACKEND_URL}/ads`, {
        params: {
          profile_id: selectedProfile,
          session_id: `session_${Date.now()}`
        }
      });

      const stamp = Date.now();
      const withCacheBuster = (url = '') => {
        if (!url) return url;
        const separator = url.includes('?') ? '&' : '?';
        return `${url}${separator}t=${stamp}`;
      };

      const adWithCacheBuster = {
        ...response.data,
        video_url: withCacheBuster(response.data.video_url),
        hls_url: withCacheBuster(response.data.hls_url || response.data.video_url)
      };

      setCurrentAd(adWithCacheBuster);
      startPersonalizedPlayback(adWithCacheBuster.video_url, 'ready', false);
      setMessage({ type: 'success', text: `Ad served: ${response.data.ad_name}. Use the preview player below to watch the personalized pre-roll before the main content.` });
      
      // Generate VAST XML for the ad
      const vast = generateVastXml(response.data);
      setVastXml(vast);
      
      // Force video reload after state update
      // Load profile details if not already loaded
      if (!profileDetails) {
        const profileResponse = await axios.get(`${BACKEND_URL}/profiles/${selectedProfile}`);
        setProfileDetails(profileResponse.data);
      }
      
      // Refresh logs and stats
      loadLogs();
      loadStats();
      
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error.response?.data?.error || 'Failed to request ad' 
      });
    }
  };

  const handleClearLogs = async () => {
    if (window.confirm('Clear all ad request logs?')) {
      try {
        await axios.post(`${BACKEND_URL}/logs/clear`);
        setLogs([]);
        setStats(null);
        setMessage({ type: 'success', text: 'Logs cleared' });
      } catch (error) {
        setMessage({ type: 'error', text: 'Failed to clear logs' });
      }
    }
  };

  return (
    <PageWrapper>
      <Header>
        <Title>📺 Dynamic Ad Insertion</Title>
      </Header>

      {message && (
        <div style={{ maxWidth: '1400px', margin: '0 auto 20px' }}>
          <Message type={message.type}>{message.text}</Message>
        </div>
      )}

      <Container>
        <Card style={{ gridColumn: '1 / -1' }}>
          <SectionTitle>
            <Icon>🏗️</Icon>
            Reference Architecture Diagram
          </SectionTitle>
          <DiagramImage 
            src="/dai-architecture.png" 
            alt="Dynamic ad insertion architecture diagram" 
          />
        </Card>

        {/* Left Column - User Profile & Ad Request */}
        <div>
          <Card>
            <SectionTitle>
              <Icon>👤</Icon>
              User Profile Selector
            </SectionTitle>

            <Label>Select Viewer Profile</Label>
            <Select 
              value={selectedProfile} 
              onChange={(e) => handleProfileChange(e.target.value)}
            >
              <option value="">-- Select a Profile --</option>
              {profiles.map(profile => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </Select>

            {profileDetails && (
              <ProfileInfo>
                <strong>Profile Details:</strong>
                <div style={{ marginTop: '10px' }}>
                  <div><strong>Age:</strong> {profileDetails.demographics.age}</div>
                  <div><strong>Location:</strong> {profileDetails.demographics.location}</div>
                </div>
                <div style={{ marginTop: '10px' }}>
                  <strong>Interests:</strong>
                  <ProfileRow>
                    {profileDetails.interests.map((interest, idx) => (
                      <Tag key={idx}>{interest}</Tag>
                    ))}
                  </ProfileRow>
                </div>
              </ProfileInfo>
            )}

            <Button onClick={handleRequestAd}>
              🎯 Request Personalized Ad
            </Button>
          </Card>

          {currentAd && (
            <Card style={{ marginTop: '30px' }}>
              <SectionTitle>
                <Icon>📺</Icon>
                Ad Information & Preview
              </SectionTitle>

              <AdInfoPanel>
                <AdInfoRow>
                  <AdLabel>Ad Name:</AdLabel>
                  <AdValue>{currentAd.ad_name}</AdValue>
                </AdInfoRow>
                <AdInfoRow>
                  <AdLabel>Category:</AdLabel>
                  <AdValue>{currentAd.category}</AdValue>
                </AdInfoRow>
                <AdInfoRow>
                  <AdLabel>Duration:</AdLabel>
                  <AdValue>{currentAd.duration}s</AdValue>
                </AdInfoRow>
                <AdInfoRow>
                  <AdLabel>Targeting Reason:</AdLabel>
                  <AdValue>{currentAd.targeting_reason}</AdValue>
                </AdInfoRow>
              </AdInfoPanel>

              <p style={{ marginTop: '20px', color: '#4a4a4a' }}>
                The stitched player below always runs the personalized pre-roll before the hero content clip <strong>file4.mp4</strong>.
              </p>

              <div style={{ marginTop: '20px' }}>
                <Button
                  onClick={handlePlayPersonalizedStream}
                  disabled={!currentAd}
                  style={{ width: 'auto', padding: '12px 28px', margin: '20px auto 10px', display: 'block' }}
                >
                  ▶️ Preview Personalized Stream
                </Button>
                <PlaybackStatus>{playbackStatusText}</PlaybackStatus>
              </div>

              <VideoContainer>
                <VideoPlayer
                  key={playbackKey}
                  ref={personalizedVideoRef}
                  controls={Boolean(personalizedSource)}
                  preload="metadata"
                  poster={currentAd.thumbnail}
                  onPlay={handlePersonalizedPlaybackStart}
                  onEnded={handlePersonalizedPlaybackEnded}
                  onError={(e) => {
                    console.error('Personalized stream playback error:', e);
                    setMessage({ type: 'error', text: 'Personalized stream failed to load. Request a new ad and try again.' });
                  }}
                >
                  {personalizedSource && (
                    <source src={personalizedSource} type="video/mp4" />
                  )}
                  Your browser does not support the video tag.
                </VideoPlayer>
              </VideoContainer>

              {/* VAST XML Display */}
              {vastXml && (
                <div style={{ marginTop: '20px' }}>
                  <SectionTitle>
                    <Icon>📄</Icon>
                    VAST 4.0 Ad Serving XML
                  </SectionTitle>
                  <VastContainer>
                    <VastXml>{vastXml}</VastXml>
                  </VastContainer>
                  <CopyButton onClick={copyVastToClipboard}>
                    📋 Copy VAST XML to Clipboard
                  </CopyButton>
                </div>
              )}
            </Card>
          )}
        </div>

        {/* Right Column - Stats & Logs */}
        <div>
          {stats && (
            <Card>
              <SectionTitle>
                <Icon>📊</Icon>
                Ad Serving Statistics
              </SectionTitle>

              <StatsGrid>
                <StatCard>
                  <StatValue>{stats.total_requests}</StatValue>
                  <StatLabel>Total Requests</StatLabel>
                </StatCard>
                <StatCard>
                  <StatValue>{stats.unique_sessions}</StatValue>
                  <StatLabel>Unique Sessions</StatLabel>
                </StatCard>
                <StatCard>
                  <StatValue>{Object.keys(stats.ads_by_category).length}</StatValue>
                  <StatLabel>Categories</StatLabel>
                </StatCard>
              </StatsGrid>
            </Card>
          )}

          <Card style={{ marginTop: '30px' }}>
            <SectionTitle>
              <Icon>📝</Icon>
              Ad Request Logs
              <Button 
                onClick={handleClearLogs}
                style={{ 
                  marginLeft: 'auto', 
                  marginTop: 0, 
                  width: 'auto', 
                  padding: '8px 15px',
                  fontSize: '0.9rem'
                }}
              >
                Clear Logs
              </Button>
            </SectionTitle>

            <LogsContainer>
              {logs.length === 0 ? (
                <div style={{ textAlign: 'center', color: '#888', padding: '20px' }}>
                  No ad requests yet. Request an ad to see logs.
                </div>
              ) : (
                logs.map((log, idx) => (
                  <LogEntry key={idx}>
                    <LogTime>{new Date(log.timestamp).toLocaleString()}</LogTime>
                    <LogDetails>
                      <strong>{log.profile_name}</strong> → <strong>{log.ad_name}</strong>
                      <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '5px' }}>
                        Category: {log.ad_category} • Duration: {log.duration}s
                      </div>
                    </LogDetails>
                  </LogEntry>
                ))
              )}
            </LogsContainer>
          </Card>
        </div>
      </Container>
    </PageWrapper>
  );
}

export default DynamicAdInsertion;
