import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styled from 'styled-components';

const PageWrapper = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 50px;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  color: #1a1a1a;
  margin-bottom: 10px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const Subtitle = styled.p`
  font-size: 1.1rem;
  color: #666;
`;

const Section = styled.div`
  background: white;
  border-radius: 12px;
  padding: 30px;
  margin-bottom: 30px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
`;

const SectionTitle = styled.h2`
  font-size: 1.5rem;
  color: #333;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const Icon = styled.span`
  font-size: 1.8rem;
`;

const TextArea = styled.textarea`
  width: 100%;
  padding: 16px;
  border: 2px solid #ddd;
  border-radius: 8px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  min-height: 120px;
  margin-bottom: 20px;
  
  &:focus {
    outline: none;
    border-color: #667eea;
  }
`;

const Button = styled.button`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 14px 32px;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s;
  
  &:hover {
    transform: translateY(-2px);
  }
  
  &:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
  }
`;

const Message = styled.div`
  padding: 12px 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  background: ${props => props.type === 'error' ? '#fee' : props.type === 'success' ? '#efe' : '#e7f3ff'};
  color: ${props => props.type === 'error' ? '#c33' : props.type === 'success' ? '#363' : '#0066cc'};
  border-left: 4px solid ${props => props.type === 'error' ? '#c33' : props.type === 'success' ? '#363' : '#0066cc'};
`;

const VideoPreview = styled.div`
  margin-top: 30px;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
`;

const VideoTitle = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 15px 20px;
  font-weight: 600;
  font-size: 1.1rem;
`;

const VideoPlayer = styled.video`
  width: 100%;
  max-height: 600px;
  display: block;
`;

const PromptDisplay = styled.div`
  background: white;
  padding: 15px 20px;
  color: #666;
  font-style: italic;
  border-top: 1px solid #eee;
`;

const HistoryGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 20px;
`;

const HistoryCard = styled.div`
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s;
  cursor: pointer;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
  }
`;

const HistoryVideoThumb = styled.video`
  width: 100%;
  height: 180px;
  object-fit: cover;
  background: #000;
`;

const HistoryInfo = styled.div`
  padding: 15px;
`;

const HistoryPrompt = styled.div`
  font-size: 0.9rem;
  color: #333;
  margin-bottom: 8px;
  font-weight: 500;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
`;

const HistoryMeta = styled.div`
  font-size: 0.8rem;
  color: #999;
  margin-top: 8px;
`;

const DeleteButton = styled.button`
  background: #ff4444;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 6px 12px;
  font-size: 0.85rem;
  cursor: pointer;
  margin-top: 8px;
  transition: background 0.2s;
  
  &:hover {
    background: #cc0000;
  }
`;

const LoadingSpinner = styled.div`
  text-align: center;
  padding: 40px;
  font-size: 1.2rem;
  color: #667eea;
  
  &:after {
    content: '⏳';
    animation: spin 2s linear infinite;
  }
  
  @keyframes spin {
    0% { content: '⏳'; }
    25% { content: '⌛'; }
    50% { content: '⏳'; }
    75% { content: '⌛'; }
    100% { content: '⏳'; }
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 60px 20px;
  color: #999;
`;

const FieldLabel = styled.label`
  display: block;
  font-weight: 600;
  color: #444;
  margin-bottom: 8px;
`;

const ReferenceGallery = styled.div`
  border-top: 1px solid #eee;
  padding: 20px;
  background: #fafbff;
`;

const GalleryTitle = styled.h3`
  margin: 0 0 12px;
  font-size: 1rem;
  color: #333;
`;

const GalleryGrid = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
`;

const GalleryImage = styled.img`
  width: 110px;
  height: 110px;
  object-fit: cover;
  border-radius: 10px;
  border: 1px solid #ddd;
  background: #fff;
`;

const ReferenceBadge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  color: #3741a0;
  background: rgba(102, 126, 234, 0.12);
  padding: 4px 10px;
  border-radius: 999px;
  margin-top: 10px;
`;

const ReferenceStrip = styled.div`
  display: flex;
  gap: 6px;
  margin-top: 10px;
`;

const ReferenceStripThumb = styled.img`
  width: 48px;
  height: 48px;
  border-radius: 8px;
  object-fit: cover;
  border: 1px solid #eee;
`;


const BACKEND_URL = 'http://localhost:5009';

function VideoGeneration() {
  const [prompt, setPrompt] = useState('');
  const [generating, setGenerating] = useState(false);
  const [message, setMessage] = useState(null);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/history`);
      setHistory(response.data.history || []);
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const handleGenerate = async () => {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt.length) {
      setMessage({ type: 'error', text: 'Enter a prompt to generate a video.' });
      return;
    }

    setGenerating(true);
    setMessage({ type: 'info', text: 'Starting video generation... This may take 1-2 minutes.' });
    setCurrentVideo(null);

    try {
      const response = await axios.post(`${BACKEND_URL}/generate-video`, { prompt: trimmedPrompt });

      const generationId = response.data.id;
      
      setMessage({ 
        type: 'info', 
        text: 'Video generation in progress... Please wait.' 
      });

      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await axios.get(`${BACKEND_URL}/check-status/${generationId}`);
          const statusData = statusResponse.data;
          
          if (statusData.status === 'completed') {
            clearInterval(pollInterval);
            setMessage({ 
              type: 'success', 
              text: 'Video generated successfully!' 
            });
            setCurrentVideo(statusData);
            setGenerating(false);
            loadHistory();
          } else if (statusData.status === 'failed') {
            clearInterval(pollInterval);
            setMessage({ 
              type: 'error', 
              text: `Generation failed: ${statusData.error || 'Unknown error'}` 
            });
            setGenerating(false);
          }
          // If still pending, continue polling
        } catch (error) {
          clearInterval(pollInterval);
          setMessage({ 
            type: 'error', 
            text: 'Error checking video status' 
          });
          setGenerating(false);
        }
      }, 5000); // Poll every 5 seconds
      
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error.response?.data?.error || 'Failed to start video generation' 
      });
      setGenerating(false);
    }
  };

  const handleDeleteFromHistory = async (generationId, promptText) => {
    const promptLabel = promptText && promptText.trim().length > 0
      ? promptText
      : 'Prompt unavailable';

    if (!window.confirm(`Delete this video?\n"${promptLabel}"`)) {
      return;
    }

    try {
      await axios.delete(`${BACKEND_URL}/history/${generationId}?delete_s3=false`);
      setMessage({ type: 'success', text: 'Video removed from history' });
      loadHistory();
      
      if (currentVideo && currentVideo.id === generationId) {
        setCurrentVideo(null);
      }
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: 'Failed to delete from history' 
      });
    }
  };

  return (
    <PageWrapper>
      <Header>
        <Title>🎬 AI Based Video Generation</Title>
        <Subtitle>
          Describe the motion you need and let Amazon Nova Reel synthesize a six-second cinematic clip.
        </Subtitle>
      </Header>

      {message && (
        <Message type={message.type}>{message.text}</Message>
      )}

      <Section>
        <SectionTitle>
          <Icon>✨</Icon>
          Design Your Video Brief
        </SectionTitle>

        <FieldLabel htmlFor="videoPrompt">Describe the motion (prompt)</FieldLabel>
        <TextArea
          id="videoPrompt"
          placeholder="Describe the video you want to generate...&#10;&#10;Example: Capture a drone fly-through over neon-lit skyscrapers at night with smooth parallax."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={generating}
        />
        <Button
          onClick={handleGenerate}
          disabled={generating || !prompt.trim()}
        >
          {generating ? 'Generating Video...' : '🎥 Generate Video'}
        </Button>

        {generating && (
          <LoadingSpinner>
            Generating your video...
          </LoadingSpinner>
        )}

        {currentVideo && (
          <VideoPreview>
            <VideoTitle>Generated Video</VideoTitle>
            <VideoPlayer controls autoPlay loop crossOrigin="anonymous">
              <source src={`${BACKEND_URL}/video/${currentVideo.id}`} type="video/mp4" />
              Your browser does not support the video tag.
            </VideoPlayer>
            <PromptDisplay>
              <strong>Prompt:</strong>{' '}
              {currentVideo.prompt && currentVideo.prompt.trim().length > 0
                ? currentVideo.prompt
                : 'Prompt unavailable'}
            </PromptDisplay>
            {currentVideo.reference_images && currentVideo.reference_images.length > 0 && (
              <ReferenceGallery>
                <GalleryTitle>Reference Images</GalleryTitle>
                <GalleryGrid>
                  {currentVideo.reference_images
                    .filter((img) => img.url)
                    .map((img) => (
                      <GalleryImage
                        key={img.s3_key || img.url}
                        src={img.url}
                        alt={img.filename || 'Reference image'}
                      />
                    ))}
                </GalleryGrid>
              </ReferenceGallery>
            )}
          </VideoPreview>
        )}
      </Section>

      <Section>
        <SectionTitle>
          <Icon>📚</Icon>
          Generation History ({history.length})
        </SectionTitle>

        {history.length === 0 ? (
          <EmptyState>
            <div style={{ fontSize: '3rem', marginBottom: '20px' }}>🎬</div>
            <h3>No videos generated yet</h3>
            <p>Enter a prompt above to generate your first AI video</p>
          </EmptyState>
        ) : (
          <HistoryGrid>
            {history.map((item) => (
              <HistoryCard key={item.id} onClick={() => setCurrentVideo(item)}>
                <HistoryVideoThumb controls crossOrigin="anonymous">
                  <source src={`${BACKEND_URL}/video/${item.id}`} type="video/mp4" />
                </HistoryVideoThumb>
                <HistoryInfo>
                  <HistoryPrompt>
                    {item.prompt && item.prompt.trim().length > 0
                      ? item.prompt
                      : 'Prompt unavailable'}
                  </HistoryPrompt>
                  <HistoryMeta>
                    {new Date(item.timestamp).toLocaleString()} • {item.resolution || '1280x720'}
                  </HistoryMeta>
                  {item.reference_images && item.reference_images.length > 0 && (
                    <>
                      <ReferenceBadge>
                        🖼️ {item.reference_images.length} reference image
                        {item.reference_images.length > 1 ? 's' : ''}
                      </ReferenceBadge>
                      <ReferenceStrip>
                        {item.reference_images
                          .filter((img) => img.url)
                          .slice(0, 3)
                          .map((img) => (
                            <ReferenceStripThumb
                              key={img.s3_key || img.url}
                              src={img.url}
                              alt={img.filename || 'Reference image thumbnail'}
                            />
                          ))}
                      </ReferenceStrip>
                    </>
                  )}
                  <DeleteButton 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteFromHistory(item.id, item.prompt);
                    }}
                  >
                    🗑️ Delete
                  </DeleteButton>
                </HistoryInfo>
              </HistoryCard>
            ))}
          </HistoryGrid>
        )}
      </Section>
    </PageWrapper>
  );
}

export default VideoGeneration;
