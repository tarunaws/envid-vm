import React, { useEffect, useMemo, useState, useRef, useCallback } from 'react';
import styled from 'styled-components';

const API_BASE = process.env.REACT_APP_INTERACTIVE_API || 'http://localhost:5055';
const HOTSPOTS_ENABLED = false; // disable bounding-box overlays on the video surface

const Page = styled.section`
  max-width: 1200px;
  margin: 0 auto;
  padding: 72px clamp(20px, 5vw, 60px) 96px;
`;

const Header = styled.header`
  margin-bottom: 32px;
`;

const Title = styled.h1`
  margin: 0 0 0.5rem 0;
  font-size: clamp(2rem, 4vw, 2.6rem);
  font-weight: 800;
  color: #f8fafc;
`;

const Lead = styled.p`
  margin: 0;
  color: var(--text-subtle);
  font-size: 1.05rem;
  line-height: 1.7;
`;

const Layout = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(300px, 1fr);
  gap: 32px;
  align-items: start;
`;

const VideoPanel = styled.div`
  background: rgba(15, 23, 42, 0.7);
  border: 1px solid rgba(var(--accent-2-rgb), 0.22);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 30px 60px rgba(2, 6, 23, 0.45);
`;

const VideoWrap = styled.div`
  position: relative;
  border-radius: 14px;
  overflow: hidden;
  line-height: 0;
`;

const VideoPlayer = styled.video`
  width: 100%;
  border-radius: 14px;
  display: block;
`;

const Hotspot = styled.button`
  position: absolute;
  border: 2px solid rgba(248, 250, 252, 0.8);
  border-radius: 12px;
  background: rgba(59, 130, 246, 0.18);
  color: #f8fafc;
  font-weight: 600;
  padding: 0.2rem 0.4rem;
  font-size: 0.75rem;
  cursor: pointer;
  backdrop-filter: blur(4px);
  transition: background 0.15s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  &:hover {
    background: rgba(59, 130, 246, 0.35);
  }
`;

const Sidebar = styled.aside`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const Panel = styled.div`
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid rgba(var(--accent-2-rgb), 0.25);
  border-radius: 14px;
  padding: 20px;
  box-shadow: 0 18px 36px rgba(2, 6, 23, 0.4);
`;

const CartPanel = styled(Panel)`
  margin-top: 18px;
`;

const PanelTitle = styled.h2`
  margin: 0 0 0.5rem 0;
  font-size: 1.05rem;
`;

const PanelSub = styled.p`
  margin: 0 0 1rem 0;
  color: var(--text-muted);
  font-size: 0.9rem;
`;

const ProductList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const ProductCard = styled.div`
  display: flex;
  gap: 12px;
  padding: 12px;
  border-radius: 12px;
  background: rgba(30, 41, 59, 0.65);
  border: 1px solid rgba(59, 130, 246, 0.15);
`;

const ProductImage = styled.img`
  width: 56px;
  height: 56px;
  object-fit: cover;
  border-radius: 10px;
`;

const ProductInfo = styled.div`
  flex: 1;
`;

const ProductName = styled.p`
  margin: 0;
  font-weight: 600;
`;

const ProductMeta = styled.p`
  margin: 4px 0 0;
  color: rgba(226, 232, 240, 0.75);
  font-size: 0.85rem;
`;

const ProductDescription = styled.p`
  margin: 4px 0 0;
  color: rgba(226, 232, 240, 0.6);
  font-size: 0.8rem;
  line-height: 1.4;
`;

const DetailsToggle = styled.button`
  margin-top: 6px;
  background: none;
  border: none;
  color: var(--text-subtle);
  font-weight: 600;
  font-size: 0.8rem;
  cursor: pointer;
  padding: 0;
  text-decoration: underline;
  &:hover {
    color: var(--accent);
  }
`;

const ExpandedInfo = styled.div`
  margin-top: 8px;
  padding: 10px;
  border-radius: 10px;
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(59, 130, 246, 0.2);
`;

const Tagline = styled.p`
  margin: 0 0 6px;
  font-weight: 600;
  color: var(--text);
`;

const ElevatorPitch = styled.p`
  margin: 0 0 8px;
  color: rgba(226, 232, 240, 0.78);
  font-size: 0.82rem;
  line-height: 1.5;
`;

const BulletList = styled.ul`
  margin: 0 0 8px 18px;
  color: rgba(226, 232, 240, 0.78);
  font-size: 0.8rem;
  line-height: 1.4;
`;

const SocialCaption = styled.p`
  margin: 0 0 8px;
  color: rgba(191, 219, 254, 0.95);
  font-size: 0.8rem;
`;

const KeywordGroup = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
`;

const KeywordPill = styled.span`
  font-size: 0.72rem;
  padding: 0.1rem 0.55rem;
  border-radius: 999px;
  background: rgba(var(--accent-rgb), 0.16);
  color: var(--text-muted);
`;

const ExpandedMeta = styled.div`
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.78rem;
  color: rgba(226, 232, 240, 0.68);
`;

const Chip = styled.span`
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  border-radius: 999px;
  padding: 0.2rem 0.6rem;
  background: rgba(45, 212, 191, 0.15);
  color: #99f6e4;
`;

const ActionButton = styled.button`
  border: none;
  border-radius: 10px;
  padding: 0.45rem 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  background: linear-gradient(135deg, var(--button-accent), var(--button-accent-2));
  color: var(--on-accent);
  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 30px rgba(var(--button-accent-2-rgb), 0.32);
  }
  &:disabled {
    opacity: 0.45;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
  }
`;

const CartList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const EmptyState = styled.p`
  margin: 0;
  color: rgba(226, 232, 240, 0.7);
  font-size: 0.9rem;
`;

const CartItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  border-radius: 10px;
  background: rgba(30, 41, 59, 0.7);
  border: 1px solid rgba(59, 130, 246, 0.15);
`;

const RemoveButton = styled.button`
  background: transparent;
  border: none;
  color: #f87171;
  font-weight: 600;
  cursor: pointer;
`;

const CheckoutButton = styled(ActionButton)`
  width: 100%;
  margin-top: 12px;
`;

const StatusBanner = styled.div`
  margin-top: 12px;
  padding: 0.75rem;
  border-radius: 10px;
  background: rgba(34, 197, 94, 0.15);
  border: 1px solid rgba(34, 197, 94, 0.4);
  color: #bbf7d0;
  font-weight: 600;
`;

const ErrorState = styled.div`
  padding: 1rem;
  border-radius: 12px;
  background: rgba(248, 113, 113, 0.15);
  border: 1px solid rgba(248, 113, 113, 0.35);
  color: #fecdd3;
`;

const LoadingState = styled.p`
  margin: 0;
  color: rgba(226, 232, 240, 0.75);
`;

function formatTime(ms) {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds % 60;
  return `${minutes}:${remaining.toString().padStart(2, '0')}`;
}

function formatPrice(price) {
  if (price === null || price === undefined) {
    return 'Included';
  }
  if (typeof price === 'number') {
    return price === 0 ? 'Free' : `$${price.toFixed(2)}`;
  }
  const value = `${price}`.trim();
  return value.length ? value : 'Included';
}

function InteractiveShoppable() {
  const [catalog, setCatalog] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentMs, setCurrentMs] = useState(0);
  const [cart, setCart] = useState([]);
  const [orderComplete, setOrderComplete] = useState(false);
  const [videoStarted, setVideoStarted] = useState(false);
  const [expandedItems, setExpandedItems] = useState({});
  const videoRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const response = await fetch(`${API_BASE}/interactive/catalog`);
        if (!response.ok) {
          throw new Error('Failed to load catalog');
        }
        const data = await response.json();
        if (!cancelled) {
          setCatalog(data.items || []);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || 'Unable to fetch catalog');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      const position = videoRef.current.currentTime * 1000;
      setCurrentMs(position);
      if (!videoStarted && position > 0) {
        setVideoStarted(true);
      }
    }
  }, [videoStarted]);

  const availableItems = useMemo(() => {
    if (!videoStarted) {
      return [];
    }
    return [...catalog]
      .filter(item => currentMs >= item.startMs)
      .sort((a, b) => (b.startMs ?? 0) - (a.startMs ?? 0));
  }, [catalog, currentMs, videoStarted]);

  const activeItems = useMemo(() => (
    catalog.filter(item => currentMs >= item.startMs && currentMs <= item.endMs)
  ), [catalog, currentMs]);

  const addToCart = (item) => {
    setOrderComplete(false);
    setCart(prev => {
      if (prev.find(entry => entry.id === item.id)) {
        return prev;
      }
      return [...prev, item];
    });
  };

  const removeFromCart = (id) => {
    setCart(prev => prev.filter(item => item.id !== id));
  };

  const handleCheckout = () => {
    if (!cart.length) {
      return;
    }
    setOrderComplete(true);
    setCart([]);
    setExpandedItems({});
  };

  const toggleDetails = (id) => {
    setExpandedItems(prev => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <Page>
      <Header>
        <Title>Interactive & Shoppable Demo</Title>
        <Lead>Tag live video with purchasable overlays, sync each SKU to a local catalog, and simulate a zero-price checkout for showcase environments.</Lead>
      </Header>

      {error && (
        <ErrorState>
          {error}
        </ErrorState>
      )}

      <Layout>
        <VideoPanel>
          <VideoWrap>
            <VideoPlayer
              ref={videoRef}
              controls
              src={`${API_BASE}/interactive/video`}
              onTimeUpdate={handleTimeUpdate}
              crossOrigin="anonymous"
            />
            {HOTSPOTS_ENABLED && activeItems.map(item => (
              <Hotspot
                key={item.id}
                style={{
                  left: `${item.boundingBox.Left * 100}%`,
                  top: `${item.boundingBox.Top * 100}%`,
                  width: `${item.boundingBox.Width * 100}%`,
                  height: `${item.boundingBox.Height * 100}%`,
                }}
                onClick={() => addToCart(item)}
                title={`Add ${item.name}`}
              >
                {item.label}
              </Hotspot>
            ))}
          </VideoWrap>

          <CartPanel>
            <PanelTitle>Demo Cart</PanelTitle>
            <PanelSub>Cart totals stay at $0 to mimic sandbox checkout.</PanelSub>
            <CartList>
              {cart.length === 0 && <EmptyState>No items added yet.</EmptyState>}
              {cart.map(item => {
                const copy = item.aiCopy || {};
                const bulletPoints = Array.isArray(copy.bulletPoints)
                  ? copy.bulletPoints.filter(point => typeof point === 'string' && point.trim())
                  : [];
                const keywords = Array.isArray(copy.keywords)
                  ? copy.keywords.filter(keyword => typeof keyword === 'string' && keyword.trim())
                  : [];

                return (
                  <CartItem key={item.id}>
                    <div>
                      <strong>{item.name}</strong>
                      <ProductMeta>{item.label}</ProductMeta>
                      <DetailsToggle type="button" onClick={() => toggleDetails(item.id)}>
                        {expandedItems[item.id] ? 'Hide details' : 'Show details'}
                      </DetailsToggle>
                      {expandedItems[item.id] && (
                        <ExpandedInfo>
                          {copy.tagline && <Tagline>{copy.tagline}</Tagline>}
                          {copy.elevatorPitch && (
                            <ElevatorPitch>{copy.elevatorPitch}</ElevatorPitch>
                          )}
                          {bulletPoints.length > 0 && (
                            <BulletList>
                              {bulletPoints.map(point => (
                                <li key={point}>{point}</li>
                              ))}
                            </BulletList>
                          )}
                          {copy.socialCaption && (
                            <SocialCaption>{copy.socialCaption}</SocialCaption>
                          )}
                          {keywords.length > 0 && (
                            <KeywordGroup>
                              {keywords.map(keyword => (
                                <KeywordPill key={keyword}>{keyword}</KeywordPill>
                              ))}
                            </KeywordGroup>
                          )}
                          {item.description && (
                            <ProductDescription>{item.description}</ProductDescription>
                          )}
                          <ExpandedMeta>
                            <span>Scene unlock: {formatTime(item.startMs || 0)}</span>
                            <span>Price: {formatPrice(item.price)}</span>
                          </ExpandedMeta>
                        </ExpandedInfo>
                      )}
                    </div>
                    <RemoveButton type="button" onClick={() => removeFromCart(item.id)}>Remove</RemoveButton>
                  </CartItem>
                );
              })}
            </CartList>
            <CheckoutButton type="button" onClick={handleCheckout} disabled={!cart.length}>
              Complete $0 Purchase
            </CheckoutButton>
            {orderComplete && (
              <StatusBanner>
                Purchase confirmed. Receipt emailed with $0 total.
              </StatusBanner>
            )}
          </CartPanel>
        </VideoPanel>

        <Sidebar>
          <Panel>
            <PanelTitle>Entire Shoppable Catalog</PanelTitle>
            {loading ? (
              <LoadingState>Loading catalog…</LoadingState>
            ) : (
              <ProductList>
                {!videoStarted && (
                  <EmptyState>Press play to reveal products as their scenes begin.</EmptyState>
                )}
                {videoStarted && availableItems.length === 0 && (
                  <EmptyState>Keep watching to unlock more items.</EmptyState>
                )}
                {availableItems.map(item => (
                  <ProductCard key={item.id}>
                    <ProductImage src={item.image} alt={item.name} onError={(e) => { e.currentTarget.src = '/usecases/placeholder.svg'; }} />
                    <ProductInfo>
                      <ProductName>{item.name}</ProductName>
                      <ProductMeta>{item.description}</ProductMeta>
                      <Chip>{formatTime(item.timestampMs)}</Chip>
                    </ProductInfo>
                    <ActionButton type="button" onClick={() => addToCart(item)}>
                      {item.price === 0 ? 'Add (Free)' : 'Add to Cart'}
                    </ActionButton>
                  </ProductCard>
                ))}
              </ProductList>
            )}
          </Panel>
        </Sidebar>
      </Layout>
    </Page>
  );
}

export default InteractiveShoppable;
