import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

import styled from 'styled-components';

const ChatContainer = styled.div`
  position: fixed;
  bottom: ${props => props.$isOpen ? '20px' : '-600px'};
  right: 20px;
  width: 400px;
  height: 500px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  display: flex;
  flex-direction: column;
  transition: bottom 0.3s ease-in-out;
  z-index: 1000;
  
  @media (max-width: 768px) {
    width: calc(100% - 40px);
    height: 400px;
  }
`;

const ChatHeader = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 16px;
  border-radius: 16px 16px 0 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ChatTitle = styled.h3`
  margin: 0;
  font-size: 1.1rem;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const CloseButton = styled.button`
  background: rgba(255, 255, 255, 0.2);
  border: none;
  color: white;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  transition: background 0.2s;
  
  &:hover {
    background: rgba(255, 255, 255, 0.3);
  }
`;

const ChatMessages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const Message = styled.div`
  background: ${props => props.isUser ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#f0f0f0'};
  color: ${props => props.isUser ? 'white' : '#333'};
  padding: 12px 16px;
  border-radius: ${props => props.isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px'};
  max-width: 80%;
  align-self: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  word-wrap: break-word;
  line-height: 1.6;
  font-size: 0.95rem;
`;

const TypingIndicator = styled.div`
  background: #f0f0f0;
  color: #666;
  padding: 12px 16px;
  border-radius: 16px 16px 16px 4px;
  max-width: 80%;
  align-self: flex-start;
  
  &::after {
    content: '...';
    animation: typing 1.5s infinite;
  }
  
  @keyframes typing {
    0%, 20% { content: '.'; }
    40% { content: '..'; }
    60%, 100% { content: '...'; }
  }
`;

const ChatInputContainer = styled.div`
  padding: 16px;
  border-top: 1px solid #e0e0e0;
  display: flex;
  gap: 8px;
`;

const ChatInput = styled.input`
  flex: 1;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 24px;
  font-size: 0.95rem;
  outline: none;
  transition: border-color 0.2s;
  
  &:focus {
    border-color: #667eea;
  }
`;

const SendButton = styled.button`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 0 20px;
  border-radius: 24px;
  cursor: pointer;
  font-weight: 600;
  transition: opacity 0.2s;
  
  &:hover:not(:disabled) {
    opacity: 0.9;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ToggleButton = styled.button`
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  cursor: pointer;
  font-size: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  transition: transform 0.2s;
  z-index: 999;
  
  &:hover {
    transform: scale(1.1);
  }
`;

const SuggestedQuestions = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 0 16px 8px;
`;

const SuggestionButton = styled.button`
  background: #f8f9fa;
  border: 2px solid #e0e0e0;
  padding: 10px 12px;
  border-radius: 12px;
  cursor: pointer;
  font-size: 0.85rem;
  color: #666;
  text-align: left;
  transition: all 0.2s;
  
  &:hover {
    border-color: #667eea;
    color: #667eea;
    background: #f0f0ff;
  }
`;

const BACKEND_URL = 'http://localhost:5008';

function AskMe({ isOpenProp = false, onClose = null }) {
  const [isOpen, setIsOpen] = useState(isOpenProp);
  const [messages, setMessages] = useState([
    {
      text: "👋 Hi! I'm AskMe, your MediaGenAI assistant. Ask me about any use case or feature!",
      isUser: false
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const suggestedQuestions = [
    "What is AI Subtitle Generation?",
    "How does Semantic Search work?",
  "Tell me about Personalized Trailer",
    "What are all the available use cases?"
  ];

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    setIsOpen(isOpenProp);
  }, [isOpenProp]);

  const scrollToBottom = () => {
    const node = messagesEndRef.current;
    if (node && typeof node.scrollIntoView === 'function') {
      node.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleToggle = () => {
    const newState = !isOpen;
    setIsOpen(newState);
    if (onClose && !newState) {
      onClose();
    }
  };

  const handleSend = async () => {
    if (!inputText.trim() || isTyping) return;

    const userMessage = inputText.trim();
    setMessages(prev => [...prev, { text: userMessage, isUser: true }]);
    setInputText('');
    setIsTyping(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/askme`, {
        question: userMessage
      });

      setMessages(prev => [...prev, {
        text: response.data.answer,
        isUser: false,
        sources: response.data.sources
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        text: "Sorry, I encountered an error. Please try again.",
        isUser: false
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestion = (question) => {
    setInputText(question);
  };

  return (
    <>
  <ChatContainer $isOpen={isOpen}>
        <ChatHeader>
          <ChatTitle>
            <span>🤖</span>
            AskMe
          </ChatTitle>
          <CloseButton onClick={handleToggle}>×</CloseButton>
        </ChatHeader>
        
        <ChatMessages>
          {messages.map((msg, idx) => (
            <Message key={idx} isUser={msg.isUser}>
              {msg.text}
            </Message>
          ))}
          {isTyping && <TypingIndicator>Thinking</TypingIndicator>}
          <div ref={messagesEndRef} />
        </ChatMessages>

        {messages.length === 1 && !isTyping && (
          <SuggestedQuestions>
            {suggestedQuestions.map((q, idx) => (
              <SuggestionButton key={idx} onClick={() => handleSuggestion(q)}>
                {q}
              </SuggestionButton>
            ))}
          </SuggestedQuestions>
        )}
        
        <ChatInputContainer>
          <ChatInput
            type="text"
            placeholder="Ask me anything..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isTyping}
          />
          <SendButton onClick={handleSend} disabled={isTyping || !inputText.trim()}>
            Send
          </SendButton>
        </ChatInputContainer>
      </ChatContainer>
      
      {!isOpen && (
        <ToggleButton onClick={handleToggle}>
          💬
        </ToggleButton>
      )}
    </>
  );
}

export default AskMe;
