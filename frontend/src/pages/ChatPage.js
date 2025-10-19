import React, { useState, useRef, useEffect } from 'react';
import { chatWithAI } from '../services/api';

const ChatPage = () => {
  const [messages, setMessages] = useState([
    {
      type: 'ai',
      content: "Hello! I'm your AI furniture assistant. What kind of furniture are you looking for today?",
      products: []
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { type: 'user', content: input, products: [] };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await chatWithAI(input);
      const aiMessage = {
        type: 'ai',
        content: response.ai_response,
        products: response.recommended_products || []
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'ai',
        content: 'Sorry, I encountered an error. Please try again.',
        products: []
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const getImageUrl = (imagesString) => {
    try {
      const images = JSON.parse(imagesString.replace(/'/g, '"'));
      return images[0] || '/placeholder.jpg';
    } catch {
      return '/placeholder.jpg';
    }
  };

  return (
    <div className="chat-container">
      <h2 className="page-title">🤖 AI Furniture Recommendations</h2>
      
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.type}`}>
            <div className="message-header">
              <span className="message-icon">
                {message.type === 'ai' ? '🤖' : '👤'}
              </span>
              <span className="message-sender">
                {message.type === 'ai' ? 'AI Assistant' : 'You'}
              </span>
            </div>
            
            <div className="message-content">
              <p>{message.content}</p>
            </div>

            {message.products && message.products.length > 0 && (
              <div className="recommended-products">
                <h4>Recommended Products:</h4>
                <div className="products-grid">
                  {message.products.map((product) => (
                    <div key={product.uniq_id} className="product-card">
                      <img
                        src={getImageUrl(product.images)}
                        alt={product.title}
                        className="product-image"
                      />
                      <div className="product-info">
                        <h5 className="product-title">{product.title}</h5>
                        <p className="product-price">${product.price}</p>
                        <span className="product-category">{product.main_category}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
        
        {loading && (
          <div className="message ai">
            <div className="message-header">
              <span className="message-icon">🤖</span>
              <span className="message-sender">AI Assistant</span>
            </div>
            <div className="message-content">
              <p>Thinking... 🤔</p>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <textarea
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask me about furniture recommendations..."
          rows="3"
        />
        <button
          className="send-button"
          onClick={handleSend}
          disabled={loading || !input.trim()}
        >
          Send 📤
        </button>
      </div>
    </div>
  );
};

export default ChatPage;
