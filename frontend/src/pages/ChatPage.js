import React, { useState, useRef, useEffect } from 'react';

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
    
    const currentInput = input;
    setInput('');
    setLoading(true);

    try {
      // Demo response - replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const demoProducts = [
        {
          uniq_id: '1',
          title: 'Modern Ergonomic Chair',
          price: '$299',
          main_category: 'Furniture',
          brand: 'ComfortCorp',
          images: "['https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400']"
        },
        {
          uniq_id: '2', 
          title: 'Coffee Table Set',
          price: '$199',
          main_category: 'Furniture',
          brand: 'ModernHome',
          images: "['https://images.unsplash.com/photo-1549497538-303791108f95?w=400']"
        }
      ];

      const aiMessage = {
        type: 'ai',
        content: `Great choice! For "${currentInput}", I recommend these products. They offer excellent quality and style for your needs.`,
        products: demoProducts
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
      return images[0] || 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400';
    } catch {
      return 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400';
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
                        onError={(e) => {e.target.src = 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400'}}
                      />
                      <div className="product-info">
                        <h5 className="product-title">{product.title}</h5>
                        <p className="product-price">{product.price}</p>
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
