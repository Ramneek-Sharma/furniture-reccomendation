import React, { useState } from 'react';

const SearchPage = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      // Demo results - replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const demoResults = [
        {
          uniq_id: '1',
          title: 'Modern Ergonomic Office Chair',
          price: '$299',
          main_category: 'Furniture',
          brand: 'ComfortCorp',
          images: "['https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400']",
          similarity_score: 0.95
        },
        {
          uniq_id: '2',
          title: 'Adjustable Standing Desk',
          price: '$599', 
          main_category: 'Furniture',
          brand: 'WorkSpace',
          images: "['https://images.unsplash.com/photo-1549497538-303791108f95?w=400']",
          similarity_score: 0.87
        },
        {
          uniq_id: '3',
          title: 'Storage Cabinet with Shelves',
          price: '$199',
          main_category: 'Storage',
          brand: 'OrganizeIt',
          images: "['https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400']",
          similarity_score: 0.73
        }
      ];
      
      setResults(demoResults);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
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
    <div className="search-container">
      <h2 className="page-title">🔍 Search Products</h2>
      
      <div className="search-input-container">
        <input
          type="text"
          className="search-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for furniture..."
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
        />
        <button 
          className="search-button"
          onClick={handleSearch}
          disabled={loading || !query.trim()}
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {results.length > 0 && (
        <div className="search-results">
          <h3>Found {results.length} products</h3>
          <div className="products-grid">
            {results.map((product) => (
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
                  <p className="product-brand">Brand: {product.brand}</p>
                  {product.similarity_score && (
                    <p className="similarity-score">
                      Match: {(product.similarity_score * 100).toFixed(1)}%
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchPage;
