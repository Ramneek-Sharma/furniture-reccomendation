const API_BASE_URL = 'http://localhost:8000';

export const searchProducts = async (query, maxResults = 10) => {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      max_results: maxResults,
    }),
  });
  
  if (!response.ok) {
    throw new Error('Search request failed');
  }
  
  return response.json();
};

export const getRecommendations = async (productId, numRecommendations = 5) => {
  const response = await fetch(`${API_BASE_URL}/recommendations`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      product_id: productId,
      num_recommendations: numRecommendations,
    }),
  });
  
  if (!response.ok) {
    throw new Error('Recommendations request failed');
  }
  
  return response.json();
};

export const chatWithAI = async (message, userPreferences = {}) => {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      user_preferences: userPreferences,
    }),
  });
  
  if (!response.ok) {
    throw new Error('Chat request failed');
  }
  
  return response.json();
};

export const classifyImage = async (imageUrl) => {
  const response = await fetch(`${API_BASE_URL}/classify-image`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_url: imageUrl,
    }),
  });
  
  if (!response.ok) {
    throw new Error('Image classification failed');
  }
  
  return response.json();
};

export const getAnalytics = async () => {
  const response = await fetch(`${API_BASE_URL}/analytics`);
  
  if (!response.ok) {
    throw new Error('Analytics request failed');
  }
  
  return response.json();
};

export const getAllProducts = async () => {
  const response = await fetch(`${API_BASE_URL}/products`);
  
  if (!response.ok) {
    throw new Error('Products request failed');
  }
  
  return response.json();
};
