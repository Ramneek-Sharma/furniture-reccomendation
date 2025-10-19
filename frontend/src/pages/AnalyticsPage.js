import React, { useState, useEffect } from 'react';

const AnalyticsPage = () => {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        // Demo data - replace with actual API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const demoAnalytics = {
          total_products: 312,
          categories: {
            'Home & Kitchen': 98,
            'Furniture': 76,
            'Storage & Organization': 54,
            'Patio, Lawn & Garden': 42,
            'Home Decor': 32,
            'Office Products': 10
          },
          brands: {
            'GOYMFK': 15,
            'subrtex': 12,
            'MUYETOL': 10,
            'VEWETOL': 8,
            'JOIN IRON Store': 7,
            'Other Brands': 260
          },
          price_ranges: {
            '$0-$25': 89,
            '$25-$50': 78,
            '$50-$100': 69,
            '$100-$200': 54,
            '$200+': 22
          },
          materials: {
            'Metal': 45,
            'Wood': 38,
            'Plastic': 32,
            'Fabric': 28,
            'Glass': 15,
            'Other': 154
          },
          colors: {
            'Black': 42,
            'White': 38,
            'Brown': 35,
            'Gray': 28,
            'Blue': 22,
            'Other': 147
          },
          statistics: {
            avg_price: 87.45,
            median_price: 49.99,
            products_with_price: 215
          }
        };
        
        setAnalytics(demoAnalytics);
      } catch (error) {
        console.error('Error fetching analytics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  if (loading) {
    return <div className="loading">Loading analytics... ⏳</div>;
  }

  if (!analytics) {
    return <div className="error">Failed to load analytics data.</div>;
  }

  return (
    <div className="analytics-container">
      <h2 className="page-title">📊 Product Analytics Dashboard</h2>

      <div className="stats-grid">
        <div className="stat-card">
          <h3>{analytics.total_products}</h3>
          <p>Total Products</p>
        </div>
        <div className="stat-card">
          <h3>${analytics.statistics?.avg_price?.toFixed(2)}</h3>
          <p>Average Price</p>
        </div>
        <div className="stat-card">
          <h3>{Object.keys(analytics.categories || {}).length}</h3>
          <p>Categories</p>
        </div>
        <div className="stat-card">
          <h3>{Object.keys(analytics.brands || {}).length}</h3>
          <p>Brands</p>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-container">
          <h3>Top Categories</h3>
          <div className="chart">
            {Object.entries(analytics.categories || {}).slice(0, 5).map(([category, count]) => (
              <div key={category} className="bar-item">
                <span className="bar-label">{category}</span>
                <div className="bar">
                  <div 
                    className="bar-fill" 
                    style={{ width: `${(count / Math.max(...Object.values(analytics.categories))) * 100}%` }}
                  ></div>
                  <span className="bar-value">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="chart-container">
          <h3>Price Distribution</h3>
          <div className="chart">
            {Object.entries(analytics.price_ranges || {}).map(([range, count]) => (
              <div key={range} className="bar-item">
                <span className="bar-label">{range}</span>
                <div className="bar">
                  <div 
                    className="bar-fill" 
                    style={{ width: `${(count / Math.max(...Object.values(analytics.price_ranges))) * 100}%` }}
                  ></div>
                  <span className="bar-value">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="chart-container">
          <h3>Top Brands</h3>
          <div className="chart">
            {Object.entries(analytics.brands || {}).slice(0, 5).map(([brand, count]) => (
              <div key={brand} className="bar-item">
                <span className="bar-label">{brand}</span>
                <div className="bar">
                  <div 
                    className="bar-fill" 
                    style={{ width: `${(count / Math.max(...Object.values(analytics.brands))) * 100}%` }}
                  ></div>
                  <span className="bar-value">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="chart-container">
          <h3>Popular Materials</h3>
          <div className="chart">
            {Object.entries(analytics.materials || {}).slice(0, 5).map(([material, count]) => (
              <div key={material} className="bar-item">
                <span className="bar-label">{material}</span>
                <div className="bar">
                  <div 
                    className="bar-fill" 
                    style={{ width: `${(count / Math.max(...Object.values(analytics.materials))) * 100}%` }}
                  ></div>
                  <span className="bar-value">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPage;
