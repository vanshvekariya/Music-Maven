import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 180000, // 3 minutes for AI processing
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.status, error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health check
  async checkHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // Process query
  async processQuery(query, maxResults = 10) {
    const response = await api.post('/query', {
      query,
      max_results: maxResults,
    });
    return response.data;
  },

  // Get system info
  async getSystemInfo() {
    const response = await api.get('/system/info');
    return response.data;
  },

  // Get example queries
  async getExamples() {
    const response = await api.get('/examples');
    return response.data;
  },
};

export default api;
