// api.ts
const BASE_URL = 'http://192.168.34.152:5000/api'; // ⚠️ use your local IP

export const API = {
  get: async (endpoint: string) => {
    const res = await fetch(`${BASE_URL}${endpoint}`);
    return res.json();
  },

  post: async (endpoint: string, data: any) => {
    const res = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return res.json();
  },
  
  // Add PUT, DELETE, etc. as needed
};
