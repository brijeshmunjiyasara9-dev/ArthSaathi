// src/services/api.js — Axios API client for ArthSaathi

import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Chat ──────────────────────────────────────────────────────────────────────

export const startChat = async (userName = 'Guest', userId = null) => {
  const { data } = await api.post('/api/chat/start', {
    user_name: userName,
    user_id: userId,
  });
  return data; // { session_id, message }
};

export const sendMessage = async (sessionId, message) => {
  const { data } = await api.post('/api/chat/message', {
    session_id: sessionId,
    message,
  });
  return data; // { session_id, reply, step, is_complete, predictions, advice }
};

export const getChatHistory = async (sessionId) => {
  const { data } = await api.get(`/api/chat/${sessionId}`);
  return data; // { session_id, messages, profile, predictions }
};

export const getGlobalHistory = async () => {
  const { data } = await api.get('/api/assessments/global');
  return data; 
};

// ── Users ────────────────────────────────────────────────────────────────────

export const createUser = async (payload) => {
  const { data } = await api.post('/api/users', payload);
  return data;
};

export const getUser = async (userId) => {
  const { data } = await api.get(`/api/users/${userId}`);
  return data;
};

// ── Assessments ───────────────────────────────────────────────────────────────

export const getUserAssessments = async (userId) => {
  const { data } = await api.get(`/api/assessments/user/${userId}`);
  return data;
};

// ── Health check ───────────────────────────────────────────────────────────────

export const checkHealth = async () => {
  const { data } = await api.get('/health');
  return data;
};

export default api;
