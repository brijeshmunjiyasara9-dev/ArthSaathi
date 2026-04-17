// src/pages/History.jsx
// Past assessments list for a user (demo with localStorage session IDs)

import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getChatHistory, getGlobalHistory } from '../services/api';
import StressGauge from '../components/StressGauge';
import AdviceCard from '../components/AdviceCard';

export default function History() {
  const [sessions, setSessions]         = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [detail, setDetail]             = useState(null);
  const [loading, setLoading]           = useState(false);
  const [sessionMeta, setSessionMeta]   = useState({}); 

  // Load session IDs from backend globally
  useEffect(() => {
    getGlobalHistory()
      .then(data => {
        if (!data || data.length === 0) {
          setSessions([]);
          return;
        }
        
        // Save the raw session info metadata since we get it from DB for free
        const validSids = [];
        const metadataMap = {};
        
        for (const record of data) {
           if (record.conversation_id) {
             validSids.push(record.conversation_id);
             metadataMap[record.conversation_id] = {
                completed: true,
                composite: record.composite_score,
                isStressed: record.financial_stress_prob > 0.5 || record.food_stress_prob > 0.5 || record.debt_stress_prob > 0.5 || record.health_stress_prob > 0.2
             };
           }
        }
        setSessions(validSids);
        setSessionMeta(metadataMap);
      })
      .catch(e => {
        console.error("Failed to fetch global history", e);
        setSessions([]);
      });
  }, []);

  async function loadSession(sessionId) {
    setLoading(true);
    setSelectedSession(sessionId);
    try {
      const data = await getChatHistory(sessionId);
      setDetail(data);
      // Cache metadata for the sidebar card
      if (data) {
        setSessionMeta(prev => ({
          ...prev,
          [sessionId]: {
            completed:  data.predictions != null,
            composite:  data.predictions?.composite_stress_score,
            isStressed: data.predictions?.is_stressed,
          },
        }));
      }
    } catch {
      setDetail(null);
    } finally {
      setLoading(false);
    }
  }

  function clearHistory() {
    // Note: since this is tied to global DB right now rather than local storage,
    // "Clearing" just hides it until refresh if we don't send a DELETE to backend.
    localStorage.removeItem('arthsaathi_sessions');
    setSessions([]);
    setSelectedSession(null);
    setDetail(null);
  }

  if (sessions.length === 0) {
    return (
      <div className="container" style={{ padding: '80px 20px', textAlign: 'center' }}>
        <div style={{ fontSize: '4rem', marginBottom: 16 }}>🗂️</div>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: 10 }}>No past assessments</h2>
        <p style={{ color: 'var(--color-text-muted)', marginBottom: 28 }}>
          Your session history will appear here after you complete an assessment.
        </p>
        <Link to="/chat" className="btn btn-primary">
          Start Your First Assessment →
        </Link>
      </div>
    );
  }

  return (
    <div className="container" style={{ padding: '32px 20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 className="section-title">Assessment History</h1>
          <p className="section-subtitle">{sessions.length} past session{sessions.length !== 1 ? 's' : ''}</p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <Link to="/chat" className="btn btn-primary" style={{ fontSize: '0.88rem' }}>
            + New Assessment
          </Link>
          <button onClick={clearHistory} className="btn btn-ghost" style={{ fontSize: '0.88rem', color: 'var(--color-red)', borderColor: 'rgba(244,63,94,0.3)' }}>
            🗑 Clear
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: selectedSession ? '280px 1fr' : '1fr', gap: 20 }}>
        {/* Session list */}
        <div>
          {sessions.map((sid, i) => {
            const meta = sessionMeta[sid] || {};
            return (
              <div
                key={sid}
                className="history-card"
                style={{
                  marginBottom: 10,
                  borderColor: selectedSession === sid ? 'var(--color-primary)' : 'var(--color-border)',
                  background:  selectedSession === sid ? 'var(--color-surface-2)' : '',
                  cursor: 'pointer',
                }}
                onClick={() => loadSession(sid)}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.92rem', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                    Session #{sessions.length - i}
                    {meta.completed && (
                      <span style={{ fontSize: '0.65rem', fontWeight: 700, padding: '2px 7px', borderRadius: 9999, background: 'rgba(16,185,129,0.15)', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)' }}>✓ Done</span>
                    )}
                    {meta.isStressed && (
                      <span style={{ fontSize: '0.65rem', fontWeight: 700, padding: '2px 7px', borderRadius: 9999, background: 'rgba(244,63,94,0.12)', color: '#f43f5e', border: '1px solid rgba(244,63,94,0.25)' }}>⚠ Stressed</span>
                    )}
                  </div>
                  <div style={{ fontSize: '0.73rem', color: 'var(--color-text-dim)', fontFamily: 'monospace' }}>{sid.slice(0, 12)}…</div>
                  {meta.composite != null && (
                    <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
                      Composite: <strong style={{ color: 'var(--color-accent)' }}>{meta.composite.toFixed(2)}</strong> / 4
                    </div>
                  )}
                </div>
                <div style={{ color: 'var(--color-text-muted)', fontSize: '1.1rem' }}>›</div>
              </div>
            );
          })}
        </div>

        {/* Detail panel */}
        {selectedSession && (
          <div>
            {loading ? (
              <div style={{ display: 'flex', justifyContent: 'center', padding: 60 }}>
                <div className="spinner" style={{ width: 36, height: 36 }} />
              </div>
            ) : detail ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                {detail.predictions && (
                  <div className="card">
                    <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: 'var(--color-accent)' }}>
                      📊 Stress Analysis
                    </h3>
                    <StressGauge predictions={detail.predictions} />
                  </div>
                )}
                {detail.messages?.length > 0 && (
                  <div className="card">
                    <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: 'var(--color-accent)' }}>
                      💬 Conversation ({detail.messages.length} messages)
                    </h3>
                    <div style={{ maxHeight: 280, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
                      {detail.messages.map((m) => (
                        <div key={m.id} style={{
                          padding: '8px 12px',
                          borderRadius: 8,
                          fontSize: '0.85rem',
                          background: m.role === 'user' ? 'rgba(79,142,247,0.1)' : 'var(--color-surface-2)',
                          border: `1px solid ${m.role === 'user' ? 'rgba(79,142,247,0.2)' : 'var(--color-border)'}`,
                          textAlign: m.role === 'user' ? 'right' : 'left',
                        }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--color-text-dim)', marginBottom: 3 }}>
                            {m.role === 'user' ? '👤 You' : '🪙 ArthSaathi'}
                          </div>
                          {m.content.slice(0, 200)}{m.content.length > 200 ? '…' : ''}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {/* Advice from last assistant message */}
                {detail.messages?.find(m => m.role === 'assistant' && m.content.includes('Action Plan')) && (
                  <AdviceCard
                    advice={detail.messages.filter(m => m.role === 'assistant').pop()?.content}
                    predictions={detail.predictions}
                  />
                )}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: 40, color: 'var(--color-text-muted)' }}>
                Could not load session details.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
