// src/pages/Chat.jsx
// Main chat interface with sidebar (progress + gauges) and chat window

import { useState, useEffect, useRef } from 'react';
import { startChat, sendMessage } from '../services/api';
import MessageBubble, { TypingIndicator } from '../components/MessageBubble';
import ProgressBar from '../components/ProgressBar';
import StressGauge from '../components/StressGauge';
import AdviceCard from '../components/AdviceCard';

const TOTAL_STEPS = 15;

export default function Chat() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [step, setStep] = useState(1);
  const [isComplete, setIsComplete] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [advice, setAdvice] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'results'

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    initSession();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    if (isComplete) setActiveTab('results');
  }, [isComplete]);

  async function initSession() {
    try {
      setLoading(true);
      const data = await startChat('Guest');
      setSessionId(data.session_id);
      setMessages([{ role: 'assistant', content: data.message, id: 0 }]);
      // Do NOT save to localStorage here — only save when assessment is complete
    } catch (e) {
      setError('Could not connect to the server. Make sure the backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || isTyping || !sessionId) return;

    const userMsg = { role: 'user', content: text, id: Date.now(), isNew: true };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);
    setError('');

    try {
      const data = await sendMessage(sessionId, text);
      setStep(data.step || step + 1);

      const botMsg = {
        role: 'assistant',
        content: data.reply,
        id: Date.now() + 1,
        isNew: true,
      };
      setMessages(prev => [...prev, botMsg]);

      if (data.is_complete) {
        setIsComplete(true);
        if (data.predictions) setPredictions(data.predictions);
        if (data.advice) setAdvice(data.advice);

        // Only save to history after full assessment completes
        const sid = sessionId;
        if (sid) {
          const prev = JSON.parse(localStorage.getItem('arthsaathi_sessions') || '[]');
          // Remove any existing entry for this session, then prepend (newest first)
          const updated = [sid, ...prev.filter(s => s !== sid)].slice(0, 50);
          localStorage.setItem('arthsaathi_sessions', JSON.stringify(updated));
        }
      }
    } catch (e) {
      setError('Failed to send message. Please try again.');
    } finally {
      setIsTyping(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleNewChat() {
    setSessionId(null);
    setMessages([]);
    setInput('');
    setStep(1);
    setIsComplete(false);
    setPredictions(null);
    setAdvice(null);
    setError('');
    setActiveTab('chat');
    initSession();
  }

  // ── Sidebar ──────────────────────────────────────────────────────────────

  function Sidebar() {
    return (
      <div className="chat-sidebar">
        {/* Bot identity */}
        <div style={{ textAlign: 'center', padding: '8px 0 16px', borderBottom: '1px solid var(--color-border)' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: 6 }}>🪙</div>
          <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--color-accent)' }}>ArthSaathi</div>
          <div style={{ fontSize: '0.78rem', color: 'var(--color-text-dim)' }}>Financial Wellness Advisor</div>
          {isComplete && (
            <span className="badge badge-green" style={{ marginTop: 8 }}>✓ Assessment Complete</span>
          )}
        </div>

        {/* Progress */}
        <ProgressBar currentStep={Math.min(step, TOTAL_STEPS)} totalSteps={TOTAL_STEPS} />

        {/* Step info */}
        {!isComplete && (
          <div style={{
            background: 'var(--color-surface-2)',
            border: '1px solid var(--color-border)',
            borderRadius: 10,
            padding: '12px 14px',
            fontSize: '0.83rem',
          }}>
            <div style={{ color: 'var(--color-text-muted)', marginBottom: 4 }}>Current step</div>
            <div style={{ fontWeight: 600, color: 'var(--color-text)' }}>
              {step <= 1 ? '🏠 Location' :
               step <= 3 ? '👨‍👩‍👧 Household Info' :
               step <= 8 ? '💰 Income & Expenses' :
               step <= 12 ? '❤️ Health & Insurance' :
               '📋 Final Details'}
            </div>
          </div>
        )}

        {/* Results preview in sidebar */}
        {predictions && (
          <div>
            <div style={{ fontSize: '0.78rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: 10 }}>
              Your Scores
            </div>
            {[
              { label: '💰 Financial', key: 'financial_stress', color: '#4f8ef7' },
              { label: '🍛 Food', key: 'food_stress', color: '#f59e0b' },
              { label: '📄 Debt', key: 'debt_stress', color: '#f43f5e' },
              { label: '❤️ Health', key: 'health_stress', color: '#10b981' },
            ].map(({ label, key, color }) => (
              <div key={key} style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: 4 }}>
                  <span style={{ color: 'var(--color-text-muted)' }}>{label}</span>
                  <span style={{ color, fontWeight: 700 }}>{Math.round((predictions[key] || 0) * 100)}%</span>
                </div>
                <div style={{ height: 5, background: 'var(--color-surface-3)', borderRadius: 9999, overflow:'hidden' }}>
                  <div style={{ width: `${(predictions[key]||0)*100}%`, height: '100%', background: color, borderRadius: 9999 }} />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* New chat button */}
        <button onClick={handleNewChat} className="btn btn-ghost" style={{ marginTop: 'auto', width: '100%', justifyContent: 'center', fontSize: '0.85rem' }}>
          🔄 New Assessment
        </button>
      </div>
    );
  }

  // ── Chat panel ────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100vh - 72px)' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="spinner" style={{ width: 40, height: 40, margin: '0 auto 16px' }} />
          <div style={{ color: 'var(--color-text-muted)' }}>Connecting to ArthSaathi...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-layout">
      <Sidebar />

      <div className="chat-main">
        {/* Header */}
        <div className="chat-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 38, height: 38, borderRadius: '50%', background: 'linear-gradient(135deg,#4f8ef7,#38bdf8)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.1rem' }}>
              🪙
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>ArthSaathi Advisor</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--color-green)', display: 'flex', alignItems: 'center', gap: 5 }}>
                <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--color-green)', animation: 'pulse-glow 2s infinite' }} />
                Online
              </div>
            </div>
          </div>

          {/* Tab switcher (visible when complete) */}
          {isComplete && (
            <div style={{ display: 'flex', gap: 4, background: 'var(--color-surface-2)', borderRadius: 9999, padding: 4 }}>
              {['chat', 'results'].map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  style={{
                    padding: '6px 16px',
                    borderRadius: 9999,
                    fontSize: '0.82rem',
                    fontWeight: 600,
                    background: activeTab === tab ? 'var(--gradient-primary)' : 'transparent',
                    color: activeTab === tab ? '#fff' : 'var(--color-text-muted)',
                    transition: 'all 0.2s',
                    textTransform: 'capitalize',
                  }}
                >
                  {tab === 'chat' ? '💬 Chat' : '📊 Results'}
                </button>
              ))}
            </div>
          )}

          <button onClick={handleNewChat} className="btn btn-ghost" style={{ fontSize: '0.82rem', padding: '8px 14px' }}>
            🔄 Restart
          </button>
        </div>

        {/* Body */}
        {activeTab === 'chat' ? (
          <>
            {/* Messages */}
            <div className="chat-messages">
              {messages.map((msg, i) => (
                <MessageBubble
                  key={msg.id || i}
                  role={msg.role}
                  content={msg.content}
                  isNew={msg.isNew}
                />
              ))}
              {isTyping && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>

            {/* Error */}
            {error && (
              <div style={{ padding: '8px 16px', margin: '0 16px 8px', background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.3)', borderRadius: 8, fontSize: '0.85rem', color: 'var(--color-red)' }}>
                ⚠️ {error}
              </div>
            )}

            {/* Input */}
            {!isComplete && (
              <div className="chat-input-row">
                <textarea
                  ref={textareaRef}
                  className="chat-textarea"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your answer here… (Enter to send)"
                  rows={1}
                  disabled={isTyping}
                />
                <button
                  className="send-btn"
                  onClick={handleSend}
                  disabled={!input.trim() || isTyping}
                  title="Send"
                >
                  ➤
                </button>
              </div>
            )}

            {isComplete && (
              <div style={{ padding: '14px 16px', borderTop: '1px solid var(--color-border)', textAlign: 'center', fontSize: '0.88rem', color: 'var(--color-text-muted)' }}>
                Assessment complete! Switch to{' '}
                <button onClick={() => setActiveTab('results')} style={{ color: 'var(--color-primary)', background: 'none', fontWeight: 600 }}>Results tab</button>
                {' '}to view your scores and advice.
              </div>
            )}
          </>
        ) : (
          /* Results tab */
          <div style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
            {predictions && (
              <div style={{ marginBottom: 28 }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 16, color: 'var(--color-accent)' }}>
                  📊 Your Stress Analysis
                </h3>
                <StressGauge predictions={predictions} />
              </div>
            )}
            {advice && (
              <div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 16, color: 'var(--color-accent)' }}>
                  🪙 Personalised Advice
                </h3>
                <AdviceCard advice={advice} predictions={predictions} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
