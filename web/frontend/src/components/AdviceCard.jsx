// src/components/AdviceCard.jsx
// Displays GPT-generated financial advice + v4 ML stress summary

import { useState } from 'react';

function formatAdvice(text) {
  if (!text) return '';
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.*?)__/g, '<u>$1</u>')
    .replace(/^#{1,3} (.+)$/gm, '<h4 style="color:var(--color-accent);margin:14px 0 6px">$1</h4>')
    .replace(/^(\d+\.) /gm, '<br/>$1 ')
    .replace(/^[•·*-] (.+)$/gm, '• $1')
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>');
}

// ── Stress level config ──────────────────────────────────────────────────────
const LEVEL_CONFIG = [
  { label: 'No Stress Detected',  emoji: '✅', bg: 'rgba(16,185,129,0.12)',  border: 'rgba(16,185,129,0.35)',  color: '#10b981' },
  { label: 'Mild Stress',         emoji: '⚠️', bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.35)',  color: '#f59e0b' },
  { label: 'Moderate Stress',     emoji: '🔶', bg: 'rgba(251,146,60,0.12)',  border: 'rgba(251,146,60,0.35)',  color: '#fb923c' },
  { label: 'Severe Stress',       emoji: '🚨', bg: 'rgba(244,63,94,0.12)',   border: 'rgba(244,63,94,0.35)',   color: '#f43f5e' },
];

const DOMAIN_LABELS = {
  financial_stress: { label: 'Financial',  icon: '💰' },
  food_stress:      { label: 'Food',        icon: '🍛' },
  debt_stress:      { label: 'Debt',        icon: '📄' },
  health_stress:    { label: 'Health',      icon: '❤️' },
};

// ── Stress Summary Banner ────────────────────────────────────────────────────
function StressBanner({ predictions }) {
  if (!predictions) return null;
  const level   = predictions.stress_level ?? 0;
  const cfg     = LEVEL_CONFIG[Math.min(level, 3)];
  const domains = predictions.stressed_domains || [];

  return (
    <div style={{
      background: cfg.bg,
      border: `1px solid ${cfg.border}`,
      borderRadius: 12,
      padding: '14px 18px',
      marginBottom: 18,
      display: 'flex',
      alignItems: 'flex-start',
      gap: 14,
    }}>
      <span style={{ fontSize: '1.6rem', lineHeight: 1 }}>{cfg.emoji}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 700, color: cfg.color, fontSize: '1rem', marginBottom: 6 }}>
          {cfg.label}
        </div>

        {/* Stressed domains */}
        {domains.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
            {domains.map(d => {
              const dm = DOMAIN_LABELS[d] || { label: d, icon: '📌' };
              return (
                <span key={d} style={{
                  fontSize: '0.76rem', fontWeight: 600,
                  padding: '3px 10px', borderRadius: 9999,
                  background: `${cfg.color}22`,
                  border: `1px solid ${cfg.color}44`,
                  color: cfg.color,
                }}>
                  {dm.icon} {dm.label}
                </span>
              );
            })}
          </div>
        )}

        {/* Health guard message */}
        {predictions.health_stress_message && (
          <div style={{
            fontSize: '0.8rem', color: 'var(--color-text-muted)',
            background: 'rgba(255,255,255,0.04)',
            borderRadius: 8, padding: '8px 12px', marginBottom: 6,
            borderLeft: '3px solid var(--color-accent)',
          }}>
            ℹ️ {predictions.health_stress_message}
          </div>
        )}

        {/* Input warnings */}
        {predictions.input_warnings?.length > 0 && (
          <details style={{ marginTop: 6 }}>
            <summary style={{ fontSize: '0.75rem', color: 'var(--color-text-dim)', cursor: 'pointer' }}>
              ⚡ {predictions.input_warnings.length} input value{predictions.input_warnings.length > 1 ? 's' : ''} outside typical range
            </summary>
            <ul style={{ margin: '6px 0 0 16px', padding: 0, fontSize: '0.72rem', color: 'var(--color-text-dim)' }}>
              {predictions.input_warnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}

// ── Main AdviceCard ──────────────────────────────────────────────────────────
export default function AdviceCard({ advice, predictions }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(advice || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!advice) return null;

  return (
    <div className="advice-card animate-fade-in">
      {/* Stress summary banner */}
      <StressBanner predictions={predictions} />

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <span style={{ fontSize: '1.4rem' }}>🪙</span>
            <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--color-accent)' }}>
              ArthSaathi Advice
            </span>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--color-text-dim)' }}>
            Personalized financial wellness assessment
          </div>
        </div>
        <button
          onClick={handleCopy}
          style={{
            padding: '7px 14px',
            background: copied ? 'rgba(16,185,129,0.15)' : 'var(--color-surface-3)',
            border: `1px solid ${copied ? 'rgba(16,185,129,0.4)' : 'var(--color-border)'}`,
            borderRadius: 9999,
            color: copied ? 'var(--color-green)' : 'var(--color-text-muted)',
            fontSize: '0.8rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {copied ? '✓ Copied' : '📋 Copy'}
        </button>
      </div>

      <div className="divider" />

      {/* Advice text */}
      <div
        className="advice-content"
        dangerouslySetInnerHTML={{ __html: formatAdvice(advice) }}
      />

      {/* Actions */}
      <div style={{ marginTop: 20, display: 'flex', gap: 10 }}>
        <button
          onClick={() => {
            const blob = new Blob([advice], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = 'ArthSaathi_Advice.txt';
            a.click(); URL.revokeObjectURL(url);
          }}
          className="btn btn-ghost"
          style={{ fontSize: '0.85rem' }}
        >
          📥 Download Report
        </button>
      </div>
    </div>
  );
}
