// src/components/StressGauge.jsx
// Visual stress score display with animated bars

const GAUGES = [
  { key: 'financial_stress', label: 'Financial Stress', icon: '💰', color: '#4f8ef7' },
  { key: 'food_stress',      label: 'Food Security',   icon: '🍛', color: '#f59e0b' },
  { key: 'debt_stress',      label: 'Debt Burden',     icon: '📄', color: '#f43f5e' },
  { key: 'health_stress',    label: 'Health Stress',   icon: '❤️', color: '#10b981' },
];

const DOMAIN_ICONS = {
  financial_stress: '💰',
  food_stress:      '🍛',
  debt_stress:      '📄',
  health_stress:    '❤️',
};

function riskLabel(prob) {
  if (prob == null) return { text: 'N/A',      color: 'var(--color-text-dim)' };
  if (prob < 0.3)   return { text: 'Low Risk',  color: '#10b981' };
  if (prob < 0.6)   return { text: 'Moderate',  color: '#f59e0b' };
  return               { text: 'High Risk', color: '#f43f5e' };
}

function GaugeCard({ label, icon, color, value }) {
  // value may be null (e.g. health_stress when guard triggered)
  const isNull = value == null;
  const pct    = isNull ? 0 : Math.round(value * 100);
  const { text, color: rColor } = riskLabel(value);

  return (
    <div className="gauge-card" style={{ animation: 'fadeInUp 0.5s ease forwards', opacity: isNull ? 0.6 : 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="gauge-label">{icon} {label}</div>
          <div className="gauge-pct" style={{ color }}>
            {isNull ? '—' : pct}<span style={{ fontSize: '1rem', color: 'var(--color-text-muted)' }}>{!isNull && '%'}</span>
          </div>
        </div>
        <span
          style={{
            fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase',
            color: rColor, background: `${rColor}22`,
            padding: '3px 10px', borderRadius: 9999,
            border: `1px solid ${rColor}44`,
          }}
        >
          {text}
        </span>
      </div>
      <div className="gauge-bar-track">
        <div
          className="gauge-bar-fill"
          style={{
            width: `${pct}%`,
            background: isNull
              ? 'var(--color-surface-3)'
              : `linear-gradient(90deg, ${color}88, ${color})`,
          }}
        />
      </div>
    </div>
  );
}

function CompositeScore({ score }) {
  const normalized = ((score || 0) / 4) * 100;
  const colors = ['#10b981', '#f59e0b', '#fb923c', '#f43f5e'];
  const colorIdx = Math.min(Math.floor(score || 0), 3);
  const c = colors[colorIdx];

  return (
    <div
      style={{
        background: `linear-gradient(145deg, ${c}11, var(--color-surface-2))`,
        border: `1px solid ${c}33`,
        borderRadius: 'var(--radius-lg)',
        padding: '20px 24px',
        display: 'grid',
        gridTemplateColumns: '1fr auto',
        alignItems: 'center',
        gap: 16,
        marginBottom: 16,
      }}
    >
      <div>
        <div style={{ fontSize: '0.8rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: 6 }}>
          🎯 Composite Stress Score
        </div>
        <div style={{ height: 10, background: 'var(--color-surface-3)', borderRadius: 9999, overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${normalized}%`, background: `linear-gradient(90deg, ${c}88, ${c})`, borderRadius: 9999, transition: 'width 1.2s ease' }} />
        </div>
      </div>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '2.5rem', fontWeight: 800, color: c, lineHeight: 1 }}>
          {(score || 0).toFixed(1)}
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--color-text-dim)' }}>out of 4</div>
      </div>
    </div>
  );
}

export default function StressGauge({ predictions }) {
  if (!predictions) return null;

  const stressed = predictions.stressed_domains || [];

  return (
    <div>
      {/* Stressed domain chips — from v4 API */}
      {stressed.length > 0 && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 8,
          marginBottom: 14, padding: '10px 14px',
          background: 'rgba(244,63,94,0.07)',
          border: '1px solid rgba(244,63,94,0.2)',
          borderRadius: 10,
        }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)', alignSelf: 'center' }}>Active stress:</span>
          {stressed.map(d => (
            <span key={d} style={{
              fontSize: '0.78rem', fontWeight: 700,
              padding: '3px 10px', borderRadius: 9999,
              background: 'rgba(244,63,94,0.15)',
              border: '1px solid rgba(244,63,94,0.3)',
              color: '#f43f5e',
            }}>
              {DOMAIN_ICONS[d] || '📌'} {d.replace('_stress', '').replace('_', ' ')}
            </span>
          ))}
        </div>
      )}

      <CompositeScore score={predictions.composite_stress_score} stressLevel={predictions.stress_level} />
      <div className="gauge-grid">
        {GAUGES.map(g => (
          <GaugeCard
            key={g.key}
            label={g.label}
            icon={g.icon}
            color={g.color}
            value={predictions[g.key]}
          />
        ))}
      </div>
    </div>
  );
}
