// src/components/ProgressBar.jsx
// Shows conversation step progress as a dot bar

export default function ProgressBar({ currentStep, totalSteps = 15 }) {
  const pct = Math.round((currentStep / totalSteps) * 100);
  return (
    <div style={{ padding: '12px 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)', fontWeight: 600 }}>
          PROGRESS
        </span>
        <span style={{ fontSize: '0.78rem', color: 'var(--color-primary)', fontWeight: 700 }}>
          {currentStep}/{totalSteps}
        </span>
      </div>
      {/* Dot track */}
      <div className="step-bar">
        {Array.from({ length: totalSteps }).map((_, i) => (
          <div
            key={i}
            className={`step-dot ${i < currentStep - 1 ? 'done' : i === currentStep - 1 ? 'active' : ''}`}
          />
        ))}
      </div>
      {/* Percentage bar */}
      <div style={{ height: 3, background: 'var(--color-surface-3)', borderRadius: 9999, marginTop: 8 }}>
        <div
          style={{
            height: '100%',
            width: `${pct}%`,
            background: 'var(--gradient-primary)',
            borderRadius: 9999,
            transition: 'width 0.6s ease',
          }}
        />
      </div>
    </div>
  );
}
