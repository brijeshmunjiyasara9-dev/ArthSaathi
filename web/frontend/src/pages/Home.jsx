// src/pages/Home.jsx
// Landing page with hero, features, and CTA

import { Link } from 'react-router-dom';

const FEATURES = [
  {
    icon: '🤖',
    title: 'AI-Powered Analysis',
    desc: 'Our ML models trained on millions of Indian household records give you accurate stress predictions.',
  },
  {
    icon: '💬',
    title: 'Friendly Conversation',
    desc: 'Just answer a few simple questions. No forms, no jargon — just a friendly chat.',
  },
  {
    icon: '📊',
    title: '4 Stress Dimensions',
    desc: 'Financial, food security, debt burden, and health stress — analysed separately and together.',
  },
  {
    icon: '🇮🇳',
    title: 'Indian Context',
    desc: 'Advice tailored to Indian households with relevant government schemes like PM-JAY, PMJJBY, PPF.',
  },
  {
    icon: '🔒',
    title: 'Private & Secure',
    desc: 'Your data stays on our servers and is never shared. Each session is fully encrypted.',
  },
  {
    icon: '📱',
    title: 'Instant Advice',
    desc: 'Get a full financial health report with actionable steps in under 5 minutes.',
  },
];

function HeroVisual() {
  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: 380 }}>
      {/* Glow ring */}
      <div style={{
        position: 'absolute', inset: -30,
        background: 'radial-gradient(ellipse, rgba(79,142,247,0.15) 0%, transparent 70%)',
        borderRadius: '50%',
        animation: 'pulse-glow 3s ease infinite',
      }} />

      {/* Main card */}
      <div className="card" style={{ position: 'relative', zIndex: 1, padding: 28 }}>
        <div style={{ textAlign: 'center', marginBottom: 20 }}>
          <div style={{ fontSize: '3rem', marginBottom: 8 }}>🪙</div>
          <div style={{ fontWeight: 700, fontSize: '1.15rem', color: 'var(--color-accent)' }}>ArthSaathi</div>
          <div style={{ fontSize: '0.83rem', color: 'var(--color-text-muted)' }}>Financial Wellness Score</div>
        </div>

        {/* Mock gauge bars */}
        {[
          { label: '💰 Financial', pct: 68, color: '#4f8ef7' },
          { label: '🍛 Food Security', pct: 34, color: '#f59e0b' },
          { label: '📄 Debt Burden', pct: 52, color: '#f43f5e' },
          { label: '❤️ Health', pct: 22, color: '#10b981' },
        ].map(({ label, pct, color }) => (
          <div key={label} style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 5 }}>
              <span style={{ color: 'var(--color-text-muted)' }}>{label}</span>
              <span style={{ color, fontWeight: 700 }}>{pct}%</span>
            </div>
            <div style={{ height: 6, background: 'var(--color-surface-3)', borderRadius: 9999, overflow: 'hidden' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 9999 }} />
            </div>
          </div>
        ))}

        <div style={{
          marginTop: 18, padding: '12px 16px',
          background: 'rgba(16,185,129,0.1)',
          border: '1px solid rgba(16,185,129,0.25)',
          borderRadius: 10,
          fontSize: '0.83rem', color: 'var(--color-green)',
          fontWeight: 600, textAlign: 'center',
        }}>
          ✅ Your financial health looks stable!
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <>
      {/* Hero */}
      <section className="hero">
        <div className="container">
          <div className="hero-grid">
            {/* Left copy */}
            <div className="animate-fade-up">
              <div className="badge badge-blue" style={{ marginBottom: 20 }}>
                🇮🇳 Built for Indian Households
              </div>
              <h1 className="hero-title">
                Know Your{' '}
                <span className="text-gradient">Financial Health</span>{' '}
                in Minutes
              </h1>
              <p className="hero-subtitle">
                ArthSaathi uses AI trained on CMIE data from millions of Indian households
                to assess your financial, food, debt, and health stress and gives you
                personalised advice in simple language.
              </p>
              <div className="hero-actions">
                <Link to="/chat" className="btn btn-primary" style={{ fontSize: '1rem', padding: '14px 32px' }}>
                  Start Free Assessment →
                </Link>
                <Link to="/history" className="btn btn-ghost" style={{ fontSize: '1rem' }}>
                  View History
                </Link>
              </div>
              <div style={{ marginTop: 32, display: 'flex', gap: 28 }}>
                {[
                  { num: '10M+', label: 'Households analysed' },
                  { num: '4', label: 'Stress dimensions' },
                  { num: '5 min', label: 'To full report' },
                ].map(({ num, label }) => (
                  <div key={label}>
                    <div style={{ fontSize: '1.4rem', fontWeight: 800 }} className="text-gradient">{num}</div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)' }}>{label}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right visual */}
            <div className="hero-visual" style={{ justifyContent: 'center', display: 'flex' }}>
              <HeroVisual />
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section style={{ background: 'var(--color-surface)', padding: '60px 0 80px', borderTop: '1px solid var(--color-border)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 48 }}>
            <div className="badge badge-gold" style={{ marginBottom: 14 }}>Why ArthSaathi</div>
            <h2 className="section-title">Everything you need to understand your <span className="text-gradient">financial health</span></h2>
            <p className="section-subtitle" style={{ maxWidth: 520, margin: '10px auto 0' }}>
              Powered by machine learning models trained on CMIE household panel data across all Indian states.
            </p>
          </div>
          <div className="features-grid">
            {FEATURES.map(f => (
              <div key={f.title} className="feature-card">
                <span className="feature-icon">{f.icon}</span>
                <div className="feature-title">{f.title}</div>
                <div className="feature-desc">{f.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ padding: '80px 0', textAlign: 'center' }}>
        <div className="container">
          <h2 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: 14 }}>
            Ready to check your <span className="text-gold-gradient">ArthSwaasthya</span>?
          </h2>
          <p style={{ color: 'var(--color-text-muted)', marginBottom: 32, maxWidth: 480, margin: '0 auto 32px' }}>
            It's free, takes 5 minutes, and gives you personalised advice based on real data.
          </p>
          <Link to="/chat" className="btn btn-primary" style={{ fontSize: '1.05rem', padding: '15px 40px' }}>
            Start Your Assessment →
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--color-border)', padding: '28px 0', textAlign: 'center' }}>
        <div className="container">
          <div style={{ fontSize: '0.85rem', color: 'var(--color-text-dim)' }}>
            © 2026 ArthSaathi • Built with CMIE data • Powered by AI
          </div>
        </div>
      </footer>
    </>
  );
}
