// src/components/MessageBubble.jsx
// Renders a single chat message bubble

import { useState, useEffect } from 'react';

function formatBotText(text) {
  // Convert basic markdown-like formatting to JSX-friendly HTML
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.*?)__/g, '<em>$1</em>')
    .replace(/^#{1,3} (.+)$/gm, '<strong>$1</strong>')
    .replace(/^[•·] /gm, '• ')
    .replace(/✅/g, '✅')
    .replace(/⚠️/g, '⚠️')
    .replace(/🏠/g, '🏠')
    .replace(/📊/g, '📊')
    .replace(/🇮🇳/g, '🇮🇳')
    .replace(/💪/g, '💪');
}

export default function MessageBubble({ role, content, isNew = false }) {
  const [visible, setVisible] = useState(!isNew);

  useEffect(() => {
    if (isNew) {
      const t = setTimeout(() => setVisible(true), 50);
      return () => clearTimeout(t);
    }
  }, [isNew]);

  const isUser = role === 'user';

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        opacity: visible ? 1 : 0,
        transform: visible ? 'none' : `translateY(${isUser ? '-' : ''}10px)`,
        transition: 'opacity 0.3s ease, transform 0.3s ease',
      }}
    >
      {!isUser && (
        <div style={{
          width: 36, height: 36,
          borderRadius: '50%',
          background: 'linear-gradient(135deg,#4f8ef7,#38bdf8)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '1rem', flexShrink: 0, marginRight: 10, marginTop: 2,
          boxShadow: '0 0 12px rgba(79,142,247,0.3)',
        }}>
          🪙
        </div>
      )}
      <div className={isUser ? 'bubble-user' : 'bubble-bot'}>
        {isUser ? (
          <span>{content}</span>
        ) : (
          <span
            dangerouslySetInnerHTML={{ __html: formatBotText(content) }}
          />
        )}
      </div>
      {isUser && (
        <div style={{
          width: 36, height: 36,
          borderRadius: '50%',
          background: 'var(--color-surface-3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '1rem', flexShrink: 0, marginLeft: 10, marginTop: 2,
        }}>
          👤
        </div>
      )}
    </div>
  );
}

export function TypingIndicator() {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
      <div style={{
        width: 36, height: 36,
        borderRadius: '50%',
        background: 'linear-gradient(135deg,#4f8ef7,#38bdf8)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '1rem', flexShrink: 0,
      }}>
        🪙
      </div>
      <div className="bubble-bot" style={{ padding: '14px 20px' }}>
        <div className="typing-indicator">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  );
}
