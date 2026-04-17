// src/App.jsx — Main application with routing and navbar

import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import Home from './pages/Home';
import Chat from './pages/Chat';
import History from './pages/History';
import './index.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="container">
        <NavLink to="/" className="navbar-brand" style={{ textDecoration: 'none' }}>
          <span className="logo-icon">🪙</span>
          ArthSaathi
        </NavLink>
        <div className="navbar-links">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            Home
          </NavLink>
          <NavLink
            to="/chat"
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            💬 Chat
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            📋 History
          </NavLink>
          <NavLink to="/chat" className="btn btn-primary" style={{ padding: '9px 20px', fontSize: '0.88rem', marginLeft: 8 }}>
            Start Assessment →
          </NavLink>
        </div>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/"        element={<Home />} />
        <Route path="/chat"    element={<Chat />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </BrowserRouter>
  );
}
