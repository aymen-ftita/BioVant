import { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import './Login.css';

const Login = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [avatarState, setAvatarState] = useState('idle'); // idle | covering | peeking
  const avatarRef = useRef(null);
  const pupilLRef = useRef(null);
  const pupilRRef = useRef(null);
  const emailRef = useRef(null);

  const maxEyeMove = 6;
  const eyeCenterL = { x: 75, y: 85 };
  const eyeCenterR = { x: 125, y: 85 };

  const updateEyePosition = useCallback((clientX, clientY) => {
    if (avatarState === 'covering' || !avatarRef.current) return;
    const box = avatarRef.current.getBoundingClientRect();
    const avatarCenterX = box.left + box.width / 2;
    const avatarCenterY = box.top + box.height / 2;
    const angle = Math.atan2(clientY - avatarCenterY, clientX - avatarCenterX);
    const dist = Math.min(maxEyeMove, Math.hypot(clientX - avatarCenterX, clientY - avatarCenterY) / 50);
    const moveX = Math.cos(angle) * dist;
    const moveY = Math.sin(angle) * dist;
    if (pupilLRef.current) pupilLRef.current.setAttribute('transform', `translate(${moveX}, ${moveY})`);
    if (pupilRRef.current) pupilRRef.current.setAttribute('transform', `translate(${moveX}, ${moveY})`);
  }, [avatarState]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (document.activeElement !== emailRef.current) {
        updateEyePosition(e.clientX, e.clientY);
      }
    };
    document.addEventListener('mousemove', handleMouseMove);
    return () => document.removeEventListener('mousemove', handleMouseMove);
  }, [updateEyePosition]);

  const handleEmailFocus = () => {
    setAvatarState('idle');
  };

  const handlePasswordFocus = () => {
    setAvatarState('covering');
    if (pupilLRef.current) pupilLRef.current.setAttribute('transform', 'translate(0, 3)');
    if (pupilRRef.current) pupilRRef.current.setAttribute('transform', 'translate(0, 3)');
  };

  const handlePasswordBlur = () => {
    setAvatarState('idle');
  };

  const handleTogglePassword = () => {
    if (!showPassword) {
      setShowPassword(true);
      setAvatarState('peeking');
    } else {
      setShowPassword(false);
      setAvatarState('covering');
    }
  };

  const [showErrorModal, setShowErrorModal] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Attempt authentic JWT login against backend
    const params = new URLSearchParams();
    params.append('username', email);
    params.append('password', password);

    try {
      const res = await axios.post('http://localhost:8000/token', params, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });
      
      const data = res.data;
      // Save valid token in localStorage
      localStorage.setItem('token', data.access_token);
      
      if (onLogin) {
        onLogin({
          id: data.user.id,
          username: data.user.first_name ? `${data.user.first_name} ${data.user.last_name}` : data.user.email.split('@')[0],
          role: data.user.role
        });
      }
    } catch (err) {
      console.warn('Login failed, showing wrong credentials popup:', err);
      setShowErrorModal(true);
    }
  };

  const handleDemoMode = () => {
    localStorage.setItem('token', 'demo-session-token');
    if (onLogin) {
      onLogin({
        id: 99,
        username: 'Visiteur Démo',
        role: 'demo'
      });
    }
  };

  const avatarClass = `doctor-avatar ${avatarState === 'covering' ? 'covering-eyes' : avatarState === 'peeking' ? 'peeking-r' : ''}`;

  return (
    <div className="login-container">
      <div className={avatarClass} ref={avatarRef}>
        <svg className="doctor-svg" viewBox="0 0 200 200">
          <path className="coat" d="M40,200 Q40,160 60,140 L140,140 Q160,160 160,200 Z" />
          <path className="shirt" d="M85,140 L115,140 L100,160 Z" />
          <path className="tie" d="M100,148 L95,155 L100,165 L105,155 Z" />
          <circle className="skin" cx="100" cy="85" r="55" />
          <path className="hair" d="M45,85 Q45,30 100,30 Q155,30 155,85 L145,85 Q145,50 100,50 Q55,50 55,85 Z" />
          <circle className="skin" cx="45" cy="90" r="10" />
          <circle className="skin" cx="155" cy="90" r="10" />
          <g id="eyes">
            <g id="eye-l">
              <circle className="eye-white" cx="75" cy="85" r="15" />
              <circle className="pupil" ref={pupilLRef} cx="75" cy="85" r="7" />
              <path className="eye-closed" d="M60,85 Q75,95 90,85" />
            </g>
            <g id="eye-r">
              <circle className="eye-white" cx="125" cy="85" r="15" />
              <circle className="pupil" ref={pupilRRef} cx="125" cy="85" r="7" />
              <path className="eye-closed" d="M110,85 Q125,95 140,85" />
            </g>
          </g>
          <path d="M85,115 Q100,125 115,115" fill="none" stroke="#4a3728" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </div>

      <div className="login-card">
        <div className="login-header">
          <h2>Bienvenue</h2>
          <p>Connectez-vous à votre espace Hypnora</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="email">Email Professionnel</label>
            <div className="input-wrapper">
              <input
                type="email"
                id="email"
                ref={emailRef}
                placeholder="docteur@hopital.fr"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                onFocus={handleEmailFocus}
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="password">Mot de passe</label>
            <div className="input-wrapper">
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onFocus={handlePasswordFocus}
                onBlur={handlePasswordBlur}
                required
              />
              <button type="button" className="toggle-password" onClick={handleTogglePassword}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" />
                </svg>
              </button>
            </div>
          </div>

          <button type="submit" className="btn-login">S'identifier</button>
        </form>
      </div>

      {showErrorModal && (
        <div className="landing-modal-overlay" style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, animation: 'fadeIn 0.2s ease' }}>
          <div className="login-card" style={{ width: '400px', margin: '20px', padding: '32px', textAlign: 'center', border: '1px solid var(--red)', boxShadow: '0 10px 30px rgba(192,57,43,0.2)', position: 'relative', background: 'var(--surface)' }}>
            <div style={{ width: '50px', height: '50px', borderRadius: '50%', background: 'rgba(192,57,43,0.1)', color: 'var(--red)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px', fontSize: '24px' }}>
              ⚠️
            </div>
            <h3 style={{ color: 'var(--text)', marginBottom: '8px', fontSize: '18px', fontWeight: '700' }}>Identifiants Incorrects</h3>
            <p style={{ color: 'var(--text2)', fontSize: '13px', marginBottom: '24px', lineHeight: '1.5' }}>
              L'email ou le mot de passe saisi ne correspond à aucun compte praticien actif.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <button 
                type="button" 
                className="btn-login" 
                onClick={handleDemoMode}
                style={{ background: 'var(--primary)', color: '#fff', fontWeight: '600' }}
              >
                🚀 Accéder au Mode Démo
              </button>
              <button 
                type="button" 
                onClick={() => setShowErrorModal(false)}
                style={{ background: 'transparent', border: '1px solid var(--border)', color: 'var(--text2)', padding: '12px', borderRadius: '8px', cursor: 'pointer', fontSize: '13px', fontWeight: '500' }}
              >
                Réessayer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Login;
