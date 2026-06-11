import { useState, useEffect } from 'react';
import { Activity, Terminal, CheckSquare, LogOut, Users, FileText, Home, ClipboardList, Building2, Sun, Moon, MessageSquare, Globe } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import './Sidebar.css';

const Sidebar = ({ user, activeTab, onTabChange, onLogout }) => {
  const { t, i18n } = useTranslation();
  const [isDark, setIsDark] = useState(() => {
    return localStorage.getItem('hypnora-theme') === 'dark';
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark-mode');
    } else {
      document.documentElement.classList.remove('dark-mode');
    }
    localStorage.setItem('hypnora-theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  const toggleLanguage = () => {
    const newLang = i18n.language === 'fr' ? 'en' : 'fr';
    i18n.changeLanguage(newLang);
  };

  return (
    <nav className="sidebar-nav">
      <div className="sidebar-brand">Hypnora<br /><em>AI</em></div>
      <div className="nav-links">
        {user && user.role === 'doctor' && (
          <>
            <button
              className={`app-tab ${activeTab === 'doctor-dashboard' ? 'active' : ''}`}
              onClick={() => onTabChange('doctor-dashboard')}
            >
              <Home size={18} />
              {t('sidebar.doctor_home')}
            </button>
            <button
              className={`app-tab ${activeTab === 'doctor' ? 'active' : ''}`}
              onClick={() => onTabChange('doctor')}
            >
              <Activity size={18} />
              {t('sidebar.new_analysis')}
            </button>
            <button
              className={`app-tab ${activeTab === 'custom-osa' ? 'active' : ''}`}
              onClick={() => onTabChange('custom-osa')}
            >
              <CheckSquare size={18} />
              {t('sidebar.custom_osa')}
            </button>
            <button
              className={`app-tab ${activeTab === 'patients' ? 'active' : ''}`}
              onClick={() => onTabChange('patients')}
            >
              <FileText size={18} />
              {t('sidebar.my_patients')}
            </button>
            <button
              className={`app-tab ${activeTab === 'conversations' ? 'active' : ''}`}
              onClick={() => onTabChange('conversations')}
            >
              <MessageSquare size={18} />
              {t('sidebar.consultations')}
            </button>
          </>
        )}

        {user && user.role === 'admin' && (
          <>
            <button
              className={`app-tab ${activeTab === 'home-dashboard' ? 'active' : ''}`}
              onClick={() => onTabChange('home-dashboard')}
            >
              <Home size={18} />
              {t('sidebar.admin_dashboard')}
            </button>
            <button
              className={`app-tab ${activeTab === 'doctors-list' ? 'active' : ''}`}
              onClick={() => onTabChange('doctors-list')}
            >
              <Users size={18} />
              {t('sidebar.doctors')}
            </button>
            <button
              className={`app-tab ${activeTab === 'hospitals' ? 'active' : ''}`}
              onClick={() => onTabChange('hospitals')}
            >
              <Building2 size={18} />
              {t('sidebar.hospitals')}
            </button>
            <button
              className={`app-tab ${activeTab === 'audit-logs' ? 'active' : ''}`}
              onClick={() => onTabChange('audit-logs')}
            >
              <ClipboardList size={18} />
              {t('sidebar.audit_logs')}
            </button>
            <button
              className={`app-tab ${activeTab === 'developer' ? 'active' : ''}`}
              onClick={() => onTabChange('developer')}
            >
              <Terminal size={18} />
              {t('sidebar.developer')}
            </button>
          </>
        )}
      </div>

      <div style={{ flex: 1 }} />

      {/* Language Toggle */}
      <button
        className="theme-toggle-btn"
        onClick={toggleLanguage}
        title={i18n.language === 'fr' ? 'Switch to English' : 'Passer en Français'}
        style={{ marginBottom: '8px' }}
      >
        <Globe size={16} />
        {i18n.language === 'fr' ? 'English' : 'Français'}
      </button>

      {/* Theme Toggle */}
      <button
        className="theme-toggle-btn"
        onClick={() => setIsDark(prev => !prev)}
        title={isDark ? t('sidebar.light_mode') : t('sidebar.dark_mode')}
      >
        {isDark ? <Sun size={16} /> : <Moon size={16} />}
        {isDark ? t('sidebar.light_mode') : t('sidebar.dark_mode')}
      </button>

      {/* Logout */}
      {user && (
        <button className="app-tab" onClick={onLogout} style={{ marginTop: '12px', color: 'var(--red)' }}>
          <LogOut size={18} />
          {t('sidebar.logout')}
        </button>
      )}

      {/* User Info */}
      {user && (
        <div style={{ marginTop: '16px', padding: '15px 0 0', borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="patient-avatar" style={{ width: '32px', height: '32px', fontSize: '12px', background: 'rgba(192,57,43,0.1)', color: 'var(--red)', fontWeight: 'bold' }}>
            {user.username[0].toUpperCase()}
          </div>
          <div style={{ fontSize: '11px' }}>
            <div style={{ fontWeight: '700' }}>Dr. {user.username}</div>
            <div style={{ color: 'var(--text3)', textTransform: 'capitalize' }}>{user.role}</div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Sidebar;
