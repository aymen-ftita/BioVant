import { Activity, Terminal, CheckSquare, LogIn, LogOut, Users, FileText } from 'lucide-react';
import './Sidebar.css';

const Sidebar = ({ user, activeTab, onTabChange, onLogout }) => {
  return (
    <nav className="sidebar-nav">
      <div className="sidebar-brand">Hypnora<br /><em>AI</em></div>
      <div className="nav-links">
        {user && user.role === 'doctor' && (
          <>
            <button
              className={`app-tab ${activeTab === 'doctor' ? 'active' : ''}`}
              onClick={() => onTabChange('doctor')}
            >
              <Activity size={18} />
              Docteur
            </button>
            <button
              className={`app-tab ${activeTab === 'custom-osa' ? 'active' : ''}`}
              onClick={() => onTabChange('custom-osa')}
            >
              <CheckSquare size={18} />
              OSA Custom
            </button>
            <button
              className={`app-tab ${activeTab === 'patients' ? 'active' : ''}`}
              onClick={() => onTabChange('patients')}
            >
              <FileText size={18} />
              Mes Patients
            </button>
          </>
        )}

        {user && user.role === 'admin' && (
          <>
            <button
              className={`app-tab ${activeTab === 'doctors-list' ? 'active' : ''}`}
              onClick={() => onTabChange('doctors-list')}
            >
              <Users size={18} />
              Médecins
            </button>
            <button
              className={`app-tab ${activeTab === 'developer' ? 'active' : ''}`}
              onClick={() => onTabChange('developer')}
            >
              <Terminal size={18} />
              Développeur
            </button>
          </>
        )}

        {!user ? (
          <button
            className={`app-tab ${activeTab === 'login' ? 'active' : ''}`}
            onClick={() => onTabChange('login')}
          >
            <LogIn size={18} />
            Connexion
          </button>
        ) : (
          <button className="app-tab" onClick={onLogout} style={{ marginTop: 'auto', color: 'var(--red)' }}>
            <LogOut size={18} />
            Déconnexion
          </button>
        )}
      </div>
      {user && (
        <div style={{ marginTop: 'auto', padding: '15px', borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '10px' }}>
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
