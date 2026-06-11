import { useState, useEffect } from 'react';
import { Activity, Users, FileText, CheckCircle, Clock } from 'lucide-react';
import axios from 'axios';
import { useTranslation } from 'react-i18next';
import './DoctorDashboard.css';

const DoctorDashboard = ({ onNavigate }) => {
  const { t } = useTranslation();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      const token = localStorage.getItem('token');
      try {
        const res = await axios.get('http://localhost:8000/doctor/stats', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setStats(res.data);
      } catch (err) {
        console.error('Failed to fetch doctor stats', err);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh', color: 'var(--text2)' }}>
        {t('common.loading')}
      </div>
    );
  }

  const { total_patients, total_psgs, osa_distribution, recent_patients } = stats || {};

  return (
    <div className="doctor-dashboard" style={{ animation: 'fadeIn 0.4s ease' }}>
      <div className="dash-header">
        <h2>{t('dashboard.welcome')}</h2>
        <p>{t('dashboard.overview')}</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(52, 152, 219, 0.1)', color: '#3498db' }}>
            <Users size={24} />
          </div>
          <div className="stat-info">
            <div className="stat-val">{total_patients || 0}</div>
            <div className="stat-lbl">{t('dashboard.patients')}</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(155, 89, 182, 0.1)', color: '#9b59b6' }}>
            <FileText size={24} />
          </div>
          <div className="stat-info">
            <div className="stat-val">{total_psgs || 0}</div>
            <div className="stat-lbl">{t('dashboard.psg')}</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(231, 76, 60, 0.1)', color: 'var(--red)' }}>
            <Activity size={24} />
          </div>
          <div className="stat-info">
            <div className="stat-val">{osa_distribution?.Severe || 0}</div>
            <div className="stat-lbl">{t('dashboard.osa_severe')}</div>
          </div>
        </div>
      </div>

      <div className="dash-content-grid">
        <div className="dash-panel">
          <h3>{t('dashboard.osa_distribution')}</h3>
          <div className="osa-bars">
            {Object.entries(osa_distribution || {}).map(([key, value]) => {
              if (key === 'Not Evaluated') return null;
              const total = total_psgs || 1;
              const pct = Math.round((value / total) * 100);
              let color = '#95a5a6';
              if (key === 'Normal') color = '#2ecc71';
              if (key === 'Mild') color = '#f1c40f';
              if (key === 'Moderate') color = '#e67e22';
              if (key === 'Severe') color = '#e74c3c';

              return (
                <div key={key} className="osa-bar-row">
                  <div className="osa-bar-lbl">{key}</div>
                  <div className="osa-bar-track">
                    <div className="osa-bar-fill" style={{ width: `${pct}%`, background: color }}></div>
                  </div>
                  <div className="osa-bar-val">{value}</div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="dash-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3>{t('dashboard.recent_patients')}</h3>
            <button className="btn-text" onClick={() => onNavigate('patients')}>{t('dashboard.see_all')}</button>
          </div>
          <div className="recent-patients-list">
            {recent_patients && recent_patients.length > 0 ? (
              recent_patients.map(p => (
                <div key={p.id} className="recent-patient-item" onClick={() => onNavigate('patients')}>
                  <div className="rp-avatar">{p.name.charAt(0).toUpperCase()}</div>
                  <div className="rp-name">{p.name}</div>
                  <Clock size={14} color="var(--text3)" style={{ marginLeft: 'auto' }} />
                </div>
              ))
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text3)', fontSize: '12px' }}>
                {t('dashboard.no_patients')}
              </div>
            )}
          </div>
          
          <button 
            className="btn-primary full-width" 
            style={{ marginTop: '20px', background: 'var(--red)', border: 'none', padding: '12px', borderRadius: '8px', color: 'white', fontWeight: 'bold', cursor: 'pointer' }}
            onClick={() => onNavigate('doctor')}
          >
            {t('dashboard.new_analysis_btn')}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DoctorDashboard;
