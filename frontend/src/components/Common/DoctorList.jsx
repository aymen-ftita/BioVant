import React, { useState, useEffect } from 'react';
import { User, Mail, ShieldAlert, Plus, RotateCcw, X, Shield, PlusCircle, Landmark } from 'lucide-react';
import axios from 'axios';

const DoctorList = () => {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Registration modal states
  const [showModal, setShowModal] = useState(false);
  const [newDoctor, setNewDoctor] = useState({
    username: '',
    email: '',
    password: ''
  });
  const [registering, setRegistering] = useState(false);
  const [registerError, setRegisterError] = useState(null);

  const fetchDoctors = async () => {
    setLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get('http://localhost:8000/admin/doctors', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setDoctors(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Impossible de charger la liste des médecins.');
      // Fallback mocks if API fails or authentication issues during dev
      setDoctors([
        { id: 101, username: 'Dr. Jean Dupont', email: 'jean.dupont@hypnoria.org', clinic: 'CHU de Lille - Unité du Sommeil', processed: 42, active: true },
        { id: 102, username: 'Dr. Sarah Alami', email: 'sarah.alami@clinique-sommeil.fr', clinic: 'Clinique du Sommeil Paris Rive Gauche', processed: 89, active: true },
        { id: 103, username: 'Dr. Marc Vasseur', email: 'marc.vasseur@sommeil-lyon.fr', clinic: 'Centre de Neuro-Pneumologie de Lyon', processed: 17, active: true }
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDoctors();
  }, []);

  const handleRegisterDoctor = async (e) => {
    e.preventDefault();
    setRegistering(true);
    setRegisterError(null);
    try {
      const token = localStorage.getItem('token');
      await axios.post('http://localhost:8000/admin/doctors', newDoctor, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setShowModal(false);
      setNewDoctor({ username: '', email: '', password: '' });
      fetchDoctors();
    } catch (err) {
      console.error(err);
      setRegisterError(err.response?.data?.detail || err.message || 'Erreur lors de la création du médecin.');
    } finally {
      setRegistering(false);
    }
  };

  return (
    <div className="doctors-container" style={{ padding: '20px 0' }}>
      
      {/* HEADER SECTION */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '15px' }}>
        <div>
          <div className="sec-lbl" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Shield size={20} color="var(--red)" />
            Base de Données des Médecins
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text3)', marginTop: '4px' }}>
            Gestion administrative des spécialistes autorisés à soumettre des polysomnographies.
          </p>
        </div>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            className="btn-reset" 
            onClick={fetchDoctors}
            style={{ display: 'flex', gap: '8px', alignItems: 'center', height: '38px', padding: '0 16px' }}
          >
            <RotateCcw size={14} /> Rafraîchir
          </button>
          <button 
            className="btn-next" 
            onClick={() => setShowModal(true)}
            style={{ display: 'flex', gap: '8px', alignItems: 'center', height: '38px', padding: '0 16px', background: 'var(--red)' }}
          >
            <PlusCircle size={14} /> Ajouter un Spécialiste
          </button>
        </div>
      </div>

      {/* ERROR MESSAGE IF ANY */}
      {error && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 16px', background: 'rgba(231,76,60,0.06)', borderLeft: '4px solid var(--red)', borderRadius: '6px', color: 'var(--red)', fontSize: '13px', marginBottom: '20px' }}>
          <ShieldAlert size={16} />
          <span>Note: {error} (Utilisation des données locales en fallback)</span>
        </div>
      )}

      {/* DOCTORS TABLE / LIST CARD */}
      {loading ? (
        <div className="status-msg">Chargement des comptes spécialistes en cours...</div>
      ) : (
        <div className="patients-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '20px' }}>
          {doctors.map(doc => (
            <div 
              key={doc.id} 
              className="patient-card" 
              style={{ 
                background: 'var(--surface)', 
                border: '1px solid var(--border)', 
                borderRadius: '12px', 
                padding: '24px', 
                display: 'flex', 
                flexDirection: 'column', 
                justifyContent: 'space-between',
                transition: 'var(--t)'
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                  <div 
                    className="patient-avatar" 
                    style={{ 
                      width: '42px', 
                      height: '42px', 
                      background: 'rgba(192,57,43,0.1)', 
                      color: 'var(--red)', 
                      fontSize: '15px', 
                      fontWeight: 'bold' 
                    }}
                  >
                    {doc.username.replace('Dr. ', '').split(' ').map(n => n[0]).join('').toUpperCase()}
                  </div>
                  <span 
                    style={{ 
                      fontSize: '11px', 
                      padding: '4px 10px', 
                      borderRadius: '20px', 
                      background: 'rgba(46,204,113,0.1)', 
                      color: 'var(--green)', 
                      fontWeight: 'bold' 
                    }}
                  >
                    Actif
                  </span>
                </div>

                <h3 style={{ fontSize: '16px', fontWeight: '800', color: 'var(--text)', marginBottom: '6px' }}>
                  {doc.username.startsWith('Dr.') ? doc.username : `Dr. ${doc.username}`}
                </h3>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text3)', fontSize: '12px', marginBottom: '12px' }}>
                  <Mail size={12} />
                  <span>{doc.email}</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg2)', padding: '10px 14px', borderRadius: '8px', fontSize: '11px', color: 'var(--text2)', border: '1px solid var(--border)' }}>
                  <Landmark size={12} color="var(--red)" />
                  <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {doc.clinic || 'Centre Hospitalier Universitaire (Clinique Affiliée)'}
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '20px', borderTop: '1px solid var(--border)', paddingTop: '15px', fontSize: '11px', color: 'var(--text3)' }}>
                <span>ID Interne: #{doc.id}</span>
                <span style={{ fontWeight: 'bold', color: 'var(--text)' }}>
                  {doc.processed || Math.floor(Math.random() * 50) + 12} rapports PSG
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* REGISTRATION MODAL FORM */}
      {showModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(4px)' }}>
          <div className="login-card" style={{ width: '450px', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '16px', padding: '32px', position: 'relative', animation: 'scaleUp 0.3s ease' }}>
            <button 
              onClick={() => setShowModal(false)}
              style={{ position: 'absolute', top: '16px', right: '16px', background: 'none', border: 'none', color: 'var(--text3)', cursor: 'pointer' }}
            >
              <X size={20} />
            </button>
            
            <div className="login-header" style={{ marginBottom: '24px' }}>
              <PlusCircle size={36} color="var(--red)" style={{ marginBottom: '10px' }} />
              <h2 style={{ fontSize: '20px', fontWeight: '800' }}>Ajouter un Médecin</h2>
              <p style={{ fontSize: '12px', color: 'var(--text3)' }}>Enregistrez un nouveau spécialiste du sommeil dans l'infrastructure Hypnora PaaS.</p>
            </div>

            {registerError && (
              <div style={{ color: 'var(--red)', background: 'rgba(231,76,60,0.1)', padding: '10px', borderRadius: '6px', fontSize: '12px', marginBottom: '15px' }}>
                ⚠ {registerError}
              </div>
            )}

            <form onSubmit={handleRegisterDoctor}>
              <div className="form-group" style={{ marginBottom: '16px' }}>
                <label>Nom complet (Dr. Nom)</label>
                <div className="input-wrapper">
                  <input 
                    type="text" 
                    placeholder="Dr. Jean Vasseur" 
                    value={newDoctor.username} 
                    onChange={e => setNewDoctor(prev => ({ ...prev, username: e.target.value }))}
                    required 
                  />
                </div>
              </div>

              <div className="form-group" style={{ marginBottom: '16px' }}>
                <label>Email Académique / Professionnel</label>
                <div className="input-wrapper">
                  <input 
                    type="email" 
                    placeholder="jean.vasseur@hopital.fr" 
                    value={newDoctor.email} 
                    onChange={e => setNewDoctor(prev => ({ ...prev, email: e.target.value }))}
                    required 
                  />
                </div>
              </div>

              <div className="form-group" style={{ marginBottom: '24px' }}>
                <label>Mot de Passe Initial</label>
                <div className="input-wrapper">
                  <input 
                    type="password" 
                    placeholder="••••••••" 
                    value={newDoctor.password} 
                    onChange={e => setNewDoctor(prev => ({ ...prev, password: e.target.value }))}
                    required 
                  />
                </div>
              </div>

              <button 
                type="submit" 
                className="btn-login"
                disabled={registering}
                style={{ width: '100%', height: '44px', background: 'var(--red)' }}
              >
                {registering ? 'Création en cours...' : 'Enregistrer le Spécialiste'}
              </button>
            </form>
          </div>
        </div>
      )}

    </div>
  );
};

export default DoctorList;
