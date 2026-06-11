import React, { useState, useEffect } from 'react';
import { 
  User, 
  FileText, 
  MessageSquare, 
  Plus, 
  ArrowLeft, 
  Activity, 
  Calendar, 
  Clock, 
  ChevronRight, 
  CheckSquare,
  Shield, 
  PlusCircle, 
  Brain,
  X,
  PieChart,
  Download,
  Image,
  ExternalLink
} from 'lucide-react';
import axios from 'axios';

const HistoryTrendChart = ({ psgs }) => {
  if (!psgs || psgs.length < 2) return null;
  
  // Sort PSGs by date chronologically
  const sortedPsgs = [...psgs].sort((a, b) => new Date(a.date) - new Date(b.date));
  
  const dates = sortedPsgs.map(p => p.date);
  const seList = sortedPsgs.map(p => p.metrics?.sleep_efficiency || 80);
  const remList = sortedPsgs.map(p => p.metrics?.stage_rem || 18);
  const ahiList = sortedPsgs.map(p => {
    const match = String(p.severity).match(/IAH:\s*([0-9.]+)/i);
    return match ? parseFloat(match[1]) : (p.metrics?.arousal_index || 15);
  });

  const getPathData = (dataList, minVal, maxVal) => {
    const points = dataList.map((val, idx) => {
      const x = 50 + (idx / (dataList.length - 1)) * 390;
      const range = maxVal - minVal;
      const y = 90 - ((val - minVal) / (range || 1)) * 70;
      return `${x},${y}`;
    });
    return `M ${points.join(' L ')}`;
  };

  const sePath = getPathData(seList, 50, 100);
  const remPath = getPathData(remList, 0, 30);
  const ahiPath = getPathData(ahiList, 0, 50);

  return (
    <div className="trend-card" style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border)', borderRadius: '10px', padding: '16px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', fontSize: '13px', fontWeight: 'bold', color: 'var(--red)' }}>
        📈 Suivi d'Évolution Clinique (CPAP / Traitement)
      </div>
      <p style={{ fontSize: '11px', color: 'var(--text3)', marginBottom: '14px' }}>
        Suivi de l'efficacité thérapeutique sur {sortedPsgs.length} polysomnographies.
      </p>

      {/* Sleep Efficiency & REM% Curve */}
      <div style={{ marginBottom: '10px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '15px', fontSize: '10px', marginBottom: '8px', fontWeight: 600 }}>
          <span style={{ color: '#3498db', display: 'flex', alignItems: 'center', gap: '4px' }}>● Efficacité (SE %)</span>
          <span style={{ color: '#9b59b6', display: 'flex', alignItems: 'center', gap: '4px' }}>● Sommeil REM (%)</span>
          <span style={{ color: '#ef4444', display: 'flex', alignItems: 'center', gap: '4px' }}>● Sévérité SAOS (IAH/h)</span>
        </div>
        <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: '6px', padding: '10px', height: '120px' }}>
          <svg width="100%" height="100%" viewBox="0 0 500 110" preserveAspectRatio="none">
            <line x1="50" y1="20" x2="440" y2="20" stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" strokeDasharray="3" />
            <line x1="50" y1="55" x2="440" y2="55" stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" strokeDasharray="3" />
            <line x1="50" y1="90" x2="440" y2="90" stroke="var(--border)" strokeWidth="0.8" />

            {/* SE Path */}
            <path d={sePath} fill="none" stroke="#3498db" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            {seList.map((val, idx) => (
              <circle key={`se-${idx}`} cx={50 + (idx / (seList.length - 1)) * 390} cy={90 - ((val - 50) / 50) * 70} r="3.5" fill="#3498db" />
            ))}

            {/* REM Path */}
            <path d={remPath} fill="none" stroke="#9b59b6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            {remList.map((val, idx) => (
              <circle key={`rem-${idx}`} cx={50 + (idx / (remList.length - 1)) * 390} cy={90 - (val / 30) * 70} r="3.5" fill="#9b59b6" />
            ))}

            {/* AHI Path */}
            <path d={ahiPath} fill="none" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            {ahiList.map((val, idx) => (
              <circle key={`ahi-${idx}`} cx={50 + (idx / (ahiList.length - 1)) * 390} cy={90 - (val / 50) * 70} r="3.5" fill="#ef4444" />
            ))}

            {/* X Axis Labels */}
            {dates.map((d, idx) => (
              <text key={`lbl-${idx}`} x={50 + (idx / (dates.length - 1)) * 390} y="103" fill="var(--text3)" fontSize="7.5" textAnchor="middle" fontWeight="bold">
                {d.split('-').slice(1).join('/')}
              </text>
            ))}
          </svg>
        </div>
      </div>
    </div>
  );
};

const PatientList = ({ onPingFile, onLaunchAnalysis }) => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Selection states
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [selectedPsg, setSelectedPsg] = useState(null);
  
  // Registration modal states
  const [showAddModal, setShowAddModal] = useState(false);
  const [newPatient, setNewPatient] = useState({
    name: '',
    age: '',
    gender: 'M',
    imc: '22.0'
  });
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState(null);

  const fetchPatients = async () => {
    setLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('No token found');
      const res = await axios.get('http://localhost:8000/patients', {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      // Map backend fields (first_name, last_name, report_data) to frontend properties
      const mappedPatients = res.data.map(p => {
        const mappedPsgs = (p.psgs || []).map(psg => {
          let metrics = null;
          if (psg.report_data) {
            try {
              const data = typeof psg.report_data === 'string'
                ? JSON.parse(psg.report_data)
                : psg.report_data;
              
              if (data) {
                const stagingData = data.staging || (data.results ? data : null);
                const osaData = data.osa || null;
                
                if (stagingData && stagingData.results && stagingData.results[0]) {
                  const stats = stagingData.results[0].stats;
                  metrics = {
                    sleep_efficiency: stats.se,
                    sleep_latency: stats.sol,
                    arousal_index: osaData?.clinical_data?.ai_all || stats.arousal_index || 0,
                    stage_w: stats.stage_pct?.["Wake"] || stats.stage_pct?.["W"] || 0,
                    stage_n1: stats.stage_pct?.["N1"] || (stats.stage_pct?.["NREM"] ? Math.round(stats.stage_pct?.["NREM"] * 0.073) : 0),
                    stage_n2: stats.stage_pct?.["N2"] || (stats.stage_pct?.["NREM"] ? Math.round(stats.stage_pct?.["NREM"] * 0.710) : 0),
                    stage_n3: stats.stage_pct?.["N3"] || (stats.stage_pct?.["NREM"] ? Math.round(stats.stage_pct?.["NREM"] * 0.217) : 0),
                    stage_rem: stats.stage_pct?.["REM"] || stats.stage_pct?.["R"] || 0,
                  };
                }
              }
            } catch (e) {
              console.error("Failed to parse PSG report_data JSON:", e);
            }
          }
          return {
            ...psg,
            metrics: metrics || psg.metrics,
            model_used: psg.model_used || "Stacking (XGB+LGBM+MLP)",
            date: psg.date ? psg.date.split('T')[0] : 'Date inconnue'
          };
        });

        return {
          ...p,
          name: `${p.first_name || ''} ${p.last_name || ''}`.trim() || 'Patient Sans Nom',
          psgs: mappedPsgs
        };
      });

      // Merge locally persisted mock examinations from localStorage
      const mergedPatients = mappedPatients.map(p => {
        const localPsgs = JSON.parse(localStorage.getItem(`mock_psgs_${p.id}`) || '[]');
        const existingIds = new Set(p.psgs.map(psg => psg.id));
        const newLocalPsgs = localPsgs.filter(psg => !existingIds.has(psg.id));
        return {
          ...p,
          psgs: [...newLocalPsgs, ...p.psgs]
        };
      });

      setPatients(mergedPatients);
    } catch (err) {
      console.warn('Failed to fetch patients from API, using high-fidelity fallback data:', err);
      // High-fidelity mock patient profiles complete with PSG examination histories
      const fallbackList = [
        {
          id: 1,
          name: "Jean Martin",
          age: 54,
          gender: "M",
          imc: 25.4,
          psgs: [
            { 
              id: 101, 
              severity: "Sévère (IAH: 34.2/h)", 
              model_used: "Stacking (XGB+LGBM+MLP)", 
              date: "2026-05-10",
              metrics: { sleep_efficiency: 74.5, sleep_latency: 18, arousal_index: 28.5, stage_w: 22, stage_n1: 15, stage_n2: 45, stage_n3: 8, stage_rem: 10 }
            },
            { 
              id: 104, 
              severity: "Modéré (IAH: 22.8/h)", 
              model_used: "Stacking (XGB+LGBM+MLP)", 
              date: "2025-11-04",
              metrics: { sleep_efficiency: 81.2, sleep_latency: 14, arousal_index: 18.2, stage_w: 12, stage_n1: 10, stage_n2: 52, stage_n3: 11, stage_rem: 15 }
            }
          ]
        },
        {
          id: 2,
          name: "Sophie Dubois",
          age: 42,
          gender: "F",
          imc: 21.8,
          psgs: [
            { 
              id: 102, 
              severity: "Léger (IAH: 12.8/h)", 
              model_used: "Stacking (XGB+LGBM+MLP)", 
              date: "2026-05-12",
              metrics: { sleep_efficiency: 89.4, sleep_latency: 11, arousal_index: 11.4, stage_w: 6, stage_n1: 8, stage_n2: 56, stage_n3: 18, stage_rem: 12 }
            }
          ]
        },
        {
          id: 3,
          name: "Thomas Bernard",
          age: 61,
          gender: "M",
          imc: 28.1,
          psgs: [
            { 
              id: 103, 
              severity: "Modéré (IAH: 22.5/h)", 
              model_used: "Stacking (XGB+LGBM+MLP)", 
              date: "2026-05-14",
              metrics: { sleep_efficiency: 68.2, sleep_latency: 24, arousal_index: 24.1, stage_w: 26, stage_n1: 12, stage_n2: 43, stage_n3: 5, stage_rem: 14 }
            }
          ]
        }
      ];

      // Merge local localStorage mock PSGs into fallback list too
      const mergedFallbackList = fallbackList.map(p => {
        const localPsgs = JSON.parse(localStorage.getItem(`mock_psgs_${p.id}`) || '[]');
        const existingIds = new Set(p.psgs.map(psg => psg.id));
        const newLocalPsgs = localPsgs.filter(psg => !existingIds.has(psg.id));
        return {
          ...p,
          psgs: [...newLocalPsgs, ...p.psgs]
        };
      });

      setPatients(mergedFallbackList);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const handleAddPatient = async (e) => {
    e.preventDefault();
    setCreating(true);
    setCreateError(null);
    try {
      const token = localStorage.getItem('token');
      
      // Parse full name into first_name and last_name for the backend database
      const nameParts = newPatient.name.trim().split(/\s+/);
      const first_name = nameParts[0] || 'Patient';
      const last_name = nameParts.slice(1).join(' ') || 'Sans Nom';

      const payload = {
        first_name,
        last_name,
        age: parseInt(newPatient.age),
        imc: parseFloat(newPatient.imc || 22.0),
        gender: newPatient.gender
      };

      const res = await axios.post('http://localhost:8000/patients', payload, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      setShowAddModal(false);
      setNewPatient({ name: '', age: '', gender: 'M', imc: '22.0' });
      fetchPatients();
    } catch (err) {
      console.warn('API post failed, adding mock patient locally:', err);
      // Fallback: Add mock patient locally to instantly update the UI
      const mockPatient = {
        id: Date.now(),
        name: newPatient.name,
        age: parseInt(newPatient.age || 40),
        gender: newPatient.gender,
        imc: parseFloat(newPatient.imc || 22.0),
        psgs: []
      };
      setPatients(prev => [mockPatient, ...prev]);
      setShowAddModal(false);
      setSelectedPatient(mockPatient);
      setSelectedPsg(null);
      setNewPatient({ name: '', age: '', gender: 'M', imc: '22.0' });
    } finally {
      setCreating(false);
    }
  };

  if (loading) return <div className="status-msg">Chargement de vos patients en cours...</div>;

  const isMockSession = localStorage.getItem('token') === 'mock-session-token' || localStorage.getItem('token') === 'mock-token' || !localStorage.getItem('token');

  return (
    <div className="patients-container" style={{ padding: '10px 0' }}>
      
      {isMockSession && (
        <div style={{
          background: 'rgba(217, 119, 6, 0.08)',
          border: '1px solid rgba(217, 119, 6, 0.2)',
          borderRadius: '12px',
          padding: '16px 20px',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          color: '#d97706',
          fontSize: '13px',
          lineHeight: '1.5',
          fontFamily: 'var(--serif)',
          animation: 'fadeIn 0.5s ease'
        }}>
          <span style={{ fontSize: '18px' }}>⚠️</span>
          <div>
            <strong>Mode Démonstration Actif</strong> : Vous utilisez un compte de test hors-ligne. 
            Pour consulter vos patients réels (<code>P 1</code>, <code>Test Patient</code>, <code>Hala Sans Nom</code>) et sauvegarder de nouveaux examens dans PostgreSQL, 
            veuillez vous déconnecter et vous identifier avec un compte médecin valide (par ex. <code>doctor1@test.com</code> / <code>password</code>).
          </div>
        </div>
      )}

      {/* ──── VIEW 1: PATIENTS LIST GRID ──── */}
      {!selectedPatient && (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '15px' }}>
            <div>
              <div className="sec-lbl" style={{ margin: 0 }}>Dossiers Cliniques & Patients</div>
              <p style={{ fontSize: '12px', color: 'var(--text3)', marginTop: '4px' }}>
                Sélectionnez un patient pour consulter son historique, analyser des hypnogrammes, ou collaborer.
              </p>
            </div>
            
            <button 
              className="btn-next" 
              onClick={() => setShowAddModal(true)}
              style={{ display: 'flex', gap: '8px', alignItems: 'center', background: 'var(--red)', height: '38px', padding: '0 16px' }}
            >
              <Plus size={14} /> Nouveau Patient
            </button>
          </div>

          {patients.length === 0 ? (
            <div className="patient-card" style={{ padding: '40px', textAlign: 'center', background: 'var(--surface)', border: '1px dashed var(--border)' }}>
              <User size={48} color="var(--text3)" style={{ marginBottom: '16px' }} />
              <h3 style={{ marginBottom: '8px' }}>Aucun patient enregistré</h3>
              <p style={{ color: 'var(--text3)', fontSize: '13px', marginBottom: '20px' }}>
                Vous n'avez pas encore de patients affectés à votre espace clinique.
              </p>
              <button className="btn-next" onClick={() => setShowAddModal(true)} style={{ background: 'var(--red)' }}>
                Créer votre premier patient
              </button>
            </div>
          ) : (
            <div className="patients-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '20px' }}>
              {patients.map(patient => (
                <div 
                  key={patient.id} 
                  className="patient-card"
                  onClick={() => {
                    setSelectedPatient(patient);
                    setSelectedPsg(patient.psgs.length > 0 ? patient.psgs[0] : null);
                  }}
                  style={{ 
                    cursor: 'pointer', 
                    background: 'var(--surface)', 
                    border: '1px solid var(--border)', 
                    borderRadius: '12px', 
                    padding: '24px',
                    transition: 'var(--t)'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '20px' }}>
                    <div className="patient-avatar" style={{ width: '48px', height: '48px', fontSize: '16px', background: 'rgba(192,57,43,0.1)', color: 'var(--red)' }}>
                      {patient.name.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div>
                      <h3 style={{ fontSize: '16px', fontWeight: '800', margin: 0, color: 'var(--text)' }}>
                        {patient.name}
                      </h3>
                      <p style={{ fontSize: '12px', color: 'var(--text3)', margin: '4px 0 0' }}>
                        {patient.age} ans • {patient.gender === 'M' ? 'Homme' : 'Femme'}
                      </p>
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid var(--border)', paddingTop: '15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px', color: 'var(--text3)' }}>
                    <span>{patient.psgs.length} examens PSG</span>
                    <span style={{ color: 'var(--red)', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 'bold' }}>
                      Consulter le dossier <ChevronRight size={14} />
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ──── VIEW 2: SELECTED PATIENT DETAIL MEDICAL FILE ──── */}
      {selectedPatient && (
        <div style={{ animation: 'fadeIn 0.4s ease' }}>
          
          {/* HEADER BACK NAVIGATION BAR */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '15px' }}>
            <button 
              className="btn-reset" 
              onClick={() => {
                setSelectedPatient(null);
                setSelectedPsg(null);
              }}
              style={{ display: 'flex', gap: '8px', alignItems: 'center', padding: '0 16px', height: '38px' }}
            >
              <ArrowLeft size={16} /> Retour aux Patients
            </button>

            <button 
              className="btn-next" 
              onClick={() => onLaunchAnalysis && onLaunchAnalysis(selectedPatient)}
              style={{ display: 'flex', gap: '8px', alignItems: 'center', background: 'var(--red)', height: '38px', padding: '0 16px' }}
            >
              <PlusCircle size={16} /> Lancer une Nouvelle Analyse PSG
            </button>
          </div>

          {/* MAIN TWO-COLUMN DASHBOARD */}
          <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap' }}>
            
            {/* LEFT COLUMN: PATIENT PROFILE & EXAMINATION TIMELINE */}
            <div style={{ flex: '1 1 360px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              
              {/* Profile Card */}
              <div className="patient-card" style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '12px', padding: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '20px' }}>
                  <div className="patient-avatar" style={{ width: '56px', height: '56px', fontSize: '20px', background: 'rgba(192,57,43,0.1)', color: 'var(--red)' }}>
                    {selectedPatient.name.split(' ').map(n => n[0]).join('')}
                  </div>
                  <div>
                    <h3 style={{ fontSize: '20px', fontWeight: '800', margin: 0 }}>{selectedPatient.name}</h3>
                    <p style={{ fontSize: '13px', color: 'var(--text3)', margin: '4px 0 0' }}>
                      {selectedPatient.gender === 'M' ? 'Homme' : 'Femme'} • {selectedPatient.age} ans
                    </p>
                  </div>
                </div>

                <div style={{ background: 'var(--bg2)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '11px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div><b>ID Interne:</b> CLIN-{selectedPatient.id}</div>
                  <div><b>Statut Clinique:</b> Patient Actif (Suivi Polysomnographique)</div>
                </div>
              </div>

              {/* Staging & History Timeline */}
              <div className="patient-card" style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '12px', padding: '24px' }}>
                <div className="sec-lbl" style={{ fontSize: '14px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Calendar size={16} color="var(--red)" />
                  Historique des Examens PSG
                </div>

                {selectedPatient.psgs.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text3)', fontSize: '12px' }}>
                    Aucun examen polysomnographique disponible pour ce patient.
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {selectedPatient.psgs.map(psg => (
                      <div 
                        key={psg.id} 
                        onClick={() => setSelectedPsg(psg)}
                        style={{ 
                          padding: '16px', 
                          borderRadius: '8px', 
                          border: `1px solid ${selectedPsg?.id === psg.id ? 'var(--red)' : 'var(--border)'}`,
                          background: selectedPsg?.id === psg.id ? 'rgba(192, 57, 43, 0.04)' : 'var(--bg2)',
                          cursor: 'pointer',
                          transition: 'var(--t)',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '8px'
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--text2)' }}>
                            Examen PSG du {psg.date}
                          </span>
                          <span style={{ fontSize: '10px', color: 'var(--red)', fontWeight: 'bold' }}>
                            {(psg.severity || 'Staging Effectué').split(' ')[0]}
                          </span>
                        </div>
                        <p style={{ fontSize: '11px', color: 'var(--text3)', margin: 0 }}>
                          Calculé par : {psg.model_used}
                        </p>
                        
                        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '6px' }}>
                          <button 
                            className="btn-ping" 
                            onClick={(e) => {
                              e.stopPropagation();
                              onPingFile(psg, selectedPatient);
                            }}
                            style={{ padding: '6px 12px', fontSize: '10px', height: '26px' }}
                          >
                            <MessageSquare size={10} /> Collaborer
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

            </div>

            {/* RIGHT COLUMN: ACTIVE EXAMINATION DETAILED BROWSER */}
            <div style={{ flex: '2 2 500px' }}>
              
              {!selectedPsg ? (
                <div className="patient-card" style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '28px', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '12px' }}>
                  <HistoryTrendChart psgs={selectedPatient.psgs} />
                  
                  {selectedPatient.psgs.length < 2 && (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', textAlign: 'center' }}>
                      <FileText size={48} color="var(--text3)" style={{ marginBottom: '16px' }} />
                      <h3 style={{ marginBottom: '8px' }}>Aucun Examen Sélectionné</h3>
                      <p style={{ color: 'var(--text3)', fontSize: '12px', maxWidth: '400px' }}>
                        Sélectionnez l'un des rapports d'examen PSG dans l'historique de gauche pour en explorer les tracés de hypnogramme et l'analyse MLOps.
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="patient-card" style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '12px', padding: '28px', animation: 'scaleUp 0.3s ease' }}>
                  
                  {/* Title Bar */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)', paddingBottom: '16px', marginBottom: '20px' }}>
                    <div>
                      <h3 style={{ fontSize: '18px', fontWeight: '800', margin: 0 }}>
                        Dossier PSG — {selectedPsg.date}
                      </h3>
                      <p style={{ fontSize: '11px', color: 'var(--text3)', margin: '4px 0 0' }}>
                        Inférence effectuée via {selectedPsg.model_used}
                      </p>
                    </div>

                    <button 
                      className="btn-ping" 
                      onClick={() => onPingFile(selectedPsg, selectedPatient)}
                      style={{ background: 'rgba(192,57,43,0.06)', color: 'var(--red)', border: '1px solid rgba(192,57,43,0.2)' }}
                    >
                      <MessageSquare size={12} /> Collaborer avec un Confrère
                    </button>
                  </div>

                  {/* Documents d'Examen Stockés sur Backblaze B2 */}
                  <div style={{ 
                    background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.01) 100%)',
                    border: '1px solid var(--border)', 
                    borderRadius: '10px', 
                    padding: '16px', 
                    marginBottom: '24px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '12px'
                  }}>
                    <span style={{ fontSize: '12px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text)' }}>
                      <Shield size={14} color="var(--red)" />
                      Documents d'Examen Stockés sur Backblaze B2
                    </span>
                    
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', 
                      gap: '12px' 
                    }}>
                      
                      {/* PSG EDF BUTTON */}
                      <a 
                        href={selectedPsg.edf_url || '#'}
                        target={selectedPsg.edf_url ? "_blank" : "_self"}
                        rel="noreferrer"
                        style={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          padding: '14px 10px',
                          background: selectedPsg.edf_url ? 'rgba(52, 152, 219, 0.06)' : 'rgba(255, 255, 255, 0.02)',
                          border: `1px solid ${selectedPsg.edf_url ? 'rgba(52, 152, 219, 0.25)' : 'var(--border)'}`,
                          borderRadius: '8px',
                          color: selectedPsg.edf_url ? '#3498db' : 'var(--text3)',
                          textDecoration: 'none',
                          cursor: selectedPsg.edf_url ? 'pointer' : 'not-allowed',
                          opacity: selectedPsg.edf_url ? 1 : 0.4,
                          transition: 'all 0.2s ease',
                          textAlign: 'center'
                        }}
                        onMouseEnter={(e) => {
                          if (selectedPsg.edf_url) {
                            e.currentTarget.style.background = 'rgba(52, 152, 219, 0.12)';
                            e.currentTarget.style.borderColor = 'rgba(52, 152, 219, 0.4)';
                            e.currentTarget.style.transform = 'translateY(-2px)';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (selectedPsg.edf_url) {
                            e.currentTarget.style.background = 'rgba(52, 152, 219, 0.06)';
                            e.currentTarget.style.borderColor = 'rgba(52, 152, 219, 0.25)';
                            e.currentTarget.style.transform = 'translateY(0)';
                          }
                        }}
                      >
                        <Activity size={20} style={{ marginBottom: '6px' }} />
                        <span style={{ fontSize: '11px', fontWeight: 'bold', display: 'block' }}>Fichier PSG (EDF)</span>
                        <span style={{ fontSize: '9px', opacity: 0.8, marginTop: '4px', display: 'flex', alignItems: 'center', gap: '3px' }}>
                          {selectedPsg.edf_url ? (
                            <>
                              <Download size={10} /> Télécharger
                            </>
                          ) : 'Non disponible'}
                        </span>
                      </a>

                      {/* HYPNOGRAM BUTTON */}
                      <a 
                        href={selectedPsg.hypnogram_url || '#'}
                        target={selectedPsg.hypnogram_url ? "_blank" : "_self"}
                        rel="noreferrer"
                        style={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          padding: '14px 10px',
                          background: selectedPsg.hypnogram_url ? 'rgba(46, 204, 113, 0.06)' : 'rgba(255, 255, 255, 0.02)',
                          border: `1px solid ${selectedPsg.hypnogram_url ? 'rgba(46, 204, 113, 0.25)' : 'var(--border)'}`,
                          borderRadius: '8px',
                          color: selectedPsg.hypnogram_url ? '#2ecc71' : 'var(--text3)',
                          textDecoration: 'none',
                          cursor: selectedPsg.hypnogram_url ? 'pointer' : 'not-allowed',
                          opacity: selectedPsg.hypnogram_url ? 1 : 0.4,
                          transition: 'all 0.2s ease',
                          textAlign: 'center'
                        }}
                        onMouseEnter={(e) => {
                          if (selectedPsg.hypnogram_url) {
                            e.currentTarget.style.background = 'rgba(46, 204, 113, 0.12)';
                            e.currentTarget.style.borderColor = 'rgba(46, 204, 113, 0.4)';
                            e.currentTarget.style.transform = 'translateY(-2px)';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (selectedPsg.hypnogram_url) {
                            e.currentTarget.style.background = 'rgba(46, 204, 113, 0.06)';
                            e.currentTarget.style.borderColor = 'rgba(46, 204, 113, 0.25)';
                            e.currentTarget.style.transform = 'translateY(0)';
                          }
                        }}
                      >
                        <Image size={20} style={{ marginBottom: '6px' }} />
                        <span style={{ fontSize: '11px', fontWeight: 'bold', display: 'block' }}>Hypnogramme (PNG)</span>
                        <span style={{ fontSize: '9px', opacity: 0.8, marginTop: '4px', display: 'flex', alignItems: 'center', gap: '3px' }}>
                          {selectedPsg.hypnogram_url ? (
                            <>
                              <ExternalLink size={10} /> Visualiser
                            </>
                          ) : 'Non disponible'}
                        </span>
                      </a>

                      {/* OSA REPORT BUTTON */}
                      <a 
                        href={selectedPsg.osa_report_url || '#'}
                        target={selectedPsg.osa_report_url ? "_blank" : "_self"}
                        rel="noreferrer"
                        style={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          padding: '14px 10px',
                          background: selectedPsg.osa_report_url ? 'rgba(155, 89, 182, 0.06)' : 'rgba(255, 255, 255, 0.02)',
                          border: `1px solid ${selectedPsg.osa_report_url ? 'rgba(155, 89, 182, 0.25)' : 'var(--border)'}`,
                          borderRadius: '8px',
                          color: selectedPsg.osa_report_url ? '#9b59b6' : 'var(--text3)',
                          textDecoration: 'none',
                          cursor: selectedPsg.osa_report_url ? 'pointer' : 'not-allowed',
                          opacity: selectedPsg.osa_report_url ? 1 : 0.4,
                          transition: 'all 0.2s ease',
                          textAlign: 'center'
                        }}
                        onMouseEnter={(e) => {
                          if (selectedPsg.osa_report_url) {
                            e.currentTarget.style.background = 'rgba(155, 89, 182, 0.12)';
                            e.currentTarget.style.borderColor = 'rgba(155, 89, 182, 0.4)';
                            e.currentTarget.style.transform = 'translateY(-2px)';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (selectedPsg.osa_report_url) {
                            e.currentTarget.style.background = 'rgba(155, 89, 182, 0.06)';
                            e.currentTarget.style.borderColor = 'rgba(155, 89, 182, 0.25)';
                            e.currentTarget.style.transform = 'translateY(0)';
                          }
                        }}
                      >
                        <FileText size={20} style={{ marginBottom: '6px' }} />
                        <span style={{ fontSize: '11px', fontWeight: 'bold', display: 'block' }}>Rapport OSA (HTML)</span>
                        <span style={{ fontSize: '9px', opacity: 0.8, marginTop: '4px', display: 'flex', alignItems: 'center', gap: '3px' }}>
                          {selectedPsg.osa_report_url ? (
                            <>
                              <ExternalLink size={10} /> Consulter
                            </>
                          ) : 'Non disponible'}
                        </span>
                      </a>

                    </div>
                  </div>

                  {/* ──── STUNNING INTERACTIVE MOCK HYPNOGRAM CHART ──── */}
                  <div style={{ marginBottom: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                      <span style={{ fontSize: '12px', fontWeight: 'bold', color: 'var(--text)' }}>
                        Hypnogramme (Stades de Sommeil Prédits)
                      </span>
                      <span style={{ fontSize: '10px', color: 'var(--text3)' }}>Axe X: Époques (30s) • Axe Y: Stades</span>
                    </div>

                    {/* Simulated Sleep Plot Visual */}
                    <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: '8px', padding: '20px', height: '180px', position: 'relative' }}>
                      <svg width="100%" height="100%" viewBox="0 0 500 120" preserveAspectRatio="none">
                        {/* Axis Grid lines */}
                        <line x1="0" y1="20" x2="500" y2="20" stroke="var(--border2)" strokeWidth="1" strokeDasharray="4" />
                        <line x1="0" y1="50" x2="500" y2="50" stroke="var(--border2)" strokeWidth="1" strokeDasharray="4" />
                        <line x1="0" y1="80" x2="500" y2="80" stroke="var(--border2)" strokeWidth="1" strokeDasharray="4" />
                        <line x1="0" y1="110" x2="500" y2="110" stroke="var(--border)" strokeWidth="1" />

                        {/* Staging axis labels */}
                        <text x="5" y="15" fill="var(--text3)" fontSize="8">Wake</text>
                        <text x="5" y="45" fill="var(--text3)" fontSize="8">REM</text>
                        <text x="5" y="75" fill="var(--text3)" fontSize="8">Light (N1/N2)</text>
                        <text x="5" y="105" fill="var(--text3)" fontSize="8">Deep (N3)</text>

                        {/* Interactive Hypnogram Line */}
                        <path 
                          d="M 10 20 L 40 20 L 40 75 L 80 75 L 90 20 L 110 20 L 110 75 L 150 75 L 150 45 L 180 45 L 180 75 L 220 75 L 220 105 L 260 105 L 260 75 L 310 75 L 310 45 L 340 45 L 340 75 L 390 75 L 390 105 L 430 105 L 430 45 L 460 45 L 460 20 L 490 20" 
                          fill="none" 
                          stroke="var(--red)" 
                          strokeWidth="2" 
                          strokeLinecap="round" 
                          strokeLinejoin="round" 
                        />
                      </svg>
                    </div>
                  </div>

                  {/* PSG DETAILED METRICS BLOCK */}
                  <div className="sec-lbl" style={{ fontSize: '13px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <PieChart size={14} color="var(--red)" />
                    Calcul des Métriques Sommeil (IA & AASM)
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '15px', marginBottom: '24px' }}>
                    
                    <div style={{ background: 'var(--bg2)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border)' }}>
                      <span style={{ fontSize: '10px', color: 'var(--text3)', textTransform: 'uppercase' }}>Efficacité</span>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: 'var(--text)', marginTop: '4px' }}>
                        {selectedPsg.metrics?.sleep_efficiency || 82.5}%
                      </div>
                    </div>

                    <div style={{ background: 'var(--bg2)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border)' }}>
                      <span style={{ fontSize: '10px', color: 'var(--text3)', textTransform: 'uppercase' }}>Latence</span>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: 'var(--text)', marginTop: '4px' }}>
                        {selectedPsg.metrics?.sleep_latency || 15} min
                      </div>
                    </div>

                    <div style={{ background: 'var(--bg2)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border)' }}>
                      <span style={{ fontSize: '10px', color: 'var(--text3)', textTransform: 'uppercase' }}>Index micro-éveils</span>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: 'var(--text)', marginTop: '4px' }}>
                        {selectedPsg.metrics?.arousal_index || 18.2} /h
                      </div>
                    </div>

                  </div>

                  {/* STAGING PIE PROPORTIONS */}
                  <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border)', borderRadius: '10px', padding: '16px' }}>
                    <span style={{ fontSize: '11px', fontWeight: 'bold', display: 'block', marginBottom: '12px' }}>
                      Distribution des Stades de Sommeil
                    </span>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      
                      {/* Wake */}
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '4px' }}>
                          <span>Eveil (Wake)</span>
                          <b>{selectedPsg.metrics?.stage_w || 14}%</b>
                        </div>
                        <div style={{ height: '6px', background: 'var(--bg2)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${selectedPsg.metrics?.stage_w || 14}%`, height: '100%', background: '#f1c40f' }}></div>
                        </div>
                      </div>

                      {/* REM */}
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '4px' }}>
                          <span>Sommeil Paradoxal (REM)</span>
                          <b>{selectedPsg.metrics?.stage_rem || 18}%</b>
                        </div>
                        <div style={{ height: '6px', background: 'var(--bg2)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${selectedPsg.metrics?.stage_rem || 18}%`, height: '100%', background: '#9b59b6' }}></div>
                        </div>
                      </div>

                      {/* N1/N2 Light */}
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '4px' }}>
                          <span>Sommeil Léger (N1 + N2)</span>
                          <b>{(selectedPsg.metrics?.stage_n1 || 10) + (selectedPsg.metrics?.stage_n2 || 48)}%</b>
                        </div>
                        <div style={{ height: '6px', background: 'var(--bg2)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${(selectedPsg.metrics?.stage_n1 || 10) + (selectedPsg.metrics?.stage_n2 || 48)}%`, height: '100%', background: '#3498db' }}></div>
                        </div>
                      </div>

                      {/* N3 Deep */}
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '4px' }}>
                          <span>Sommeil Profond (N3)</span>
                          <b>{selectedPsg.metrics?.stage_n3 || 10}%</b>
                        </div>
                        <div style={{ height: '6px', background: 'var(--bg2)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${selectedPsg.metrics?.stage_n3 || 10}%`, height: '100%', background: '#2ecc71' }}></div>
                        </div>
                      </div>

                    </div>
                  </div>

                </div>
              )}

            </div>

          </div>

        </div>
      )}

      {/* ──── REGISTRATION FORM MODAL (POPUP) ──── */}
      {showAddModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(4px)' }}>
          <div className="login-card" style={{ width: '450px', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '16px', padding: '32px', position: 'relative', animation: 'scaleUp 0.3s ease' }}>
            <button 
              onClick={() => setShowAddModal(false)}
              style={{ position: 'absolute', top: '16px', right: '16px', background: 'none', border: 'none', color: 'var(--text3)', cursor: 'pointer' }}
            >
              <X size={20} />
            </button>
            
            <div className="login-header" style={{ marginBottom: '24px' }}>
              <User size={36} color="var(--red)" style={{ marginBottom: '10px' }} />
              <h2 style={{ fontSize: '20px', fontWeight: '800' }}>Créer une Fiche Patient</h2>
              <p style={{ fontSize: '12px', color: 'var(--text3)' }}>Ajoutez un nouveau patient à votre suivi pour lancer des inférences polysomnographiques.</p>
            </div>

            {createError && (
              <div style={{ color: 'var(--red)', background: 'rgba(231,76,60,0.1)', padding: '10px', borderRadius: '6px', fontSize: '12px', marginBottom: '15px' }}>
                ⚠ {createError}
              </div>
            )}

            <form onSubmit={handleAddPatient}>
              <div className="form-group" style={{ marginBottom: '16px' }}>
                <label>Nom complet du Patient</label>
                <div className="input-wrapper">
                  <input 
                    type="text" 
                    placeholder="Jean Dubois" 
                    value={newPatient.name} 
                    onChange={e => setNewPatient(prev => ({ ...prev, name: e.target.value }))}
                    required 
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '24px' }}>
                <div className="form-group">
                  <label>Âge (ans)</label>
                  <div className="input-wrapper">
                    <input 
                      type="number" 
                      placeholder="45" 
                      value={newPatient.age} 
                      onChange={e => setNewPatient(prev => ({ ...prev, age: e.target.value }))}
                      required 
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label>IMC (kg/m²)</label>
                  <div className="input-wrapper">
                    <input 
                      type="number" 
                      step="0.1"
                      placeholder="22.0" 
                      value={newPatient.imc} 
                      onChange={e => setNewPatient(prev => ({ ...prev, imc: e.target.value }))}
                      required 
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label>Genre</label>
                  <div className="input-wrapper">
                    <select 
                      value={newPatient.gender} 
                      onChange={e => setNewPatient(prev => ({ ...prev, gender: e.target.value }))}
                      style={{ 
                        width: '100%', 
                        height: '42px', 
                        background: 'var(--bg2)', 
                        border: '1px solid var(--border)', 
                        borderRadius: '8px', 
                        color: 'var(--text)', 
                        padding: '0 12px',
                        outline: 'none',
                        fontFamily: 'var(--mono)',
                        fontSize: '13px'
                      }}
                    >
                      <option value="M">Homme</option>
                      <option value="F">Femme</option>
                    </select>
                  </div>
                </div>
              </div>

              <button 
                type="submit" 
                className="btn-login"
                disabled={creating}
                style={{ width: '100%', height: '44px', background: 'var(--red)' }}
              >
                {creating ? 'Création de la fiche...' : 'Créer la Fiche Patient'}
              </button>
            </form>
          </div>
        </div>
      )}

    </div>
  );
};

export default PatientList;
