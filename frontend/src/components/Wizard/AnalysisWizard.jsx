import { useState, useEffect } from 'react';
import { CloudUpload, Activity, Layers, FileText } from 'lucide-react';
import ScanningAnimation from './ScanningAnimation';
import AnalysisSimulation from '../AnalysisSimulation/AnalysisSimulation';
import { AnalysisResults } from '../Results';
import axios from 'axios';
import './Wizard.css';

const AnalysisWizard = ({ onAnalysisComplete, onStartBgUpload, preselectedPatient, onClearPreselectedPatient }) => {
  const [step, setStep] = useState(1);
  const [channels, setChannels] = useState('5');
  const [classes, setClasses] = useState('3');
  const [file, setFile] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [serverStatus, setServerStatus] = useState('checking');
  const [error, setError] = useState(null);

  const [analysisData, setAnalysisData] = useState(null);
  const [activePsgId, setActivePsgId] = useState(null);

  // Simulation State
  const [simState, setSimState] = useState({ visible: false, activeStep: 0, progress: 0 });

  useEffect(() => {
    const checkServer = async () => {
      try {
        const res = await fetch('http://localhost:8000/docs');
        if (res.ok) setServerStatus('online');
        else setServerStatus('offline');
      } catch (err) {
        setServerStatus('offline');
      }
    };
    checkServer();
    const timer = setInterval(checkServer, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    setStep(1);
    setFile(null);
    setError(null);
    setAnalysisData(null);
    setActivePsgId(null);
  }, [preselectedPatient]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.name.toLowerCase().endsWith('.edf')) {
        setError('Please upload an EDF file.');
        return;
      }
      setError(null);
      setFile(selectedFile);
      setIsScanning(true);
      setTimeout(() => setIsScanning(false), 2000); // Initial fast scan animation
    }
  };

  const delay = (ms) => new Promise(r => setTimeout(r, ms));

  const startAnalysis = async () => {
    if (!file) return;
    setIsScanning(true);
    setError(null);

    // Hide wizard panels and show simulation
    setSimState({ visible: true, activeStep: 0, progress: 10 });
    await delay(700);
    setSimState({ visible: true, activeStep: 1, progress: 25 });
    await delay(900);
    setSimState({ visible: true, activeStep: 2, progress: 42 });
    await delay(700);
    setSimState({ visible: true, activeStep: 3, progress: 88 });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('models', 'LSTM'); // Using LSTM by default for now
    formData.append('channels', channels);
    formData.append('classes', classes);

    try {
      const res = await axios.post('http://localhost:8000/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      if (res.data.results) {
        setSimState({ visible: true, activeStep: 4, progress: 100 });
        await delay(1000); // let user see 100%
        setSimState({ visible: false, activeStep: 4, progress: 100 });
        
        // 1. Instantly set analysis data and step to 4 so results render without delay!
        setAnalysisData(res.data);
        setStep(4);
        
        if (onAnalysisComplete) {
          onAnalysisComplete(res.data, null);
        }

        // 2. Perform the database save immediately (without the huge file payload) so it is instant
        if (preselectedPatient) {
          const patientId = preselectedPatient.id;
          (async () => {
            try {
              console.log('[Wizard] Starting instant metadata PSG save for patient ID:', patientId);
              const token = localStorage.getItem('token');
              const saveFormData = new FormData();
              saveFormData.append('severity', 'Staging Effectué');
              saveFormData.append('report_data', JSON.stringify(res.data));

              // Call backend to create the database record immediately
              const saveRes = await axios.post(`http://localhost:8000/patients/${patientId}/psgs`, saveFormData, {
                headers: {
                  'Content-Type': 'multipart/form-data',
                  Authorization: `Bearer ${token}`
                }
              });
              
              if (saveRes.data && saveRes.data.id) {
                const psgId = saveRes.data.id;
                console.log('[Wizard] Successfully saved PSG record metadata with ID:', psgId);
                setActivePsgId(psgId);
                if (onAnalysisComplete) {
                  onAnalysisComplete(res.data, psgId);
                }

                // If a file is uploaded, kick off the background upload task
                if (file && onStartBgUpload) {
                  onStartBgUpload(file, preselectedPatient.name || 'Patient', psgId);
                }
              } else {
                throw new Error('No PSG ID returned from backend');
              }
            } catch (saveErr) {
              console.warn('[Wizard] Failed to auto-save PSG record to database, using local backup store:', saveErr);
              
              // Fallback: Save locally in localStorage so it persists dynamically for the mock/fallback flow
              try {
                const stats = res.data.results?.[0]?.stats || {};
                const stage_pct = stats.stage_pct || {};
                
                const mockPsg = {
                  id: Date.now(),
                  patient_id: patientId,
                  severity: 'Staging Effectué',
                  model_used: 'LSTM (5 Canaux, 3 Classes)',
                  date: new Date().toISOString().split('T')[0],
                  report_data: JSON.stringify(res.data),
                  metrics: {
                    sleep_efficiency: stats.se || 80,
                    sleep_latency: stats.sol || 15,
                    arousal_index: 0,
                    stage_w: stage_pct["Wake"] || stage_pct["W"] || 0,
                    stage_n1: stage_pct["N1"] || (stage_pct["NREM"] ? Math.round(stage_pct["NREM"] * 0.07) : 0),
                    stage_n2: stage_pct["N2"] || (stage_pct["NREM"] ? Math.round(stage_pct["NREM"] * 0.71) : 0),
                    stage_n3: stage_pct["N3"] || (stage_pct["NREM"] ? Math.round(stage_pct["NREM"] * 0.22) : 0),
                    stage_rem: stage_pct["REM"] || stage_pct["R"] || 0,
                  }
                };
                
                const existing = JSON.parse(localStorage.getItem(`mock_psgs_${patientId}`) || '[]');
                existing.unshift(mockPsg);
                localStorage.setItem(`mock_psgs_${patientId}`, JSON.stringify(existing));
                console.log('[Wizard] Successfully saved PSG record to localStorage!');
              } catch (localErr) {
                console.error('[Wizard] Failed to backup to localStorage:', localErr);
              }
            }
          })();
        }
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Error connecting to server.');
      setSimState({ visible: false, activeStep: 0, progress: 0 });
    } finally {
      setIsScanning(false);
    }
  };

  if (simState.visible) {
    return (
      <div className="wizard-container" style={{ padding: '0', background: 'transparent', boxShadow: 'none', border: 'none' }}>
        <AnalysisSimulation activeStep={simState.activeStep} progress={simState.progress} visible={true} />
      </div>
    );
  }

  const isMockSession = localStorage.getItem('token') === 'mock-session-token' || localStorage.getItem('token') === 'mock-token' || !localStorage.getItem('token');

  return (
    <div className="wizard-container">
      <div className="wizard-header">
        <div className={`wiz-step ${step === 1 ? 'active' : step > 1 ? 'done' : ''}`} onClick={() => setStep(1)}>
          <span className="wiz-circle">1</span>
          <span className="wiz-label">CANAUX</span>
        </div>
        <div className="wiz-line"></div>
        <div className={`wiz-step ${step === 2 ? 'active' : step > 2 ? 'done' : ''}`} onClick={() => setStep(2)}>
          <span className="wiz-circle">2</span>
          <span className="wiz-label">CLASSES</span>
        </div>
        <div className="wiz-line"></div>
        <div className={`wiz-step ${step === 3 ? 'active' : step > 3 ? 'done' : ''}`} onClick={() => setStep(3)}>
          <span className="wiz-circle">3</span>
          <span className="wiz-label">FICHIER EDF</span>
        </div>
        <div className="wiz-line"></div>
        <div className={`wiz-step ${step === 4 ? 'active' : ''}`}>
          <span className="wiz-circle">4</span>
          <span className="wiz-label">RÉSULTATS</span>
        </div>
      </div>

      <div className="wiz-body">
        {isMockSession && (
          <div style={{
            background: 'rgba(217, 119, 6, 0.08)',
            border: '1px solid rgba(217, 119, 6, 0.2)',
            borderRadius: '10px',
            padding: '14px 18px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            color: '#d97706',
            fontSize: '12.5px',
            lineHeight: '1.45',
            fontFamily: 'var(--serif)',
            animation: 'fadeIn 0.5s ease'
          }}>
            <span style={{ fontSize: '16px' }}>⚠️</span>
            <div>
              <strong>Mode Démonstration Actif</strong> : Vous utilisez un compte de test hors-ligne. 
              Pour sauvegarder les examens dans la base de données, déconnectez-vous et identifiez-vous avec un compte médecin valide (par ex. <code>doctor1@test.com</code> / <code>password</code>).
            </div>
          </div>
        )}

        {error && <div className="error-bar visible">⚠ {error}</div>}
        
        {/* STEP 1: Channels */}
        {step === 1 && (
          <div className="wiz-panel active">
            <div className="wiz-info-box">
              <div className="wiz-info-icon" style={{color: '#1d4ed8', background: 'rgba(29,78,216,0.1)'}}>
                <Activity size={24} />
              </div>
              <div>
                <div className="wiz-info-title">Quels canaux PSG sont disponibles ?</div>
                <div className="wiz-info-text">
                  Plus de canaux = meilleure précision. 5 canaux est le standard clinique complet.
                </div>
              </div>
            </div>
            <div className="card-group">
              <div className={`sel-card ${channels === '5' ? 'active' : ''}`} onClick={() => setChannels('5')}>
                <div className="sel-card-icon"><Layers size={24} /></div>
                <div className="sel-card-title">5 Canaux</div>
                <div className="sel-card-desc">EEG×2 + EOG×2 + EMG</div>
                <div className="sel-card-tag green">RECOMMANDÉ</div>
              </div>
              <div className={`sel-card ${channels === '2' ? 'active' : ''}`} onClick={() => setChannels('2')}>
                <div className="sel-card-icon"><Activity size={24} /></div>
                <div className="sel-card-title">2 Canaux</div>
                <div className="sel-card-desc">EEG×2 uniquement</div>
                <div className="sel-card-tag blue">PORTABLE</div>
              </div>
            </div>
            <div className="wiz-actions">
              <button className="btn-next" onClick={() => setStep(2)}>SUIVANT &gt;</button>
            </div>
          </div>
        )}

        {/* STEP 2: Classes */}
        {step === 2 && (
          <div className="wiz-panel active">
            <div className="wiz-info-box">
              <div className="wiz-info-icon" style={{color: '#6d28d9', background: 'rgba(109,40,217,0.1)'}}>
                <Layers size={24} />
              </div>
              <div>
                <div className="wiz-info-title">Combien de stades de sommeil ?</div>
                <div className="wiz-info-text">
                  La classification en 3 classes est plus robuste pour le diagnostic global.
                </div>
              </div>
            </div>
            <div className="card-group">
              <div className={`sel-card ${classes === '3' ? 'active' : ''}`} onClick={() => setClasses('3')}>
                <div className="sel-card-title">3 Classes</div>
                <div className="sel-card-desc">Wake / NREM / REM</div>
                <div className="sel-card-tag green">RECOMMANDÉ</div>
              </div>
              <div className={`sel-card ${classes === '5' ? 'active' : ''}`} onClick={() => setClasses('5')}>
                <div className="sel-card-title">5 Classes</div>
                <div className="sel-card-desc">W / N1 / N2 / N3 / R</div>
                <div className="sel-card-tag purple">RECHERCHE</div>
              </div>
            </div>
            <div className="wiz-actions" style={{justifyContent: 'space-between'}}>
              <button className="btn-prev" onClick={() => setStep(1)}>&lt; RETOUR</button>
              <button className="btn-next" onClick={() => setStep(3)}>SUIVANT &gt;</button>
            </div>
          </div>
        )}

        {/* STEP 3: Upload */}
        {step === 3 && (
          <div className="wiz-panel active">
             <div className="wiz-info-box">
              <div className="wiz-info-icon" style={{color: '#c0392b', background: 'rgba(192,57,43,0.1)'}}>
                <FileText size={24} />
              </div>
              <div>
                <div className="wiz-info-title">Chargez le fichier PSG (.edf)</div>
                <div className="wiz-info-text">
                  L'analyse est effectuée localement sur votre machine.
                </div>
              </div>
            </div>

            <div className="wiz-summary">
              <div className="wiz-sum-title">RÉCAPITULATIF</div>
              <div className="wiz-sum-grid">
                <div className="wiz-sum-row">
                  <span className="wiz-sum-lbl"><span className="dot green"></span> Canaux</span>
                  <b>{channels} canaux</b>
                </div>
                <div className="wiz-sum-row">
                  <span className="wiz-sum-lbl"><span className="dot blue"></span> Classes</span>
                  <b>{classes} classes</b>
                </div>
                <div className="wiz-sum-row">
                  <span className="wiz-sum-lbl"><span className="dot grey"></span> Fichier</span>
                  <i>{file ? file.name : 'Aucun fichier'}</i>
                </div>
                {preselectedPatient && (
                  <div className="wiz-sum-row" style={{ borderTop: '1px dashed var(--border)', paddingTop: '10px', marginTop: '10px' }}>
                    <span className="wiz-sum-lbl" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span className="dot red"></span> Patient Cible
                    </span>
                    <b style={{ color: 'var(--red)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {preselectedPatient.name}
                      <button 
                        onClick={(e) => { e.stopPropagation(); onClearPreselectedPatient && onClearPreselectedPatient(); }}
                        style={{ background: 'rgba(231,76,60,0.1)', border: 'none', color: 'var(--red)', borderRadius: '50%', width: '16px', height: '16px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '9px', cursor: 'pointer' }}
                        title="Détacher le patient"
                      >
                        ✕
                      </button>
                    </b>
                  </div>
                )}
              </div>
            </div>

            <div className="srv-row" style={{marginTop: '20px', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '10px'}}>
              <div className="srv-chip">
                <div className={`dot ${serverStatus === 'online' ? 'online' : 'offline'}`}></div>
                <span>Serveur {serverStatus}</span>
              </div>
            </div>

            <div className="drop-zone" onClick={() => document.getElementById('file-input').click()}>
              <ScanningAnimation isRunning={isScanning} />
              {!isScanning && (
                <div className="dz-body">
                  <div className="dz-icon"><CloudUpload size={40} color="var(--red)" style={{opacity: 0.7}} /></div>
                  <div className="dz-title">Déposez le fichier EDF ici</div>
                  <div className="dz-sub">ou cliquez pour <b>parcourir</b></div>
                </div>
              )}
              <input type="file" id="file-input" accept=".edf" style={{display: 'none'}} onChange={handleFileChange} />
            </div>

            <div className="wiz-actions" style={{justifyContent: 'space-between'}}>
              <button className="btn-prev" onClick={() => setStep(2)}>&lt; RETOUR</button>
              <button 
                className={`btn-next ${(!file || isScanning) ? 'disabled' : ''}`} 
                disabled={!file || isScanning}
                onClick={startAnalysis}
              >
                {isScanning ? 'ANALYSE EN COURS...' : '▷ ANALYSER'}
              </button>
            </div>
          </div>
        )}

        {/* STEP 4: Results */}
        {step === 4 && analysisData && (
          <div className="wiz-panel active">
            <AnalysisResults 
              analysisData={analysisData} 
              activePsgId={activePsgId}
              patient={preselectedPatient}
            />
            
            <div className="wiz-actions" style={{justifyContent: 'space-between', marginTop: '40px'}}>
              <button className="btn-prev" onClick={() => setStep(3)}>&lt; NOUVEL UPLOAD</button>
              <button className="btn-next" onClick={() => {
                setStep(1);
                setFile(null);
                setAnalysisData(null);
                setActivePsgId(null);
                onClearPreselectedPatient && onClearPreselectedPatient();
              }}>TERMINER</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisWizard;
