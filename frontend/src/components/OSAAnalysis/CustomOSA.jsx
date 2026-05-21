import React, { useState } from 'react';
import axios from 'axios';
import { 
  FileDown, 
  FileBarChart, 
  AlertCircle, 
  User, 
  Activity, 
  Moon, 
  Zap, 
  RotateCcw, 
  CheckCircle,
  Clock
} from 'lucide-react';
import './OSAAnalysis.css';

const CustomOSA = () => {
  const [activeTab, setActiveTab] = useState('manual'); // 'manual' or 'file'
  
  // File Upload states
  const [file, setFile] = useState(null);
  const [fileFeatures, setFileFeatures] = useState(null);
  
  // Manual Input states
  const [manualFeatures, setManualFeatures] = useState({
    age: 50,
    gender: 'M',
    bmi: 28.0,
    tst: 420,
    tib: 480,
    sol: 15,
    se: 87.5,
    waso: 30,
    spt: 460,
    n1: 7.3,
    n2: 50,
    n3: 18,
    rem: 22,
    n3min: 75.6,
    remlat: 90,
    n3lat: 20,
    avgsat: 94,
    minsat: 85,
    pctsa90: '',
    pctsa85: '',
    pctsa95: '',
    ai_all: '',
    ai_nrem: '',
    ai_rem: '',
    frag: '',
    wakebouts: '',
    remcycles: '',
    remt1p: '',
    remt34p: ''
  });

  const [osaResults, setOsaResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (field, val) => {
    setManualFeatures(prev => ({
      ...prev,
      [field]: val
    }));
  };

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setLoading(true);
    setError(null);
    setFileFeatures(null);
    setOsaResults(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await axios.post('http://localhost:8000/parse_features_file', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setFileFeatures(res.data.features);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Error parsing file.');
    } finally {
      setLoading(false);
    }
  };

  const handlePredictManual = async () => {
    setLoading(true);
    setError(null);
    setOsaResults(null);

    // Map input fields to what predict_osa_custom expects
    const payload = {
      features: {
        age_s2: manualFeatures.age,
        gender: manualFeatures.gender,
        bmi_s2: manualFeatures.bmi,
        tst_min: manualFeatures.tst,
        tib_min: manualFeatures.tib,
        sol_min: manualFeatures.sol,
        sleep_efficiency: manualFeatures.se,
        waso_min: manualFeatures.waso,
        spt_min: manualFeatures.spt,
        N1_pct: manualFeatures.n1,
        N2_pct: manualFeatures.n2,
        N3_pct: manualFeatures.n3,
        REM_pct: manualFeatures.rem,
        timest34: manualFeatures.n3min,
        rem_latency_min: manualFeatures.remlat,
        n3_latency_min: manualFeatures.n3lat,
        avgsat: manualFeatures.avgsat,
        minsat: manualFeatures.minsat,
        pctsa90h: manualFeatures.pctsa90,
        pctsa85h: manualFeatures.pctsa85,
        pctsa95h: manualFeatures.pctsa95,
        ai_all: manualFeatures.ai_all,
        ai_nrem: manualFeatures.ai_nrem,
        ai_rem: manualFeatures.ai_rem,
        frag_index: manualFeatures.frag,
        n_wake_bouts: manualFeatures.wakebouts,
        n_rem_cycles: manualFeatures.remcycles,
        remt1p: manualFeatures.remt1p,
        remt34p: manualFeatures.remt34p
      }
    };

    try {
      const res = await axios.post('http://localhost:8000/predict_osa_custom', payload);
      setOsaResults(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Error predicting OSA.');
    } finally {
      setLoading(false);
    }
  };

  const handlePredictFile = async () => {
    if (!fileFeatures) return;
    setLoading(true);
    setError(null);
    setOsaResults(null);
    try {
      const payload = { features: fileFeatures };
      const res = await axios.post('http://localhost:8000/predict_osa_custom', payload);
      setOsaResults(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Error predicting OSA.');
    } finally {
      setLoading(false);
    }
  };

  const resetCustomFileUpload = () => {
    setFile(null);
    setFileFeatures(null);
    setOsaResults(null);
    setError(null);
  };

  function getClinicalInterpretations(feats) {
    const list = [];
    const se = parseFloat(feats.sleep_efficiency || feats.slpeffp || 100);
    const waso = parseFloat(feats.waso_min || feats.waso || 0);
    const rem = parseFloat(feats.REM_pct || feats.timeremp || 20);
    const sol = parseFloat(feats.sol_min || feats.slplatp || 15);
    const avgsat = parseFloat(feats.avgsat || 95);
    const minsat = parseFloat(feats.minsat || 85);
    const ai = parseFloat(feats.ai_all || 0);

    if (se < 85) list.push("Efficacité du sommeil réduite (<85%) — fragmentation significative");
    if (waso > 60) list.push("WASO élevé (>60 min) — réveils nocturnes fréquents");
    if (rem < 15) list.push("Temps REM insuffisant (<15%) — perturbation du sommeil paradoxal");
    if (sol > 30) list.push("Latence d'endormissement prolongée (>30 min) — insomnie possible");
    if (avgsat < 90) list.push("Désaturation globale (SpO₂ moyenne < 90%) — hypoxémie persistante");
    if (minsat < 80) list.push("Nadir nocturne très bas (SpO₂ min < 80%) — apnées obstructives sévères");
    if (ai > 15) list.push("Index d'éveil élevé (>15/h) — instabilité majeure de l'architecture du sommeil");
    
    if (list.length === 0) {
      list.push("Architecture et oxymétrie du sommeil dans les limites de la normale");
    }
    return list;
  }

  return (
    <div className="custom-osa-section" id="custom-osa-section" style={{ background: 'var(--surface)', padding: '24px', borderRadius: '12px', border: '1px solid var(--border)', maxWidth: '1200px', margin: '0 auto' }}>
      
      {/* HEADER WITH TITLE & TABS */}
      <div className="custom-osa-header" style={{ borderBottom: '1px solid var(--border)', paddingBottom: '20px', marginBottom: '24px' }}>
        <div className="custom-osa-title-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px', marginBottom: '20px' }}>
          <div>
            <div className="custom-osa-title" style={{ fontSize: '20px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text)' }}>
              <FileBarChart size={22} color="var(--red)" />
              Prédiction OSA Personnalisée
            </div>
            <div className="custom-osa-subtitle" style={{ fontSize: '12px', color: 'var(--text3)', marginTop: '4px' }}>
              Saisissez les données manuellement ou importez un fichier CSV/XML — sans fichier EDF requis
            </div>
          </div>
        </div>

        {/* Dynamic Tab Switcher */}
        <div className="custom-osa-tabs" style={{ display: 'flex', borderBottom: '1.5px solid var(--border2)', marginTop: '10px' }}>
          <button 
            className={`co-tab ${activeTab === 'manual' ? 'active' : ''}`} 
            onClick={() => { setActiveTab('manual'); setOsaResults(null); setError(null); }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style={{ marginRight: '8px' }}><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            Saisie Manuelle
          </button>
          <button 
            className={`co-tab ${activeTab === 'file' ? 'active' : ''}`} 
            onClick={() => { setActiveTab('file'); setOsaResults(null); setError(null); }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style={{ marginRight: '8px' }}><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
            Import CSV / XML
          </button>
        </div>
      </div>

      {/* ──── TAB 1: MANUAL INPUT PANEL ──── */}
      {activeTab === 'manual' && (
        <div className="co-panel active">
          <div className="osa-form">
            
            {/* Demographics */}
            <div className="osa-section-title" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', color: 'var(--red)', textTransform: 'uppercase', letterSpacing: '0.5px', margin: '24px 0 12px' }}>
              <User size={15} />
              Données Démographiques
            </div>
            <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <label>Âge (ans)
                <input type="number" value={manualFeatures.age} onChange={(e) => handleInputChange('age', e.target.value)} min="18" max="100" />
              </label>
              <label>Sexe
                <select value={manualFeatures.gender} onChange={(e) => handleInputChange('gender', e.target.value)}>
                  <option value="M">Homme</option>
                  <option value="F">Femme</option>
                </select>
              </label>
              <label>IMC (kg/m²)
                <input type="number" step="0.1" value={manualFeatures.bmi} onChange={(e) => handleInputChange('bmi', e.target.value)} min="15" max="60" />
              </label>
            </div>

            {/* Sleep Architecture */}
            <div className="osa-section-title" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', color: 'var(--red)', textTransform: 'uppercase', letterSpacing: '0.5px', margin: '24px 0 12px' }}>
              <Moon size={15} />
              Architecture du Sommeil
            </div>
            <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <label>TST (min)
                <input type="number" value={manualFeatures.tst} onChange={(e) => handleInputChange('tst', e.target.value)} min="0" max="600" step="0.1" />
                <span className="osa-hint">Temps total de sommeil</span>
              </label>
              <label>TIB (min)
                <input type="number" value={manualFeatures.tib} onChange={(e) => handleInputChange('tib', e.target.value)} min="0" max="700" step="0.1" />
                <span className="osa-hint">Temps au lit</span>
              </label>
              <label>SOL (min)
                <input type="number" value={manualFeatures.sol} onChange={(e) => handleInputChange('sol', e.target.value)} min="0" max="120" step="0.1" />
                <span className="osa-hint">Latence d'endormissement</span>
              </label>
              <label>Eff. Sommeil (%)
                <input type="number" value={manualFeatures.se} onChange={(e) => handleInputChange('se', e.target.value)} min="0" max="100" step="0.1" />
                <span className="osa-hint">TST / TIB × 100</span>
              </label>
              <label>WASO (min)
                <input type="number" value={manualFeatures.waso} onChange={(e) => handleInputChange('waso', e.target.value)} min="0" max="300" step="0.1" />
                <span className="osa-hint">Éveil après endormissement</span>
              </label>
              <label>SPT (min)
                <input type="number" value={manualFeatures.spt} onChange={(e) => handleInputChange('spt', e.target.value)} min="0" max="600" step="0.1" />
                <span className="osa-hint">Période de sommeil</span>
              </label>
            </div>

            {/* Stage Percentages */}
            <div className="osa-section-title" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', color: 'var(--red)', textTransform: 'uppercase', letterSpacing: '0.5px', margin: '24px 0 12px' }}>
              <Activity size={15} />
              Distribution des Stades (% du TST)
            </div>
            <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <label>N1 %
                <input type="number" value={manualFeatures.n1} onChange={(e) => handleInputChange('n1', e.target.value)} min="0" max="100" step="0.1" />
                <span className="osa-hint">Sommeil léger N1</span>
              </label>
              <label>N2 %
                <input type="number" value={manualFeatures.n2} onChange={(e) => handleInputChange('n2', e.target.value)} min="0" max="100" step="0.1" />
                <span className="osa-hint">Sommeil N2</span>
              </label>
              <label>N3 %
                <input type="number" value={manualFeatures.n3} onChange={(e) => handleInputChange('n3', e.target.value)} min="0" max="100" step="0.1" />
                <span className="osa-hint">Sommeil profond</span>
              </label>
              <label>REM %
                <input type="number" value={manualFeatures.rem} onChange={(e) => handleInputChange('rem', e.target.value)} min="0" max="100" step="0.1" />
                <span className="osa-hint">Sommeil paradoxal</span>
              </label>
              <label>N3 durée (min)
                <input type="number" value={manualFeatures.n3min} onChange={(e) => handleInputChange('n3min', e.target.value)} min="0" max="300" step="0.1" />
                <span className="osa-hint">timest34 (durée N3)</span>
              </label>
            </div>

            {/* Latencies */}
            <div className="osa-section-title" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', color: 'var(--red)', textTransform: 'uppercase', letterSpacing: '0.5px', margin: '24px 0 12px' }}>
              <Clock size={15} />
              Latences
            </div>
            <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <label>Latence REM (min)
                <input type="number" value={manualFeatures.remlat} onChange={(e) => handleInputChange('remlat', e.target.value)} min="0" max="300" step="0.1" />
                <span className="osa-hint">Temps jusqu'au 1er REM</span>
              </label>
              <label>Latence N3 (min)
                <input type="number" value={manualFeatures.n3lat} onChange={(e) => handleInputChange('n3lat', e.target.value)} min="-1" max="300" step="0.1" />
                <span className="osa-hint">-1 si non applicable</span>
              </label>
            </div>

            {/* Oximetry */}
            <div className="osa-section-title" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', color: 'var(--red)', textTransform: 'uppercase', letterSpacing: '0.5px', margin: '24px 0 12px' }}>
              <Zap size={15} />
              Oxymétrie Nocturne
            </div>
            <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <label>SpO₂ Moy. (%)
                <input type="number" value={manualFeatures.avgsat} onChange={(e) => handleInputChange('avgsat', e.target.value)} min="70" max="100" step="0.1" />
                <span className="osa-hint">Saturation moyenne</span>
              </label>
              <label>SpO₂ Min. (%)
                <input type="number" value={manualFeatures.minsat} onChange={(e) => handleInputChange('minsat', e.target.value)} min="50" max="100" step="0.1" />
                <span className="osa-hint">Nadir nocturne</span>
              </label>
              <label>% Temps &lt;90%
                <input type="number" value={manualFeatures.pctsa90} onChange={(e) => handleInputChange('pctsa90', e.target.value)} placeholder="auto" min="0" max="100" step="0.1" />
                <span className="osa-hint">Hypoxémie modérée</span>
              </label>
              <label>% Temps &lt;85%
                <input type="number" value={manualFeatures.pctsa85} onChange={(e) => handleInputChange('pctsa85', e.target.value)} placeholder="auto" min="0" max="100" step="0.1" />
                <span className="osa-hint">Hypoxémie sévère</span>
              </label>
              <label>% Temps &lt;95%
                <input type="number" value={manualFeatures.pctsa95} onChange={(e) => handleInputChange('pctsa95', e.target.value)} placeholder="auto" min="0" max="100" step="0.1" />
                <span className="osa-hint">Désaturation légère</span>
              </label>
            </div>

            {/* Arousal Indices */}
            <div className="osa-section-title" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', color: 'var(--red)', textTransform: 'uppercase', letterSpacing: '0.5px', margin: '24px 0 12px' }}>
              <AlertCircle size={15} />
              Indices d'Éveils (Arousal) <span style={{ fontSize: '11px', color: 'var(--text3)', textTransform: 'lowercase', marginLeft: '5px' }}>(optionnel)</span>
            </div>
            <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <label>Index Global (AI)
                <input type="number" value={manualFeatures.ai_all} onChange={(e) => handleInputChange('ai_all', e.target.value)} placeholder="auto" min="0" max="200" step="0.1" />
                <span className="osa-hint">Événements/h total</span>
              </label>
              <label>AI NREM
                <input type="number" value={manualFeatures.ai_nrem} onChange={(e) => handleInputChange('ai_nrem', e.target.value)} placeholder="auto" min="0" max="200" step="0.1" />
                <span className="osa-hint">Éveils en NREM</span>
              </label>
              <label>AI REM
                <input type="number" value={manualFeatures.ai_rem} onChange={(e) => handleInputChange('ai_rem', e.target.value)} placeholder="auto" min="0" max="200" step="0.1" />
                <span className="osa-hint">Éveils en REM</span>
              </label>
            </div>

            {/* Fragmentation */}
            <div className="osa-section-title" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', color: 'var(--red)', textTransform: 'uppercase', letterSpacing: '0.5px', margin: '24px 0 12px' }}>
              <CheckCircle size={15} />
              Fragmentation & Cycles <span style={{ fontSize: '11px', color: 'var(--text3)', textTransform: 'lowercase', marginLeft: '5px' }}>(optionnel)</span>
            </div>
            <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '30px' }}>
              <label>Index Fragmentation
                <input type="number" value={manualFeatures.frag} onChange={(e) => handleInputChange('frag', e.target.value)} placeholder="auto" min="0" max="200" step="0.1" />
                <span className="osa-hint">Transitions/h</span>
              </label>
              <label>Nb Éveils
                <input type="number" value={manualFeatures.wakebouts} onChange={(e) => handleInputChange('wakebouts', e.target.value)} placeholder="auto" min="0" max="500" step="1" />
                <span className="osa-hint">Wake bouts</span>
              </label>
              <label>Cycles REM
                <input type="number" value={manualFeatures.remcycles} onChange={(e) => handleInputChange('remcycles', e.target.value)} placeholder="auto" min="0" max="20" step="1" />
                <span className="osa-hint">Nb cycles REM</span>
              </label>
              <label>REM 1er Tiers (%)
                <input type="number" value={manualFeatures.remt1p} onChange={(e) => handleInputChange('remt1p', e.target.value)} placeholder="auto" min="0" max="100" step="0.1" />
                <span className="osa-hint">% REM 1er tiers</span>
              </label>
              <label>REM 2/3 Dernier (%)
                <input type="number" value={manualFeatures.remt34p} onChange={(e) => handleInputChange('remt34p', e.target.value)} placeholder="auto" min="0" max="100" step="0.1" />
                <span className="osa-hint">% REM dernier 2/3</span>
              </label>
            </div>

            <button 
              className={`btn-next ${loading ? 'running' : ''}`} 
              onClick={handlePredictManual} 
              disabled={loading}
              style={{ width: '100%', justifyContent: 'center', height: '48px', fontSize: '14px', fontWeight: 'bold' }}
            >
              {loading ? 'Évaluation en cours...' : '▷ Évaluer le Risque OSA'}
            </button>
          </div>
        </div>
      )}

      {/* ──── TAB 2: FILE UPLOAD PANEL ──── */}
      {activeTab === 'file' && (
        <div className="co-panel active">
          {!fileFeatures ? (
            <div className="drop-zone" onClick={() => document.getElementById('osa-file-input').click()} style={{ marginBottom: '30px', border: '2px dashed var(--border2)', borderRadius: '12px', padding: '40px 20px', cursor: 'pointer', textAlign: 'center', transition: 'var(--t)' }}>
              <div className="dz-body">
                <div className="dz-icon" style={{ marginBottom: '15px' }}><FileDown size={48} color="var(--blue)" style={{ opacity: 0.7 }} /></div>
                <div className="dz-title" style={{ fontSize: '15px', fontWeight: 'bold', marginBottom: '6px' }}>
                  {file ? file.name : "Déposez votre fichier CSV ou XML ici"}
                </div>
                <div className="dz-sub" style={{ fontSize: '13px', color: 'var(--text3)' }}>
                  Fichier de features extraites d'un rapport PSG — ou <b style={{ color: 'var(--red)' }}>parcourir</b>
                </div>
              </div>
              <input type="file" id="osa-file-input" accept=".csv,.xml" style={{ display: 'none' }} onChange={handleFileChange} />
            </div>
          ) : (
            <div style={{ marginTop: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div className="sec-lbl" style={{ margin: 0 }}>Features Extraites ({Object.keys(fileFeatures).length})</div>
                <button className="btn-reset" onClick={resetCustomFileUpload} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 14px', fontSize: '11px', background: 'none', border: '1px solid var(--border)', color: 'var(--text2)', borderRadius: '6px', cursor: 'pointer' }}>
                  <RotateCcw size={12} /> Réinitialiser
                </button>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px', marginBottom: '24px', maxHeight: '300px', overflowY: 'auto', padding: '12px', background: 'var(--bg2)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                {Object.entries(fileFeatures).map(([k, v]) => (
                  <div key={k} style={{ fontSize: '12px', display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: '4px' }}>
                    <span style={{ color: 'var(--text3)', fontFamily: 'var(--mono)' }}>{k}</span>
                    <span style={{ fontWeight: 'bold', color: 'var(--text)' }}>
                      {typeof v === 'number' ? v.toFixed(2) : String(v)}
                    </span>
                  </div>
                ))}
              </div>

              <button 
                className="btn-next" 
                onClick={handlePredictFile} 
                style={{ width: '100%', justifyContent: 'center', height: '48px', fontSize: '14px', fontWeight: 'bold' }}
              >
                <FileBarChart size={18} style={{ marginRight: '8px' }} /> Évaluer le Risque OSA
              </button>
            </div>
          )}
        </div>
      )}

      {loading && activeTab === 'file' && (
        <div style={{ textAlign: 'center', margin: '30px', color: 'var(--text3)' }}>Traitement en cours...</div>
      )}
      
      {error && (
        <div className="error-bar visible" style={{ marginTop: '20px', padding: '12px', background: 'rgba(231,76,60,0.1)', color: 'var(--red)', borderLeft: '4px solid var(--red)', borderRadius: '4px', fontSize: '13px' }}>
          ⚠ {error}
        </div>
      )}

      {/* ──── SHARED DIAGNOSTIC OSA CLINICAL REPORT ──── */}
      {osaResults && (
        <div className="osa-report" style={{ marginTop: '40px', display: 'block !important' }}>
          
          {/* Header Card */}
          <div className="osa-report-header" style={{ padding: '24px 28px' }}>
            <div className="osa-report-left">
              <div className="osa-report-title" style={{ fontSize: '20px', fontWeight: '800' }}>Rapport de Sévérité OSA — Mode Personnalisé</div>
              <div className="osa-model-badge" style={{ marginTop: '8px' }}>{osaResults.model_used || "Stacking Ensemble"}</div>
            </div>
            <div className="osa-severity-block">
              <div className="osa-severity-label">Sévérité Prédite</div>
              <div className={`osa-severity-badge sev-${osaResults.severity?.toLowerCase()}`}>
                {osaResults.severity}
              </div>
            </div>
          </div>

          {/* Probability Distribution */}
          <div className="osa-proba-section" style={{ padding: '24px 28px', borderBottom: '1px dashed var(--border2)' }}>
            <div className="osa-proba-title" style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold', color: 'var(--text3)', marginBottom: '16px' }}>
              Distribution de Confiance
            </div>
            <div className="osa-proba-bars">
              {Object.entries(osaResults.probabilities || {}).map(([className, prob]) => {
                const pct = (prob * 100).toFixed(1);
                return (
                  <div key={className} className="osa-proba-bar-row" style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '12px' }}>
                    <span className="osa-proba-label" style={{ width: '80px', fontSize: '12px', fontWeight: 'bold' }}>{className}</span>
                    <div className="osa-proba-bar-track" style={{ flex: 1, height: '18px', background: 'var(--bg2)', borderRadius: '9px', overflow: 'hidden', border: '1px solid var(--border)' }}>
                      <div 
                        className={`osa-proba-bar-fill fill-${className.toLowerCase()}`} 
                        style={{ width: `${pct}%`, height: '100%', borderRadius: '9px', transition: 'width 0.6s cubic-bezier(0.1, 0.8, 0.2, 1)' }} 
                      />
                    </div>
                    <span className="osa-proba-value" style={{ width: '50px', textAlign: 'right', fontSize: '12px', fontWeight: 'bold', fontFamily: 'var(--mono)' }}>{pct}%</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* SHAP Explainable Factors */}
          {osaResults.shap_explanations && osaResults.shap_explanations.length > 0 && (
            <div className="osa-shap-section" style={{ padding: '24px 28px', borderBottom: '1px dashed var(--border2)' }}>
              <div className="osa-shap-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '10px', marginBottom: '20px' }}>
                <div>
                  <div className="osa-shap-title" style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold', color: 'var(--text3)' }}>
                    Explications SHAP — Facteurs Déterminants
                  </div>
                  <div className="osa-shap-subtitle" style={{ fontSize: '11px', color: 'var(--text3)', marginTop: '4px' }}>
                    Impact de chaque variable sur la prédiction de sévérité
                  </div>
                </div>
                <div className="osa-shap-legend" style={{ display: 'flex', gap: '15px', fontSize: '11px' }}>
                  <span className="shap-leg-neg" style={{ color: 'var(--green)', fontWeight: 'bold' }}>← Réduit le risque</span>
                  <span className="shap-leg-pos" style={{ color: 'var(--red)', fontWeight: 'bold' }}>Aggrave le risque →</span>
                </div>
              </div>
              
              <div className="osa-shaps" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {osaResults.shap_explanations.map((item, idx) => {
                  const val = parseFloat(item.impact || item.value || 0);
                  const isPositive = val >= 0;
                  const absPct = Math.min(Math.abs(val) * 100, 100);
                  
                  return (
                    <div key={idx} className="osa-shap-row" style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                      <span className="osa-shap-feature-name" style={{ width: '180px', fontSize: '11px', color: 'var(--text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontFamily: 'var(--mono)' }}>
                        {item.feature || item.name} ({typeof item.value === 'number' ? item.value.toFixed(2) : String(item.value)})
                      </span>
                      <div className="osa-shap-bar-container" style={{ flex: 1, display: 'flex', height: '14px', background: 'var(--bg2)', borderRadius: '7px', overflow: 'hidden', border: '1px solid var(--border)' }}>
                        {isPositive ? (
                          <>
                            <div className="osa-shap-bar-half left" style={{ width: '50%' }} />
                            <div className="osa-shap-bar-half right" style={{ width: '50%', background: 'none' }}>
                              <div 
                                className="osa-shap-fill positive" 
                                style={{ width: `${absPct}%`, height: '100%', background: 'var(--red)', borderRadius: '0 7px 7px 0', transition: 'width 0.6s ease' }} 
                              />
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="osa-shap-bar-half left" style={{ width: '50%', display: 'flex', justifyContent: 'flex-end' }}>
                              <div 
                                className="osa-shap-fill negative" 
                                style={{ width: `${absPct}%`, height: '100%', background: 'var(--green)', borderRadius: '7px 0 0 7px', transition: 'width 0.6s ease' }} 
                              />
                            </div>
                            <div className="osa-shap-bar-half right" style={{ width: '50%' }} />
                          </>
                        )}
                      </div>
                      <span className={`osa-shap-impact-value ${isPositive ? 'positive' : 'negative'}`} style={{ width: '50px', textAlign: 'right', fontSize: '11px', fontWeight: 'bold', fontFamily: 'var(--mono)', color: isPositive ? 'var(--red)' : 'var(--green)' }}>
                        {isPositive ? '+' : ''}{val.toFixed(3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Clinical Takeaways */}
          <div className="osa-interpretation-section" style={{ padding: '24px 28px', background: 'var(--bg2)' }}>
            <div className="sec-lbl" style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold', color: 'var(--text3)', marginBottom: '14px' }}>
              Interprétation Clinique
            </div>
            <ul style={{ paddingLeft: '20px', marginTop: '10px', fontSize: '13px', color: 'var(--text2)', listStyleType: 'disc' }}>
              {getClinicalInterpretations(osaResults.used_features || manualFeatures).map((item, i) => (
                <li key={i} style={{ marginBottom: '8px', lineHeight: '1.5' }}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

    </div>
  );
};

export default CustomOSA;
