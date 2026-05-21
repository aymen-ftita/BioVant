import React, { useState, useEffect } from 'react';
import Hypnogram from './Hypnogram';
import axios from 'axios';
import './Results.css';
import '../OSAAnalysis/OSAAnalysis.css';

const SC = { Wake: '#c0392b', NREM: '#1d4ed8', REM: '#047857', N1: '#d97706', N2: '#1d4ed8', N3: '#6d28d9' };
const fmt = (n) => (n != null ? n : '–');

const AnalysisResults = ({ analysisData, activePsgId, patient }) => {
  const [osaResults, setOsaResults] = useState(null);
  const [isPredictingOsa, setIsPredictingOsa] = useState(false);
  const [osaError, setOsaError] = useState(null);

  // Extracted features state (auto-fetched from hypnogram)
  const [features, setFeatures] = useState(null);
  const [featuresLoading, setFeaturesLoading] = useState(false);
  const [featuresError, setFeaturesError] = useState(null);

  // Full OSA form — matches vanilla OSAAnalysis.html exactly
  const [formData, setFormData] = useState({
    age: patient?.age ? String(patient.age) : '50',
    gender: patient?.gender || 'M',
    bmi: patient?.imc ? String(patient.imc) : '28.0',
    avgsat: '94', minsat: '85',
    pctsa90h: '', pctsa85h: '', pctsa95h: '',
    ai_all: '', ai_nrem: '', ai_rem: ''
  });

  // Pre-fill form when target patient is selected or changes
  useEffect(() => {
    if (patient) {
      setFormData(prev => ({
        ...prev,
        age: patient.age ? String(patient.age) : prev.age,
        gender: patient.gender || prev.gender,
        bmi: patient.imc ? String(patient.imc) : prev.bmi,
      }));
    }
  }, [patient]);

  const primaryResult = analysisData?.results?.[0];
  // Convert to plain Array so JSON.stringify works correctly (avoid typed arrays)
  const stages_int = primaryResult?.stages_int ? Array.from(primaryResult.stages_int) : null;
  const class_names = primaryResult?.stats?.class_names || [];

  // Reset OSA & features when a new analysis arrives
  useEffect(() => {
    setOsaResults(null);
    setOsaError(null);
    setFeatures(null);
    setFeaturesError(null);
  }, [analysisData]);

  // Auto-fetch hypnogram features
  useEffect(() => {
    if (!stages_int || stages_int.length === 0 || !class_names || class_names.length === 0) return;
    const fetchFeatures = async () => {
      setFeaturesLoading(true);
      setFeaturesError(null);
      try {
        const res = await axios.post('http://localhost:8000/extract_features', {
          stages_int: stages_int,
          class_names: class_names,
        });
        setFeatures(res.data);
      } catch (err) {
        console.error('Feature extraction error:', err.response?.data || err.message);
        const detail = err.response?.data?.detail;
        setFeaturesError(typeof detail === 'string' ? detail : (detail ? JSON.stringify(detail) : err.message));
      } finally {
        setFeaturesLoading(false);
      }
    };
    fetchFeatures();
  }, [primaryResult]); // depend on the whole result object to avoid stale closure

  if (!analysisData || !analysisData.results || analysisData.results.length === 0) return null;

  const stages = primaryResult?.stages || [];
  const stats = primaryResult?.stats || {};
  const model_info = primaryResult?.model_info || {};
  const stage_minutes = stats?.stage_minutes || {};
  const stage_pct = stats?.stage_pct || {};

  const handlePredictOsa = async () => {
    if (!stages_int || stages_int.length === 0) {
      setOsaError('Données de stages manquantes. Lancez d\'abord une analyse.');
      return;
    }
    setIsPredictingOsa(true);
    setOsaError(null);
    setOsaResults(null);
    try {
      const payload = {
        stages_int: stages_int,
        class_names: class_names,
        clinical_data: {
          age: formData.age || null,
          gender: formData.gender,
          bmi: formData.bmi || null,
          avgsat: formData.avgsat || null,
          minsat: formData.minsat || null,
          pctsa90h: formData.pctsa90h || null,
          pctsa85h: formData.pctsa85h || null,
          pctsa95h: formData.pctsa95h || null,
          ai_all: formData.ai_all || null,
          ai_nrem: formData.ai_nrem || null,
          ai_rem: formData.ai_rem || null,
        }
      };
      console.log('[OSA] Sending payload — stages:', stages_int.length, 'classes:', class_names);
      const res = await axios.post('http://localhost:8000/predict_osa', payload);
      console.log('[OSA] Response:', res.data);
      setOsaResults(res.data);

      // Auto-update database PSG record with completed OSA results
      if (activePsgId) {
        try {
          const token = localStorage.getItem('token');
          await axios.put(`http://localhost:8000/psgs/${activePsgId}`, {
            severity: res.data.severity,
            report_data: JSON.stringify({
              staging: analysisData,
              osa: res.data
            })
          }, {
            headers: {
              Authorization: `Bearer ${token}`
            }
          });
          console.log('[OSA] Successfully updated database PSG record severity & report_data!');
        } catch (dbErr) {
          console.error('[OSA] Failed to update database PSG record:', dbErr);
        }
      }
    } catch (err) {
      console.error('[OSA] Error:', err.response?.data || err.message);
      const detail = err.response?.data?.detail;
      setOsaError(typeof detail === 'string' ? detail : (detail ? JSON.stringify(detail) : (err.message || 'Erreur serveur.')));
    } finally {
      setIsPredictingOsa(false);
    }
  };

  const classColors = { Normal: '#059669', Mild: '#d97706', Moderate: '#ea580c', Severe: '#dc2626' };

  return (
    <div id="results" className="results-container visible" style={{ marginTop: '40px' }}>
      <div style={{ fontFamily: 'var(--serif)', fontSize: '20px', fontWeight: '700', color: 'var(--red)', marginBottom: '16px', borderBottom: '1px solid var(--border)', paddingBottom: '8px' }}>
        Résultats de l'Analyse ({model_info.type})
      </div>

      {/* ── HYPNOGRAM ── */}
      <div className="sec-lbl">Hypnogramme</div>
      <Hypnogram stages={stages} classNames={stats.class_names} />

      {/* ── AASM METRICS ── */}
      <div className="sec-lbl" style={{ marginTop: '30px' }}>Métriques AASM</div>
      <div className="stats-grid">
        <StatCard lbl="Efficacité Sommeil" val={stats.se} unit="%" note="Normal ≥85%" cls={stats.se >= 85 ? 'good' : stats.se >= 75 ? 'warn' : 'danger'} />
        <StatCard lbl="Temps Total Sommeil" val={fmt(stats.tst)} unit="min" note={`${(stats.tst / 60).toFixed(1)}h`} />
        <StatCard lbl="Temps au Lit" val={fmt(stats.tib)} unit="min" note={`${(stats.tib / 60).toFixed(1)}h`} />
        <StatCard lbl="Latence Endormissement" val={stats.sol} unit="min" note="Normal 10–20 min" cls={stats.sol > 20 ? 'warn' : stats.sol < 5 ? 'danger' : 'good'} />
        <StatCard lbl="Latence REM" val={stats.rem_latency != null ? stats.rem_latency : 'N/A'} unit="min" note="Normal 90–120 min" cls={stats.rem_latency != null && stats.rem_latency < 60 ? 'danger' : ''} />
        <StatCard lbl="WASO" val={stats.waso} unit="min" note="Éveil Intra-Sommeil" cls={stats.waso > 30 ? 'warn' : ''} />
      </div>

      {/* ── STAGE BREAKDOWN ── */}
      <div className="sec-lbl" style={{ marginTop: '30px' }}>Répartition des Stades</div>
      <div className="breakdown-grid">
        {(stats.class_names || ["Wake", "NREM", "REM"]).map(st => {
          const color = SC[st] || SC['NREM'];
          const abbr = st === "Wake" ? "W" : st.replace("NREM", "NR").replace("REM", "R");
          return (
            <div key={st} className="stage-card">
              <div className="stage-sw" style={{ background: `${color}18`, color: color, border: `1.5px solid ${color}33` }}>{abbr}</div>
              <div>
                <div className="stage-name">{st}</div>
                <div className="stage-min" style={{ color: color }}>
                  {fmt(stage_minutes[st])}
                  <span style={{ fontSize: '11px', fontWeight: 400, color: 'var(--text3)', fontFamily: 'var(--mono)' }}> min</span>
                </div>
                <div className="stage-pct">{stage_pct[st] ?? '0'}% {st !== "Wake" ? "du TST" : "du TIB"}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* ── ALERTS ── */}
      {stats.alerts && stats.alerts.length > 0 && (
        <>
          <div className="sec-lbl" style={{ marginTop: '30px' }}>Alertes Cliniques</div>
          <div className="alerts-section">
            {stats.alerts.map((a, i) => (
              <div key={i} className="alert-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>
                {a}
              </div>
            ))}
          </div>
        </>
      )}

      {/* ── AUTO-EXTRACTED FEATURES FROM HYPNOGRAM ── */}
      <div className="extracted-features-wrapper" style={{ marginTop: '30px' }}>
        <div className="sec-lbl">
          Features Extraites de l'Hypnogramme
          {features && !featuresLoading && (
            <span className="extracted-features-badge" style={{ marginLeft: '10px', background: 'var(--red-s)', color: 'var(--red)', padding: '2px 10px', borderRadius: '20px', fontSize: '10px', fontWeight: 700 }}>
              {features.metadata?.n_features || '—'} features
            </span>
          )}
        </div>

        {featuresLoading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '16px', color: 'var(--text3)', fontSize: '12px' }}>
            <div className="sim-spinner-el" /> Extraction des features en cours…
          </div>
        )}

        {featuresError && (
          <div style={{ color: 'var(--red)', fontSize: '12px', padding: '10px' }}>⚠ {featuresError}</div>
        )}

        {features && !featuresLoading && (
          <div style={{ marginTop: '10px' }}>
            {/* Render each feature group */}
            {['timing', 'stages', 'latencies', 'fragmentation', 'rem_distribution'].map(groupKey => {
              const groupData = features[groupKey];
              if (!groupData || groupData.length === 0) return null;
              const titles = {
                timing: '⏱ Temporalité',
                stages: '📊 Distribution des Stades',
                latencies: '➡ Latences',
                fragmentation: '🔔 Fragmentation & Cycles',
                rem_distribution: '📈 Distribution REM'
              };
              return (
                <div key={groupKey} className="feat-group" style={{ marginBottom: '16px' }}>
                  <div className="feat-group-title" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text2)', letterSpacing: '1px', textTransform: 'uppercase', marginBottom: '8px' }}>
                    {titles[groupKey]}
                  </div>
                  <div className="feat-group-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '8px' }}>
                    {groupData.map((feat, i) => (
                      <div key={i} className="feat-card" style={{
                        padding: '10px 12px',
                        background: 'var(--surface)',
                        border: '1px solid var(--border)',
                        borderRadius: '8px',
                        animation: `fadeUp .3s ease ${i * 40}ms both`
                      }}>
                        <div style={{ fontSize: '9px', color: 'var(--text3)', letterSpacing: '1px', textTransform: 'uppercase', marginBottom: '4px' }}>{feat.name}</div>
                        <div style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'var(--mono)', color: 'var(--text)' }}>
                          {feat.value === -1 ? 'N/A' : feat.value}
                          <span style={{ fontSize: '10px', color: 'var(--text3)', fontWeight: 400, marginLeft: '3px' }}>{feat.unit}</span>
                        </div>
                        {feat.note && <div style={{ fontSize: '9px', color: 'var(--text3)', marginTop: '3px' }}>{feat.note}</div>}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}

            {/* Metadata */}
            {features.metadata && (
              <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', fontSize: '10px', color: 'var(--text3)', marginTop: '8px', padding: '8px 0', borderTop: '1px solid var(--border)' }}>
                <span>● {features.metadata.n_epochs} époques</span>
                <span>● {features.metadata.is_3class ? '3 classes' : '5 classes'} ({features.metadata.class_names?.join(' / ')})</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── OSA PREDICTION PANEL — Full form matching vanilla OSAAnalysis.html ── */}
      <div className="osa-panel" style={{ marginTop: '40px', padding: '20px', background: 'var(--surface)', borderRadius: '12px', border: '1px solid var(--border)' }}>
        <div className="sec-lbl" style={{ marginBottom: '15px' }}>Prédiction SAOS (Apnée du Sommeil)</div>
        <p style={{ fontSize: '12px', color: 'var(--text2)', marginBottom: '20px' }}>
          Renseignez les données cliniques du patient pour affiner l'évaluation de la sévérité de l'apnée obstructive du sommeil.
        </p>

        {/* Demographics */}
        <div className="osa-section-title" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text2)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          Données Démographiques
        </div>
        <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', marginBottom: '20px' }}>
          <label className="osa-label">Âge (ans)<input type="number" value={formData.age} min="18" max="100" onChange={e => setFormData({...formData, age: e.target.value})} /></label>
          <label className="osa-label">Sexe
            <select value={formData.gender} onChange={e => setFormData({...formData, gender: e.target.value})}>
              <option value="M">Homme</option>
              <option value="F">Femme</option>
            </select>
          </label>
          <label className="osa-label">IMC (kg/m²)<input type="number" step="0.1" value={formData.bmi} min="15" max="60" onChange={e => setFormData({...formData, bmi: e.target.value})} /></label>
        </div>

        {/* Oximetry */}
        <div className="osa-section-title" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text2)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
          Oxymétrie Nocturne
        </div>
        <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', marginBottom: '20px' }}>
          <label className="osa-label">SpO₂ Moy. (%)<input type="number" step="0.1" value={formData.avgsat} min="70" max="100" onChange={e => setFormData({...formData, avgsat: e.target.value})} /><span className="osa-hint">Saturation moyenne</span></label>
          <label className="osa-label">SpO₂ Min. (%)<input type="number" step="0.1" value={formData.minsat} min="50" max="100" onChange={e => setFormData({...formData, minsat: e.target.value})} /><span className="osa-hint">Nadir nocturne</span></label>
          <label className="osa-label">% Temps &lt;90%<input type="number" step="0.1" value={formData.pctsa90h} placeholder="auto" min="0" max="100" onChange={e => setFormData({...formData, pctsa90h: e.target.value})} /><span className="osa-hint">Hypoxémie modérée</span></label>
          <label className="osa-label">% Temps &lt;85%<input type="number" step="0.1" value={formData.pctsa85h} placeholder="auto" min="0" max="100" onChange={e => setFormData({...formData, pctsa85h: e.target.value})} /><span className="osa-hint">Hypoxémie sévère</span></label>
          <label className="osa-label">% Temps &lt;95%<input type="number" step="0.1" value={formData.pctsa95h} placeholder="auto" min="0" max="100" onChange={e => setFormData({...formData, pctsa95h: e.target.value})} /><span className="osa-hint">Désaturation légère</span></label>
        </div>

        {/* Arousal Indices */}
        <div className="osa-section-title" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text2)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 01-3.46 0"/></svg>
          Indices d'Éveils (Arousal)
          <span style={{ fontSize: '9px', color: 'var(--text3)', fontWeight: 400 }}>(optionnel — améliore la précision)</span>
        </div>
        <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', marginBottom: '20px' }}>
          <label className="osa-label">Index Global (AI)<input type="number" step="0.1" value={formData.ai_all} placeholder="auto" min="0" max="200" onChange={e => setFormData({...formData, ai_all: e.target.value})} /><span className="osa-hint">Événements/h total</span></label>
          <label className="osa-label">AI NREM<input type="number" step="0.1" value={formData.ai_nrem} placeholder="auto" min="0" max="200" onChange={e => setFormData({...formData, ai_nrem: e.target.value})} /><span className="osa-hint">Éveils en NREM</span></label>
          <label className="osa-label">AI REM<input type="number" step="0.1" value={formData.ai_rem} placeholder="auto" min="0" max="200" onChange={e => setFormData({...formData, ai_rem: e.target.value})} /><span className="osa-hint">Éveils en REM</span></label>
        </div>

        {osaError && <div className="error-bar visible" style={{ marginBottom: '15px' }}>⚠ {osaError}</div>}

        <button className="btn-analyse" onClick={handlePredictOsa} disabled={isPredictingOsa} style={{ width: '100%' }}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /><path d="M12 8v4l3 3" /></svg>
          {isPredictingOsa ? 'Analyse en cours...' : 'Générer le Rapport Clinique SAOS'}
        </button>
      </div>

      {/* ── FULL OSA REPORT — outside osa-panel for correct layout ── */}
      {osaResults && (
          <div className="osa-report" style={{ marginTop: '30px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
            {/* Header: Severity + Model */}
            <div className="osa-report-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <div style={{ fontFamily: 'var(--serif)', fontSize: '18px', fontWeight: 700 }}>Rapport de Sévérité OSA</div>
                <div style={{ fontSize: '10px', color: 'var(--text3)', marginTop: '4px' }}>Modèle: {osaResults.model_used || 'Stacking Ensemble'}</div>
              </div>
              <div className={`osa-severity-badge sev-${getSevClass(osaResults.severity)}`}>
                {osaResults.severity}
              </div>
            </div>

            {/* Probability Distribution */}
            {osaResults.probabilities && (
              <div style={{ marginBottom: '24px' }}>
                <div style={{ fontSize: '10px', letterSpacing: '2px', textTransform: 'uppercase', color: 'var(--text3)', marginBottom: '12px' }}>Distribution de Confiance</div>
                {['Normal', 'Mild', 'Moderate', 'Severe'].map(cls => {
                  const pct = (osaResults.probabilities[cls] || 0) * 100;
                  return (
                    <div key={cls} className="osa-proba-row" style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                      <div style={{ width: '70px', fontSize: '11px', fontWeight: 600, color: classColors[cls] }}>{cls}</div>
                      <div style={{ flex: 1, height: '20px', background: 'var(--bg2)', borderRadius: '4px', overflow: 'hidden', position: 'relative' }}>
                        <div style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: classColors[cls],
                          borderRadius: '4px',
                          transition: 'width 0.6s ease',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'flex-end',
                          paddingRight: '6px',
                          fontSize: '9px',
                          color: '#fff',
                          fontWeight: 700,
                          minWidth: pct > 5 ? 'auto' : '0'
                        }}>
                          {pct > 5 ? `${pct.toFixed(1)}%` : ''}
                        </div>
                      </div>
                      <div style={{ width: '50px', textAlign: 'right', fontSize: '11px', fontFamily: 'var(--mono)', fontWeight: 700 }}>{pct.toFixed(1)}%</div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* SHAP Explanations */}
            {osaResults.shap_explanations && osaResults.shap_explanations.length > 0 && (
              <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap' }}>
                  <div>
                    <div style={{ fontSize: '10px', letterSpacing: '2px', textTransform: 'uppercase', color: 'var(--text3)' }}>Explications SHAP — Facteurs Déterminants</div>
                    <div style={{ fontSize: '9px', color: 'var(--text3)', marginTop: '2px' }}>Impact de chaque variable sur la prédiction de sévérité</div>
                  </div>
                  <div style={{ display: 'flex', gap: '12px', fontSize: '9px' }}>
                    <span style={{ color: '#059669' }}>← Réduit le risque</span>
                    <span style={{ color: '#dc2626' }}>Aggrave le risque →</span>
                  </div>
                </div>
                {(() => {
                  const maxImp = Math.max(...osaResults.shap_explanations.map(x => Math.abs(x.impact)), 0.1);
                  return osaResults.shap_explanations.map((sh, idx) => {
                    const isPos = sh.impact > 0;
                    const pct = (Math.abs(sh.impact) / maxImp * 45).toFixed(1);
                    return (
                      <div key={idx} className="shap-item" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', fontSize: '11px' }}>
                        <div style={{ width: '140px', fontSize: '10px', color: 'var(--text2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {sh.feature} <span style={{ color: 'var(--text3)' }}>({sh.value})</span>
                        </div>
                        <div style={{ flex: 1, height: '14px', background: 'var(--bg2)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${pct}%`,
                            height: '100%',
                            background: isPos ? '#dc2626' : '#059669',
                            borderRadius: '3px',
                            transition: `width 0.4s ease ${idx * 30}ms`,
                            float: isPos ? 'left' : 'right',
                          }} />
                        </div>
                        <div style={{ width: '55px', textAlign: 'right', fontSize: '10px', fontFamily: 'var(--mono)', color: isPos ? '#dc2626' : '#059669' }}>{sh.impact.toFixed(3)}</div>
                      </div>
                    );
                  });
                })()}
              </div>
            )}

            {/* Interpretation */}
            {osaResults.interpretation && osaResults.interpretation.length > 0 && (
              <div style={{ borderTop: '1px solid var(--border)', paddingTop: '16px' }}>
                <div style={{ fontSize: '10px', letterSpacing: '2px', textTransform: 'uppercase', color: 'var(--text3)', marginBottom: '10px' }}>Interprétation Clinique</div>
                {osaResults.interpretation.map((item, i) => (
                  <div key={i} style={{ fontSize: '12px', color: 'var(--text2)', marginBottom: '8px', paddingLeft: '12px', borderLeft: '2px solid var(--red)' }}>
                    {item.text || item}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
    </div>
  );
};

function getSevClass(severity) {
  if (!severity) return 'normal';
  const s = severity.toLowerCase();
  if (s.includes('severe')) return 'severe';
  if (s.includes('moderate')) return 'moderate';
  if (s.includes('mild')) return 'mild';
  return 'normal';
}

const StatCard = ({ lbl, val, unit, note, cls = '' }) => (
  <div className={`stat-card ${cls}`}>
    <div className="stat-lbl">{lbl}</div>
    <div className="stat-val">{val}<span className="stat-unit">{unit}</span></div>
    <div className="stat-note">{note}</div>
  </div>
);

export default AnalysisResults;
