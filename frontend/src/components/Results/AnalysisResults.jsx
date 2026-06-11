import React, { useState, useEffect, useRef } from 'react';
import Hypnogram from './Hypnogram';
import axios from 'axios';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import { useTranslation } from 'react-i18next';
import './Results.css';
import '../OSAAnalysis/OSAAnalysis.css';

const SC = { Wake: '#c0392b', NREM: '#1d4ed8', REM: '#047857', N1: '#d97706', N2: '#1d4ed8', N3: '#6d28d9' };
const fmt = (n) => (n != null ? n : '–'); 
const AnalysisResults = ({ analysisData, activePsgId, patient, onStartHypnogramUpload, onStartHypnogramAnnotatedUpload, onStartOsaReportUpload }) => {
  const { t } = useTranslation();
  const [osaResults, setOsaResults] = useState(null);
  const [isPredictingOsa, setIsPredictingOsa] = useState(false);
  const [osaError, setOsaError] = useState(null);
  const [hypnogramBlob, setHypnogramBlob] = useState(null);
  const reportRef = useRef(null);

  // Extracted features state (auto-fetched from hypnogram)
  const [features, setFeatures] = useState(null);
  const [featuresLoading, setFeaturesLoading] = useState(false);
  const [featuresError, setFeaturesError] = useState(null);

  const generatePDF = async () => {
    if (!reportRef.current) return;
    try {
      const canvas = await html2canvas(reportRef.current, { scale: 2 });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      
      pdf.text(`Rapport Médical - ${patient?.first_name || 'Patient'} ${patient?.last_name || ''}`, 10, 10);
      pdf.addImage(imgData, 'PNG', 0, 20, pdfWidth, pdfHeight);
      pdf.save(`Rapport_PSG_${patient?.first_name || 'Patient'}.pdf`);
    } catch (err) {
      console.error('Erreur lors de la génération du PDF:', err);
      alert('Erreur lors de la génération du PDF.');
    }
  };

  const [reportConfig, setReportConfig] = useState({
    hypnogram: true,
    aasm: true,
    osa: true,
    shap: true,
    spo2: true,
    normative: true
  });

  const getNormativeComparison = (metricKey, val) => {
    if (!reportConfig.normative) return null;
    const age = patient?.age ? Number(patient.age) : 50;
    const bmi = patient?.imc ? Number(patient.imc) : 28.0;

    let norms = {
      se: { min: 80, max: 90, unit: '%' },
      sol: { min: 10, max: 25, unit: ' min' },
      rem: { min: 18, max: 24, unit: '%' },
      deep: { min: 10, max: 18, unit: '%' }
    };

    if (age > 65) {
      norms.se = { min: 72, max: 85, unit: '%' };
      norms.deep = { min: 4, max: 12, unit: '%' };
      norms.sol = { min: 15, max: 30, unit: ' min' };
    } else if (age < 35) {
      norms.se = { min: 86, max: 94, unit: '%' };
      norms.deep = { min: 15, max: 23, unit: '%' };
    }

    if (bmi > 30) {
      norms.se.min -= 5;
      norms.se.max -= 3;
    }

    const norm = norms[metricKey];
    if (!norm) return null;

    const isLow = val < norm.min;
    const isHigh = val > norm.max;
    const color = isLow || isHigh ? '#d97706' : '#059669';
    const arrow = isLow ? '↓' : isHigh ? '↑' : '✓';
    const status = isLow ? 'Inférieur' : isHigh ? 'Supérieur' : 'Normal';

    return (
      <div style={{ fontSize: '9px', color: color, marginTop: '3px', fontWeight: 600 }}>
        SHHS matched: {norm.min}-{norm.max}{norm.unit} {arrow} ({status})
      </div>
    );
  };

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
    setHypnogramBlob(null);
  }, [analysisData]);

  // Upload Hypnogram as soon as both activePsgId and hypnogramBlob are ready
  useEffect(() => {
    if (hypnogramBlob && activePsgId && onStartHypnogramUpload) {
      onStartHypnogramUpload(activePsgId, hypnogramBlob);
      setHypnogramBlob(null); // Upload once
    }
  }, [hypnogramBlob, activePsgId, onStartHypnogramUpload]);

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

  const exportFeatures = (format) => {
    // ─── Use the EXACT same export format as vanilla-version buildOsaCSV / buildOsaXML ───
    // This ensures CSV/XML re-imported in CustomOSA produces identical predictions.
    
    // If osaResults are available (after OSA prediction), use those — they contain
    // the actual used_features dict that was sent to the model with correct SHHS column names.
    // If no osaResults yet, build a features dict from extract_features + form inputs.
    const data = osaResults;
    
    const timestamp = () => new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const downloadFile = (filename, content, mimeType) => {
      const blob = new Blob([content], { type: mimeType });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(a.href); }, 100);
    };
    
    if (data) {
      // ── EXPORT FROM OSA RESULTS (matches vanilla buildOsaCSV / buildOsaXML exactly) ──
      if (format === 'csv') {
        const rows = [];
        rows.push(['Section', 'Key', 'Value']);
        rows.push(['Prediction', 'Severity', data.severity]);
        rows.push(['Prediction', 'Model', data.model_used || 'XGBoost']);
        for (const [cls, prob] of Object.entries(data.probabilities || {})) {
          rows.push(['Probability', cls, (prob * 100).toFixed(1) + '%']);
        }
        if (data.aasm_features) {
          for (const [group, feats] of Object.entries(data.aasm_features)) {
            for (const [k, v] of Object.entries(feats)) {
              rows.push(['AASM_' + group, k, v]);
            }
          }
        }
        if (data.used_features) {
          for (const [k, v] of Object.entries(data.used_features)) {
            rows.push(['ModelFeature', k, v]);
          }
        }
        if (data.shap_explanations) {
          data.shap_explanations.forEach(sh => {
            rows.push(['SHAP', sh.feature, sh.impact.toFixed(4)]);
          });
        }
        const csvContent = rows.map(r => r.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
        downloadFile(`osa_report_${timestamp()}.csv`, csvContent, 'text/csv');
      } else if (format === 'xml') {
        let xml = '<?xml version="1.0" encoding="UTF-8"?>\n<OSAReport>\n';
        xml += `  <GeneratedAt>${new Date().toISOString()}</GeneratedAt>\n`;
        xml += `  <Title>Rapport OSA</Title>\n`;
        xml += `  <Model>${data.model_used || 'XGBoost'}</Model>\n`;
        xml += `  <Severity>${data.severity}</Severity>\n`;
        xml += '  <Probabilities>\n';
        for (const [cls, prob] of Object.entries(data.probabilities || {})) {
          xml += `    <Class name="${cls}" probability="${(prob * 100).toFixed(1)}"/>\n`;
        }
        xml += '  </Probabilities>\n';
        if (data.aasm_features) {
          xml += '  <AASMFeatures>\n';
          for (const [group, feats] of Object.entries(data.aasm_features)) {
            xml += `    <Group name="${group}">\n`;
            for (const [k, v] of Object.entries(feats)) {
              xml += `      <Feature name="${k}" value="${v}"/>\n`;
            }
            xml += '    </Group>\n';
          }
          xml += '  </AASMFeatures>\n';
        }
        if (data.used_features) {
          xml += '  <ModelFeatures>\n';
          for (const [k, v] of Object.entries(data.used_features)) {
            xml += `    <Feature name="${k}" value="${v}"/>\n`;
          }
          xml += '  </ModelFeatures>\n';
        }
        if (data.shap_explanations) {
          xml += '  <SHAPExplanations>\n';
          data.shap_explanations.forEach(sh => {
            xml += `    <Factor feature="${sh.feature}" value="${sh.value}" impact="${sh.impact.toFixed(4)}"/>\n`;
          });
          xml += '  </SHAPExplanations>\n';
        }
        if (data.interpretation) {
          xml += '  <Interpretation>\n';
          data.interpretation.forEach(item => {
            xml += `    <Finding type="${item.type}">${item.text}</Finding>\n`;
          });
          xml += '  </Interpretation>\n';
        }
        xml += '</OSAReport>\n';
        downloadFile(`osa_report_${timestamp()}.xml`, xml, 'application/xml');
      }
    } else if (features) {
      // ── EXPORT EXTRACTED FEATURES ONLY (before OSA prediction) ──
      // Build used_features dict matching model column names from extract_features response
      const flatData = {};
      
      // Grab all extracted features from backend
      const allGroups = ['timing', 'stages', 'latencies', 'fragmentation', 'rem_distribution'];
      allGroups.forEach(groupKey => {
        const groupList = features[groupKey] || [];
        groupList.forEach(item => {
          if (item.key) {
            flatData[item.key] = item.value === -1 ? 0 : item.value;
          }
        });
      });
      
      // Add clinical data from form
      flatData['age_s2'] = Number(formData.age) || patient?.age || 50;
      flatData['gender'] = formData.gender || patient?.gender || 'M';
      flatData['bmi_s2'] = Number(formData.bmi) || patient?.imc || 28.0;
      flatData['avgsat'] = Number(formData.avgsat) || 94.0;
      flatData['minsat'] = Number(formData.minsat) || 85.0;
      flatData['pctsa90h'] = Number(formData.pctsa90h) || 0.0;
      flatData['pctsa85h'] = Number(formData.pctsa85h) || 0.0;
      flatData['pctsa95h'] = Number(formData.pctsa95h) || 0.0;
      flatData['ai_all'] = Number(formData.ai_all) || 0.0;
      flatData['ai_nrem'] = Number(formData.ai_nrem) || 0.0;
      flatData['ai_rem'] = Number(formData.ai_rem) || 0.0;
      
      // Add SHHS PSG column aliases so predict_osa_custom doesn't fall back to medians
      if (flatData['sleep_efficiency'] !== undefined) flatData['slpeffp'] = flatData['sleep_efficiency'];
      if (flatData['sol_min'] !== undefined) flatData['slplatp'] = flatData['sol_min'];
      if (flatData['N1_pct'] !== undefined) flatData['timest1p'] = flatData['N1_pct'];
      if (flatData['N2_pct'] !== undefined) flatData['timest2p'] = flatData['N2_pct'];
      if (flatData['REM_pct'] !== undefined) flatData['timeremp'] = flatData['REM_pct'];
      if (flatData['waso_min'] !== undefined) flatData['waso'] = flatData['waso_min'];
      const n3_min = stats.stage_minutes && stats.stage_minutes['N3'] ? stats.stage_minutes['N3'] : 0;
      if (flatData['timest34'] === undefined) flatData['timest34'] = n3_min;
      
      if (format === 'csv') {
        // Use Section,Key,Value format (same as vanilla buildOsaCSV for ModelFeature rows)
        const rows = [];
        rows.push(['Section', 'Key', 'Value']);
        for (const [k, v] of Object.entries(flatData)) {
          rows.push(['ModelFeature', k, v]);
        }
        const csvContent = rows.map(r => r.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
        downloadFile(`features_psg_${activePsgId || 'export'}.csv`, csvContent, 'text/csv');
      } else if (format === 'xml') {
        const escapeXml = (unsafe) => {
          if (unsafe == null) return '';
          return String(unsafe).replace(/[<>&'"]/g, function (c) {
            switch (c) {
              case '<': return '&lt;';
              case '>': return '&gt;';
              case '&': return '&amp;';
              case '\'': return '&apos;';
              case '"': return '&quot;';
              default: return c;
            }
          });
        };
        let xml = '<?xml version="1.0" encoding="UTF-8"?>\n<OSAReport>\n';
        xml += `  <GeneratedAt>${new Date().toISOString()}</GeneratedAt>\n`;
        xml += '  <ModelFeatures>\n';
        for (const [k, v] of Object.entries(flatData)) {
          xml += `    <Feature name="${escapeXml(k)}" value="${escapeXml(v)}"/>\n`;
        }
        xml += '  </ModelFeatures>\n';
        xml += '</OSAReport>\n';
        downloadFile(`features_psg_${activePsgId || 'export'}.xml`, xml, 'application/xml');
      }
    }
  };

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

      // Trigger OSA report upload in the background
      if (onStartOsaReportUpload && activePsgId) {
        try {
          const htmlReport = generateOsaHtmlReport(patient, analysisData, res.data);
          const htmlBlob = new Blob([htmlReport], { type: 'text/html' });
          onStartOsaReportUpload(activePsgId, htmlBlob);
        } catch (repErr) {
          console.error('[OSA] HTML generation or background upload failed:', repErr);
        }
      }

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
    <div id="results" className="results-container visible" style={{ marginTop: '40px' }} ref={reportRef}>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '15px' }}>
        <button 
          className="btn-primary" 
          onClick={generatePDF}
          style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', fontSize: '13px' }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
          {t('common.download_pdf')}
        </button>
      </div>

      {/* ── REPORT CUSTOMIZATION PANEL ── */}
      <div className="report-config-panel" style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: '12px',
        padding: '16px',
        marginBottom: '25px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--red)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          ⚙️ Personnaliser le Rapport de Diagnostic
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '15px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontWeight: 600, cursor: 'pointer' }}>
            <input type="checkbox" checked={reportConfig.hypnogram} onChange={e => setReportConfig({ ...reportConfig, hypnogram: e.target.checked })} />
            Hypnogramme
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontWeight: 600, cursor: 'pointer' }}>
            <input type="checkbox" checked={reportConfig.aasm} onChange={e => setReportConfig({ ...reportConfig, aasm: e.target.checked })} />
            Métriques AASM
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontWeight: 600, cursor: 'pointer' }}>
            <input type="checkbox" checked={reportConfig.osa} onChange={e => setReportConfig({ ...reportConfig, osa: e.target.checked })} />
            Sévérité SAOS
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontWeight: 600, cursor: 'pointer' }}>
            <input type="checkbox" checked={reportConfig.shap} onChange={e => setReportConfig({ ...reportConfig, shap: e.target.checked })} />
            Explications SHAP
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontWeight: 600, cursor: 'pointer' }}>
            <input type="checkbox" checked={reportConfig.spo2} onChange={e => setReportConfig({ ...reportConfig, spo2: e.target.checked })} />
            Signaux bruts & Apnées (SpO₂)
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontWeight: 600, cursor: 'pointer' }}>
            <input type="checkbox" checked={reportConfig.normative} onChange={e => setReportConfig({ ...reportConfig, normative: e.target.checked })} />
            Comparaison Normative (SHHS)
          </label>
        </div>
      </div>

      <div style={{ fontFamily: 'var(--serif)', fontSize: '20px', fontWeight: '700', color: 'var(--red)', marginBottom: '16px', borderBottom: '1px solid var(--border)', paddingBottom: '8px' }}>
        Résultats de l'Analyse ({model_info.type})
      </div>

      {/* ── HYPNOGRAM ── */}
      {reportConfig.hypnogram && (
        <>
          <div className="sec-lbl">Architecture du Sommeil et Signaux</div>
          <Hypnogram
            stages={stages}
            classNames={stats.class_names}
            spo2={reportConfig.spo2 ? analysisData.spo2 : null}
            apneaTimeline={reportConfig.spo2 ? analysisData.apnea_timeline : null}
            activePsgId={activePsgId}
            onExport={setHypnogramBlob}
            onExportAnnotated={onStartHypnogramAnnotatedUpload}
          />
        </>
      )}

      {/* ── AASM METRICS ── */}
      {reportConfig.aasm && (
        <>
          <div className="sec-lbl" style={{ marginTop: '30px' }}>Métriques AASM</div>
          <div className="stats-grid">
            <StatCard lbl="Efficacité Sommeil" val={stats.se} unit="%" note="Normal ≥85%" cls={stats.se >= 85 ? 'good' : stats.se >= 75 ? 'warn' : 'danger'} customNode={getNormativeComparison('se', stats.se)} />
            <StatCard lbl="Temps Total Sommeil" val={fmt(stats.tst)} unit="min" note={`${(stats.tst / 60).toFixed(1)}h`} />
            <StatCard lbl="Temps au Lit" val={fmt(stats.tib)} unit="min" note={`${(stats.tib / 60).toFixed(1)}h`} />
            <StatCard lbl="Latence Endormissement" val={stats.sol} unit="min" note="Normal 10–20 min" cls={stats.sol > 20 ? 'warn' : stats.sol < 5 ? 'danger' : 'good'} customNode={getNormativeComparison('sol', stats.sol)} />
            <StatCard lbl="Latence REM" val={stats.rem_latency != null ? stats.rem_latency : 'N/A'} unit="min" note="Normal 90–120 min" cls={stats.rem_latency != null && stats.rem_latency < 60 ? 'danger' : ''} customNode={stats.rem_latency != null ? getNormativeComparison('deep', stats.rem_latency) : null} />
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
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" /></svg>
                    {a}
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      {/* ── AUTO-EXTRACTED FEATURES FROM HYPNOGRAM ── */}
      {reportConfig.aasm && (
        <div className="extracted-features-wrapper" style={{ marginTop: '30px' }}>
          <div className="sec-lbl" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
            <span>Features Extraites de l'Hypnogramme</span>
            {features && !featuresLoading && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="extracted-features-badge" style={{ background: 'var(--red-s)', color: 'var(--red)', padding: '2px 10px', borderRadius: '20px', fontSize: '10px', fontWeight: 700 }}>
                  {features.metadata?.n_features || '—'} features
                </span>
                <button 
                  onClick={() => exportFeatures('csv')} 
                  style={{ background: 'none', border: '1px solid var(--border)', color: 'var(--text2)', borderRadius: '6px', fontSize: '10px', padding: '3px 10px', cursor: 'pointer', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '4px' }}
                >
                  📥 Exporter CSV
                </button>
                <button 
                  onClick={() => exportFeatures('xml')} 
                  style={{ background: 'none', border: '1px solid var(--border)', color: 'var(--text2)', borderRadius: '6px', fontSize: '10px', padding: '3px 10px', cursor: 'pointer', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '4px' }}
                >
                  📥 Exporter XML
                </button>
              </div>
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
      )}

      {/* ── OSA PREDICTION PANEL — Full form matching vanilla OSAAnalysis.html ── */}
      {reportConfig.osa && (
        <div className="osa-panel" style={{ marginTop: '40px', padding: '20px', background: 'var(--surface)', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <div className="sec-lbl" style={{ marginBottom: '15px' }}>Prédiction SAOS (Apnée du Sommeil)</div>
          <p style={{ fontSize: '12px', color: 'var(--text2)', marginBottom: '20px' }}>
            Renseignez les données cliniques du patient pour affiner l'évaluation de la sévérité de l'apnée obstructive du sommeil.
          </p>

          {/* Demographics */}
          <div className="osa-section-title" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text2)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" /><circle cx="12" cy="7" r="4" /></svg>
            Données Démographiques
          </div>
          <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', marginBottom: '20px' }}>
            <label className="osa-label">Âge (ans)<input type="number" value={formData.age} min="18" max="100" onChange={e => setFormData({ ...formData, age: e.target.value })} /></label>
            <label className="osa-label">Sexe
              <select value={formData.gender} onChange={e => setFormData({ ...formData, gender: e.target.value })}>
                <option value="M">Homme</option>
                <option value="F">Femme</option>
              </select>
            </label>
            <label className="osa-label">IMC (kg/m²)<input type="number" step="0.1" value={formData.bmi} min="15" max="60" onChange={e => setFormData({ ...formData, bmi: e.target.value })} /></label>
          </div>

          {/* Oximetry */}
          <div className="osa-section-title" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text2)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
            Oxymétrie Nocturne
          </div>
          <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', marginBottom: '20px' }}>
            <label className="osa-label">SpO₂ Moy. (%)<input type="number" step="0.1" value={formData.avgsat} min="70" max="100" onChange={e => setFormData({ ...formData, avgsat: e.target.value })} /><span className="osa-hint">Saturation moyenne</span></label>
            <label className="osa-label">SpO₂ Min. (%)<input type="number" step="0.1" value={formData.minsat} min="50" max="100" onChange={e => setFormData({ ...formData, minsat: e.target.value })} /><span className="osa-hint">Nadir nocturne</span></label>
            <label className="osa-label">% Temps &lt;90%<input type="number" step="0.1" value={formData.pctsa90h} placeholder="auto" min="0" max="100" onChange={e => setFormData({ ...formData, pctsa90h: e.target.value })} /><span className="osa-hint">Hypoxémie modérée</span></label>
            <label className="osa-label">% Temps &lt;85%<input type="number" step="0.1" value={formData.pctsa85h} placeholder="auto" min="0" max="100" onChange={e => setFormData({ ...formData, pctsa85h: e.target.value })} /><span className="osa-hint">Hypoxémie sévère</span></label>
            <label className="osa-label">% Temps &lt;95%<input type="number" step="0.1" value={formData.pctsa95h} placeholder="auto" min="0" max="100" onChange={e => setFormData({ ...formData, pctsa95h: e.target.value })} /><span className="osa-hint">Désaturation légère</span></label>
          </div>

          {/* Arousal Indices */}
          <div className="osa-section-title" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text2)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9" /><path d="M13.73 21a2 2 0 01-3.46 0" /></svg>
            Indices d'Éveils (Arousal)
            <span style={{ fontSize: '9px', color: 'var(--text3)', fontWeight: 400 }}>(optionnel — améliore la précision)</span>
          </div>
          <div className="osa-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', marginBottom: '20px' }}>
            <label className="osa-label">Index Global (AI)<input type="number" step="0.1" value={formData.ai_all} placeholder="auto" min="0" max="200" onChange={e => setFormData({ ...formData, ai_all: e.target.value })} /><span className="osa-hint">Événements/h total</span></label>
            <label className="osa-label">AI NREM<input type="number" step="0.1" value={formData.ai_nrem} placeholder="auto" min="0" max="200" onChange={e => setFormData({ ...formData, ai_nrem: e.target.value })} /><span className="osa-hint">Éveils en NREM</span></label>
            <label className="osa-label">AI REM<input type="number" step="0.1" value={formData.ai_rem} placeholder="auto" min="0" max="200" onChange={e => setFormData({ ...formData, ai_rem: e.target.value })} /><span className="osa-hint">Éveils en REM</span></label>
          </div>

          {osaError && <div className="error-bar visible" style={{ marginBottom: '15px' }}>⚠ {osaError}</div>}

          <button className="btn-analyse" onClick={handlePredictOsa} disabled={isPredictingOsa} style={{ width: '100%' }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /><path d="M12 8v4l3 3" /></svg>
            {isPredictingOsa ? 'Analyse en cours...' : 'Générer le Rapport Clinique SAOS'}
          </button>
        </div>
      )}

      {osaResults && (
        <div className="osa-report" style={{ marginTop: '30px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
          <div className="osa-report-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <div style={{ fontFamily: 'var(--serif)', fontSize: '18px', fontWeight: 700 }}>Rapport de Sévérité OSA</div>
              <div style={{ fontSize: '10px', color: 'var(--text3)', marginTop: '4px' }}>Modèle: {osaResults.model_used || 'Stacking Ensemble'}</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
              <button 
                onClick={() => exportFeatures('csv')} 
                style={{ background: 'none', border: '1px solid var(--border)', color: 'var(--text2)', borderRadius: '6px', fontSize: '10px', padding: '4px 10px', cursor: 'pointer', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                📥 Exporter CSV
              </button>
              <button 
                onClick={() => exportFeatures('xml')} 
                style={{ background: 'none', border: '1px solid var(--border)', color: 'var(--text2)', borderRadius: '6px', fontSize: '10px', padding: '4px 10px', cursor: 'pointer', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                📥 Exporter XML
              </button>
              <div className={`osa-severity-badge sev-${getSevClass(osaResults.severity)}`} style={{ marginLeft: '6px' }}>
                {osaResults.severity}
              </div>
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
          {reportConfig.shap && osaResults.shap_explanations && osaResults.shap_explanations.length > 0 && (
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

// Beautiful clinical HTML OSA report generator for Backblaze B2 Storage
const generateOsaHtmlReport = (patient, staging, osa) => {
  const patientName = patient ? `${patient.first_name} ${patient.last_name}` : 'Patient Anonyme';
  const patientAge = patient?.age || '—';
  const patientGender = patient?.gender === 'M' ? 'Homme' : patient?.gender === 'F' ? 'Femme' : '—';
  const patientBmi = patient?.imc || '—';
  const dateStr = new Date().toLocaleDateString('fr-FR', {
    year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'
  });

  const severity = osa.severity || 'Normal';
  const modelUsed = osa.model_used || 'Stacking Ensemble (XGB+LGBM+MLP)';

  // Staging Stats
  const stats = staging?.results?.[0]?.stats || {};
  const se = stats.se != null ? `${stats.se}%` : '—';
  const tst = stats.tst != null ? `${stats.tst} min` : '—';
  const tib = stats.tib != null ? `${stats.tib} min` : '—';
  const sol = stats.sol != null ? `${stats.sol} min` : '—';
  const rem_latency = stats.rem_latency != null ? `${stats.rem_latency} min` : '—';
  const waso = stats.waso != null ? `${stats.waso} min` : '—';

  // Probabilities
  const classColors = { Normal: '#059669', Mild: '#d97706', Moderate: '#ea580c', Severe: '#dc2626' };
  let probaHtml = '';
  if (osa.probabilities) {
    probaHtml = Object.entries(osa.probabilities).map(([cls, val]) => {
      const pct = (val * 100).toFixed(1);
      const color = classColors[cls] || '#1d4ed8';
      return `
        <div class="proba-row" style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
          <div class="proba-label" style="width: 80px; font-size: 12px; font-weight: 600; color: ${color};">${cls}</div>
          <div class="proba-bar-bg" style="flex: 1; height: 18px; background: rgba(255,255,255,0.03); border-radius: 4px; overflow: hidden;">
            <div class="proba-bar-fill" style="height: 100%; border-radius: 4px; width: ${pct}%; background-color: ${color};"></div>
          </div>
          <div class="proba-val" style="width: 50px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: #fff;">${pct}%</div>
        </div>
      `;
    }).join('');
  }

  // SHAP features
  let shapHtml = '';
  if (osa.shap_explanations && osa.shap_explanations.length > 0) {
    const maxImp = Math.max(...osa.shap_explanations.map(x => Math.abs(x.impact)), 0.1);
    shapHtml = osa.shap_explanations.map((sh) => {
      const isPos = sh.impact > 0;
      const pct = (Math.abs(sh.impact) / maxImp * 100).toFixed(1);
      const color = isPos ? '#dc2626' : '#059669';
      const barStyle = isPos
        ? `margin-left: 50%; width: ${pct / 2}%; background-color: ${color};`
        : `margin-left: ${50 - pct / 2}%; width: ${pct / 2}%; background-color: ${color};`;

      return `
        <div class="shap-row" style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px; font-size: 12px;">
          <div class="shap-feat" style="width: 220px; color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            ${sh.feature} <span class="shap-feat-val" style="color: #64748b; font-size: 11px;">(${sh.value})</span>
          </div>
          <div class="shap-bar-container" style="flex: 1; height: 12px; background: rgba(255,255,255,0.02); border-radius: 3px; position: relative;">
            <div class="shap-bar-fill" style="height: 100%; border-radius: 3px; position: absolute; ${barStyle}"></div>
          </div>
          <div class="shap-val" style="width: 60px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: ${color};">${sh.impact.toFixed(4)}</div>
        </div>
      `;
    }).join('');
  }

  // Interpretations
  let interpretationHtml = '';
  if (osa.interpretation) {
    interpretationHtml = osa.interpretation.map(item => {
      const text = item.text || item;
      return `<div class="interpretation-item" style="font-size: 13px; color: #94a3b8; margin-bottom: 10px; padding-left: 12px; border-left: 2px solid #ef4444;">${text}</div>`;
    }).join('');
  }

  return `<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Rapport Clinique SAOS - ${patientName}</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --surface: #151c2c;
      --surface-hover: #1e293b;
      --border: #24324f;
      --text: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --primary: #6366f1;
      --red: #ef4444;
      --red-s: rgba(239, 68, 68, 0.15);
      --font-sans: 'Plus Jakarta Sans', sans-serif;
      --font-serif: 'Outfit', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      line-height: 1.6;
      padding: 40px 20px;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 40px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    header {
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
      margin-bottom: 30px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }
    .logo-area h1 {
      font-family: var(--font-serif);
      font-size: 28px;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.5px;
    }
    .logo-area h1 span {
      color: var(--red);
    }
    .meta-time {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .sev-badge {
      display: inline-block;
      font-family: var(--font-serif);
      font-size: 16px;
      font-weight: 700;
      padding: 8px 20px;
      border-radius: 30px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .sev-normal { background: rgba(5, 150, 105, 0.15); color: #10b981; border: 1.5px solid rgba(5, 150, 105, 0.3); }
    .sev-mild { background: rgba(217, 119, 6, 0.15); color: #fbbf24; border: 1.5px solid rgba(217, 119, 6, 0.3); }
    .sev-moderate { background: rgba(234, 88, 12, 0.15); color: #f97316; border: 1.5px solid rgba(234, 88, 12, 0.3); }
    .sev-severe { background: rgba(220, 38, 38, 0.2); color: #ef4444; border: 1.5px solid rgba(220, 38, 38, 0.4); }

    .section-title {
      font-family: var(--font-serif);
      font-size: 18px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 16px;
      border-left: 3px solid var(--red);
      padding-left: 12px;
      margin-top: 30px;
    }
    .grid-patient {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 30px;
    }
    .card-info {
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }
    .card-info-lbl {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .card-info-val {
      font-size: 18px;
      font-weight: 700;
      color: #fff;
    }

    .grid-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 30px;
    }
    .stat-box {
      background: rgba(255,255,255,0.01);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      text-align: center;
    }
    .stat-lbl {
      font-size: 10px;
      color: var(--text-secondary);
      margin-bottom: 4px;
    }
    .stat-val {
      font-family: var(--font-mono);
      font-size: 16px;
      font-weight: 700;
      color: #fff;
    }
    footer {
      border-top: 1px solid var(--border);
      padding-top: 20px;
      margin-top: 40px;
      text-align: center;
      font-size: 11px;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-area">
        <h1>Hypnoria<span>.</span></h1>
        <div class="meta-time">Généré le ${dateStr}</div>
      </div>
      <div class="sev-badge sev-${severity.toLowerCase()}">
        SAOS: ${severity}
      </div>
    </header>

    <div class="section-title">Patient & Données Cliniques</div>
    <div class="grid-patient">
      <div class="card-info">
        <div class="card-info-lbl">Patient</div>
        <div class="card-info-val">${patientName}</div>
      </div>
      <div class="card-info">
        <div class="card-info-lbl">Âge / Sexe</div>
        <div class="card-info-val">${patientAge} ans / ${patientGender}</div>
      </div>
      <div class="card-info">
        <div class="card-info-lbl">IMC</div>
        <div class="card-info-val">${patientBmi} kg/m²</div>
      </div>
    </div>

    <div class="section-title">Statistiques Polysomnographie (Staging)</div>
    <div class="grid-stats">
      <div class="stat-box">
        <div class="stat-lbl">Efficacité Sommeil</div>
        <div class="stat-val">${se}</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Temps Total Sommeil</div>
        <div class="stat-val">${tst}</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Temps au Lit</div>
        <div class="stat-val">${tib}</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Latence Endormissement</div>
        <div class="stat-val">${sol}</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Latence REM</div>
        <div class="stat-val">${rem_latency}</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">WASO</div>
        <div class="stat-val">${waso}</div>
      </div>
    </div>

    <div class="section-title">Distribution de Confiance du Modèle</div>
    <div style="margin-bottom: 30px; background: rgba(255,255,255,0.01); border: 1px solid var(--border); border-radius: 12px; padding: 20px;">
      ${probaHtml}
      <div style="font-size: 10px; color: var(--text-muted); margin-top: 15px; text-align: right;">Algorithme: ${modelUsed}</div>
    </div>

    ${shapHtml ? `
    <div class="section-title">Analyse SHAP des Facteurs Sévérité</div>
    <div style="margin-bottom: 30px; background: rgba(255,255,255,0.01); border: 1px solid var(--border); border-radius: 12px; padding: 20px;">
      <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted); margin-bottom: 15px;">
        <span>← Réduit le risque (Protecteur)</span>
        <span>Aggrave le risque (Inducteur) →</span>
      </div>
      ${shapHtml}
    </div>
    ` : ''}

    ${interpretationHtml ? `
    <div class="section-title">Interprétation Clinique</div>
    <div style="margin-bottom: 10px; background: rgba(255,255,255,0.01); border: 1px solid var(--border); border-radius: 12px; padding: 20px;">
      ${interpretationHtml}
    </div>
    ` : ''}

    <footer>
      Ce document est un rapport clinique généré automatiquement par Hypnoria suite à l'analyse prédictive.
    </footer>
  </div>
</body>
</html>`;
};

function getSevClass(severity) {
  if (!severity) return 'normal';
  const s = severity.toLowerCase();
  if (s.includes('severe')) return 'severe';
  if (s.includes('moderate')) return 'moderate';
  if (s.includes('mild')) return 'mild';
  return 'normal';
}

const StatCard = ({ lbl, val, unit, note, cls = '', customNode = null }) => (
  <div className={`stat-card ${cls}`}>
    <div className="stat-lbl">{lbl}</div>
    <div className="stat-val">{val}<span className="stat-unit">{unit}</span></div>
    <div className="stat-note">{note}</div>
    {customNode}
  </div>
);

export default AnalysisResults;
