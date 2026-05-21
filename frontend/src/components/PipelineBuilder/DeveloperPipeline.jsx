import React, { useEffect, useRef, useCallback, useState } from 'react';
import Drawflow from 'drawflow';
import 'drawflow/dist/drawflow.min.css';
import './PipelineBuilder.css';
import axios from 'axios';

/**
 * DeveloperPipeline — full port of vanilla PipelineBuilder with real /analyze calls
 */

const API = 'http://localhost:8000';

let _dfNodeCounter = 0;

function getNodeTemplate(name) {
  let html = '';
  let inputs = 1;
  let outputs = 1;

  if (name === 'patient_data') {
    inputs = 0;
    _dfNodeCounter++;
    const fid = `df-file-${_dfNodeCounter}`;
    html = `<div><div class="title-box">📁 Patient Data</div><div><input type="file" class="df-file-input" id="${fid}" accept=".edf" style="font-size:10px; max-width:180px; margin-top:8px;"></div></div>`;
  } else if (name === '2_channels') {
    html = `<div><div class="title-box">⚡ 2 Canaux</div><div style="font-size:11px;color:var(--text3);">EEG</div></div>`;
  } else if (name === '5_channels') {
    html = `<div><div class="title-box">⚡ 5 Canaux</div><div style="font-size:11px;color:var(--text3);">EEG, EOG, EMG</div></div>`;
  } else if (name === '3_classes') {
    html = `<div><div class="title-box">📊 3 Classes</div><div style="font-size:11px;color:var(--text3);">Wake, NREM, REM</div></div>`;
  } else if (name === '5_classes') {
    html = `<div><div class="title-box">📊 5 Classes</div><div style="font-size:11px;color:var(--text3);">W, N1, N2, N3, R</div></div>`;
  } else if (name.startsWith('model_')) {
    outputs = 0;
    const m = name.split('_')[1];
    const mname = m === 'bilstm' ? 'Bi-LSTM' : m === 'cnn' ? '1D-CNN' : m === 'transformer' ? 'Transformer' : 'Stacking';
    html = `<div><div class="title-box">🧠 Modèle IA</div><div style="color:var(--red);font-weight:700;">${mname}</div></div>`;
  }

  return { html, inputs, outputs };
}

function getConfidence(ch, cls) {
  if (ch === '5' && cls === '3') return 98;
  if (ch === '5' && cls === '5') return 94;
  if (ch === '2' && cls === '3') return 91;
  return 88;
}

function getConfColor(conf) {
  if (conf >= 95) return '#059669';
  if (conf >= 90) return '#d97706';
  return '#dc2626';
}

function getModelType(nodeName) {
  if (nodeName === 'model_bilstm') return 'LSTM';
  if (nodeName === 'model_cnn') return 'CNN';
  if (nodeName === 'model_transformer') return 'Transformer';
  if (nodeName === 'model_stacking') return 'Stacking';
  return null;
}

/** Walk backwards from a model node through the graph to collect all configs */
function tracePathsBack(nodeId, nodes, currentCfg, domContainer) {
  const nid = String(nodeId);
  const node = nodes[nid];
  if (!node) return [];
  const cfg = { ...currentCfg };
  if (node.name === '2_channels') cfg.channels = '2';
  if (node.name === '5_channels') cfg.channels = '5';
  if (node.name === '3_classes') cfg.classes = '3';
  if (node.name === '5_classes') cfg.classes = '5';

  if (node.name === 'patient_data') {
    let fileName = '(aucun fichier)';
    let file = null;
    // Get file from the actual DOM node inside drawflow container
    const nodeEl = domContainer ? domContainer.querySelector(`#node-${nid}`) : document.querySelector(`#node-${nid}`);
    if (nodeEl) {
      const fileInput = nodeEl.querySelector('.df-file-input');
      if (fileInput && fileInput.files && fileInput.files[0]) {
        file = fileInput.files[0];
        fileName = file.name;
      }
    }
    return [{ ...cfg, dataNodeId: nid, fileName, file }];
  }

  const paths = [];
  const inputKeys = Object.keys(node.inputs || {});
  for (const key of inputKeys) {
    const conns = node.inputs[key].connections || [];
    for (const conn of conns) {
      const subPaths = tracePathsBack(String(conn.node), nodes, { ...cfg }, domContainer);
      paths.push(...subPaths);
    }
  }
  return paths;
}

// ─── Stage Color Map ──────────────────────────────────────────────────────────
const SC = { Wake: '#c0392b', NREM: '#1d4ed8', REM: '#047857', N1: '#d97706', N2: '#1d4ed8', N3: '#6d28d9' };
const fmt = (n) => (n != null ? n : '–');

// ─── Pipeline Result Card ─────────────────────────────────────────────────────
const PipelineResultCard = ({ result, groupLabel }) => {
  const { stages, stats, model_info } = result;
  const classNames = stats.class_names || [];

  return (
    <div style={{ marginBottom: '40px', padding: '20px', background: 'var(--surface)', borderRadius: '12px', border: '1px solid var(--border)', boxShadow: 'var(--sh)' }}>
      <div style={{ fontFamily: 'var(--serif)', fontSize: '15px', fontWeight: 700, color: 'var(--text)', marginBottom: '12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <span>🧠 {model_info.type}</span>
        <span style={{ fontSize: '11px', fontWeight: 400, color: 'var(--text2)', fontFamily: 'var(--mono)' }}>
          {model_info.channels}ch · {model_info.classes}cls · {groupLabel}
        </span>
      </div>

      {/* Stage distribution bar */}
      <div style={{ display: 'flex', height: '14px', borderRadius: '6px', overflow: 'hidden', marginBottom: '10px' }}>
        {classNames.map(st => {
          const pct = stats.stage_pct[st] || 0;
          return <div key={st} title={`${st}: ${pct}%`} style={{ width: `${pct}%`, background: SC[st] || '#888', transition: 'width .5s ease' }} />;
        })}
      </div>

      {/* Key metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: '8px', marginBottom: '10px' }}>
        {[
          { lbl: 'TST', val: fmt(stats.tst), unit: 'min' },
          { lbl: 'TIB', val: fmt(stats.tib), unit: 'min' },
          { lbl: 'Efficacité', val: fmt(stats.se), unit: '%' },
          { lbl: 'Lat. Endorm.', val: fmt(stats.sol), unit: 'min' },
          { lbl: 'Lat. REM', val: stats.rem_latency != null ? fmt(stats.rem_latency) : 'N/A', unit: 'min' },
          { lbl: 'WASO', val: fmt(stats.waso), unit: 'min' },
        ].map(m => (
          <div key={m.lbl} style={{ padding: '8px 10px', background: 'var(--bg)', borderRadius: '8px', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '9px', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '3px' }}>{m.lbl}</div>
            <div style={{ fontSize: '15px', fontWeight: 700, fontFamily: 'var(--mono)' }}>{m.val}<span style={{ fontSize: '9px', color: 'var(--text3)', marginLeft: '2px' }}>{m.unit}</span></div>
          </div>
        ))}
      </div>

      {/* Stage breakdown */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {classNames.map(st => (
          <div key={st} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', padding: '3px 8px', borderRadius: '20px', background: `${SC[st] || '#888'}18`, color: SC[st] || '#888', border: `1px solid ${SC[st] || '#888'}33` }}>
            <span style={{ fontWeight: 700 }}>{st}</span>
            <span style={{ color: 'var(--text3)' }}>{stats.stage_pct[st] || 0}%</span>
          </div>
        ))}
      </div>

      {/* Alerts */}
      {stats.alerts && stats.alerts.length > 0 && (
        <div style={{ marginTop: '10px' }}>
          {stats.alerts.map((a, i) => (
            <div key={i} style={{ fontSize: '11px', color: '#b45309', background: '#fef3c7', padding: '5px 10px', borderRadius: '6px', marginBottom: '4px' }}>⚠ {a}</div>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Main Component ───────────────────────────────────────────────────────────
const DeveloperPipeline = () => {
  const wrapperRef = useRef(null);
  const containerRef = useRef(null);
  const editorRef = useRef(null);
  const [pipelineJobs, setPipelineJobs] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [runProgress, setRunProgress] = useState('');
  const [pipelineResults, setPipelineResults] = useState([]); // [{groupLabel, result}]
  const [pipelineError, setPipelineError] = useState(null);

  const updatePipelinePreview = useCallback(() => {
    if (!editorRef.current) return;
    try {
      const exportdata = editorRef.current.export();
      const nodes = exportdata.drawflow.Home.data;

      const modelNodes = [];
      for (const id in nodes) {
        if (nodes[id].name.startsWith('model_')) modelNodes.push(id);
      }

      if (modelNodes.length === 0) {
        setPipelineJobs([]);
        return;
      }

      const jobs = [];
      for (const mId of modelNodes) {
        const mType = getModelType(nodes[mId].name);
        const paths = tracePathsBack(mId, nodes, { channels: '5', classes: '3' }, containerRef.current);
        for (const p of paths) {
          jobs.push({
            fileName: p.fileName || '(aucun fichier)',
            channels: p.channels,
            classes: p.classes,
            model: mType,
            file: p.file,
            dataNodeId: p.dataNodeId,
          });
        }
      }
      setPipelineJobs(jobs);
    } catch (e) {
      console.warn('Could not update pipeline preview', e);
    }
  }, []);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.innerHTML = '';
    }

    const drawflowElement = document.createElement('div');
    drawflowElement.id = 'drawflow';
    drawflowElement.className = 'drawflow';
    drawflowElement.style.width = '100%';
    drawflowElement.style.height = '100%';

    if (containerRef.current) {
      containerRef.current.appendChild(drawflowElement);
    }

    const editor = new Drawflow(drawflowElement);
    editor.reroute = true;
    editor.start();
    editorRef.current = editor;

    editor.on('connectionCreated', updatePipelinePreview);
    editor.on('connectionRemoved', updatePipelinePreview);
    editor.on('nodeCreated', () => setTimeout(updatePipelinePreview, 100));
    editor.on('nodeRemoved', updatePipelinePreview);

    const wrapper = wrapperRef.current;
    const handleDragOver = (ev) => ev.preventDefault();

    const handleDrop = (ev) => {
      ev.preventDefault();
      const name = ev.dataTransfer.getData('node');
      if (!name || !editorRef.current) return;

      const ed = editorRef.current;
      if (ed.editor_mode === 'fixed') return;

      const pos_x =
        ev.clientX * (ed.precanvas.clientWidth / (ed.precanvas.clientWidth * ed.zoom)) -
        ed.precanvas.getBoundingClientRect().x * (ed.precanvas.clientWidth / (ed.precanvas.clientWidth * ed.zoom));
      const pos_y =
        ev.clientY * (ed.precanvas.clientHeight / (ed.precanvas.clientHeight * ed.zoom)) -
        ed.precanvas.getBoundingClientRect().y * (ed.precanvas.clientHeight / (ed.precanvas.clientHeight * ed.zoom));

      const { html, inputs, outputs } = getNodeTemplate(name);
      if (!html) return;

      ed.addNode(name, inputs, outputs, pos_x, pos_y, name, { nodeName: name }, html);
      setTimeout(updatePipelinePreview, 100);
    };

    const handleFileInput = (e) => {
      if (e.target && e.target.classList.contains('df-file-input')) {
        setTimeout(updatePipelinePreview, 50);
      }
    };
    drawflowElement.addEventListener('change', handleFileInput);

    if (wrapper) {
      wrapper.addEventListener('drop', handleDrop);
      wrapper.addEventListener('dragover', handleDragOver);
    }

    return () => {
      if (wrapper) {
        wrapper.removeEventListener('drop', handleDrop);
        wrapper.removeEventListener('dragover', handleDragOver);
      }
      drawflowElement.removeEventListener('change', handleFileInput);
      if (containerRef.current && drawflowElement.parentNode === containerRef.current) {
        containerRef.current.removeChild(drawflowElement);
      }
      editorRef.current = null;
    };
  }, [updatePipelinePreview]);

  const handleDragStart = useCallback((e) => {
    const nodeType = e.currentTarget.getAttribute('data-node');
    if (nodeType) e.dataTransfer.setData('node', nodeType);
  }, []);

  const clearPipeline = useCallback(() => {
    if (editorRef.current) {
      editorRef.current.clear();
      setPipelineJobs([]);
      setPipelineResults([]);
      setPipelineError(null);
    }
  }, []);

  const startPipelineAnalysis = useCallback(async () => {
    // Re-read jobs fresh from DOM (file objects need to come from DOM)
    if (!editorRef.current) return;

    let freshJobs = [];
    try {
      const exportdata = editorRef.current.export();
      const nodes = exportdata.drawflow.Home.data;
      const modelNodes = Object.keys(nodes).filter(id => nodes[id].name.startsWith('model_'));

      for (const mId of modelNodes) {
        const mType = getModelType(nodes[mId].name);
        const paths = tracePathsBack(mId, nodes, { channels: '5', classes: '3' }, containerRef.current);
        for (const p of paths) {
          freshJobs.push({ ...p, model: mType });
        }
      }
    } catch (e) {
      setPipelineError('Erreur lors de la lecture du pipeline.');
      return;
    }

    if (freshJobs.length === 0) {
      setPipelineError("Pipeline incomplet : connectez 'Patient Data' → Canaux → Classes → Modèle.");
      return;
    }

    const jobsWithFile = freshJobs.filter(j => j.file);
    if (jobsWithFile.length === 0) {
      setPipelineError("Aucun fichier EDF sélectionné dans le nœud 'Patient Data'.");
      return;
    }

    // Group jobs by (dataNodeId|channels|classes) to batch model calls
    const groups = {};
    for (const j of freshJobs) {
      if (!j.file) continue;
      const key = `${j.dataNodeId}|${j.channels}|${j.classes}`;
      if (!groups[key]) groups[key] = { file: j.file, fileName: j.fileName, channels: j.channels, classes: j.classes, models: [] };
      if (j.model && !groups[key].models.includes(j.model)) groups[key].models.push(j.model);
    }

    setIsRunning(true);
    setPipelineResults([]);
    setPipelineError(null);

    const groupKeys = Object.keys(groups);
    const allResults = [];

    try {
      for (let gi = 0; gi < groupKeys.length; gi++) {
        const grp = groups[groupKeys[gi]];
        setRunProgress(`Analyse ${gi + 1}/${groupKeys.length} — ${grp.fileName} (${grp.channels}ch · ${grp.classes}cls)`);

        const formData = new FormData();
        formData.append('file', grp.file);
        formData.append('models', grp.models.join(','));
        formData.append('channels', grp.channels);
        formData.append('classes', grp.classes);

        const res = await axios.post(`${API}/analyze`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });

        if (res.data.results) {
          const groupLabel = `${grp.fileName} · ${grp.channels}ch · ${grp.classes}cls`;
          for (const r of res.data.results) {
            allResults.push({ groupLabel, result: r });
          }
        }
      }

      setPipelineResults(allResults);
      setRunProgress(`✓ Terminé — ${allResults.length} résultat(s)`);
    } catch (err) {
      console.error(err);
      let msg = 'Erreur serveur.';
      if (err.response?.data?.detail) {
        msg = typeof err.response.data.detail === 'string' ? err.response.data.detail : JSON.stringify(err.response.data.detail);
      } else if (err.message) {
        msg = err.message;
      }
      setPipelineError(msg);
      setRunProgress('');
    } finally {
      setIsRunning(false);
    }
  }, []);

  return (
    <div style={{ marginTop: '20px' }}>
      {/* Real-time confidence preview */}
      {pipelineJobs.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '2px', color: 'var(--text3)', marginBottom: '10px' }}>
            📊 Pipeline Preview — Configurations Détectées
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
            {pipelineJobs.map((job, idx) => {
              const conf = getConfidence(job.channels, job.classes);
              const color = getConfColor(conf);
              return (
                <div key={idx} style={{ flex: '0 0 auto', minWidth: '180px', padding: '14px 18px', borderRadius: '12px', background: 'var(--surface)', border: `1.5px solid ${color}44`, transition: 'all .2s' }}>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: '24px', fontWeight: 800, color, marginBottom: '6px' }}>{conf}%</div>
                  <div style={{ fontFamily: 'var(--serif)', fontSize: '13px', fontWeight: 700, color: 'var(--text)', marginBottom: '3px' }}>🧠 {job.model}</div>
                  <div style={{ fontSize: '10px', color: 'var(--text3)', fontFamily: 'var(--mono)' }}>{job.channels}ch · {job.classes}cls</div>
                  <div style={{ fontSize: '10px', color: job.file ? '#059669' : 'var(--text3)', marginTop: '4px', fontWeight: job.file ? 600 : 400, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '200px' }}>
                    {job.file ? `✓ ${job.fileName}` : `📁 ${job.fileName}`}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Drawflow canvas */}
      <div className="df-container" ref={wrapperRef}>
        <div className="df-sidebar">
          <h3>1. Data</h3>
          <div className="df-item" draggable="true" data-node="patient_data" onDragStart={handleDragStart}>📁 Patient Data</div>

          <h3>2. Canaux</h3>
          <div className="df-item" draggable="true" data-node="2_channels" onDragStart={handleDragStart}>⚡ 2 Canaux (EEG)</div>
          <div className="df-item" draggable="true" data-node="5_channels" onDragStart={handleDragStart}>⚡ 5 Canaux (EEG+EOG+EMG)</div>

          <h3>3. Classes</h3>
          <div className="df-item" draggable="true" data-node="3_classes" onDragStart={handleDragStart}>📊 3 Classes</div>
          <div className="df-item" draggable="true" data-node="5_classes" onDragStart={handleDragStart}>📊 5 Classes</div>

          <h3>4. Modèle IA</h3>
          <div className="df-item" draggable="true" data-node="model_bilstm" onDragStart={handleDragStart}>🧠 Bi-LSTM</div>
          <div className="df-item" draggable="true" data-node="model_cnn" onDragStart={handleDragStart}>🧠 CNN</div>
          <div className="df-item" draggable="true" data-node="model_transformer" onDragStart={handleDragStart}>🧠 Transformer</div>
          <div className="df-item" draggable="true" data-node="model_stacking" onDragStart={handleDragStart}>🧠 Stacking Ensemble</div>
        </div>

        <div ref={containerRef} style={{ width: '100%', height: '100%', flex: 1, position: 'relative' }} />
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap', marginTop: '12px' }}>
        <button className="btn-run-pipeline" onClick={startPipelineAnalysis} disabled={isRunning} style={{ opacity: isRunning ? 0.7 : 1 }}>
          {isRunning ? (
            <>
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" style={{ animation: 'spin 1s linear infinite' }}><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48 2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48 2.83-2.83" /></svg>
              Analyse en cours…
            </>
          ) : (
            <>
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3" /></svg>
              Lancer l'Analyse du Pipeline
            </>
          )}
        </button>
        <button className="btn-reset" onClick={clearPipeline} style={{ padding: '10px 20px', fontSize: '12px' }} disabled={isRunning}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-3.51" /></svg>
          Vider le Pipeline
        </button>
        {runProgress && (
          <span style={{ fontSize: '11px', color: runProgress.startsWith('✓') ? '#059669' : 'var(--text2)', fontFamily: 'var(--mono)' }}>
            {runProgress}
          </span>
        )}
      </div>

      {/* Error */}
      {pipelineError && (
        <div style={{ marginTop: '12px', padding: '10px 14px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '8px', fontSize: '12px', color: '#dc2626' }}>
          ⚠ {pipelineError}
        </div>
      )}

      {/* Results */}
      {pipelineResults.length > 0 && (
        <div style={{ marginTop: '30px' }}>
          <div style={{ fontFamily: 'var(--serif)', fontSize: '18px', fontWeight: 900, color: 'var(--text)', marginBottom: '20px', paddingBottom: '10px', borderBottom: '2px solid var(--border)' }}>
            Résultats du Pipeline — {pipelineResults.length} analyse{pipelineResults.length > 1 ? 's' : ''}
          </div>
          {pipelineResults.map(({ groupLabel, result }, idx) => (
            <PipelineResultCard key={idx} result={result} groupLabel={groupLabel} />
          ))}
        </div>
      )}
    </div>
  );
};

export default DeveloperPipeline;
