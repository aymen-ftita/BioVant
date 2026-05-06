/**
 * PipelineBuilder Component Logic
 * Handles Drawflow initialization, node dragging, and pipeline execution.
 */
let editor;

function initDrawflow() {
  const id = document.getElementById("drawflow");
  if (!id) return;
  if (typeof Drawflow === 'undefined') {
      console.error("Drawflow library not loaded.");
      return;
  }
  editor = new Drawflow(id);
  editor.reroute = true;
  editor.start();

  editor.on('connectionCreated', () => updatePipelinePreview());
  editor.on('connectionRemoved', () => updatePipelinePreview());
  editor.on('nodeCreated', () => setTimeout(updatePipelinePreview, 100));
  editor.on('nodeRemoved', () => updatePipelinePreview());
}

function clearPipeline() {
  if (!editor) return;
  editor.clear();
  const preview = document.getElementById("pipe-confidence-preview");
  if (preview) preview.style.display = "none";
  const results = document.getElementById("pipeline-results");
  if (results) { results.innerHTML = ""; results.style.display = "none"; }
}

function updatePipelinePreview() {
  if (!editor) return;
  const exportdata = editor.export();
  const nodes = exportdata.drawflow.Home.data;

  const modelNodes = [];
  for (const id in nodes) {
    if (nodes[id].name.startsWith('model_')) modelNodes.push(id);
  }

  const preview = document.getElementById("pipe-confidence-preview");
  const container = document.getElementById("pipe-conf-cards");
  if (!preview || !container) return;

  if (modelNodes.length === 0) {
    preview.style.display = "none";
    return;
  }

  const jobs = [];
  for (const mId of modelNodes) {
    const mType = getModelType(nodes[mId].name);
    const paths = tracePathsBack(mId, nodes, { channels: '5', classes: '3' });
    for (const p of paths) {
      jobs.push({ fileName: p.fileName || '(aucun fichier)', channels: p.channels, classes: p.classes, model: mType });
    }
  }

  if (jobs.length === 0) {
    preview.style.display = "none";
    return;
  }

  preview.style.display = "block";
  container.innerHTML = "";

  jobs.forEach(j => {
    const conf = getConfidence(j.channels, j.classes, j.model);
    const color = getConfColor(conf);
    const card = document.createElement("div");
    card.style.cssText = `flex:0 0 auto; min-width:180px; padding:14px 18px; border-radius:12px; background:var(--surface); border:1.5px solid var(--border); transition:var(--t);`;
    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-family:var(--mono); font-size:22px; font-weight:800; color:${color};">${conf}%</span>
      </div>
      <div style="font-family:var(--serif); font-size:13px; font-weight:700; color:var(--text); margin-bottom:4px;">🧠 ${j.model}</div>
      <div style="font-size:10px; color:var(--text3); font-family:var(--mono);">${j.channels}ch · ${j.classes}cls</div>
      <div style="font-size:10px; color:var(--text2); margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:200px;">📁 ${j.fileName}</div>
    `;
    container.appendChild(card);
  });
}

function allowDrop(ev) { ev.preventDefault(); }
function drag(ev) { ev.dataTransfer.setData("node", ev.target.getAttribute("data-node")); }
function drop(ev) {
  ev.preventDefault();
  var data = ev.dataTransfer.getData("node");
  addNodeToDrawFlow(data, ev.clientX, ev.clientY);
}

let _dfNodeCounter = 0;
function addNodeToDrawFlow(name, pos_x, pos_y) {
  if (!editor || editor.editor_mode === 'fixed') return;

  pos_x = pos_x * (editor.precanvas.clientWidth / (editor.precanvas.clientWidth * editor.zoom)) - (editor.precanvas.getBoundingClientRect().x * (editor.precanvas.clientWidth / (editor.precanvas.clientWidth * editor.zoom)));
  pos_y = pos_y * (editor.precanvas.clientHeight / (editor.precanvas.clientHeight * editor.zoom)) - (editor.precanvas.getBoundingClientRect().y * (editor.precanvas.clientHeight / (editor.precanvas.clientHeight * editor.zoom)));

  let html = "";
  let inputs = 1;
  let outputs = 1;
  _dfNodeCounter++;

  if (name === 'patient_data') {
    inputs = 0;
    const fid = `df-file-${_dfNodeCounter}`;
    html = `<div><div class="title-box">📁 Patient Data</div><div><input type="file" class="df-file-input" id="${fid}" accept=".edf" style="font-size:10px; max-width:180px; margin-top:8px;"></div></div>`;
  } else if (name === '2_channels') {
    html = `<div><div class="title-box">⚡ 2 Canaux</div><div style="font-size:11px; color:var(--text3);">EEG</div></div>`;
  } else if (name === '5_channels') {
    html = `<div><div class="title-box">⚡ 5 Canaux</div><div style="font-size:11px; color:var(--text3);">EEG, EOG, EMG</div></div>`;
  } else if (name === '3_classes') {
    html = `<div><div class="title-box">📊 3 Classes</div><div style="font-size:11px; color:var(--text3);">Wake, NREM, REM</div></div>`;
  } else if (name === '5_classes') {
    html = `<div><div class="title-box">📊 5 Classes</div><div style="font-size:11px; color:var(--text3);">W, N1, N2, N3, R</div></div>`;
  } else if (name.startsWith('model_')) {
    outputs = 0;
    let m = name.split('_')[1];
    let mname = m === 'bilstm' ? 'Bi-LSTM' : m === 'cnn' ? '1D-CNN' : m === 'transformer' ? 'Transformer' : 'Stacking';
    html = `<div><div class="title-box">🧠 Modèle IA</div><div style="color:var(--red); font-weight:700;">${mname}</div></div>`;
  }

  editor.addNode(name, inputs, outputs, pos_x, pos_y, name, { "nodeName": name }, html);
}

function getConfidence(ch, cls, model) {
  let base = 90;
  if (ch === '5' && cls === '3') base = 98;
  return base;
}

function getConfColor(conf) {
  if (conf >= 95) return '#059669';
  return '#d97706';
}

function getModelType(nodeName) {
  if (nodeName === 'model_bilstm') return 'LSTM';
  if (nodeName === 'model_cnn') return 'CNN';
  if (nodeName === 'model_transformer') return 'Transformer';
  if (nodeName === 'model_stacking') return 'Stacking';
  return null;
}

function tracePathsBack(nodeId, nodes, currentCfg) {
    const nid = String(nodeId);
    const node = nodes[nid];
    if (!node) return [];
    const cfg = { ...currentCfg };
    if (node.name === '2_channels') cfg.channels = '2';
    if (node.name === '5_channels') cfg.channels = '5';
    if (node.name === '3_classes') cfg.classes = '3';
    if (node.name === '5_classes') cfg.classes = '5';

    if (node.name === 'patient_data') {
        const nodeEl = document.querySelector(`#node-${nid}`);
        const fileInput = nodeEl ? nodeEl.querySelector('.df-file-input') : null;
        const file = fileInput && fileInput.files && fileInput.files[0];
        return [{ ...cfg, dataNodeId: nid, file: file, fileName: file ? file.name : null }];
    }

    const paths = [];
    const inputKeys = Object.keys(node.inputs || {});
    for (const key of inputKeys) {
        const conns = node.inputs[key].connections || [];
        for (const conn of conns) {
            const subPaths = tracePathsBack(String(conn.node), nodes, { ...cfg });
            paths.push(...subPaths);
        }
    }
    return paths;
}

async function startPipelineAnalysis() {
    // Similar logic to main.js but scoped
}
