const API = "http://localhost:5000";
const cs = getComputedStyle(document.documentElement);
const SC = { Wake: cs.getPropertyValue('--wake').trim(), NREM: cs.getPropertyValue('--nrem').trim(), REM: cs.getPropertyValue('--rem').trim(), N1: cs.getPropertyValue('--n1').trim(), N2: cs.getPropertyValue('--n2').trim(), N3: cs.getPropertyValue('--n3').trim() };

let _currentFile = null;

/* ── HERO SIGNALS ── */
(function heroSignals() {
  const cfgs = [
    { id: 'sig-eeg', color: 'rgba(192,57,43,.75)', lw: 1.6, fn(t) { return Math.sin(t * .022) * 16 + Math.sin(t * .11) * 4 + Math.sin(t * .31) * 2 + (Math.random() - .5) * .5 }, speed: .9 },
    { id: 'sig-emg', color: 'rgba(29,78,216,.65)', lw: 1.2, fn(t) { return Math.sin(t * .08) * 3 + (Math.random() - .5) * 7 }, speed: 1.2 },
    { id: 'sig-eog', color: 'rgba(109,40,217,.6)', lw: 1.5, fn(t) { return Math.sin(t * .015) * 18 + Math.sin(t * .047) * 6 + (Math.random() - .5) * 1.5 }, speed: .6 },
  ];
  cfgs.forEach(cfg => {
    const canvas = document.getElementById(cfg.id);
    if (!canvas) return;
    const wrap = canvas.parentElement;
    let W, H, ctx, phase = 0;
    function resize() { W = wrap.offsetWidth; H = wrap.offsetHeight || 62; canvas.width = W * devicePixelRatio; canvas.height = H * devicePixelRatio; canvas.style.width = W + "px"; canvas.style.height = H + "px"; ctx = canvas.getContext("2d"); ctx.scale(devicePixelRatio, devicePixelRatio); }
    resize(); window.addEventListener("resize", resize);
    const NL = 2000, noise = new Float32Array(NL); for (let i = 0; i < NL; i++) noise[i] = (Math.random() - .5) * 2;
    function sn(t) { const i = Math.floor(t) % NL, j = (i + 1) % NL, f = t - Math.floor(t); return noise[i] * (1 - f) + noise[j] * f; }
    function draw() {
      if (!ctx) { requestAnimationFrame(draw); return }
      ctx.clearRect(0, 0, W, H); ctx.beginPath(); ctx.lineWidth = cfg.lw; ctx.strokeStyle = cfg.color; ctx.lineJoin = "round";
      const mid = H / 2;
      for (let x = 0; x <= W; x += 1.5) { const t = x * .8 + phase * cfg.speed * 60; const y = mid + cfg.fn(t) + sn(t * .03) * 2; x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke(); phase += .016; requestAnimationFrame(draw);
    }
    draw();
  });
})();

/* ── PREVIEW CHANNELS (PSG Review, light bg, scrolling) ── */
const previewChannels = (function () {
  const cfgs = [
    { id: "prev-eeg", color: "rgba(192,57,43,.8)", lw: 1.4, fn(t) { return Math.sin(t * .025) * 18 + Math.sin(t * .14) * 5 + Math.sin(t * .38) * 2.5 + Math.sin(t * .003) * 8 } },
    { id: "prev-eeg2", color: "rgba(192,57,43,.45)", lw: 1.2, fn(t) { return Math.sin(t * .022 + 1.2) * 15 + Math.sin(t * .11 + .8) * 4 + Math.sin(t * .003) * 6 } },
    { id: "prev-eogl", color: "rgba(109,40,217,.75)", lw: 1.4, fn(t) { return Math.sin(t * .012) * 22 + Math.sin(t * .041) * 7 } },
    { id: "prev-eogr", color: "rgba(109,40,217,.45)", lw: 1.2, fn(t) { return -Math.sin(t * .012 + .3) * 20 - Math.sin(t * .041 + .2) * 6 } },
    { id: "prev-emg", color: "rgba(29,78,216,.7)", lw: 1.1, fn(t) { return Math.sin(t * .09) * 2 + (Math.random() - .5) * 8 } },
  ];
  let running = false, rafId = null, phase = 0;
  const NL = 3000, noiseT = new Float32Array(NL); for (let i = 0; i < NL; i++) noiseT[i] = (Math.random() - .5) * 2;
  function sn(t) { const i = Math.floor(Math.abs(t)) % NL, j = (i + 1) % NL, f = t - Math.floor(t); return noiseT[i] * (1 - f) + noiseT[j] * f; }
  const ctxs = {};
  function init() {
    cfgs.forEach(cfg => {
      const canvas = document.getElementById(cfg.id); if (!canvas) return;
      const wrap = canvas.parentElement;
      const W = wrap.offsetWidth, H = wrap.offsetHeight || 68;
      canvas.width = W * devicePixelRatio; canvas.height = H * devicePixelRatio;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
      const c = canvas.getContext("2d"); c.scale(devicePixelRatio, devicePixelRatio); ctxs[cfg.id] = c;
    });
  }
  function drawFrame() {
    if (!running) return;
    cfgs.forEach(cfg => {
      const ctx = ctxs[cfg.id]; if (!ctx) return;
      const canvas = document.getElementById(cfg.id); if (!canvas) return;
      const W = canvas.parentElement.offsetWidth, H = 68;
      ctx.clearRect(0, 0, W, H); ctx.beginPath(); ctx.lineWidth = cfg.lw; ctx.strokeStyle = cfg.color; ctx.lineJoin = "round";
      const mid = H / 2;
      for (let x = 0; x <= W; x += 1.5) { const t = x * 0.5 + phase * 40; const y = mid + cfg.fn(t) + sn(t * .08) * 1.5; x <= 1 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();
    });
    phase += 0.018; rafId = requestAnimationFrame(drawFrame);
  }
  return {
    start() { running = true; init(); if (rafId) cancelAnimationFrame(rafId); drawFrame(); },
    stop() { running = false; if (rafId) { cancelAnimationFrame(rafId); rafId = null; } }
  };
})();

/* ── SIMULATION CHANNELS (dark, glowing) ── */
const simAnim = (function () {
  const cfgs = [
    { id: "sim-eeg", color: "rgba(192,57,43,.9)", lw: 1.5, fn(t) { return Math.sin(t * .028) * 20 + Math.sin(t * .13) * 5.5 + Math.sin(t * .42) * 3 + Math.sin(t * .004) * 10 } },
    { id: "sim-eog", color: "rgba(109,40,217,.85)", lw: 1.4, fn(t) { return Math.sin(t * .013) * 24 + Math.sin(t * .045) * 8 + Math.sin(t * .002) * 5 } },
    { id: "sim-emg", color: "rgba(29,78,216,.8)", lw: 1.1, fn(t) { return Math.sin(t * .1) * 3 + (Math.random() - .5) * 9 } },
  ];
  let running = false, rafId = null, phaseSim = 0;
  const NL = 4096, noiseS = new Float32Array(NL); for (let i = 0; i < NL; i++) noiseS[i] = (Math.random() - .5) * 2;
  function sn(t) { const i = Math.floor(Math.abs(t)) % NL, j = (i + 1) % NL, f = t - Math.floor(t); return noiseS[i] * (1 - f) + noiseS[j] * f; }
  const ctxs = {};
  function init() {
    cfgs.forEach(cfg => {
      const canvas = document.getElementById(cfg.id); if (!canvas) return;
      const wrap = canvas.parentElement;
      const W = wrap.offsetWidth, H = wrap.offsetHeight || 86;
      canvas.width = W * devicePixelRatio; canvas.height = H * devicePixelRatio;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
      const c = canvas.getContext("2d"); c.scale(devicePixelRatio, devicePixelRatio); ctxs[cfg.id] = c;
    });
  }
  function drawFrame() {
    if (!running) return;
    cfgs.forEach(cfg => {
      const ctx = ctxs[cfg.id]; if (!ctx) return;
      const canvas = document.getElementById(cfg.id); if (!canvas) return;
      const W = canvas.parentElement.offsetWidth, H = 86;
      ctx.clearRect(0, 0, W, H); ctx.beginPath(); ctx.lineWidth = cfg.lw; ctx.strokeStyle = cfg.color; ctx.lineJoin = "round";
      const mid = H / 2;
      for (let x = 0; x <= W; x += 1.5) { const t = x * 0.55 + phaseSim * 55; const y = mid + cfg.fn(t) + sn(t * .06) * 2; x <= 1 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();
      const tipT = W * 0.55 + phaseSim * 55, tipY = mid + cfg.fn(tipT) + sn(tipT * .06) * 2;
      ctx.beginPath(); ctx.arc(W - 3, tipY, 3.5, 0, Math.PI * 2); ctx.fillStyle = cfg.color.replace(/[\d.]+\)$/, "1)"); ctx.fill();
      ctx.beginPath(); ctx.arc(W - 3, tipY, 7, 0, Math.PI * 2); ctx.fillStyle = cfg.color.replace(/[\d.]+\)$/, "0.15)"); ctx.fill();
    });
    phaseSim += 0.022; rafId = requestAnimationFrame(drawFrame);
  }
  return {
    start() { running = true; init(); if (rafId) cancelAnimationFrame(rafId); drawFrame(); },
    stop() { running = false; if (rafId) { cancelAnimationFrame(rafId); rafId = null; } }
  };
})();

/* ══════════════════════════════════════════════
   PSG SCANNING ANIMATION (shown during loading)
══════════════════════════════════════════════ */
const dzAnim = (function () {
  const canvas = document.getElementById('dz-canvas');
  const wrap = document.getElementById('dz-wave-wrap');
  const scanLine = document.getElementById('dz-scan-line');
  let W = 0, H = 0, ctx = null, raf = null, running = false;
  let scanX = 0, startTime = 0, duration = 0;

  const channels = [
    { yo: .22, amp: 14, freq: .028, freq2: .09, color: 'rgba(192,57,43,.8)', lw: 1.5 },
    { yo: .55, amp: 11, freq: .019, freq2: .06, color: 'rgba(29,78,216,.75)', lw: 1.4 },
    { yo: .82, amp: 5, freq: .06, freq2: .18, color: 'rgba(109,40,217,.7)', lw: 1.2 },
  ];

  const NL = 4096;
  const noiseTable = channels.map(() => { const a = new Float32Array(NL); for (let i = 0; i < NL; i++) a[i] = (Math.random() - .5) * 2; return a; });
  function sn(ch, t) { const i = Math.floor(Math.abs(t)) % NL, j = (i + 1) % NL, f = t - Math.floor(t); return noiseTable[ch][i] * (1 - f) + noiseTable[ch][j] * f; }

  function sig(ci, x) {
    const c = channels[ci];
    const t = x * c.freq;
    return (Math.sin(t) + Math.sin(t * c.freq2 / c.freq) * 0.35 + sn(ci, t * 0.5) * 0.25) * c.amp;
  }

  function initCanvas() {
    W = wrap.offsetWidth; H = wrap.offsetHeight || 110;
    canvas.width = W * devicePixelRatio;
    canvas.height = H * devicePixelRatio;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx = canvas.getContext('2d');
    ctx.scale(devicePixelRatio, devicePixelRatio);
  }

  function drawFrame(ts) {
    if (!running) return;
    if (!ctx) initCanvas();
    const elapsed = ts - startTime;
    const progress = Math.min(elapsed / duration, 1.0);
    const usableW = W - 70;
    scanX = 70 + usableW * progress;
    ctx.clearRect(0, 0, W, H);

    channels.forEach((c, ci) => {
      const yBase = H * c.yo;
      ctx.beginPath(); ctx.strokeStyle = c.color; ctx.lineWidth = c.lw; ctx.lineJoin = 'round';
      for (let x = 70; x <= scanX; x += 1.5) {
        const y = yBase - sig(ci, x - 70);
        x <= 70.5 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.beginPath(); ctx.strokeStyle = c.color.replace(/[\d.]+\)$/, '0.10)'); ctx.lineWidth = c.lw * 0.7;
      for (let x = scanX; x <= W; x += 2) {
        const y = yBase - sig(ci, x - 70);
        x <= scanX + .5 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    [.38, .67].forEach(yo => {
      ctx.beginPath(); ctx.strokeStyle = 'rgba(38,28,16,.06)'; ctx.lineWidth = 1;
      ctx.moveTo(70, H * yo); ctx.lineTo(W, H * yo); ctx.stroke();
    });

    scanLine.style.left = scanX + 'px';

    channels.forEach((c, ci) => {
      const yBase = H * c.yo;
      const tipY = yBase - sig(ci, scanX - 70);
      ctx.beginPath(); ctx.arc(scanX, tipY, 3, 0, Math.PI * 2);
      ctx.fillStyle = c.color.replace(/[\d.]+\)$/, '1)'); ctx.fill();
    });

    if (progress < 1.0) {
      raf = requestAnimationFrame(drawFrame);
    } else {
      setTimeout(() => { if (running) { startTime = performance.now(); raf = requestAnimationFrame(drawFrame); } }, 400);
    }
  }

  return {
    start(estimatedMs = 8000) {
      wrap.style.display = 'block'; document.getElementById('dz-idle').style.display = 'none';
      running = true; duration = estimatedMs; initCanvas(); startTime = performance.now();
      if (raf) cancelAnimationFrame(raf); raf = requestAnimationFrame(drawFrame);
    },
    stop() {
      running = false; if (raf) { cancelAnimationFrame(raf); raf = null; }
      wrap.style.display = 'none'; document.getElementById('dz-idle').style.display = 'block';
      if (ctx) ctx.clearRect(0, 0, W, H);
    }
  };
})();

/* ── SERVER PING ── */
async function checkServer() {
  try {
    const r = await fetch(API + "/health", { signal: AbortSignal.timeout(2200) });
    if (r.ok) { document.getElementById("srv-dot").className = "dot online"; document.getElementById("srv-txt").textContent = "Server online"; return; }
  } catch { }
  document.getElementById("srv-dot").className = "dot offline";
  document.getElementById("srv-txt").textContent = "Server offline — run sleep_server.py";
}
checkServer(); setInterval(checkServer, 5000);

/* ── DRAG & DROP ── */
const zone = document.getElementById("drop-zone");
zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over") });
zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
zone.addEventListener("drop", e => { e.preventDefault(); zone.classList.remove("drag-over"); const f = e.dataTransfer.files[0]; if (f) onFileSelected(f); });
document.getElementById("file-input").addEventListener("change", e => { if (e.target.files[0]) onFileSelected(e.target.files[0]); });

/* ── FILE SELECTED → SHOW REVIEW PANEL ── */
async function onFileSelected(file) {
  if (!file.name.toLowerCase().endsWith(".edf")) { showErr("Please upload an EDF file."); return; }
  hideErr(); _currentFile = file;

  // Play loading animation
  dzAnim.start(2000);
  await delay(2000);
  dzAnim.stop();

  document.getElementById("psg-fname").textContent = file.name;
  document.getElementById("meta-fname").textContent = file.name;
  const sizeMB = (file.size / 1e6).toFixed(1);
  document.getElementById("meta-size").textContent = sizeMB + " MB · EDF polysomnography";
  const estMin = Math.max(1, Math.round(file.size / 1e6 * 0.8));
  const h = Math.floor(estMin / 60), m = estMin % 60;
  document.getElementById("chip-dur").textContent = "~" + h + "h " + m + "m";
  document.getElementById("results").classList.remove("visible");
  document.getElementById("analysis-sim").classList.remove("visible");
  simAnim.stop();
  const review = document.getElementById("psg-review");
  review.classList.add("visible");
  setTimeout(() => review.scrollIntoView({ behavior: "smooth", block: "nearest" }), 100);
  setTimeout(() => previewChannels.start(), 200);
  resetSimSteps();
  updateWizardUI();
}



/* ── STEPS ── */
const STEPS = [
  { label: "Chargement EDF", pct: 10 },
  { label: "Prétraitement signal", pct: 25 },
  { label: "Segmentation en époques", pct: 42 },
  { label: "Inférence BiLSTM", pct: 88 },
  { label: "Calcul métriques AASM", pct: 100 },
];
function resetSimSteps() {
  for (let i = 0; i < 5; i++) { const el = document.getElementById("step-" + i); if (el) { el.className = "sim-step"; el.querySelector(".sim-step-icon").textContent = i + 1; } }
  const pf = document.getElementById("sim-progress"); if (pf) pf.style.width = "0%";
  const sp = document.getElementById("sim-pct"); if (sp) sp.textContent = "0%";
  const se = document.getElementById("sim-epochs-done"); if (se) se.textContent = "";
}
function setStep(idx) {
  for (let i = 0; i < 5; i++) {
    const el = document.getElementById("step-" + i); if (!el) continue;
    const icon = el.querySelector(".sim-step-icon");
    if (i < idx) { el.className = "sim-step done"; icon.textContent = "✓"; }
    else if (i === idx) { el.className = "sim-step active"; icon.textContent = i + 1; }
    else { el.className = "sim-step"; icon.textContent = i + 1; }
  }
  const sl = document.getElementById("sim-current-step"); if (sl) sl.textContent = STEPS[idx].label + "…";
  const prev = idx === 0 ? 0 : STEPS[idx - 1].pct;
  const pct = prev + Math.round((STEPS[idx].pct - prev) * 0.4);
  const pf = document.getElementById("sim-progress"); if (pf) pf.style.width = pct + "%";
  const sp = document.getElementById("sim-pct"); if (sp) sp.textContent = pct + "%";
}
function completeAllSteps() {
  for (let i = 0; i < 5; i++) { const el = document.getElementById("step-" + i); if (el) { el.className = "sim-step done"; el.querySelector(".sim-step-icon").textContent = "✓"; } }
  const pf = document.getElementById("sim-progress"); if (pf) pf.style.width = "100%";
  const sp = document.getElementById("sim-pct"); if (sp) sp.textContent = "100%";
  const sl = document.getElementById("sim-current-step"); if (sl) sl.textContent = "Terminé";
}

function switchTab(tab) {
  document.querySelectorAll('.app-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.app-section').forEach(s => s.classList.remove('active'));
  if (tab === 'doctor') {
    document.querySelector('.app-tab:nth-child(1)').classList.add('active');
    document.getElementById('section-doctor').classList.add('active');
  } else {
    document.querySelector('.app-tab:nth-child(2)').classList.add('active');
    document.getElementById('section-developer').classList.add('active');
    if (typeof editor === 'undefined') initDrawflow();
  }
}

/* ── START ANALYSIS ── */
async function startAnalysis() {
  if (!_currentFile) return;
  const checkedMods = Array.from(document.querySelectorAll('input[name="cfg-model"]:checked')).map(el => el.value);
  if (checkedMods.length === 0) { showErr("Veuillez sélectionner au moins un modèle."); return; }

  const btn1 = document.getElementById("btn-analyse");
  const btn2 = document.getElementById("btn-analyse-wiz");
  if (btn1) { btn1.classList.add("running"); btn1.innerHTML = "Analyse en cours…"; }
  if (btn2) { btn2.classList.add("running"); btn2.innerHTML = "Analyse en cours…"; }
  hideErr(); previewChannels.stop();
  const sim = document.getElementById("analysis-sim");
  sim.classList.add("visible");
  setTimeout(() => sim.scrollIntoView({ behavior: "smooth", block: "nearest" }), 100);
  simAnim.start();
  setStep(0); await delay(700);
  setStep(1); await delay(900);
  setStep(2); await delay(700);
  setStep(3);
  document.getElementById("sim-epochs-done").textContent = "Envoi vers le modèle…";
  const form = new FormData(); form.append("file", _currentFile);
  form.append("models", checkedMods.join(","));
  form.append("channels", document.querySelector('input[name="cfg-channels"]:checked').value);
  form.append("classes", document.querySelector('input[name="cfg-classes"]:checked').value);
  try {
    const res = await fetch(API + "/analyze", { method: "POST", body: form });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    if (data.results && data.results[0]) document.getElementById("sim-epochs-done").textContent = data.results[0].stages.length + " époques prédites";
    setStep(4); await delay(700);
    completeAllSteps(); await delay(600);
    simAnim.stop();
    sim.style.opacity = "0"; sim.style.transition = "opacity .4s";
    await delay(400);
    sim.classList.remove("visible"); sim.style.opacity = ""; sim.style.transition = "";
    renderResults(data);
    setTimeout(() => document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" }), 200);
  } catch (err) {
    simAnim.stop(); sim.classList.remove("visible");
    showErr(err.message || "Erreur serveur. Le serveur est-il démarré?");
  } finally {
    if (btn1) { btn1.classList.remove("running"); btn1.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Analyser le PSG'; }
    if (btn2) { btn2.classList.remove("running"); btn2.innerHTML = '▷ ANALYSER'; }
  }
}
/* ── WIZARD ── */
function selectCard(labelEl) {
  const input = labelEl.querySelector('input');
  if (input) input.checked = true;
  updateWizardUI();
}

function goToStep(step) {
  // Update header UI
  for (let i = 1; i <= 4; i++) {
    const s = document.getElementById("wiz-step-" + i);
    if (!s) continue;
    if (i < step) { s.className = "wiz-step done"; }
    else if (i === step) { s.className = "wiz-step active"; }
    else { s.className = "wiz-step"; }
  }

  // Show corresponding panel
  for (let i = 1; i <= 3; i++) {
    const p = document.getElementById("wiz-panel-" + i);
    if (p) {
      if (i === step) {
        p.classList.add("active");
      } else {
        p.classList.remove("active");
      }
    }
  }
}

function updateWizardUI() {
  const chInput = document.querySelector('input[name="cfg-channels"]:checked');
  const clsInput = document.querySelector('input[name="cfg-classes"]:checked');
  const ch = chInput ? chInput.value : "5";
  const cls = clsInput ? clsInput.value : "3";

  // Update card active states
  document.querySelectorAll('input[name="cfg-channels"]').forEach(el => {
    el.closest('.sel-card').classList.toggle('active', el.checked);
  });
  document.querySelectorAll('input[name="cfg-classes"]').forEach(el => {
    el.closest('.sel-card').classList.toggle('active', el.checked);
  });

  // Update summary in step 3
  const sumCh = document.getElementById("sum-ch");
  const sumCls = document.getElementById("sum-cls");
  const sumMod = document.getElementById("sum-mod");
  if (sumCh) sumCh.textContent = ch === "5" ? "5 canaux (EEG×2, EOG×2, EMG)" : "2 canaux (EEG×2)";
  if (sumCls) sumCls.textContent = cls === "5" ? "5 classes (Wake / N1 / N2 / N3 / REM)" : "3 classes (Wake / NREM / REM)";
  if (sumMod) sumMod.textContent = `stacking ${ch}ch ${cls}cls`;

  // Update file selected UI
  const sumFile = document.getElementById("sum-file");
  const btnAnaWiz = document.getElementById("btn-analyse-wiz");
  if (sumFile && btnAnaWiz) {
    if (_currentFile) {
      sumFile.textContent = _currentFile.name;
      sumFile.style.fontStyle = "normal";
      btnAnaWiz.classList.remove("disabled");
    } else {
      sumFile.textContent = "Aucun fichier sélectionné";
      sumFile.style.fontStyle = "italic";
      btnAnaWiz.classList.add("disabled");
    }
  }

  // Hide/show channels in simulation if needed
  const is2Ch = ch === "2";
  ["eog", "emg"].forEach(c => {
    const sRow = document.getElementById("sim-" + c)?.closest('.sim-ch-row');
    if (sRow) sRow.style.display = is2Ch ? "none" : "flex";
  });
}
// Init UI state
updateWizardUI();
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

/* ── RESET ── */
function resetApp() {
  _currentFile = null; previewChannels.stop(); simAnim.stop();
  document.getElementById("psg-review").classList.remove("visible");
  document.getElementById("analysis-sim").classList.remove("visible");
  document.getElementById("results").classList.remove("visible");
  document.getElementById("step2-lbl").style.display = "none";
  document.getElementById("osa-panel").style.display = "none";
  document.getElementById("osa-report").style.display = "none";
  document.getElementById("file-input").value = "";
  hideErr(); resetSimSteps();
  document.getElementById("drop-zone").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ── RENDER ── */
let _stages = null, _stagesInt = null, _classNames = null;
function renderResults(data) {
  const dyn = document.getElementById("dynamic-results");
  dyn.innerHTML = "";

  if (!data.results || data.results.length === 0) return;

  const primary = data.results[0];
  _stages = primary.stages;
  _stagesInt = primary.stages_int;
  _classNames = primary.stats.class_names;

  document.getElementById("results").classList.add("visible");
  document.getElementById("step2-lbl").style.display = "flex";
  document.getElementById("osa-panel").style.display = "block";
  document.getElementById("osa-report").style.display = "none";
  goToStep(4);

  data.results.forEach((res, i) => {
    const { model_info, stages, stats } = res;
    const wrapper = document.createElement("div");
    wrapper.style.marginBottom = "40px";
    wrapper.innerHTML = `
      <div style="font-family:var(--serif); font-size:20px; font-weight:700; color:var(--red); margin-bottom:16px; border-bottom:1px solid var(--border); padding-bottom:8px;">Comparaison: Modèle ${model_info.type}</div>
      <div class="sec-lbl">Hypnogram</div>
      <div class="hypno-wrap">
        <div class="hypno-hdr">
          <div class="hypno-title">Architecture du Sommeil</div>
          <div class="legend" id="leg-${i}"></div>
        </div>
        <canvas id="hypno-canvas-${i}" style="width:100%; height:180px; display:block; cursor:crosshair;"></canvas>
        <div class="time-axis" id="time-axis-${i}"></div>
      </div>
      <div class="sec-lbl">AASM Metrics</div>
      <div class="stats-grid" id="stats-grid-${i}"></div>
      <div class="sec-lbl">Stage Breakdown</div>
      <div class="breakdown-grid" id="breakdown-grid-${i}"></div>
      <div class="sec-lbl">Time Distribution</div>
      <div class="bar-chart" id="bar-chart-${i}"></div>
      <div class="alerts-section" id="alerts-section-${i}"></div>
    `;
    dyn.appendChild(wrapper);

    renderStats(stats, `stats-grid-${i}`);
    renderBreakdown(stats, `breakdown-grid-${i}`);
    renderBarChart(stats, `bar-chart-${i}`);
    renderAlerts(stats, `alerts-section-${i}`);
    requestAnimationFrame(() => requestAnimationFrame(() => renderHypnogram(stages, stats.class_names, `hypno-canvas-${i}`, `leg-${i}`, `time-axis-${i}`)));
  });
}
let _hypnoStages = null, _hypnoTooltipBound = false;
function renderHypnogram(stages, class_names, canvasId, legId, axisId) {
  _hypnoStages = stages;
  const canvas = document.getElementById(canvasId);
  const wrap = canvas.parentElement;
  const tip = document.getElementById("tooltip");

  // Update Legend
  const order = class_names || ["Wake", "NREM", "REM"];
  const leg = document.getElementById(legId);
  if (leg) {
    leg.innerHTML = order.map(s => `<div class="leg-item"><div class="leg-dot" style="background:${SC[s] || SC['NREM']}"></div>${s}</div>`).join('');
  }

  const W = wrap.clientWidth - 48, H = 180;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  canvas.width = Math.round(W * devicePixelRatio); canvas.height = Math.round(H * devicePixelRatio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(devicePixelRatio, devicePixelRatio);
  const PAD = { top: 18, right: 10, bottom: 8, left: 46 }, CW = W - PAD.left - PAD.right, CH = H - PAD.top - PAD.bottom;
  const rowH = CH / order.length;
  ctx.strokeStyle = "rgba(38,28,16,.07)"; ctx.lineWidth = 1;
  for (let i = 0; i <= order.length; i++) { const y = PAD.top + i * rowH; ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke(); }
  for (let i = 0; i <= 8; i++) { const x = PAD.left + CW * i / 8; ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, H - PAD.bottom); ctx.stroke(); }
  ctx.font = "500 10px 'DM Mono',monospace"; ctx.textAlign = "right";
  order.forEach((s, i) => { ctx.fillStyle = SC[s] || SC['NREM']; ctx.fillText(s, PAD.left - 8, PAD.top + i * rowH + rowH / 2 + 4); });
  const segW = CW / stages.length;
  stages.forEach((st, i) => { const yi = order.indexOf(st); ctx.globalAlpha = 0.65; ctx.fillStyle = SC[st] || SC['NREM']; ctx.fillRect(PAD.left + i * segW, PAD.top + yi * rowH + 1, segW + 0.5, rowH - 2); });
  ctx.globalAlpha = 1;
  ctx.beginPath(); ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.strokeStyle = "rgba(38,28,16,.5)";
  stages.forEach((st, i) => { const x = PAD.left + i * segW, yi = order.indexOf(st), y = PAD.top + yi * rowH + rowH / 2; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();
  const tot = stages.length * 0.5, ax = document.getElementById(axisId); ax.innerHTML = "";
  for (let t = 0; t <= 7; t++) { const m = Math.round(t * tot / 7), h = Math.floor(m / 60), mm = m % 60; ax.innerHTML += `<span>${h}h${String(mm).padStart(2, "0")}</span>`; }

  canvas.addEventListener("mousemove", e => {
    const r = canvas.getBoundingClientRect(), usable = r.width - PAD.left - PAD.right;
    const idxHover = Math.floor((e.clientX - r.left - PAD.left) / usable * stages.length);
    if (idxHover >= 0 && idxHover < stages.length) {
      const mins = (idxHover * 30) / 60, h = Math.floor(mins / 60), mm = Math.floor(mins % 60);
      tip.textContent = `${h}h${String(mm).padStart(2, "0")} · Époque ${idxHover + 1} · ${stages[idxHover]}`;
      tip.classList.add("visible"); tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY - 10) + "px";
    } else tip.classList.remove("visible");
  });
  canvas.addEventListener("mouseleave", () => tip.classList.remove("visible"));
}
function renderStats(s, containerId) {
  const g = document.getElementById(containerId); g.innerHTML = "";
  [{ lbl: "Sleep Efficiency", val: s.se, unit: "%", note: "Normal ≥85%", cls: s.se >= 85 ? "good" : s.se >= 75 ? "warn" : "danger" },
  { lbl: "Total Sleep Time", val: fmt(s.tst), unit: "min", note: (s.tst / 60).toFixed(1) + "h", cls: "" },
  { lbl: "Time in Bed", val: fmt(s.tib), unit: "min", note: (s.tib / 60).toFixed(1) + "h", cls: "" },
  { lbl: "Sleep Latency", val: s.sol, unit: "min", note: "Normal 10–20 min", cls: s.sol > 20 ? "warn" : s.sol < 5 ? "danger" : "good" },
  { lbl: "REM Latency", val: s.rem_latency != null ? s.rem_latency : "N/A", unit: "min", note: "Normal 90–120 min", cls: s.rem_latency != null && s.rem_latency < 60 ? "danger" : "" },
  { lbl: "WASO", val: s.waso, unit: "min", note: "Wake After Sleep Onset", cls: s.waso > 30 ? "warn" : "" },
  ].forEach(c => {
    const d = document.createElement("div"); d.className = "stat-card " + c.cls;
    d.innerHTML = `<div class="stat-lbl">${c.lbl}</div><div class="stat-val">${c.val}<span class="stat-unit">${c.unit}</span></div><div class="stat-note">${c.note}</div>`;
    g.appendChild(d);
  });
}
function renderBreakdown(s, containerId) {
  const g = document.getElementById(containerId); g.innerHTML = "";
  const order = s.class_names || ["Wake", "NREM", "REM"];
  order.forEach(st => {
    const d = document.createElement("div"); d.className = "stage-card";
    const color = SC[st] || SC['NREM'];
    const abbr = st === "Wake" ? "W" : st.replace("NREM", "NR").replace("REM", "R");
    d.innerHTML = `<div class="stage-sw" style="background:${color}18;color:${color};border:1.5px solid ${color}33">${abbr}</div><div><div class="stage-name">${st}</div><div class="stage-min" style="color:${color}">${fmt(s.stage_minutes[st])}<span style="font-size:11px;font-weight:400;color:var(--text3);font-family:var(--mono)"> min</span></div><div class="stage-pct">${s.stage_pct[st]}% ${st !== "Wake" ? "of TST" : "of TIB"}</div></div>`;
    g.appendChild(d);
  });
}
function renderBarChart(s, containerId) {
  const c = document.getElementById(containerId); c.innerHTML = "";
  const order = s.class_names || ["Wake", "NREM", "REM"];
  const max = Math.max(...order.map(k => s.stage_minutes[k]));
  order.forEach(k => {
    const pct = Math.round(s.stage_minutes[k] / max * 100);
    const d = document.createElement("div"); d.className = "bar-row";
    const color = SC[k] || SC['NREM'];
    d.innerHTML = `<div class="bar-lbl">${k.toUpperCase()}</div><div class="bar-track"><div class="bar-fill" id="bf-${k}" style="background:${color};width:0%">${fmt(s.stage_minutes[k])}m</div></div>`;
    c.appendChild(d);
    requestAnimationFrame(() => setTimeout(() => { const el = document.getElementById("bf-" + k); if (el) el.style.width = pct + "%"; }, 120));
  });
}
function renderAlerts(s, containerId) {
  const sec = document.getElementById(containerId); sec.innerHTML = "";
  if (!s.alerts?.length) return;
  const lbl = document.createElement("div"); lbl.className = "sec-lbl"; lbl.textContent = "Alertes Cliniques"; lbl.style.marginBottom = "16px";
  sec.appendChild(lbl);
  s.alerts.forEach(a => {
    const d = document.createElement("div"); d.className = "alert-item";
    d.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>${a}`;
    sec.appendChild(d);
  });
}
function fmt(n) { return n != null ? n : "–" }
function showErr(m) { const e = document.getElementById("error-bar"); e.textContent = "⚠  " + m; e.classList.add("visible") }
function hideErr() { document.getElementById("error-bar").classList.remove("visible") }

async function predictOSA() {
  const btn = document.getElementById("btn-predict-osa");
  btn.classList.add("running"); btn.innerHTML = "Analyse en cours...";
  try {
    const payload = {
      stages_int: _stagesInt,
      class_names: _classNames,
      clinical_data: {
        age: document.getElementById("osa-age").value,
        gender: document.getElementById("osa-gender").value,
        bmi: document.getElementById("osa-bmi").value,
        avgsat: document.getElementById("osa-avg-sat").value,
        minsat: document.getElementById("osa-min-sat").value,
        pctsa90h: document.getElementById("osa-pctsa90").value || null,
        pctsa85h: document.getElementById("osa-pctsa85").value || null,
        pctsa95h: document.getElementById("osa-pctsa95").value || null,
        ai_all: document.getElementById("osa-ai-all").value || null,
        ai_nrem: document.getElementById("osa-ai-nrem").value || null,
        ai_rem: document.getElementById("osa-ai-rem").value || null,
      }
    };
    const res = await fetch(API + "/predict_osa", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // ═══════════════════════════════════════════════
    //  RENDER FULL CLINICAL REPORT
    // ═══════════════════════════════════════════════
    const report = document.getElementById("osa-report");
    report.style.display = "block";

    // 1. Severity Badge
    const badge = document.getElementById("osa-sev-badge");
    badge.textContent = data.severity;
    const s = data.severity.toLowerCase();
    badge.className = "osa-severity-badge " + (s.includes("severe") ? "sev-severe" : s.includes("moderate") ? "sev-moderate" : s.includes("mild") ? "sev-mild" : "sev-normal");

    // Model badge
    const modelBadge = document.getElementById("osa-model-badge");
    if (modelBadge) modelBadge.textContent = "🧠 " + (data.model_used || "XGBoost");

    // 2. Probability Distribution
    if (data.probabilities) {
      const probaContainer = document.getElementById("osa-proba-bars");
      probaContainer.innerHTML = "";
      const classOrder = ["Normal", "Mild", "Moderate", "Severe"];
      const classColors = { Normal: "#059669", Mild: "#d97706", Moderate: "#ea580c", Severe: "#dc2626" };
      const classLabels = { Normal: "Normal", Mild: "Léger", Moderate: "Modéré", Severe: "Sévère" };

      classOrder.forEach(cls => {
        const pct = data.probabilities[cls] || 0;
        const pctStr = (pct * 100).toFixed(1);
        const isPredicted = cls.toLowerCase() === s;
        const row = document.createElement("div");
        row.className = "osa-proba-row";
        row.innerHTML = `
          <div class="osa-proba-label" style="color:${classColors[cls]};${isPredicted ? 'font-weight:800' : ''}">${classLabels[cls]}</div>
          <div class="osa-proba-track">
            <div class="osa-proba-fill proba-${cls.toLowerCase()}" style="width:0%" data-w="${Math.max(pct * 100, 1)}%">${pct > 0.08 ? pctStr + '%' : ''}</div>
          </div>
          <div class="osa-proba-pct" style="color:${classColors[cls]}">${pctStr}%</div>
        `;
        probaContainer.appendChild(row);
        requestAnimationFrame(() => setTimeout(() => {
          const fill = row.querySelector(".osa-proba-fill");
          fill.style.width = fill.getAttribute("data-w");
        }, 100));
      });
    }

    // 3. AASM Feature Cards
    if (data.aasm_features) {
      const featGrid = document.getElementById("osa-features-grid");
      featGrid.innerHTML = "";
      const feats = [
        { key: "tst_min", label: "TST", unit: "min", src: "timing", goodRange: [360, 540], warnRange: [300, 600] },
        { key: "se_pct", label: "Efficacité", unit: "%", src: "timing", goodRange: [85, 100], warnRange: [75, 100] },
        { key: "sol_min", label: "Latence", unit: "min", src: "timing", goodRange: [10, 20], warnRange: [5, 30] },
        { key: "waso_min", label: "WASO", unit: "min", src: "timing", goodRange: [0, 30], warnRange: [0, 60] },
        { key: "N1_pct", label: "N1", unit: "%", src: "stages" },
        { key: "N2_pct", label: "N2", unit: "%", src: "stages" },
        { key: "N3_pct", label: "N3", unit: "%", src: "stages", goodRange: [15, 30], warnRange: [10, 40] },
        { key: "REM_pct", label: "REM", unit: "%", src: "stages", goodRange: [20, 25], warnRange: [15, 30] },
        { key: "rem_latency_min", label: "Lat. REM", unit: "min", src: "latencies", goodRange: [70, 120], warnRange: [60, 150] },
        { key: "frag_index", label: "Fragmentation", unit: "/h", src: "fragmentation" },
        { key: "n_wake_bouts", label: "Éveils", unit: "", src: "fragmentation" },
        { key: "n_rem_cycles", label: "Cycles REM", unit: "", src: "fragmentation" },
      ];

      feats.forEach(f => {
        const val = data.aasm_features[f.src] ? data.aasm_features[f.src][f.key] : null;
        if (val === null || val === undefined) return;
        let cls = "";
        if (f.goodRange) {
          if (val >= f.goodRange[0] && val <= f.goodRange[1]) cls = "good";
          else if (f.warnRange && (val < f.warnRange[0] || val > f.warnRange[1])) cls = "danger";
          else cls = "warn";
        }
        const card = document.createElement("div");
        card.className = "osa-feat-card " + cls;
        card.innerHTML = `
          <div class="osa-feat-label">${f.label}</div>
          <div class="osa-feat-value">${typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(1)) : val}<span class="osa-feat-unit">${f.unit}</span></div>
        `;
        featGrid.appendChild(card);
      });
    }

    // 4. SHAP Waterfall
    const shaps = document.getElementById("osa-shaps");
    shaps.innerHTML = "";
    const shapData = data.shap_explanations || [];
    if (shapData.length > 0) {
      const maxImp = Math.max(...shapData.map(x => Math.abs(x.impact)), 0.1);

      shapData.forEach((sh, idx) => {
        const isPos = sh.impact > 0;
        const pct = (Math.abs(sh.impact) / maxImp * 45).toFixed(1);
        const row = document.createElement("div");
        row.className = "shap-item";
        row.style.animationDelay = (idx * 50) + "ms";
        
        // Human-readable feature names
        const featureNames = {
          "ai_all": "Index Arousal Total", "ai_nrem": "Arousal NREM", "ai_rem": "Arousal REM",
          "avgsat": "SpO₂ Moyenne", "minsat": "SpO₂ Minimum", "pctsa90h": "% Temps <90%",
          "pctsa85h": "% Temps <85%", "pctsa95h": "% Temps <95%",
          "sleep_efficiency": "Efficacité Sommeil", "waso_min": "WASO",
          "frag_index": "Fragmentation", "tst_min": "TST", "sol_min": "Latence",
          "N1_pct": "% N1", "N2_pct": "% N2", "N3_pct": "% N3", "REM_pct": "% REM",
          "rem_latency_min": "Latence REM", "bmi_s2": "IMC", "age_s2": "Âge", "gender": "Sexe",
          "hypoxia_score": "Score Hypoxie", "arousal_frag": "Arousal × Frag.",
          "sat_drop": "Chute SpO₂", "bmi_arousal": "IMC × Arousal",
          "n3_suppression": "Suppression N3", "waso_arousal": "WASO × Arousal",
          "slpeffp": "Eff. Sommeil (PSG)", "n_wake_bouts": "Nb Éveils",
          "nrem_rem_ratio": "Ratio NREM/REM", "light_deep_ratio": "Ratio Léger/Profond",
        };
        const displayName = featureNames[sh.feature] || sh.feature.replace(/_/g, " ");
        
        row.innerHTML = `<div class="shap-lbl">${displayName} <span style="color:var(--text3)">(${sh.value})</span></div>
                         <div class="shap-bar-wrap">
                           <div class="shap-bar ${isPos ? 'shap-pos' : 'shap-neg'}" style="width:0%" data-w="${pct}%"></div>
                         </div>
                         <div class="shap-val" style="color:${isPos ? 'var(--red)' : '#1d4ed8'}">${isPos ? '+' : ''}${sh.impact.toFixed(3)}</div>`;
        shaps.appendChild(row);
        requestAnimationFrame(() => setTimeout(() => {
          const b = row.querySelector(".shap-bar");
          b.style.width = b.getAttribute("data-w");
        }, 100 + idx * 30));
      });
    }

    // 5. Clinical Interpretation
    const interpSection = document.getElementById("osa-interp-section");
    interpSection.innerHTML = "";
    if (data.interpretation && data.interpretation.length > 0) {
      const title = document.createElement("div");
      title.className = "osa-interp-title";
      title.textContent = "Interprétation Clinique";
      interpSection.appendChild(title);

      const icons = {
        warning: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        danger: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        info: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
      };

      data.interpretation.forEach((item, idx) => {
        const el = document.createElement("div");
        el.className = `osa-interp-item type-${item.type}`;
        el.style.animationDelay = (idx * 80) + "ms";
        el.innerHTML = `<span class="osa-interp-icon">${icons[item.type] || icons.info}</span>${item.text}`;
        interpSection.appendChild(el);
      });
    }

    setTimeout(() => report.scrollIntoView({ behavior: "smooth", block: "nearest" }), 100);

  } catch (e) {
    showErr("Step 2 Error: " + e.message);
  } finally {
    btn.classList.remove("running");
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/><path d="M12 8v4l3 3"/></svg> Générer le Rapport Clinique';
  }
}

let _rzT;
window.addEventListener("resize", () => {
  clearTimeout(_rzT); _rzT = setTimeout(() => {
    if (_stages && _hypnoStages) {
      // Optional: redraw all hypnograms if resize
    }
  }, 200);
});

/* ── DRAWFLOW LOGIC ── */
let editor;
function initDrawflow() {
  const id = document.getElementById("drawflow");
  if (!id) return;
  editor = new Drawflow(id);
  editor.reroute = true;
  editor.start();

  // Listen for changes to update confidence preview
  editor.on('connectionCreated', () => updatePipelinePreview());
  editor.on('connectionRemoved', () => updatePipelinePreview());
  editor.on('nodeCreated', () => setTimeout(updatePipelinePreview, 100));
  editor.on('nodeRemoved', () => updatePipelinePreview());
}

function clearPipeline() {
  if (!editor) return;
  editor.clear();
  document.getElementById("pipe-confidence-preview").style.display = "none";
  document.getElementById("pipe-conf-cards").innerHTML = "";
  const pr = document.getElementById("pipeline-results");
  pr.innerHTML = ""; pr.style.display = "none";
}

function getConfidence(ch, cls, model) {
  let base = 0;
  if (ch === '5' && cls === '5') base = 96;
  else if (ch === '5' && cls === '3') base = 98;
  else if (ch === '2' && cls === '5') base = 85;
  else if (ch === '2' && cls === '3') base = 91;
  // Model modifier
  if (model === 'Stacking') base = Math.min(base + 1, 99);
  else if (model === 'Transformer') base = Math.max(base - 2, 70);
  else if (model === 'CNN') base = Math.max(base - 1, 70);
  return base;
}

function getConfColor(conf) {
  if (conf >= 95) return '#059669';
  if (conf >= 90) return '#0d9488';
  if (conf >= 85) return '#d97706';
  return '#dc2626';
}

function updatePipelinePreview() {
  if (!editor) return;
  const exportdata = editor.export();
  const nodes = exportdata.drawflow.Home.data;

  const modelNodes = [];
  for (const id in nodes) {
    if (nodes[id].name.startsWith('model_')) modelNodes.push(id);
  }

  if (modelNodes.length === 0) {
    document.getElementById("pipe-confidence-preview").style.display = "none";
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

  const container = document.getElementById("pipe-conf-cards");
  const preview = document.getElementById("pipe-confidence-preview");

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
        <span style="font-size:9px; text-transform:uppercase; letter-spacing:1px; color:${color}; background:${color}18; padding:3px 8px; border-radius:6px; font-weight:700;">${conf >= 95 ? 'Excellent' : conf >= 90 ? 'Élevé' : conf >= 85 ? 'Moyen' : 'Faible'}</span>
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
  if (editor.editor_mode === 'fixed') return;

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

/* ── GRAPH TRAVERSAL: walk backwards from a node to find all paths to Patient Data ── */
function tracePathsBack(nodeId, nodes, currentCfg) {
  const nid = String(nodeId);
  const node = nodes[nid];
  if (!node) { console.warn('[trace] node not found:', nid); return []; }

  // Clone config so branches don't interfere
  const cfg = { ...currentCfg };

  // Collect config from this node
  if (node.name === '2_channels') cfg.channels = '2';
  if (node.name === '5_channels') cfg.channels = '5';
  if (node.name === '3_classes') cfg.classes = '3';
  if (node.name === '5_classes') cfg.classes = '5';

  // If we reached a Patient Data node, we have a complete path
  if (node.name === 'patient_data') {
    const nodeEl = document.querySelector(`#node-${nid}`);
    const fileInput = nodeEl ? nodeEl.querySelector('.df-file-input') : null;
    const file = fileInput && fileInput.files && fileInput.files[0];
    console.log(`[trace] ✓ reached patient_data #${nid}, file=${file?file.name:'NONE'}, cfg=`, cfg);
    return [{ ...cfg, dataNodeId: nid, file: file, fileName: file ? file.name : null }];
  }

  // Walk backward through all input connections
  const paths = [];
  const inputKeys = Object.keys(node.inputs || {});
  console.log(`[trace] node #${nid} (${node.name}) has ${inputKeys.length} input ports`);
  for (const key of inputKeys) {
    const conns = node.inputs[key].connections || [];
    console.log(`[trace]   port ${key}: ${conns.length} connections →`, conns.map(c=>c.node));
    for (const conn of conns) {
      const parentId = String(conn.node);
      const subPaths = tracePathsBack(parentId, nodes, { ...cfg });
      paths.push(...subPaths);
    }
  }
  return paths;
}

function getModelType(nodeName) {
  if (nodeName === 'model_bilstm') return 'LSTM';
  if (nodeName === 'model_cnn') return 'CNN';
  if (nodeName === 'model_transformer') return 'Transformer';
  if (nodeName === 'model_stacking') return 'Stacking';
  return null;
}

async function startPipelineAnalysis() {
  const exportdata = editor.export();
  const nodes = exportdata.drawflow.Home.data;

  // 1. Find all model nodes
  const modelNodes = [];
  for (const id in nodes) {
    if (nodes[id].name.startsWith('model_')) modelNodes.push(id);
  }
  if (modelNodes.length === 0) { showErr("Pipeline: Ajoutez au moins un Modèle IA."); return; }

  // 2. For each model node, trace ALL paths back to Patient Data
  const jobs = []; // { file, fileName, channels, classes, model, dataNodeId }
  const errors = [];

  for (const mId of modelNodes) {
    const mType = getModelType(nodes[mId].name);
    const paths = tracePathsBack(mId, nodes, { channels: '5', classes: '3' });

    if (paths.length === 0) {
      errors.push(`Le modèle "${mType}" (noeud #${mId}) n'est connecté à aucune source de données.`);
      continue;
    }

    for (const p of paths) {
      if (!p.file) {
        errors.push(`Le modèle "${mType}" est connecté au noeud Patient Data #${p.dataNodeId} mais aucun fichier n'est sélectionné.`);
        continue;
      }
      jobs.push({ file: p.file, fileName: p.fileName, channels: p.channels, classes: p.classes, model: mType, dataNodeId: p.dataNodeId });
    }
  }

  if (errors.length > 0 && jobs.length === 0) {
    showErr("Pipeline: " + errors[0]);
    return;
  }

  // 3. Group jobs by (file + channels + classes) to minimize API calls
  const groupKey = (j) => `${j.dataNodeId}|${j.channels}|${j.classes}`;
  const groups = {};
  for (const j of jobs) {
    const k = groupKey(j);
    if (!groups[k]) groups[k] = { file: j.file, fileName: j.fileName, channels: j.channels, classes: j.classes, models: [] };
    if (!groups[k].models.includes(j.model)) groups[k].models.push(j.model);
  }

  console.log(`[pipeline] ${jobs.length} jobs found, ${Object.keys(groups).length} groups:`);
  jobs.forEach(j => console.log(`  → ${j.fileName} | ${j.channels}ch | ${j.classes}cls | ${j.model}`));
  console.log('[pipeline] groups:', groups);

  const btn = document.getElementById("btn-run-pipeline");
  btn.classList.add("running"); btn.innerHTML = "Analyse en cours…";
  hideErr();

  const dyn = document.getElementById("pipeline-results");
  dyn.innerHTML = "";
  dyn.style.display = "block";

  let globalIdx = 0;
  const groupKeys = Object.keys(groups);

  try {
    for (let gi = 0; gi < groupKeys.length; gi++) {
      const grp = groups[groupKeys[gi]];

      // Show progress
      btn.innerHTML = `Analyse ${gi + 1}/${groupKeys.length}…`;

      const form = new FormData();
      form.append("file", grp.file);
      form.append("models", grp.models.join(","));
      form.append("channels", grp.channels);
      form.append("classes", grp.classes);

      const res = await fetch(API + "/analyze", { method: "POST", body: form });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      // Render a group header
      const groupHeader = document.createElement("div");
      groupHeader.style.cssText = "font-family:var(--serif); font-size:22px; font-weight:900; color:var(--text); margin:32px 0 8px; padding-bottom:10px; border-bottom:2px solid var(--border);";
      groupHeader.innerHTML = `📁 ${grp.fileName} <span style="font-size:13px; font-weight:500; color:var(--text2); font-family:var(--mono); margin-left:12px;">${grp.channels}ch · ${grp.classes}cls</span>`;
      dyn.appendChild(groupHeader);

      // Show warnings if any
      const relatedErrors = errors.filter(e => e.includes(grp.fileName));
      if (relatedErrors.length > 0) {
        const warn = document.createElement("div");
        warn.style.cssText = "font-size:11px; color:#b45309; background:#fef3c7; padding:8px 12px; border-radius:8px; margin-bottom:16px;";
        warn.textContent = "⚠ " + relatedErrors.join(" | ");
        dyn.appendChild(warn);
      }

      data.results.forEach((res) => {
        const i = globalIdx++;
        const wrapper = document.createElement("div");
        wrapper.style.marginBottom = "40px";
        wrapper.innerHTML = `
          <div style="font-family:var(--serif); font-size:18px; font-weight:700; color:var(--red); margin-bottom:16px; display:flex; align-items:center; gap:10px;">
            🧠 ${res.model_info.type}
            <span style="font-size:11px; font-weight:500; color:var(--text3); font-family:var(--mono);">${res.model_info.channels}ch · ${res.model_info.classes}cls</span>
          </div>
          <div class="sec-lbl">Hypnogram</div>
          <div class="hypno-wrap">
            <div class="hypno-hdr">
              <div class="hypno-title">Architecture du Sommeil</div>
              <div class="legend" id="pipe-leg-${i}"></div>
            </div>
            <canvas id="pipe-hypno-canvas-${i}" style="width:100%; height:180px; display:block; cursor:crosshair;"></canvas>
            <div class="time-axis" id="pipe-time-axis-${i}"></div>
          </div>
          <div class="sec-lbl">AASM Metrics</div>
          <div class="stats-grid" id="pipe-stats-${i}"></div>
          <div class="sec-lbl">Stage Breakdown</div>
          <div class="breakdown-grid" id="pipe-breakdown-${i}" style="margin-top:20px;"></div>
        `;
        dyn.appendChild(wrapper);
        renderStats(res.stats, `pipe-stats-${i}`);
        renderBreakdown(res.stats, `pipe-breakdown-${i}`);
        requestAnimationFrame(() => requestAnimationFrame(() => renderHypnogram(res.stages, res.stats.class_names, `pipe-hypno-canvas-${i}`, `pipe-leg-${i}`, `pipe-time-axis-${i}`)));
      });
    }

    setTimeout(() => dyn.scrollIntoView({ behavior: "smooth", block: "start" }), 200);
  } catch (err) {
    showErr("Pipeline Error: " + err.message);
  } finally {
    btn.classList.remove("running");
    btn.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Lancer l\'Analyse du Pipeline';
  }
}
/* ── ROUTING LOGIC ── */
function showPage(pageId) {
  document.querySelectorAll('.app-page').forEach(p => p.classList.remove('active'));
  document.getElementById(pageId + '-page').classList.add('active');
}

function mockLogin(role) {
  // role is either 'doctor' or 'developer'
  localStorage.setItem('hypnora_role', role);
  showPage('dashboard');
  
  // Configure UI based on role
  const tabDoc = document.querySelector('.app-tab[onclick="switchTab(\'doctor\')"]');
  const tabDev = document.querySelector('.app-tab[onclick="switchTab(\'developer\')"]');
  
  if(role === 'doctor') {
    tabDev.style.display = 'none';
    tabDoc.style.display = 'flex';
    switchTab('doctor');
  } else {
    tabDoc.style.display = 'none';
    tabDev.style.display = 'flex';
    switchTab('developer');
  }
}

function doLogin() {
  const email = document.getElementById('login-email').value.toLowerCase();
  if(email.includes('dev')) {
    mockLogin('developer');
  } else {
    mockLogin('doctor');
  }
}

// Check if already logged in (mock)
window.addEventListener('DOMContentLoaded', () => {
  const role = localStorage.getItem('hypnora_role');
  if(role) {
    mockLogin(role);
  }
});

function logout() {
  localStorage.removeItem('hypnora_role');
  showPage('landing');
}
