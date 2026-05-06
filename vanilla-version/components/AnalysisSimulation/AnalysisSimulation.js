/**
 * AnalysisSimulation Component Logic
 * Handles the animated progress and steps during PSG analysis.
 */
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

const STEPS = [
  { label: "Chargement EDF", pct: 10 },
  { label: "Prétraitement signal", pct: 25 },
  { label: "Segmentation en époques", pct: 42 },
  { label: "Inférence BiLSTM", pct: 88 },
  { label: "Calcul métriques AASM", pct: 100 },
];

function resetSimSteps() {
  for (let i = 0; i < 5; i++) {
    const el = document.getElementById("step-" + i);
    if (el) {
      el.className = "sim-step";
      const icon = el.querySelector(".sim-step-icon");
      if (icon) icon.textContent = i + 1;
    }
  }
  const pf = document.getElementById("sim-progress"); if (pf) pf.style.width = "0%";
  const sp = document.getElementById("sim-pct"); if (sp) sp.textContent = "0%";
  const se = document.getElementById("sim-epochs-done"); if (se) se.textContent = "";
}

function setStep(idx) {
  for (let i = 0; i < 5; i++) {
    const el = document.getElementById("step-" + i); if (!el) continue;
    const icon = el.querySelector(".sim-step-icon");
    if (i < idx) { el.className = "sim-step done"; if (icon) icon.textContent = "✓"; }
    else if (i === idx) { el.className = "sim-step active"; if (icon) icon.textContent = i + 1; }
    else { el.className = "sim-step"; if (icon) icon.textContent = i + 1; }
  }
  const sl = document.getElementById("sim-current-step"); if (sl) sl.textContent = STEPS[idx].label + "…";
  const prev = idx === 0 ? 0 : STEPS[idx - 1].pct;
  const pct = prev + Math.round((STEPS[idx].pct - prev) * 0.4);
  const pf = document.getElementById("sim-progress"); if (pf) pf.style.width = pct + "%";
  const sp = document.getElementById("sim-pct"); if (sp) sp.textContent = pct + "%";
}

function completeAllSteps() {
  for (let i = 0; i < 5; i++) {
    const el = document.getElementById("step-" + i);
    if (el) {
      el.className = "sim-step done";
      const icon = el.querySelector(".sim-step-icon");
      if (icon) icon.textContent = "✓";
    }
  }
  const pf = document.getElementById("sim-progress"); if (pf) pf.style.width = "100%";
  const sp = document.getElementById("sim-pct"); if (sp) sp.textContent = "100%";
  const sl = document.getElementById("sim-current-step"); if (sl) sl.textContent = "Terminé";
}
