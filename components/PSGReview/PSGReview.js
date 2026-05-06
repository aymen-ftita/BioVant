/**
 * PSGReview Component Logic
 * Handles the preview of PSG signals before full analysis.
 */
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

function resetApp() {
    window._currentFile = null;
    if (typeof previewChannels !== 'undefined') previewChannels.stop();
    if (typeof simAnim !== 'undefined') simAnim.stop();
    
    document.getElementById("psg-review").classList.remove("visible");
    const sim = document.getElementById("analysis-sim");
    if (sim) sim.classList.remove("visible");
    const res = document.getElementById("results");
    if (res) res.classList.remove("visible");
    
    const fileInput = document.getElementById("file-input");
    if (fileInput) fileInput.value = "";
    
    if (typeof resetSimSteps === 'function') resetSimSteps();
    
    const dropZone = document.getElementById("drop-zone");
    if (dropZone) dropZone.scrollIntoView({ behavior: "smooth", block: "nearest" });
    
    if (typeof updateWizardUI === 'function') updateWizardUI();
}
