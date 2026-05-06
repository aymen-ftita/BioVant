/**
 * Wizard Component Logic
 * Includes Step-by-Step navigation and PSG Scanning Animation.
 */

/* ── SCANNING ANIMATION ── */
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
    if (!wrap) return;
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
    });

    scanLine.style.left = scanX + 'px';

    if (progress < 1.0) {
      raf = requestAnimationFrame(drawFrame);
    } else {
      setTimeout(() => { if (running) { startTime = performance.now(); raf = requestAnimationFrame(drawFrame); } }, 400);
    }
  }

  return {
    start(estimatedMs = 8000) {
      if (!wrap) return;
      wrap.style.display = 'block'; document.getElementById('dz-idle').style.display = 'none';
      running = true; duration = estimatedMs; initCanvas(); startTime = performance.now();
      if (raf) cancelAnimationFrame(raf); raf = requestAnimationFrame(drawFrame);
    },
    stop() {
      running = false; if (raf) { cancelAnimationFrame(raf); raf = null; }
      if (wrap) wrap.style.display = 'none'; 
      const idle = document.getElementById('dz-idle');
      if (idle) idle.style.display = 'block';
      if (ctx) ctx.clearRect(0, 0, W, H);
    }
  };
})();

/* ── WIZARD NAVIGATION ── */
function goToStep(step) {
  for (let i = 1; i <= 4; i++) {
    const s = document.getElementById("wiz-step-" + i);
    if (!s) continue;
    if (i < step) { s.className = "wiz-step done"; }
    else if (i === step) { s.className = "wiz-step active"; }
    else { s.className = "wiz-step"; }
  }

  for (let i = 1; i <= 3; i++) {
    const p = document.getElementById("wiz-panel-" + i);
    if (p) {
      if (i === step) p.classList.add("active");
      else p.classList.remove("active");
    }
  }
}

function selectCard(labelEl) {
  const input = labelEl.querySelector('input');
  if (input) input.checked = true;
  updateWizardUI();
}

function updateWizardUI() {
  const chInput = document.querySelector('input[name="cfg-channels"]:checked');
  const clsInput = document.querySelector('input[name="cfg-classes"]:checked');
  const ch = chInput ? chInput.value : "5";
  const cls = clsInput ? clsInput.value : "3";

  document.querySelectorAll('input[name="cfg-channels"]').forEach(el => {
    el.closest('.sel-card').classList.toggle('active', el.checked);
  });
  document.querySelectorAll('input[name="cfg-classes"]').forEach(el => {
    el.closest('.sel-card').classList.toggle('active', el.checked);
  });

  const sumCh = document.getElementById("sum-ch");
  const sumCls = document.getElementById("sum-cls");
  const sumMod = document.getElementById("sum-mod");
  if (sumCh) sumCh.textContent = ch === "5" ? "5 canaux (EEG×2, EOG×2, EMG)" : "2 canaux (EEG×2)";
  if (sumCls) sumCls.textContent = cls === "5" ? "5 classes (Wake / N1 / N2 / N3 / REM)" : "3 classes (Wake / NREM / REM)";
  if (sumMod) sumMod.textContent = `stacking ${ch}ch ${cls}cls`;

  const sumFile = document.getElementById("sum-file");
  const btnAnaWiz = document.getElementById("btn-analyse-wiz");
  if (sumFile && btnAnaWiz) {
    if (window._currentFile) {
      sumFile.textContent = window._currentFile.name;
      sumFile.style.fontStyle = "normal";
      btnAnaWiz.classList.remove("disabled");
    } else {
      sumFile.textContent = "Aucun fichier sélectionné";
      sumFile.style.fontStyle = "italic";
      btnAnaWiz.classList.add("disabled");
    }
  }
}

/* ── SERVER HEALTH ── */
async function checkServer() {
  const dot = document.getElementById("srv-dot");
  const txt = document.getElementById("srv-txt");
  if (!dot) return;
  try {
    const r = await fetch("http://localhost:5000/health", { signal: AbortSignal.timeout(2000) });
    if (r.ok) { dot.className = "dot online"; txt.textContent = "Server online"; return; }
  } catch { }
  dot.className = "dot offline";
  txt.textContent = "Server offline — run sleep_server.py";
}

document.addEventListener('DOMContentLoaded', () => {
    checkServer();
    setInterval(checkServer, 5000);
    updateWizardUI();
});
