import { useEffect, useRef } from 'react';
import './AnalysisSimulation.css';

const STEPS = [
  { label: 'Chargement et décodage EDF', sub: 'Lecture des canaux · Résolution des métadonnées' },
  { label: 'Prétraitement du signal', sub: 'Rééchantillonnage 125Hz→100Hz · Normalisation z-score par époque' },
  { label: 'Segmentation en époques de 30 secondes', sub: 'Découpage · Empilement → (N, 5, 3000)' },
  { label: 'Inférence modèle', sub: 'Prédiction par lots de 64 · Wake / NREM / REM' },
  { label: 'Calcul des métriques AASM', sub: 'TST · WASO · SE · Latence REM · Architecture du sommeil' },
];

const SIM_CHANNELS = [
  { label: 'EEG', color: 'rgba(192,57,43,.9)', lw: 1.5, fn: (t) => Math.sin(t * .028) * 20 + Math.sin(t * .13) * 5.5 + Math.sin(t * .42) * 3 + Math.sin(t * .004) * 10 },
  { label: 'EOG', color: 'rgba(109,40,217,.85)', lw: 1.4, fn: (t) => Math.sin(t * .013) * 24 + Math.sin(t * .045) * 8 + Math.sin(t * .002) * 5 },
  { label: 'EMG', color: 'rgba(29,78,216,.8)', lw: 1.1, fn: (t) => Math.sin(t * .1) * 3 + (Math.random() - .5) * 9 },
];

const AnalysisSimulation = ({ activeStep = 0, progress = 0, visible = false }) => {
  const canvasRefs = useRef([]);
  const rafRef = useRef(null);
  const phaseRef = useRef(0);

  useEffect(() => {
    if (!visible) return;

    const NL = 4096;
    const noiseS = new Float32Array(NL);
    for (let i = 0; i < NL; i++) noiseS[i] = (Math.random() - .5) * 2;
    const sn = (t) => { const i = Math.floor(Math.abs(t)) % NL; const j = (i + 1) % NL; const f = t - Math.floor(t); return noiseS[i] * (1 - f) + noiseS[j] * f; };

    const ctxs = [];
    const dims = [];

    canvasRefs.current.forEach((canvas, idx) => {
      if (!canvas) return;
      const wrap = canvas.parentElement;
      const W = wrap.offsetWidth;
      const H = wrap.offsetHeight || 86;
      canvas.width = W * devicePixelRatio;
      canvas.height = H * devicePixelRatio;
      canvas.style.width = W + 'px';
      canvas.style.height = H + 'px';
      const ctx = canvas.getContext('2d');
      ctx.scale(devicePixelRatio, devicePixelRatio);
      ctxs[idx] = ctx;
      dims[idx] = { W, H };
    });

    const draw = () => {
      SIM_CHANNELS.forEach((cfg, idx) => {
        const ctx = ctxs[idx];
        if (!ctx) return;
        const { W, H } = dims[idx];
        ctx.clearRect(0, 0, W, H);
        ctx.beginPath();
        ctx.lineWidth = cfg.lw;
        ctx.strokeStyle = cfg.color;
        ctx.lineJoin = 'round';
        const mid = H / 2;
        for (let x = 0; x <= W; x += 1.5) {
          const t = x * 0.55 + phaseRef.current * 55;
          const y = mid + cfg.fn(t) + sn(t * .06) * 2;
          x <= 1 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        // Tip dot
        const tipT = W * 0.55 + phaseRef.current * 55;
        const tipY = mid + cfg.fn(tipT) + sn(tipT * .06) * 2;
        ctx.beginPath(); ctx.arc(W - 3, tipY, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = cfg.color.replace(/[\d.]+\)$/, '1)'); ctx.fill();
        ctx.beginPath(); ctx.arc(W - 3, tipY, 7, 0, Math.PI * 2);
        ctx.fillStyle = cfg.color.replace(/[\d.]+\)$/, '0.15)'); ctx.fill();
      });
      phaseRef.current += 0.022;
      rafRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [visible]);

  if (!visible) return null;

  const chLabels = ['EEG', 'EOG', 'EMG'];
  const chColors = ['#c0392b', '#6d28d9', '#1d4ed8'];

  return (
    <div id="analysis-sim" className="visible">
      <div className="sim-header">
        <div className="sim-title">
          <div className="sim-spinner-el" />
          Analyse en cours
        </div>
        <div className="sim-step-label">{STEPS[activeStep]?.label || 'Initialisation'}…</div>
      </div>
      <div className="sim-channels">
        {SIM_CHANNELS.map((ch, i) => (
          <div className="sim-ch-row" key={i}>
            <div className="sim-ch-label">
              <div className="ch-name" style={{ color: chColors[i] }}>{chLabels[i]}</div>
              <div className="ch-amp">μV · 100Hz</div>
            </div>
            <div className="sim-ch-canvas-wrap-dark">
              <canvas ref={(el) => canvasRefs.current[i] = el} />
            </div>
          </div>
        ))}
      </div>
      <div className="sim-steps">
        {STEPS.map((step, i) => (
          <div className={`sim-step ${i < activeStep ? 'done' : i === activeStep ? 'active' : ''}`} key={i}>
            <div className="sim-step-icon">{i < activeStep ? '✓' : i + 1}</div>
            <div>
              <div className="sim-step-text">{step.label}</div>
              <div className="sim-step-sub">{step.sub}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="sim-progress-wrap">
        <div className="sim-progress-track">
          <div className="sim-progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <div className="sim-progress-label">
          <span>{progress}%</span>
          <span></span>
        </div>
      </div>
    </div>
  );
};

export default AnalysisSimulation;
