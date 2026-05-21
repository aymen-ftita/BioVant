import { useEffect, useRef } from 'react';
import './PSGReview.css';

const CHANNEL_CFGS = [
  { label: 'EEG', color: 'rgba(192,57,43,.8)', lw: 1.4, fn: (t) => Math.sin(t * .025) * 18 + Math.sin(t * .14) * 5 + Math.sin(t * .38) * 2.5 + Math.sin(t * .003) * 8, hz: '125→100 Hz' },
  { label: 'EEG (sec)', color: 'rgba(192,57,43,.45)', lw: 1.2, fn: (t) => Math.sin(t * .022 + 1.2) * 15 + Math.sin(t * .11 + .8) * 4 + Math.sin(t * .003) * 6, hz: '125→100 Hz' },
  { label: 'EOG (L)', color: 'rgba(109,40,217,.75)', lw: 1.4, fn: (t) => Math.sin(t * .012) * 22 + Math.sin(t * .041) * 7, hz: '50→100 Hz' },
  { label: 'EOG (R)', color: 'rgba(109,40,217,.45)', lw: 1.2, fn: (t) => -Math.sin(t * .012 + .3) * 20 - Math.sin(t * .041 + .2) * 6, hz: '50→100 Hz' },
  { label: 'EMG', color: 'rgba(29,78,216,.7)', lw: 1.1, fn: (t) => Math.sin(t * .09) * 2 + (Math.random() - .5) * 8, hz: 'Chin · 256 Hz' },
];

const PSGReview = ({ file, channels = '5', classes = '3', onAnalyze, onReset }) => {
  const canvasRefs = useRef([]);
  const rafRef = useRef(null);
  const phaseRef = useRef(0);

  useEffect(() => {
    const NL = 3000;
    const noiseT = new Float32Array(NL);
    for (let i = 0; i < NL; i++) noiseT[i] = (Math.random() - .5) * 2;
    const sn = (t) => { const i = Math.floor(Math.abs(t)) % NL; const j = (i + 1) % NL; const f = t - Math.floor(t); return noiseT[i] * (1 - f) + noiseT[j] * f; };

    const ctxs = [];
    const dims = [];

    canvasRefs.current.forEach((canvas, idx) => {
      if (!canvas) return;
      const wrap = canvas.parentElement;
      const W = wrap.offsetWidth;
      const H = wrap.offsetHeight || 68;
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
      CHANNEL_CFGS.forEach((cfg, idx) => {
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
          const t = x * 0.5 + phaseRef.current * 40;
          const y = mid + cfg.fn(t) + sn(t * .08) * 1.5;
          x <= 1 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
      phaseRef.current += 0.018;
      rafRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  const labelColors = ['#c0392b', 'rgba(192,57,43,.55)', '#6d28d9', 'rgba(109,40,217,.55)', '#1d4ed8'];

  return (
    <div id="psg-review" className="visible">
      <div className="psg-info-bar">
        <div className="psg-file-name">{file?.name || 'recording.edf'}</div>
        <div className="psg-chips">
          <span className="psg-chip">{file ? `${(file.size / 1e6).toFixed(1)} MB` : '—'}</span>
          <span className="psg-chip blue">100 Hz</span>
          <span className="psg-chip green">{channels} channels</span>
          <span className="psg-chip green">Ready</span>
        </div>
      </div>
      <div className="psg-channels">
        <div className="psg-ch-header">
          <div className="psg-ch-title">Aperçu des canaux — Signal simulé</div>
          <div className="psg-ch-note">Formes d'onde représentatives · L'analyse réelle traite le fichier EDF complet</div>
        </div>
        <div className="psg-ch-rows">
          {CHANNEL_CFGS.map((cfg, i) => (
            <div className="psg-ch-row" key={i}>
              <div className="psg-ch-label">
                <div className="ch-name" style={{ color: labelColors[i] }}>{cfg.label}</div>
                <div className="ch-hz">{cfg.hz}</div>
              </div>
              <div className="psg-ch-canvas-wrap">
                <canvas ref={(el) => canvasRefs.current[i] = el} />
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="psg-action-bar">
        <div className="psg-meta">
          <div><strong>{file?.name || '—'}</strong></div>
          <div>{file ? `${(file.size / 1e6).toFixed(1)} MB` : '—'}</div>
          <div style={{ marginTop: '4px', fontSize: '8.5px', letterSpacing: '.5px' }}>Modèle : BiLSTM-SHHS1 · Acc 95.2% · κ 0.911</div>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button className="btn-reset" onClick={onReset}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-3.51" />
            </svg>
            Reset
          </button>
          <button className="btn-analyse" onClick={onAnalyze}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            Analyser le PSG
          </button>
        </div>
      </div>
    </div>
  );
};

export default PSGReview;
