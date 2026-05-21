import { useEffect, useRef } from 'react';

const ScanningAnimation = ({ isRunning, duration = 8000 }) => {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const scanLineRef = useRef(null);
  const rafRef = useRef(null);
  const startTimeRef = useRef(0);

  const channels = [
    { yo: 0.22, amp: 14, freq: 0.028, freq2: 0.09, color: 'rgba(192,57,43,.8)', lw: 1.5 },
    { yo: 0.55, amp: 11, freq: 0.019, freq2: 0.06, color: 'rgba(29,78,216,.75)', lw: 1.4 },
    { yo: 0.82, amp: 5, freq: 0.06, freq2: 0.18, color: 'rgba(109,40,217,.7)', lw: 1.2 },
  ];

  const NL = 4096;
  const noiseTable = channels.map(() => {
    const a = new Float32Array(NL);
    for (let i = 0; i < NL; i++) a[i] = (Math.random() - 0.5) * 2;
    return a;
  });

  const sn = (ch, t) => {
    const i = Math.floor(Math.abs(t)) % NL;
    const j = (i + 1) % NL;
    const f = t - Math.floor(t);
    return noiseTable[ch][i] * (1 - f) + noiseTable[ch][j] * f;
  };

  const sig = (ci, x) => {
    const c = channels[ci];
    const t = x * c.freq;
    return (Math.sin(t) + Math.sin((t * c.freq2) / c.freq) * 0.35 + sn(ci, t * 0.5) * 0.25) * c.amp;
  };

  const drawFrame = (ts) => {
    if (!isRunning) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const wrap = wrapRef.current;
    const scanLine = scanLineRef.current;

    if (!startTimeRef.current) startTimeRef.current = ts;
    const elapsed = ts - startTimeRef.current;
    const progress = Math.min(elapsed / duration, 1.0);
    
    const W = wrap.offsetWidth;
    const H = wrap.offsetHeight || 110;
    const usableW = W - 70;
    const scanX = 70 + usableW * progress;

    ctx.clearRect(0, 0, W, H);

    channels.forEach((c, ci) => {
      const yBase = H * c.yo;
      ctx.beginPath();
      ctx.strokeStyle = c.color;
      ctx.lineWidth = c.lw;
      ctx.lineJoin = 'round';
      for (let x = 70; x <= scanX; x += 1.5) {
        const y = yBase - sig(ci, x - 70);
        x <= 70.5 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    if (scanLine) scanLine.style.left = `${scanX}px`;

    if (progress < 1.0) {
      rafRef.current = requestAnimationFrame(drawFrame);
    } else {
      setTimeout(() => {
        if (isRunning) {
          startTimeRef.current = performance.now();
          rafRef.current = requestAnimationFrame(drawFrame);
        }
      }, 400);
    }
  };

  useEffect(() => {
    if (isRunning) {
      const canvas = canvasRef.current;
      const wrap = wrapRef.current;
      if (canvas && wrap) {
        const W = wrap.offsetWidth;
        const H = wrap.offsetHeight || 110;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width = `${W}px`;
        canvas.style.height = `${H}px`;
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
      }
      startTimeRef.current = 0;
      rafRef.current = requestAnimationFrame(drawFrame);
    } else {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [isRunning]);

  return (
    <div className="dz-wave" ref={wrapRef} style={{ display: isRunning ? 'block' : 'none' }}>
      <canvas ref={canvasRef}></canvas>
      <div className="dz-wave-labels">
        <span>EEG Fpz-Cz</span>
        <span>EEG Pz-Oz</span>
        <span>EMG</span>
      </div>
      <div className="dz-scan-line" ref={scanLineRef}></div>
    </div>
  );
};

export default ScanningAnimation;
