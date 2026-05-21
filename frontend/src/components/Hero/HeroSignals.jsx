import React, { useEffect, useRef } from 'react';

const HeroSignals = () => {
  const eegRef = useRef(null);
  const emgRef = useRef(null);
  const eogRef = useRef(null);

  useEffect(() => {
    const cfgs = [
      { 
        canvas: eegRef.current, 
        color: 'rgba(192,57,43,.75)', 
        lw: 1.6, 
        fn(t) { return Math.sin(t * .022) * 16 + Math.sin(t * .11) * 4 + Math.sin(t * .31) * 2 + (Math.random() - .5) * .5 }, 
        speed: .9 
      },
      { 
        canvas: emgRef.current, 
        color: 'rgba(29,78,216,.65)', 
        lw: 1.2, 
        fn(t) { return Math.sin(t * .08) * 3 + (Math.random() - .5) * 7 }, 
        speed: 1.2 
      },
      { 
        canvas: eogRef.current, 
        color: 'rgba(109,40,217,.6)', 
        lw: 1.5, 
        fn(t) { return Math.sin(t * .015) * 18 + Math.sin(t * .047) * 6 + (Math.random() - .5) * 1.5 }, 
        speed: .6 
      },
    ];

    let rafId;

    const initCanvases = () => {
      const NL = 2000;
      const noise = new Float32Array(NL); 
      for (let i = 0; i < NL; i++) noise[i] = (Math.random() - .5) * 2;
      
      const sn = (t) => { 
        const i = Math.floor(t) % NL; 
        const j = (i + 1) % NL; 
        const f = t - Math.floor(t); 
        return noise[i] * (1 - f) + noise[j] * f; 
      };

      const runningCfgs = cfgs.map(cfg => {
        if (!cfg.canvas) return null;
        const wrap = cfg.canvas.parentElement;
        let W = wrap.offsetWidth;
        let H = wrap.offsetHeight || 62;
        const dpr = window.devicePixelRatio || 1;
        
        cfg.canvas.width = W * dpr; 
        cfg.canvas.height = H * dpr; 
        cfg.canvas.style.width = W + "px"; 
        cfg.canvas.style.height = H + "px"; 
        
        const ctx = cfg.canvas.getContext("2d"); 
        ctx.scale(dpr, dpr);
        
        return { ...cfg, ctx, W, H, phase: 0 };
      }).filter(Boolean);

      const draw = () => {
        runningCfgs.forEach(cfg => {
          cfg.ctx.clearRect(0, 0, cfg.W, cfg.H); 
          cfg.ctx.beginPath(); 
          cfg.ctx.lineWidth = cfg.lw; 
          cfg.ctx.strokeStyle = cfg.color; 
          cfg.ctx.lineJoin = "round";
          
          const mid = cfg.H / 2;
          for (let x = 0; x <= cfg.W; x += 1.5) { 
            const t = x * .8 + cfg.phase * cfg.speed * 60; 
            const y = mid + cfg.fn(t) + sn(t * .03) * 2; 
            if (x === 0) cfg.ctx.moveTo(x, y); 
            else cfg.ctx.lineTo(x, y); 
          }
          cfg.ctx.stroke(); 
          cfg.phase += .016; 
        });
        rafId = requestAnimationFrame(draw);
      };

      draw();
    };

    initCanvases();

    const handleResize = () => {
      if (rafId) cancelAnimationFrame(rafId);
      initCanvases();
    };

    window.addEventListener("resize", handleResize);

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <div className="hero-signals">
      <div className="sig-row"><span className="sig-lbl">EEG</span><canvas ref={eegRef}></canvas></div>
      <div className="sig-row"><span className="sig-lbl">EMG</span><canvas ref={emgRef}></canvas></div>
      <div className="sig-row"><span className="sig-lbl">EOG</span><canvas ref={eogRef}></canvas></div>
    </div>
  );
};

export default HeroSignals;
