import HeroSignals from './HeroSignals';
import './Hero.css';

const Hero = () => {
  return (
    <div className="hero">
      <div className="hero-inner">
        <div className="hero-text">
          <div className="eyebrow">Neural Polysomnography Analysis</div>
          <svg className="ecg-bar" viewBox="0 0 220 28" preserveAspectRatio="none">
            <path d="M0,14 L18,14 L24,14 L27,3 L30,25 L33,14 L38,14 L46,14 L50,2 L54,26 L58,14 L68,14 L75,14 L79,7 L83,21 L87,14 L100,14 L108,14 L111,3 L114,25 L117,14 L122,14 L130,14 L134,8 L138,20 L142,14 L155,14 L159,3 L163,25 L167,14 L180,14 L184,6 L188,22 L192,14 L220,14" />
          </svg>
          <h1>Hypnora</h1>
          <p className="hero-sub">
            Plateforme d'intelligence artificielle pour l'analyse clinique du sommeil.<br />
            Segmentation automatique et détection des troubles respiratoires.
          </p>
        </div>
        <HeroSignals />
      </div>
    </div>
  );
};

export default Hero;
