import React, { useState } from 'react';
import { 
  Shield, 
  Terminal, 
  Activity, 
  CheckSquare, 
  Database, 
  Cpu, 
  Users, 
  ChevronRight, 
  Lock, 
  Sparkles, 
  Clock, 
  Globe, 
  MessageSquare,
  X,
  FileText
} from 'lucide-react';
import './LandingPage.css';
import Login from '../Login';

const LandingPage = ({ onLoginSuccess }) => {
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [showAboutModal, setShowAboutModal] = useState(false);

  const handleOpenLogin = () => {
    setShowLoginModal(true);
  };

  const handleCloseLogin = () => {
    setShowLoginModal(false);
  };

  const handleOpenAbout = () => {
    setShowAboutModal(true);
  };

  const handleCloseAbout = () => {
    setShowAboutModal(false);
  };

  return (
    <div className="landing-wrapper">
      
      {/* ──── HEADER ──── */}
      <header className="landing-header">
        <div className="landing-logo">
          <span>Hypnora</span>
          <span className="logo-accent">AI</span>
        </div>
        
        <nav className="landing-nav">
          <button className="nav-link" onClick={handleOpenAbout}>À propos</button>
          <a href="#features" className="nav-link">Fonctionnalités</a>
          <a href="#architecture" className="nav-link">PaaS Architecture</a>
          <a href="#pricing" className="nav-link">Tarifs</a>
        </nav>
        
        <div className="landing-actions">
          <button className="btn-header-login" onClick={handleOpenLogin}>
            Se Connecter
          </button>
        </div>
      </header>

      {/* ──── HERO SECTION ──── */}
      <section className="landing-hero">
        <div className="hero-glow"></div>
        <div className="hero-content">
          <div className="hero-badge">
            <Sparkles size={12} />
            <span>Diagnostic Neuro-Sommeil en Temps Réel</span>
          </div>
          
          <h1 className="hero-title">
            Polysomnographie Cloud & <br />
            <span className="gradient-text">MLOps Intelligence</span>
          </h1>
          
          <p className="hero-desc">
            Le PaaS médical de référence pour l'analyse automatisée du sommeil et la prédiction de la sévérité de l'Apnée Obstructive du Sommeil (OSA) via Stacking d'Ensemble de modèles de Deep Learning et d'explications SHAP transparentes.
          </p>
          
          <div className="hero-ctas">
            <button className="btn-hero-primary" onClick={handleOpenLogin}>
              Démarrer l'Analyse <ChevronRight size={16} />
            </button>
            <button className="btn-hero-secondary" onClick={handleOpenAbout}>
              Explorer la Plateforme
            </button>
          </div>
        </div>

        {/* Mockup Dashboard Preview */}
        <div className="hero-mockup">
          <div className="mockup-header">
            <div className="mockup-dots">
              <span></span><span></span><span></span>
            </div>
            <div className="mockup-title">Hypnora AI — Cloud Node console</div>
          </div>
          <div className="mockup-body">
            <div className="mockup-grid">
              <div className="mockup-sidebar">
                <div className="m-logo">Hypnora <em>AI</em></div>
                <div className="m-nav-item active"><Activity size={12} /> Docteur</div>
                <div className="m-nav-item"><CheckSquare size={12} /> OSA Custom</div>
                <div className="m-nav-item"><Terminal size={12} /> Développeur</div>
              </div>
              <div className="mockup-main">
                <div className="mockup-panel">
                  <div className="m-panel-header">
                    <h4>Inférence Globale (XGB+LGBM+MLP)</h4>
                    <span className="m-badge">Severe OSA</span>
                  </div>
                  <div className="m-graph">
                    <div className="m-graph-bar" style={{height: '80%'}}></div>
                    <div className="m-graph-bar" style={{height: '60%'}}></div>
                    <div className="m-graph-bar" style={{height: '95%'}}></div>
                    <div className="m-graph-bar" style={{height: '40%'}}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ──── PAAS ROLE CAPABILITIES (DOCTOR VS ADMIN) ──── */}
      <section className="landing-roles">
        <div className="section-header">
          <h2 className="section-title">Deux Environnements Sur Mesure</h2>
          <p className="section-subtitle">Notre plateforme s'adapte précisément à votre profil pour maximiser l'efficacité.</p>
        </div>

        <div className="roles-grid">
          
          {/* Doctor Role Card */}
          <div className="role-card doctor">
            <div className="role-icon-wrapper">
              <Activity size={24} />
            </div>
            <h3 className="role-title">Espace Praticien Clinique</h3>
            <p className="role-description">
              Idéal pour les médecins spécialistes et les pneumologues. Prenez des décisions éclairées grâce à l'IA explicable.
            </p>
            <ul className="role-features-list">
              <li>
                <CheckSquare size={14} className="feature-check" />
                <span><b>Diagnostic OSA Customisé</b> : Saisie manuelle ou import CSV/XML des features pour évaluer le risque de sévérité.</span>
              </li>
              <li>
                <Activity size={14} className="feature-check" />
                <span><b>Section Docteur dédiée</b> : Assistant d'analyse de fichiers EDF polysomnographiques et tracés de sommeil.</span>
              </li>
              <li>
                <Users size={14} className="feature-check" />
                <span><b>Historique des Patients</b> : Base de données centralisée avec messagerie de collaboration médicale en temps réel.</span>
              </li>
            </ul>
          </div>

          {/* Admin / Developer Role Card */}
          <div className="role-card admin">
            <div className="role-icon-wrapper">
              <Terminal size={24} />
            </div>
            <h3 className="role-title">Console Admin & MLOps</h3>
            <p className="role-description">
              Pour les administrateurs réseau et les ingénieurs en Machine Learning souhaitant superviser et modéliser.
            </p>
            <ul className="role-features-list">
              <li>
                <Users size={14} className="feature-check" />
                <span><b>Base des Médecins</b> : Contrôlez les accès, vérifiez le volume des examens et enregistrez les nouveaux praticiens.</span>
              </li>
              <li>
                <Terminal size={14} className="feature-check" />
                <span><b>Pipeline de Développement</b> : Éditeur visuel interactif pour connecter les sources, filtres et modèles ML.</span>
              </li>
              <li>
                <Cpu size={14} className="feature-check" />
                <span><b>Supervision d'Inférence</b> : Suivi en direct du temps d'exécution des modèles et de l'entraînement des réseaux.</span>
              </li>
            </ul>
          </div>

        </div>
      </section>

      {/* ──── KEY FEATURES SECTION ──── */}
      <section id="features" className="landing-features">
        <div className="section-header">
          <h2 className="section-title">Une Technologie Cloud d'Avant-garde</h2>
          <p className="section-subtitle">Hypnora AI fusionne la médecine clinique du sommeil et le génie logiciel de pointe.</p>
        </div>

        <div className="features-grid">
          
          <div className="feature-item">
            <Database size={20} className="f-icon" />
            <h4>Parsing EDF & Signaux</h4>
            <p>Extraction rapide des signaux polysomnographiques multi-canaux (EEG, EOG, EMG, SpO₂) via API ultra-optimisée.</p>
          </div>

          <div className="feature-item">
            <Cpu size={20} className="f-icon" />
            <h4>Modèles de Stacking</h4>
            <p>Stacking de pointe d'algorithmes (XGBoost + LightGBM + MLP → Régression Logistique) pour une robustesse maximale.</p>
          </div>

          <div className="feature-item">
            <Sparkles size={20} className="f-icon" />
            <h4>Explicabilité SHAP</h4>
            <p>Aucun effet "boîte noire".Visualisez instantanément l'impact de chaque variable biologique sur la prédiction de l'OSA.</p>
          </div>

          <div className="feature-item">
            <MessageSquare size={20} className="f-icon" />
            <h4>Collaboration Live Chat</h4>
            <p>Partagez instantanément les rapports complexes et collaborez entre confrères en un clic depuis le dossier patient.</p>
          </div>

        </div>
      </section>

      {/* ──── SYSTEM METRICS ──── */}
      <section className="landing-metrics">
        <div className="metrics-grid">
          <div className="metric-box">
            <div className="metric-val">99.8%</div>
            <div className="metric-lbl">Disponibilité API Cloud</div>
          </div>
          <div className="metric-box">
            <div className="metric-val">&lt; 4s</div>
            <div className="metric-lbl">Temps d'inférence PSG</div>
          </div>
          <div className="metric-box">
            <div className="metric-val">50+</div>
            <div className="metric-lbl">Cliniques Connectées</div>
          </div>
          <div className="metric-box">
            <div className="metric-val">1.2M</div>
            <div className="metric-lbl">Époques de Sommeil Entraînées</div>
          </div>
        </div>
      </section>

      {/* ──── PRICING SECTION ──── */}
      <section id="pricing" className="landing-pricing">
        <div className="section-header">
          <h2 className="section-title">Plans & Abonnements PaaS</h2>
          <p className="section-subtitle">Une tarification transparente adaptée aux laboratoires de recherche comme aux cliniques privées.</p>
        </div>

        <div className="pricing-grid">
          
          <div className="pricing-card">
            <div className="p-title">Laboratoire Individuel</div>
            <div className="p-price">0€ <span>/ utilisateur</span></div>
            <p className="p-desc">Pour tester les capacités prédictives en recherche académique.</p>
            <ul className="p-features">
              <li>10 Prédictions Custom / mois</li>
              <li>Parsing XML basique</li>
              <li>SHAP explications incluses</li>
              <li>1 Compte Spécialiste</li>
            </ul>
            <button className="btn-pricing" onClick={handleOpenLogin}>Démarrer l'essai gratuit</button>
          </div>

          <div className="pricing-card active">
            <div className="p-popular">Recommandé</div>
            <div className="p-title">Clinique / Sommeil</div>
            <div className="p-price">149€ <span>/ mois</span></div>
            <p className="p-desc">Orchestration complète pour les centres du sommeil professionnels.</p>
            <ul className="p-features">
              <li>Prédictions EDF & XML illimitées</li>
              <li>Stacking Ensemble Model complet</li>
              <li>Export PDF & Rapports Cliniques</li>
              <li>Collaboration en direct entre médecins</li>
              <li>Support technique prioritaire 24/7</li>
            </ul>
            <button className="btn-pricing primary" onClick={handleOpenLogin}>Activer le Plan Pro</button>
          </div>

          <div className="pricing-card">
            <div className="p-title">Réseau Hospitalier</div>
            <div className="p-price">Sur Mesure</div>
            <p className="p-desc">Infrastructures privées sur site avec pipeline MLOps dédié.</p>
            <ul className="p-features">
              <li>Comptes Praticiens illimités</li>
              <li>API et serveurs dédiés à haute performance</li>
              <li>Conception visuelle des pipelines personnalisés</li>
              <li>SLA de 99.9% et support dédié</li>
            </ul>
            <button className="btn-pricing" onClick={handleOpenLogin}>Contacter les ventes</button>
          </div>

        </div>
      </section>

      {/* ──── FOOTER ──── */}
      <footer className="landing-footer">
        <div className="footer-top">
          <div className="footer-brand">
            <h3>Hypnora<span>AI</span></h3>
            <p>PaaS de diagnostic polysomnographique et de modélisation intelligente du sommeil.</p>
          </div>
          
          <div className="footer-links">
            <div className="footer-col">
              <h5>Produit</h5>
              <a href="#features">Fonctionnalités</a>
              <a href="#architecture">ML Architecture</a>
              <a href="#pricing">Tarifs</a>
            </div>
            
            <div className="footer-col">
              <h5>Ressources</h5>
              <button onClick={handleOpenAbout} className="f-btn-link">À propos</button>
              <a href="#help">Aide & Support</a>
              <a href="#api">Documentation API</a>
            </div>

            <div className="footer-col">
              <h5>Légal & Conformité</h5>
              <span>RGPD & CNIL</span>
              <span>HIPAA Compliant</span>
              <span>Mentions Légales</span>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <p>© 2026 Hypnora AI Cloud. Tous droits réservés. Conçu et certifié pour les cliniciens et chercheurs du sommeil.</p>
          <div className="footer-badges">
            <span className="f-badge">CE Medical Device</span>
            <span className="f-badge">SSL Secure</span>
          </div>
        </div>
      </footer>

      {/* ──── ABOUT US MODAL (POPUP) ──── */}
      {showAboutModal && (
        <div className="landing-modal-overlay" onClick={handleCloseAbout}>
          <div className="landing-modal-card about-modal" onClick={e => e.stopPropagation()}>
            <button className="modal-close-btn" onClick={handleCloseAbout}><X size={20} /></button>
            
            <div className="modal-header">
              <Sparkles size={36} className="modal-header-icon" />
              <h3>À Propos de Hypnora AI</h3>
              <p>Notre mission : Démocratiser l'IA explicable en médecine du sommeil.</p>
            </div>
            
            <div className="modal-body-scroll">
              <p>
                <b>Hypnora AI</b> est une plateforme d'infrastructure en tant que service (PaaS) conçue pour résoudre le goulot d'étranglement diagnostique des troubles respiratoires du sommeil. L'évaluation polysomnographique manuelle exigeant des heures d'analyse par patient, nos outils simplifient et automatisent ce processus.
              </p>
              
              <h4>Notre Stack Scientifique & Machine Learning</h4>
              <p>
                La plateforme s'appuie sur des réseaux de neurones profonds multicouches (MLP) et des modèles de gradient boosting de pointe (XGBoost, LightGBM) agrégés via un algorithme de Stacking Meta-Learner (Régression Logistique). Cette architecture garantit une précision globale de <b>99.8%</b>.
              </p>

              <h4>IA Explicable (eXplainable AI)</h4>
              <p>
                Parce que la confiance clinique est primordiale, nous integrons des explications basées sur la théorie des jeux coopératifs (SHAP). Chaque prédiction est accompagnée de ses facteurs déterminants pour permettre au médecin d'interpréter le verdict de l'algorithme.
              </p>

              <div className="about-stats-row">
                <div>
                  <strong>Conformité Totale</strong>
                  <p>Hébergement certifié de données de santé (HDS), conforme HIPAA & RGPD.</p>
                </div>
                <div>
                  <strong>Écosystème MLOps</strong>
                  <p>Permet aux ingénieurs d'ajuster visuellement et d'optimiser les pipelines d'inférence en direct.</p>
                </div>
              </div>
            </div>

            <button className="btn-modal-action" onClick={() => { handleCloseAbout(); handleOpenLogin(); }}>
              Se Connecter à la Console
            </button>
          </div>
        </div>
      )}

      {/* ──── LOGIN MODAL (POPUP) ──── */}
      {showLoginModal && (
        <div className="landing-modal-overlay" onClick={handleCloseLogin}>
          <div className="landing-modal-card login-modal-wrapper" onClick={e => e.stopPropagation()}>
            <button className="modal-close-btn" onClick={handleCloseLogin}><X size={20} /></button>
            <Login onLogin={(user) => {
              handleCloseLogin();
              onLoginSuccess(user);
            }} />
          </div>
        </div>
      )}

    </div>
  );
};

export default LandingPage;
