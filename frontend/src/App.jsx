import { useState, useEffect } from 'react'
import './App.css'
import axios from 'axios'

// Components — folder-based architecture mirroring vanilla-version/components
import Sidebar from './components/Sidebar'
import Hero from './components/Hero'
import LandingPage from './components/LandingPage'
import { AnalysisWizard } from './components/Wizard'
import { AnalysisResults } from './components/Results'
import { CustomOSA } from './components/OSAAnalysis'
import { DeveloperPipeline } from './components/PipelineBuilder'
import { PatientList, CollaborationChat, DoctorList, HomeDashboard, AuditLogsDashboard, HospitalsDashboard, ConsultationsView, DoctorDashboard } from './components/Common'
import { Terminal, CloudUpload, CheckCircle2, AlertTriangle, X } from 'lucide-react'

function App() {
  const [activeTab, setActiveTab] = useState('doctor')
  const [user, setUser] = useState(null)
  const [activeChat, setActiveChat] = useState(null) // {psg, patient}
  const [activeAnalysis, setActiveAnalysis] = useState(null)
  const [preselectedPatient, setPreselectedPatient] = useState(null)
  const [activePsgId, setActivePsgId] = useState(null)
  const [bgUploads, setBgUploads] = useState([])

  const handleEdfBgUpload = (file, patientName, psgId) => {
    const uploadId = Date.now();
    const token = localStorage.getItem('token');
    
    setBgUploads(prev => [...prev, {
      id: uploadId,
      fileName: file.name,
      patientName: patientName,
      progress: 0,
      status: 'uploading'
    }]);

    const uploadFormData = new FormData();
    uploadFormData.append('edf_file', file);

    // Smooth optimistic progress simulation (0% -> 95%) to show real-time progress while waiting for B2
    let currentProgress = 0;
    const progressInterval = setInterval(() => {
      if (currentProgress < 95) {
        const increment = Math.floor(Math.random() * 8) + 4; // 4% to 11% increments
        currentProgress = Math.min(currentProgress + increment, 95);
        setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, progress: currentProgress } : up));
      } else {
        clearInterval(progressInterval);
      }
    }, 120);

    console.log(`[App] Starting background B2 upload for patient "${patientName}" (PSG ID: ${psgId})`);
    axios.post(`http://localhost:8000/psgs/${psgId}/upload_edf`, uploadFormData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        Authorization: `Bearer ${token}`
      }
    }).then((uploadRes) => {
      clearInterval(progressInterval);
      console.log('[App] Background B2 EDF upload completed successfully:', uploadRes.data);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'completed', progress: 100 } : up));
      
      // Auto-remove completed item after 5 seconds to keep screen clean
      setTimeout(() => {
        setBgUploads(prev => prev.filter(up => up.id !== uploadId));
      }, 5000);
    }).catch((uploadErr) => {
      clearInterval(progressInterval);
      console.error('[App] Background B2 EDF upload failed:', uploadErr);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'failed' } : up));
    });
  };

  const handleHypnogramBgUpload = (psgId, blob) => {
    const uploadId = Date.now();
    const token = localStorage.getItem('token');
    const patientName = preselectedPatient?.name || 'Patient';
    
    setBgUploads(prev => [...prev, {
      id: uploadId,
      fileName: 'hypnogramme.png',
      patientName: patientName,
      progress: 0,
      status: 'uploading'
    }]);

    const uploadFormData = new FormData();
    uploadFormData.append('hypnogram_file', blob, 'hypnogram.png');

    // Smooth optimistic progress simulation (0% -> 95%) to show real-time progress while waiting for B2
    let currentProgress = 0;
    const progressInterval = setInterval(() => {
      if (currentProgress < 95) {
        const increment = Math.floor(Math.random() * 12) + 6; // 6% to 17% increments (smaller file uploads faster)
        currentProgress = Math.min(currentProgress + increment, 95);
        setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, progress: currentProgress } : up));
      } else {
        clearInterval(progressInterval);
      }
    }, 100);

    console.log(`[App] Starting background B2 Hypnogram upload for patient "${patientName}" (PSG ID: ${psgId})`);
    axios.post(`http://localhost:8000/psgs/${psgId}/upload_hypnogram`, uploadFormData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        Authorization: `Bearer ${token}`
      }
    }).then((uploadRes) => {
      clearInterval(progressInterval);
      console.log('[App] Background B2 Hypnogram upload completed successfully:', uploadRes.data);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'completed', progress: 100 } : up));
      
      setTimeout(() => {
        setBgUploads(prev => prev.filter(up => up.id !== uploadId));
      }, 5000);
    }).catch((uploadErr) => {
      clearInterval(progressInterval);
      console.error('[App] Background B2 Hypnogram upload failed:', uploadErr);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'failed' } : up));
    });
  };

  const handleHypnogramAnnotatedBgUpload = (psgId, blob) => {
    const uploadId = Date.now();
    const token = localStorage.getItem('token');
    const patientName = preselectedPatient?.name || 'Patient';
    
    setBgUploads(prev => [...prev, {
      id: uploadId,
      fileName: 'hypnogramme_annote.png',
      patientName: patientName,
      progress: 0,
      status: 'uploading'
    }]);

    const uploadFormData = new FormData();
    uploadFormData.append('hypnogram_file', blob, 'hypnogram_annotated.png');

    // Smooth optimistic progress simulation (0% -> 95%) to show real-time progress while waiting for B2
    let currentProgress = 0;
    const progressInterval = setInterval(() => {
      if (currentProgress < 95) {
        const increment = Math.floor(Math.random() * 12) + 6; // 6% to 17% increments
        currentProgress = Math.min(currentProgress + increment, 95);
        setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, progress: currentProgress } : up));
      } else {
        clearInterval(progressInterval);
      }
    }, 100);

    console.log(`[App] Starting background B2 Annotated Hypnogram upload for patient "${patientName}" (PSG ID: ${psgId})`);
    axios.post(`http://localhost:8000/psgs/${psgId}/upload_hypnogram_annotated`, uploadFormData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        Authorization: `Bearer ${token}`
      }
    }).then((uploadRes) => {
      clearInterval(progressInterval);
      console.log('[App] Background B2 Annotated Hypnogram upload completed successfully:', uploadRes.data);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'completed', progress: 100 } : up));
      
      setTimeout(() => {
        setBgUploads(prev => prev.filter(up => up.id !== uploadId));
      }, 5000);
    }).catch((uploadErr) => {
      clearInterval(progressInterval);
      console.error('[App] Background B2 Annotated Hypnogram upload failed:', uploadErr);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'failed' } : up));
    });
  };

  const handleOsaReportBgUpload = (psgId, blob) => {
    const uploadId = Date.now();
    const token = localStorage.getItem('token');
    const patientName = preselectedPatient?.name || 'Patient';
    
    setBgUploads(prev => [...prev, {
      id: uploadId,
      fileName: 'rapport_osa.html',
      patientName: patientName,
      progress: 0,
      status: 'uploading'
    }]);

    const uploadFormData = new FormData();
    uploadFormData.append('osa_report_file', blob, 'osa_report.html');

    // Smooth optimistic progress simulation (0% -> 95%) to show real-time progress while waiting for B2
    let currentProgress = 0;
    const progressInterval = setInterval(() => {
      if (currentProgress < 95) {
        const increment = Math.floor(Math.random() * 12) + 6; // 6% to 17% increments
        currentProgress = Math.min(currentProgress + increment, 95);
        setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, progress: currentProgress } : up));
      } else {
        clearInterval(progressInterval);
      }
    }, 100);

    console.log(`[App] Starting background B2 OSA Report upload for patient "${patientName}" (PSG ID: ${psgId})`);
    axios.post(`http://localhost:8000/psgs/${psgId}/upload_osa_report`, uploadFormData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        Authorization: `Bearer ${token}`
      }
    }).then((uploadRes) => {
      clearInterval(progressInterval);
      console.log('[App] Background B2 OSA Report upload completed successfully:', uploadRes.data);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'completed', progress: 100 } : up));
      
      setTimeout(() => {
        setBgUploads(prev => prev.filter(up => up.id !== uploadId));
      }, 5000);
    }).catch((uploadErr) => {
      clearInterval(progressInterval);
      console.error('[App] Background B2 OSA Report upload failed:', uploadErr);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'failed' } : up));
    });
  };

  useEffect(() => {
    const savedUser = localStorage.getItem('user')
    if (savedUser) {
      const parsed = JSON.parse(savedUser)
      setUser(parsed)
      setActiveTab(parsed.role === 'admin' ? 'home-dashboard' : parsed.role === 'demo' ? 'doctor' : 'doctor-dashboard')
    }
  }, [])

  const handleLogin = (mockUser) => {
    localStorage.setItem('user', JSON.stringify(mockUser))
    if (!localStorage.getItem('token')) {
      localStorage.setItem('token', 'mock-token')
    }
    setUser(mockUser)
    if (mockUser.role === 'admin') {
      setActiveTab('home-dashboard')
    } else if (mockUser.role === 'demo') {
      setActiveTab('doctor')
    } else {
      setActiveTab('doctor-dashboard')
    }
  }

  const handleLogout = () => {
    localStorage.clear()
    setUser(null)
    setActiveTab('doctor')
    setActiveAnalysis(null)
    setActivePsgId(null)
  }

  // If not logged in, render the gorgeous SaaS PaaS Landing Page!
  if (!user) {
    return <LandingPage onLoginSuccess={handleLogin} />
  }

  return (
    <div className="app-layout">
      {/* --- SIDEBAR --- */}
      <Sidebar
        user={user}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onLogout={handleLogout}
      />

      {/* --- MAIN CONTENT --- */}
      <main className="main-content">
        <div id="tooltip"></div>

        {/* ──── DOCTOR: DASHBOARD ──── */}
        {user.role === 'doctor' && activeTab === 'doctor-dashboard' && (
          <section className="app-section active">
            <div className="container">
               <DoctorDashboard onNavigate={setActiveTab} />
            </div>
          </section>
        )}

        {/* ──── DOCTOR / DEMO: PSG DIAGNOSTIC SECTION ──── */}
        {(user.role === 'doctor' || user.role === 'demo') && activeTab === 'doctor' && (
          <section className="app-section active">
            <Hero />
            <div className="container">
               <AnalysisWizard 
                 onAnalysisComplete={(data, psgId) => {
                   setActiveAnalysis(data);
                   setActivePsgId(psgId || null);
                 }} 
                 onStartBgUpload={handleEdfBgUpload}
                 onStartHypnogramBgUpload={handleHypnogramBgUpload}
                 onStartHypnogramAnnotatedUpload={handleHypnogramAnnotatedBgUpload}
                 onStartOsaReportBgUpload={handleOsaReportBgUpload}
                 preselectedPatient={preselectedPatient}
                 onClearPreselectedPatient={() => {
                   setPreselectedPatient(null);
                   setActivePsgId(null);
                 }}
               />
            </div>
          </section>
        )}

        {/* ──── DOCTOR: CUSTOM OSA PREDICTION ──── */}
        {user.role === 'doctor' && activeTab === 'custom-osa' && (
           <section className="app-section active">
              <div className="container">
                <CustomOSA />
              </div>
           </section>
        )}

        {/* ──── DOCTOR: PATIENT HISTORY & COLLABORATION ──── */}
        {user.role === 'doctor' && activeTab === 'patients' && (
          <section className="app-section active">
            <div className="container">
              <PatientList 
                onPingFile={(psg, patient) => setActiveChat({psg, patient})} 
                onLaunchAnalysis={(patient) => {
                  setPreselectedPatient(patient);
                  setActiveTab('doctor');
                  setActiveAnalysis(null);
                  setActivePsgId(null);
                }}
              />
            </div>
          </section>
        )}

        {/* ──── DOCTOR: CONSULTATIONS (MESSAGES) ──── */}
        {user.role === 'doctor' && activeTab === 'conversations' && (
          <section className="app-section active" style={{ padding: '20px' }}>
            <ConsultationsView />
          </section>
        )}

        {/* ──── ADMIN: HOME DASHBOARD ──── */}
        {user.role === 'admin' && activeTab === 'home-dashboard' && (
          <section className="app-section active">
            <div className="container">
              <HomeDashboard user={user} onTabChange={setActiveTab} />
            </div>
          </section>
        )}

        {/* ──── ADMIN: DOCTOR DATABASE MANAGEMENT ──── */}
        {user.role === 'admin' && activeTab === 'doctors-list' && (
          <section className="app-section active">
            <div className="container">
              <DoctorList />
            </div>
          </section>
        )}

        {/* ──── ADMIN: HOSPITALS / CLINICS MANAGEMENT ──── */}
        {user.role === 'admin' && activeTab === 'hospitals' && (
          <section className="app-section active">
            <div className="container">
              <HospitalsDashboard />
            </div>
          </section>
        )}

        {/* ──── ADMIN: AUDIT LOGS ──── */}
        {user.role === 'admin' && activeTab === 'audit-logs' && (
          <section className="app-section active">
            <div className="container">
              <AuditLogsDashboard />
            </div>
          </section>
        )}

        {/* ──── ADMIN / DEMO: VISUAL MLOPS PIPELINE BUILDER ──── */}
        {(user.role === 'admin' || user.role === 'demo') && activeTab === 'developer' && (
           <section className="app-section active">
              <div className="container">
                <DeveloperPipeline />
              </div>
           </section>
        )}

        {/* --- CHAT MODAL --- */}
        {activeChat && (
          <CollaborationChat 
            psg={activeChat.psg} 
            patient={activeChat.patient} 
            onClose={() => setActiveChat(null)} 
          />
        )}

      </main>

      {/* --- FLOATING BACKGROUND UPLOADS WIDGET --- */}
      {bgUploads.length > 0 && (
        <div className="floating-upload-container">
          {bgUploads.map(up => (
            <div key={up.id} className={`upload-card ${up.status}`}>
              <div className="upload-header">
                <div className="upload-icon-wrapper">
                  {up.status === 'uploading' && <CloudUpload size={18} className="pulse-animation" />}
                  {up.status === 'completed' && <CheckCircle2 size={18} />}
                  {up.status === 'failed' && <AlertTriangle size={18} />}
                </div>
                
                <div className="upload-details">
                  <h4 className="upload-title">
                    {up.status === 'uploading' && 'Envoi en arrière-plan...'}
                    {up.status === 'completed' && 'Fichier synchronisé'}
                    {up.status === 'failed' && 'Échec du transfert cloud'}
                  </h4>
                  <p className="upload-subtitle">
                    {up.patientName} — {up.fileName}
                  </p>
                </div>

                <button 
                  className="upload-close-btn"
                  onClick={() => setBgUploads(prev => prev.filter(item => item.id !== up.id))}
                >
                  <X size={14} />
                </button>
              </div>

              <div className="upload-progress-section">
                <div className="upload-progress-info">
                  <span>{up.status === 'uploading' ? 'Upload B2' : up.status === 'completed' ? 'Terminé' : 'Erreur'}</span>
                  <span>{up.progress}%</span>
                </div>
                <div className="upload-progress-bar-bg">
                  <div 
                    className="upload-progress-bar-fill" 
                    style={{ width: `${up.progress}%` }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default App
