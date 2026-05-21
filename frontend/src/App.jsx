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
import { PatientList, CollaborationChat, DoctorList } from './components/Common'
import { Terminal, CloudUpload, CheckCircle2, AlertTriangle, X } from 'lucide-react'

function App() {
  const [activeTab, setActiveTab] = useState('doctor')
  const [user, setUser] = useState(null)
  const [activeChat, setActiveChat] = useState(null) // {psg, patient}
  const [activeAnalysis, setActiveAnalysis] = useState(null)
  const [preselectedPatient, setPreselectedPatient] = useState(null)
  const [activePsgId, setActivePsgId] = useState(null)
  const [bgUploads, setBgUploads] = useState([])

  const handleBgUpload = (file, patientName, psgId) => {
    const uploadId = Date.now();
    const token = localStorage.getItem('token');
    
    // Add to active uploads tracking list
    setBgUploads(prev => [...prev, {
      id: uploadId,
      fileName: file.name,
      patientName: patientName,
      progress: 0,
      status: 'uploading'
    }]);

    const uploadFormData = new FormData();
    uploadFormData.append('edf_file', file);

    console.log(`[App] Starting background B2 upload for patient "${patientName}" (PSG ID: ${psgId})`);
    axios.post(`http://localhost:8000/psgs/${psgId}/upload_edf`, uploadFormData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        Authorization: `Bearer ${token}`
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, progress: percentCompleted } : up));
      }
    }).then((uploadRes) => {
      console.log('[App] Background B2 EDF upload completed successfully:', uploadRes.data);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'completed', progress: 100 } : up));
      
      // Auto-remove completed item after 5 seconds to keep screen clean
      setTimeout(() => {
        setBgUploads(prev => prev.filter(up => up.id !== uploadId));
      }, 5000);
    }).catch((uploadErr) => {
      console.error('[App] Background B2 EDF upload failed:', uploadErr);
      setBgUploads(prev => prev.map(up => up.id === uploadId ? { ...up, status: 'failed' } : up));
    });
  };

  useEffect(() => {
    const savedUser = localStorage.getItem('user')
    if (savedUser) {
      const parsed = JSON.parse(savedUser)
      setUser(parsed)
      setActiveTab(parsed.role === 'admin' ? 'doctors-list' : 'doctor')
    }
  }, [])

  const handleLogin = (mockUser) => {
    localStorage.setItem('user', JSON.stringify(mockUser))
    if (!localStorage.getItem('token')) {
      localStorage.setItem('token', 'mock-token')
    }
    setUser(mockUser)
    if (mockUser.role === 'admin') {
      setActiveTab('doctors-list')
    } else {
      setActiveTab('doctor')
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

        {/* ──── DOCTOR: PSG DIAGNOSTIC SECTION ──── */}
        {user.role === 'doctor' && activeTab === 'doctor' && (
          <section className="app-section active">
            <Hero />
            <div className="container">
               <AnalysisWizard 
                 onAnalysisComplete={(data, psgId) => {
                   setActiveAnalysis(data);
                   setActivePsgId(psgId || null);
                 }} 
                 onStartBgUpload={handleBgUpload}
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

        {/* ──── ADMIN: DOCTOR DATABASE MANAGEMENT ──── */}
        {user.role === 'admin' && activeTab === 'doctors-list' && (
          <section className="app-section active">
            <div className="container">
              <DoctorList />
            </div>
          </section>
        )}

        {/* ──── ADMIN: VISUAL MLOPS PIPELINE BUILDER ──── */}
        {user.role === 'admin' && activeTab === 'developer' && (
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
