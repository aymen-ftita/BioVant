import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

const resources = {
  en: {
    translation: {
      "sidebar": {
        "doctor_home": "Dashboard",
        "new_analysis": "New Analysis",
        "custom_osa": "Custom OSA",
        "my_patients": "My Patients",
        "consultations": "Consultations",
        "admin_dashboard": "Dashboard",
        "doctors": "Doctors",
        "hospitals": "Hospitals",
        "audit_logs": "Audit Logs",
        "developer": "Developer",
        "light_mode": "Light Mode",
        "dark_mode": "Dark Mode",
        "logout": "Logout"
      },
      "dashboard": {
        "welcome": "Welcome to your Clinical Space",
        "overview": "Overview of your activity and recent analyses.",
        "patients": "Tracked Patients",
        "psg": "PSG Exams",
        "osa_severe": "Severe OSA Risks",
        "osa_distribution": "Apnea Risks Distribution",
        "recent_patients": "Recent Patients",
        "see_all": "See all",
        "no_patients": "No patients registered yet.",
        "new_analysis_btn": "+ New Analysis"
      },
      "common": {
        "loading": "Loading...",
        "save": "Save",
        "cancel": "Cancel",
        "download_pdf": "Download PDF Report"
      }
    }
  },
  fr: {
    translation: {
      "sidebar": {
        "doctor_home": "Accueil",
        "new_analysis": "Nouvelle Analyse",
        "custom_osa": "OSA Custom",
        "my_patients": "Mes Patients",
        "consultations": "Avis Confraternels",
        "admin_dashboard": "Tableau de Bord",
        "doctors": "Médecins",
        "hospitals": "Cliniques",
        "audit_logs": "Journal d'Audit",
        "developer": "Développeur",
        "light_mode": "Mode Clair",
        "dark_mode": "Mode Sombre",
        "logout": "Déconnexion"
      },
      "dashboard": {
        "welcome": "Bienvenue sur votre Espace Clinique",
        "overview": "Aperçu de votre activité et des analyses récentes.",
        "patients": "Patients Suivis",
        "psg": "Examens PSG",
        "osa_severe": "Risques OSA Sévères",
        "osa_distribution": "Répartition des Risques d'Apnée",
        "recent_patients": "Derniers Patients",
        "see_all": "Voir tout",
        "no_patients": "Aucun patient enregistré.",
        "new_analysis_btn": "+ Nouvelle Analyse"
      },
      "common": {
        "loading": "Chargement...",
        "save": "Enregistrer",
        "cancel": "Annuler",
        "download_pdf": "Télécharger Rapport PDF"
      }
    }
  }
};

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: "fr", // langue par défaut
    fallbackLng: "fr",
    interpolation: {
      escapeValue: false // React s'occupe déjà de l'échappement
    }
  });

export default i18n;
