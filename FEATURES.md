# 📊 Spécifications des Fonctionnalités et Caractéristiques — Projet BioVant (Hypnoria)

Ce document fournit une description exhaustive, structurée et technique de l'ensemble des fonctionnalités, des caractéristiques et de l'architecture API de l'application **BioVant (Hypnoria)**. Ce projet est conçu comme une plateforme médicale avancée (SaaS / PaaS) dédiée à l'analyse automatique de la polysomnographie (PSG), au staging du sommeil par Deep Learning et à la prédiction de la sévérité du Syndrome d'Apnées Obstructives du Sommeil (SAOS / OSA) par Machine Learning.

---

## 🏛️ Architecture Globale & Stack Technique

L'application repose sur une architecture découplée moderne, performante et prête pour la production :

### 💻 Frontend (React SPA)
*   **Framework** : React 18+ (avec Vite pour un build ultra-rapide).
*   **Design & UI/UX** : Design premium sur-mesure (thème sombre élégant, glassmorphisme, micro-animations fluides, typographies modernes).
*   **Support Multilingue (i18n)** : Traduction intégrale de l'interface en temps réel (Français/Anglais) avec `react-i18next`.
*   **Gestion des requêtes** : Axios (avec synchronisation des tokens de sécurité et suivi optimiste des téléversements en arrière-plan).

### ⚙️ Backend (FastAPI REST API)
*   **Framework** : FastAPI (Python 3.10+), offrant des performances asynchrones élevées et une auto-documentation via OpenAPI / Swagger.
*   **Base de Données** : PostgreSQL (ou SQLite en dev) avec SQLAlchemy ORM.
*   **Stockage Cloud** : Intégration directe avec l'API Backblaze B2 pour le stockage cloud hautement disponible des fichiers volumineux (signaux bruts `.edf`, images d'hypnogrammes, rapports cliniques HTML).

### 🧠 Modèles de Deep Learning & Machine Learning
*   **Sleep Staging (Deep Learning - PyTorch)** : LSTM, CNN et Transformer combinés via un ensemble Stacking.
*   **Prédiction SAOS (Machine Learning)** : Classifieurs (XGBoost / LightGBM) avec explicabilité SHAP intégrée.

### 🐳 Infrastructure & Déploiement (Docker)
*   Conteneurisation complète avec **Docker** et **Docker Compose**.
*   **Frontend** : Déployé via une image multi-stages `Node.js` + `Nginx` pour des performances optimales.
*   **Variables d'Environnement** : Sécurisation des credentials (DB, JWT, B2) via fichiers `.env`.

---

## 👥 Rôles, Acteurs & Authentification

### 1. 🛡️ Profil Administrateur (System Admin)
*   **Tableau de bord global** : Statistiques globales sur la plateforme.
*   **Gestion des Hôpitaux** : Création et assignation de centres cliniques.
*   **Cycle de vie des Médecins** : Création, suspension, gestion des dates d'expiration des licences, réinitialisation des mots de passe.
*   **Journal d'Audit (Audit Logs)** : Traçabilité complète des actions (connexions, ajouts) exportable en CSV pour la conformité.
*   **Console MLOps** : Interface interactive permettant de monitorer les pipelines de données.

### 2. 🩺 Profil Médecin (Clinician)
*   **Tableau de Bord Personnel** : Vue globale (Dashboard) avec la distribution des risques SAOS de sa propre patientèle et un récapitulatif des derniers examens.
*   **Registre des Patients** : Tableau listant ses patients (âge, IMC) avec accès rapide aux examens.
*   **Avis Confraternels (Collaboration Chat)** :
    *   Fils de discussion sécurisés entre médecins rattachés à des enregistrements PSG spécifiques.
    *   Vue centralisée "Consultations" pour voir tous les messages non-lus.

---

## 📋 Fonctionnalités Médicales & Cliniques

### ⚡ Assistant d'Analyse (Sleep Analysis Wizard)
L'assistant guide le médecin à travers l'analyse d'un fichier EDF brut :
1. Choix des Canaux (ex: EEG+EOG+EMG ou EEG seul).
2. Choix des Classes (3 classes ou 5 classes AASM).
3. Téléversement `.edf` dynamique avec simulation temps-réel (Backend IA en action).
4. **Génération de l'Hypnogramme** interactif et extraction immédiate des **Métriques AASM** (SE%, TST, Latences, WASO).

### 🫁 Prédiction SAOS & Rapport Clinique
*   Extraction mathématique des marqueurs du sommeil (ratios NREM/REM, fragmentations).
*   Saisie des données oxymétriques.
*   Prédiction IA de la sévérité (*Normal, Léger, Modéré, Sévère*).
*   **Explicabilité SHAP** visuelle (facteurs aggravants vs protecteurs).
*   **Génération PDF** : Export complet du rapport médical grâce à `jsPDF` et `html2canvas` (incluant l'hypnogramme, l'analyse AASM et l'impact SHAP).
*   **Cloud** : Sauvegarde asynchrone des analyses en arrière-plan vers Backblaze B2.

### 🧪 Prédiction SAOS Manuelle (Custom OSA)
*   Import de rapports externes (CSV, XML).
*   Le fichier XML est pré-formaté et échappé correctement pour une interopérabilité parfaite.

---

## 🔌 API Endpoints & Intégrations Backend

Toutes les routes API sont protégées par JWT (à l'exception des routes publiques). Le préfixe de l'hôte local par défaut est `http://localhost:8000`.

### 🔑 Authentification
*   `POST /token` : Connexion, validation de la licence et du statut, génération du JWT.
*   `GET /users/me` : Renvoie le profil de l'utilisateur connecté.

### 👑 Routes Administrateur
*   `GET /admin/dashboard-stats` : Statistiques de base (nombre total de médecins, patients, hôpitaux, etc.).
*   `POST /admin/doctors` | `GET /admin/doctors` : Création et listing des comptes médicaux.
*   `PUT /admin/doctors/{doctor_id}/lifecycle` : Modification du statut (actif/suspendu) et de l'expiration de la licence.
*   `POST /admin/doctors/{doctor_id}/reset-password` : Génération d'un nouveau mot de passe temporaire pour un médecin.
*   `POST /admin/hospitals` | `GET /admin/hospitals` : Gestion des cliniques / hôpitaux rattachés.
*   `GET /admin/audit-logs` : Récupération des logs de sécurité.
*   `GET /admin/audit-logs/export` : Téléchargement du journal d'audit au format CSV.

### 🩺 Routes Médecin & Patients
*   `GET /doctor/stats` : Statistiques analytiques personnelles (distribution des risques OSA de sa patientèle).
*   `GET /doctors` : Liste les confrères actifs (pour initier un chat collaboratif).
*   `POST /patients` | `GET /patients` : Ajouter ou récupérer les patients assignés au médecin.
*   `GET /patients/{patient_id}` : Obtenir le profil détaillé d'un patient et son historique PSG.

### 📈 Gestion des Enregistrements (PSG) & Backblaze
*   `POST /patients/{patient_id}/psgs` : Créer l'entrée d'un nouvel examen (PSG).
*   `PUT /psgs/{psg_id}` : Mettre à jour la sévérité SAOS ou les résultats JSON d'un examen.
*   `POST /psgs/{psg_id}/upload_edf` : Stocke le fichier brut sur Backblaze B2.
*   `POST /psgs/{psg_id}/upload_hypnogram` : Sauvegarde l'image de l'hypnogramme brut.
*   `POST /psgs/{psg_id}/upload_hypnogram_annotated` : Sauvegarde l'hypnogramme avec les notes du médecin.
*   `POST /psgs/{psg_id}/upload_osa_report` : Exporte et stocke le rapport au format HTML.
*   `POST /psgs/{psg_id}/annotations` | `GET /psgs/{psg_id}/annotations` : Ajout et récupération de notes textuelles à des instants temporels précis (époques) sur l'hypnogramme.

### 💬 Chat Collaboratif & Consultations
*   `POST /conversations` : Crée une nouvelle discussion liée à un examen (PSG).
*   `GET /conversations` : Récupère la boîte de réception globale du médecin connecté.
*   `GET /conversations/psg/{psg_id}` : Récupère les fils de discussion liés à un PSG spécifique.
*   `GET /conversations/{conversation_id}/messages` : Lit l'historique d'un chat.
*   `POST /conversations/{conversation_id}/messages` : Envoie un nouveau message.

### 🤖 Intelligence Artificielle & Machine Learning (ML Routes)
*   `GET /health` : Vérifie que le moteur IA est prêt (modèles chargés en RAM/VRAM).
*   `POST /channels` : Analyse l'entête d'un fichier EDF pour lister ses canaux EEG/EOG.
*   `POST /analyze` : Exécute le pipeline complet de Deep Learning PyTorch (prétraitement, LSTM/CNN/Transformer, Meta-Learner) et retourne la séquence des stades (W, N1, N2, N3, R).
*   `POST /extract_features` : Calcule les marqueurs de l'architecture du sommeil depuis l'hypnogramme (Latences, fragmentation, ratios temporels).
*   `POST /predict_osa` : Effectue l'inférence XGBoost pour diagnostiquer la sévérité de l'apnée du sommeil avec valeurs SHAP.
*   `POST /predict_osa_custom` : Variante de la prédiction depuis une saisie manuelle (sans fichier EDF).
*   `POST /parse_features_file` : Parse de manière sécurisée les fichiers XML ou CSV externes pour remplir automatiquement le formulaire clinique.
