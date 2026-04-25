"""
Sleep Staging Backend Server — SHHS1 Model (5-channel, 3-class)
Run with: python sleep_server.py
Requires: flask, flask-cors, mne, torch, numpy, scipy

Model: best_LSTM_shhs.pth
  - Input  : 5 channels (EEG×2, EOG×2, EMG), 30s epochs @ 100Hz
  - Output : 3 classes — Wake (0), NREM (1), REM (2)
  - Trained: SHHS1 dataset, 50 subjects, kappa=0.91
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import joblib
import xgboost as xgb
import shap

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid
from db import supabase

app = Flask(__name__)
CORS(app)

from api.auth import auth_bp
from api.patients import patients_bp
from api.analyses import analyses_bp
from api.discussions import discussions_bp
from api.messages import messages_bp
from api.admin import admin_bp

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(patients_bp, url_prefix='/api/patients')
app.register_blueprint(analyses_bp, url_prefix='/api/analyses')
app.register_blueprint(discussions_bp, url_prefix='/api/discussions')
app.register_blueprint(messages_bp, url_prefix='/api/messages')
app.register_blueprint(admin_bp, url_prefix='/api/admin')

from models import SleepTransformer, SleepCNN, SleepLSTM
from preprocessing import EPOCH_SEC, SAMPLE_RATE, MODELS_DIR, STEP2_DIR, CHANNEL_ALIASES, CH_ORDER, resolve_channels, compute_aasm_stats, preprocess_edf
from osa_predictor import init_osa_predictor
import osa_predictor

# ─────────────────────────────────────────────
# LOAD MODEL & CACHING
# ─────────────────────────────────────────────
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

_MODEL_CACHE = {}

def get_base_model(model_type, channels, classes):
    key = f"{model_type}_{channels}ch_{classes}cls"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    
    n_ch = int(channels)
    n_cl = int(classes)
    
    if model_type == "LSTM":
        model = SleepLSTM(input_size=n_ch, num_classes=n_cl)
    elif model_type == "CNN":
        model = SleepCNN(n_channels=n_ch, num_classes=n_cl)
    elif model_type == "Transformer":
        model = SleepTransformer(n_channels=n_ch, num_classes=n_cl)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model_path = os.path.join(MODELS_DIR, key, f"best_{key}.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    _MODEL_CACHE[key] = model
    print(f"✓ Cached base model: {key}")
    return model

def get_stacking_ensemble(channels, classes):
    key = f"Stacking_{channels}ch_{classes}cls"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    
    import joblib
    pkl_path = os.path.join(MODELS_DIR, "stacking", key, f"stacking_{channels}ch_{classes}cls.pkl")
    meta_learner = joblib.load(pkl_path)
    
    # Load base models
    cnn = get_base_model("CNN", channels, classes)
    lstm = get_base_model("LSTM", channels, classes)
    transformer = get_base_model("Transformer", channels, classes)
    
    _MODEL_CACHE[key] = (meta_learner, cnn, lstm, transformer)
    print(f"✓ Cached stacking ensemble: {key}")
    return _MODEL_CACHE[key]

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".edf"):
        return jsonify({"error": "Please upload an EDF file"}), 400

    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".edf")
    os.close(tmp_fd)
    f.save(tmp_path)

    try:
        model_types_str = request.form.get("models", "LSTM")
        model_types = [m.strip() for m in model_types_str.split(",")]
        channels   = request.form.get("channels", "5")
        classes    = request.form.get("classes", "3")
        patient_id = request.form.get("patient_id")
        doctor_id  = request.form.get("doctor_id")
        
        print(f"\n[ANALYZE] {f.filename} | Models: {model_types} | Ch: {channels} | Cls: {classes}")
        X      = preprocess_edf(tmp_path, channels_str=channels)
        tensor = torch.tensor(X, dtype=torch.float32).to(device)

        class_names = ["Wake", "N1", "N2", "N3", "REM"] if classes == "5" else ["Wake", "NREM", "REM"]
        
        results = []
        for m_type in model_types:
            preds = []
            if m_type == "Stacking":
                meta_learner, cnn, lstm, transformer = get_stacking_ensemble(channels, classes)
                with torch.no_grad():
                    for i in range(0, len(tensor), 64):
                        batch = tensor[i:i + 64]
                        out_cnn = cnn(batch)
                        out_lstm = lstm(batch)
                        out_trans = transformer(batch)
                        prob_cnn = F.softmax(out_cnn, dim=1).cpu().numpy()
                        prob_lstm = F.softmax(out_lstm, dim=1).cpu().numpy()
                        prob_trans = F.softmax(out_trans, dim=1).cpu().numpy()
                        meta_X = np.concatenate([prob_cnn, prob_lstm, prob_trans], axis=1)
                        batch_preds = meta_learner.predict(meta_X)
                        preds.extend(batch_preds.tolist())
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
            else:
                model = get_base_model(m_type, channels, classes)
                with torch.no_grad():
                    for i in range(0, len(tensor), 64):
                        out = model(tensor[i:i + 64])
                        preds.extend(torch.argmax(out, dim=1).cpu().numpy().tolist())
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

            stage_labels = [class_names[p] for p in preds]
            stats        = compute_aasm_stats(stage_labels, class_names)
            
            results.append({
                "stages":     stage_labels,
                "stages_int": preds,
                "stats":      stats,
                "model_info": {
                    "name":     f"{m_type}_{channels}ch_{classes}cls",
                    "type":     m_type,
                    "channels": channels,
                    "classes":  classes,
                    "labels":   class_names,
                },
            })

        print(f"  Done → {len(tensor)} epochs processed for {len(model_types)} models.")


        # Save to Supabase if patient_id is provided
        if patient_id and doctor_id:
            for res_idx, res in enumerate(results):
                try:
                    # Generate plot
                    fig, ax = plt.subplots(figsize=(10, 3))
                    y_vals = res["stages_int"]
                    x_vals = range(len(y_vals))
                    ax.plot(x_vals, y_vals, drawstyle='steps-post', color='#c0392b')
                    ax.set_yticks(range(len(class_names)))
                    ax.set_yticklabels(class_names)
                    ax.invert_yaxis()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    img_filename = f"hypno_{uuid.uuid4()}.png"
                    img_path = os.path.join(tempfile.gettempdir(), img_filename)
                    fig.savefig(img_path, bbox_inches='tight', dpi=100)
                    plt.close(fig)
                    
                    # Upload to storage
                    with open(img_path, 'rb') as img_f:
                        supabase.storage.from_("hypnograms").upload(img_filename, img_f.read(), {"content-type": "image/png"})
                    
                    public_url = supabase.storage.from_("hypnograms").get_public_url(img_filename)
                    res["hypnogram_url"] = public_url
                    
                    # Insert analysis record
                    new_analysis = {
                        "patient_id": patient_id,
                        "doctor_id": doctor_id,
                        "edf_filename": f.filename,
                        "configuration": res["model_info"],
                        "metrics": res["stats"],
                        "hypnogram_url": public_url
                    }
                    inserted = supabase.table("analyses").insert(new_analysis).execute()
                    if inserted.data:
                        res["analysis_id"] = inserted.data[0]["id"]
                        
                except Exception as e:
                    print(f"Failed to upload to Supabase: {e}")
                    
        return jsonify({
            "results": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/predict_osa", methods=["POST"])
def predict_osa():
    init_osa_predictor()
    if osa_predictor.osa_model is None:
        return jsonify({"error": "OSA models not initialized."}), 500
    
    try:
        data = request.json
        stages_int = data.get("stages_int", [])
        class_names = data.get("class_names", [])
        clinical = data.get("clinical_data", {})
        
        if not stages_int or not class_names:
            return jsonify({"error": "Missing stages data."}), 400
            
        # 1. Compute Hypnogram Features
        hypno = np.array(stages_int)
        n = len(hypno)
        epoch_min = EPOCH_SEC / 60.0
        
        # Identify stage indices
        label_map = {name: idx for idx, name in enumerate(class_names)}
        wake_idx = label_map.get("Wake", 0)
        rem_idx = label_map.get("REM", 2 if len(class_names)==3 else 4)
        n1_idx = label_map.get("N1", 1) if "N1" in label_map else label_map.get("NREM", 1)
        n2_idx = label_map.get("N2", 2) if "N2" in label_map else label_map.get("NREM", 1)
        n3_idx = label_map.get("N3", 3) if "N3" in label_map else label_map.get("NREM", 1)
        
        sleep_epochs = np.where(hypno != wake_idx)[0]
        sleep_onset = sleep_epochs[0] if len(sleep_epochs) > 0 else n
        last_sleep = sleep_epochs[-1] if len(sleep_epochs) > 0 else n
        
        tib_min = n * epoch_min
        tst_min = len(sleep_epochs) * epoch_min
        sol_min = sleep_onset * epoch_min
        spt_min = (last_sleep - sleep_onset + 1) * epoch_min if len(sleep_epochs) > 0 else 0
        se = (tst_min / tib_min * 100) if tib_min > 0 else 0
        
        post_onset = hypno[sleep_onset:] if len(sleep_epochs) > 0 else []
        waso_min = np.sum(post_onset == wake_idx) * epoch_min
        
        n1_min = np.sum(hypno == n1_idx) * epoch_min
        n2_min = np.sum(hypno == n2_idx) * epoch_min
        n3_min = np.sum(hypno == n3_idx) * epoch_min
        rem_min = np.sum(hypno == rem_idx) * epoch_min
        
        # REM / N3 Latency
        rem_epochs = np.where(hypno == rem_idx)[0]
        n3_epochs = np.where(hypno == n3_idx)[0]
        rem_latency_min = (rem_epochs[0] - sleep_onset) * epoch_min if len(rem_epochs) > 0 else 0
        n3_latency_min = (n3_epochs[0] - sleep_onset) * epoch_min if len(n3_epochs) > 0 else 0
        
        # Bouts and Transitions
        wake_bouts = 0
        rem_bouts = 0
        shifts = 0
        transitions = {(wake_idx, n1_idx): 0, (rem_idx, wake_idx): 0, (n2_idx, wake_idx): 0, 
                       (n3_idx, wake_idx): 0, (n2_idx, rem_idx): 0, (n1_idx, wake_idx): 0}
        total_transitions_from = {wake_idx: 0, rem_idx: 0, n2_idx: 0, n3_idx: 0, n1_idx: 0}
        
        for i in range(1, len(post_onset)):
            prev, curr = post_onset[i-1], post_onset[i]
            if prev != curr:
                shifts += 1
                if (prev, curr) in transitions:
                    transitions[(prev, curr)] += 1
                if prev in total_transitions_from:
                    total_transitions_from[prev] += 1
            if prev != wake_idx and curr == wake_idx: wake_bouts += 1
            if prev != rem_idx and curr == rem_idx: rem_bouts += 1
            
        mean_wake_bout_min = (waso_min / wake_bouts) if wake_bouts > 0 else 0
        mean_rem_bout_min = (rem_min / rem_bouts) if rem_bouts > 0 else 0
        frag_index = (shifts / (tst_min / 60)) if tst_min > 0 else 0
        
        nrem_min = n1_min + n2_min + n3_min
        nrem_rem_ratio = (nrem_min / rem_min) if rem_min > 0 else 0
        light_deep_ratio = ((n1_min + n2_min) / n3_min) if n3_min > 0 else 0
        
        def tp(f, t): return transitions[(f, t)] / total_transitions_from[f] if total_transitions_from[f] > 0 else 0
        
        calc_feats = {
            "sol_min": sol_min, "tst_min": tst_min, "tib_min": tib_min, "spt_min": spt_min,
            "sleep_efficiency": se, "waso_min": waso_min,
            "N1_pct": (n1_min/tst_min*100) if tst_min else 0,
            "N2_pct": (n2_min/tst_min*100) if tst_min else 0,
            "N3_pct": (n3_min/tst_min*100) if tst_min else 0,
            "REM_pct": (rem_min/tst_min*100) if tst_min else 0,
            "rem_latency_min": rem_latency_min, "n3_latency_min": n3_latency_min,
            "frag_index": frag_index, "n_wake_bouts": wake_bouts, "mean_wake_bout_min": mean_wake_bout_min,
            "n_rem_cycles": rem_bouts, "mean_rem_bout_min": mean_rem_bout_min,
            "nrem_rem_ratio": nrem_rem_ratio, "light_deep_ratio": light_deep_ratio,
            "p_W_N1": tp(wake_idx, n1_idx), "p_REM_W": tp(rem_idx, wake_idx),
            "p_N2_W": tp(n2_idx, wake_idx), "p_N3_W": tp(n3_idx, wake_idx),
            "p_N2_REM": tp(n2_idx, rem_idx), "p_N1_W": tp(n1_idx, wake_idx)
        }
        
        # 2. Map calculated features to the exact expected redundant columns as well
        calc_feats.update({
            "slpeffp": se, "slplatp": sol_min, 
            "timest1p": calc_feats["N1_pct"], "timest2p": calc_feats["N2_pct"],
            "timest34": calc_feats["N3_pct"], "timeremp": calc_feats["REM_pct"],
            "waso": waso_min, "remt1p": calc_feats["N1_pct"], "remt34p": calc_feats["N3_pct"]
        })
        
        # 3. Handle Clinical Data
        gender_val = 1 if str(clinical.get("gender", "")).lower().startswith("m") else 2
        input_feats = {
            "age_s2": float(clinical.get("age") or osa_predictor.osa_medians.get("age_s2", 50)),
            "bmi_s2": float(clinical.get("bmi") or osa_predictor.osa_medians.get("bmi_s2", 28)),
            "gender": gender_val,
            "avgsat": float(clinical.get("avgsat") or osa_predictor.osa_medians.get("avgsat", 94)),
            "minsat": float(clinical.get("minsat") or osa_predictor.osa_medians.get("minsat", 85)),
        }
        input_feats["hypoxia_score"] = (100 - input_feats["avgsat"]) * (100 - input_feats["minsat"]) / 100
        
        # 4. Build feature vector and impute
        feature_vector = []
        for col in osa_predictor.osa_features:
            if col in calc_feats:
                feature_vector.append(calc_feats[col])
            elif col in input_feats:
                feature_vector.append(input_feats[col])
            else:
                feature_vector.append(osa_predictor.osa_medians.get(col, 0))
                
        X = pd.DataFrame([feature_vector], columns=osa_predictor.osa_features)
        
        # 5. Predict & SHAP
        pred_idx = int(osa_predictor.osa_model.predict(X)[0])
        pred_label = str(osa_predictor.osa_le.inverse_transform([pred_idx])[0]) if osa_predictor.osa_le else str(pred_idx)
        
        shap_values = osa_predictor.explainer(X)
        shap_vals = shap_values.values[0]
        # Depending on XGBoost objective, shap_values.values might have shape (n_features, n_classes)
        # We take the impacts for the predicted class if it's multi-class
        if len(shap_vals.shape) > 1:
            impacts = shap_vals[:, pred_idx]
        else:
            impacts = shap_vals
            
        feature_impacts = []
        for i, col in enumerate(osa_predictor.osa_features):
            val = float(X.iloc[0, i])
            imp = float(impacts[i])
            if abs(imp) > 0.001:
                feature_impacts.append({"feature": col, "value": round(val, 2), "impact": round(imp, 4)})
                
        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_impacts = feature_impacts[:6] # Top 6 drivers
        

        # Save to Supabase if patient_id is provided
        if patient_id and doctor_id:
            for res_idx, res in enumerate(results):
                try:
                    # Generate plot
                    fig, ax = plt.subplots(figsize=(10, 3))
                    y_vals = res["stages_int"]
                    x_vals = range(len(y_vals))
                    ax.plot(x_vals, y_vals, drawstyle='steps-post', color='#c0392b')
                    ax.set_yticks(range(len(class_names)))
                    ax.set_yticklabels(class_names)
                    ax.invert_yaxis()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    img_filename = f"hypno_{uuid.uuid4()}.png"
                    img_path = os.path.join(tempfile.gettempdir(), img_filename)
                    fig.savefig(img_path, bbox_inches='tight', dpi=100)
                    plt.close(fig)
                    
                    # Upload to storage
                    with open(img_path, 'rb') as img_f:
                        supabase.storage.from_("hypnograms").upload(img_filename, img_f.read(), {"content-type": "image/png"})
                    
                    public_url = supabase.storage.from_("hypnograms").get_public_url(img_filename)
                    res["hypnogram_url"] = public_url
                    
                    # Insert analysis record
                    new_analysis = {
                        "patient_id": patient_id,
                        "doctor_id": doctor_id,
                        "edf_filename": f.filename,
                        "configuration": res["model_info"],
                        "metrics": res["stats"],
                        "hypnogram_url": public_url
                    }
                    inserted = supabase.table("analyses").insert(new_analysis).execute()
                    if inserted.data:
                        res["analysis_id"] = inserted.data[0]["id"]
                        
                except Exception as e:
                    print(f"Failed to upload to Supabase: {e}")
                    
        return jsonify({
            "severity": pred_label,
            "shap_explanations": top_impacts
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/channels", methods=["POST"])
def list_channels():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    import tempfile, mne
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".edf")
    os.close(tmp_fd)
    f.save(tmp_path)
    try:
        raw      = mne.io.read_raw_edf(tmp_path, preload=False, verbose=False)
        channels = raw.ch_names
        sfreq    = raw.info["sfreq"]
        duration = raw.times[-1]
        try:
            resolved = resolve_channels(channels)
        except Exception as e:
            resolved = {"error": str(e)}

        # Save to Supabase if patient_id is provided
        if patient_id and doctor_id:
            for res_idx, res in enumerate(results):
                try:
                    # Generate plot
                    fig, ax = plt.subplots(figsize=(10, 3))
                    y_vals = res["stages_int"]
                    x_vals = range(len(y_vals))
                    ax.plot(x_vals, y_vals, drawstyle='steps-post', color='#c0392b')
                    ax.set_yticks(range(len(class_names)))
                    ax.set_yticklabels(class_names)
                    ax.invert_yaxis()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    img_filename = f"hypno_{uuid.uuid4()}.png"
                    img_path = os.path.join(tempfile.gettempdir(), img_filename)
                    fig.savefig(img_path, bbox_inches='tight', dpi=100)
                    plt.close(fig)
                    
                    # Upload to storage
                    with open(img_path, 'rb') as img_f:
                        supabase.storage.from_("hypnograms").upload(img_filename, img_f.read(), {"content-type": "image/png"})
                    
                    public_url = supabase.storage.from_("hypnograms").get_public_url(img_filename)
                    res["hypnogram_url"] = public_url
                    
                    # Insert analysis record
                    new_analysis = {
                        "patient_id": patient_id,
                        "doctor_id": doctor_id,
                        "edf_filename": f.filename,
                        "configuration": res["model_info"],
                        "metrics": res["stats"],
                        "hypnogram_url": public_url
                    }
                    inserted = supabase.table("analyses").insert(new_analysis).execute()
                    if inserted.data:
                        res["analysis_id"] = inserted.data[0]["id"]
                        
                except Exception as e:
                    print(f"Failed to upload to Supabase: {e}")
                    
        return jsonify({
            "channels":     channels,
            "sfreq":        sfreq,
            "duration_sec": round(duration, 1),
            "resolved":     resolved,
            "expected_slots": CH_ORDER,
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/health")
def health():
    info = {
        "status":     "ok",
        "device":     str(device),
    }
    if device.type == "cuda":
        info["gpu"]     = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return jsonify(info)


if __name__ == "__main__":
    print("\nStarting Sleep Staging Server — http://localhost:5000")
    app.run(debug=False, port=5000)
