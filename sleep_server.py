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
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
import joblib
import xgboost as xgb
import shap

app = Flask(__name__)
CORS(app)

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
            
        # ═══════════════════════════════════════════════════════════════
        # 1. COMPUTE HYPNOGRAM FEATURES (25 AASM-compliant features)
        # ═══════════════════════════════════════════════════════════════
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
        
        post_onset = hypno[sleep_onset:] if len(sleep_epochs) > 0 else np.array([], dtype=int)
        waso_min = float(np.sum(post_onset == wake_idx)) * epoch_min
        
        n1_min = float(np.sum(hypno == n1_idx)) * epoch_min
        n2_min = float(np.sum(hypno == n2_idx)) * epoch_min
        n3_min = float(np.sum(hypno == n3_idx)) * epoch_min
        rem_min = float(np.sum(hypno == rem_idx)) * epoch_min
        
        # REM / N3 Latency
        rem_epochs = np.where(hypno == rem_idx)[0]
        n3_epochs = np.where(hypno == n3_idx)[0]
        rem_latency_min = float((rem_epochs[0] - sleep_onset) * epoch_min) if len(rem_epochs) > 0 else -1.0
        n3_latency_min = float((n3_epochs[0] - sleep_onset) * epoch_min) if len(n3_epochs) > 0 else -1.0
        
        # Bouts and Transitions
        wake_bouts = 0
        rem_bouts = 0
        shifts = 0
        transitions = {(wake_idx, n1_idx): 0, (rem_idx, wake_idx): 0, (n2_idx, wake_idx): 0, 
                       (n3_idx, wake_idx): 0, (n2_idx, rem_idx): 0, (n1_idx, wake_idx): 0}
        total_transitions_from = {wake_idx: 0, rem_idx: 0, n2_idx: 0, n3_idx: 0, n1_idx: 0}
        
        for i in range(1, len(post_onset)):
            prev, curr = int(post_onset[i-1]), int(post_onset[i])
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
        
        def tp(f, t): return transitions.get((f, t), 0) / total_transitions_from.get(f, 1) if total_transitions_from.get(f, 0) > 0 else 0
        
        n1_pct = (n1_min / tst_min * 100) if tst_min else 0
        n2_pct = (n2_min / tst_min * 100) if tst_min else 0
        n3_pct = (n3_min / tst_min * 100) if tst_min else 0
        rem_pct = (rem_min / tst_min * 100) if tst_min else 0
        
        calc_feats = {
            "sol_min": sol_min, "tst_min": tst_min, "tib_min": tib_min, "spt_min": spt_min,
            "sleep_efficiency": se, "waso_min": waso_min,
            "N1_pct": n1_pct, "N2_pct": n2_pct, "N3_pct": n3_pct, "REM_pct": rem_pct,
            "rem_latency_min": rem_latency_min, "n3_latency_min": n3_latency_min,
            "frag_index": frag_index, "n_wake_bouts": wake_bouts, "mean_wake_bout_min": mean_wake_bout_min,
            "n_rem_cycles": rem_bouts, "mean_rem_bout_min": mean_rem_bout_min,
            "nrem_rem_ratio": nrem_rem_ratio, "light_deep_ratio": light_deep_ratio,
            "p_W_N1": tp(wake_idx, n1_idx), "p_REM_W": tp(rem_idx, wake_idx),
            "p_N2_W": tp(n2_idx, wake_idx), "p_N3_W": tp(n3_idx, wake_idx),
            "p_N2_REM": tp(n2_idx, rem_idx), "p_N1_W": tp(n1_idx, wake_idx)
        }
        
        # Map to PSG-scored column aliases expected by the model
        calc_feats.update({
            "slpeffp": se, "slplatp": sol_min, 
            "timest1p": n1_pct, "timest2p": n2_pct,
            "timest34": n3_pct, "timeremp": rem_pct,
            "waso": waso_min, "remt1p": n1_pct, "remt34p": n3_pct
        })
        
        # ═══════════════════════════════════════════════════════════════
        # 2. CLINICAL DATA — demographics + oximetry + arousal indices
        # ═══════════════════════════════════════════════════════════════
        def safe_float(val, fallback_key, default=0):
            if val is not None and val != "" and val != "null":
                try: return float(val)
                except: pass
            return float(osa_predictor.osa_medians.get(fallback_key, default))
        
        gender_val = 1 if str(clinical.get("gender", "")).lower().startswith("m") else 2
        
        input_feats = {
            # Demographics
            "age_s2": safe_float(clinical.get("age"), "age_s2", 50),
            "bmi_s2": safe_float(clinical.get("bmi"), "bmi_s2", 28),
            "gender": gender_val,
            # Oximetry
            "avgsat": safe_float(clinical.get("avgsat"), "avgsat", 94),
            "minsat": safe_float(clinical.get("minsat"), "minsat", 85),
            "pctsa90h": safe_float(clinical.get("pctsa90h"), "pctsa90h", 0),
            "pctsa85h": safe_float(clinical.get("pctsa85h"), "pctsa85h", 0),
            "pctsa95h": safe_float(clinical.get("pctsa95h"), "pctsa95h", 0),
            # Arousal indices
            "ai_all": safe_float(clinical.get("ai_all"), "ai_all", 0),
            "ai_nrem": safe_float(clinical.get("ai_nrem"), "ai_nrem", 0),
            "ai_rem": safe_float(clinical.get("ai_rem"), "ai_rem", 0),
        }
        
        # ═══════════════════════════════════════════════════════════════
        # 3. FEATURE ENGINEERING — interaction features
        # ═══════════════════════════════════════════════════════════════
        engineered = {}
        
        # Hypoxia severity score
        engineered["hypoxia_score"] = (
            input_feats["pctsa95h"] * 1.0 +
            input_feats["pctsa90h"] * 3.0 +
            input_feats["pctsa85h"] * 9.0
        )
        
        # Arousal × fragmentation interaction
        engineered["arousal_frag"] = input_feats["ai_all"] * calc_feats["frag_index"]
        
        # SpO2 drop depth
        engineered["sat_drop"] = input_feats["avgsat"] - input_feats["minsat"]
        
        # Arousal per wake bout
        engineered["arousal_per_bout"] = input_feats["ai_all"] / (calc_feats["n_wake_bouts"] + 1)
        
        # REM disruption ratio
        engineered["rem_nrem_arousal_ratio"] = input_feats["ai_rem"] / (input_feats["ai_nrem"] + 0.1)
        
        # WASO × arousal
        engineered["waso_arousal"] = calc_feats["waso_min"] * input_feats["ai_all"]
        
        # N3 suppression
        engineered["n3_suppression"] = calc_feats["N3_pct"] / (input_feats["ai_all"] + 0.1)
        
        # BMI × arousal
        engineered["bmi_arousal"] = input_feats["bmi_s2"] * input_feats["ai_all"]
        
        # ═══════════════════════════════════════════════════════════════
        # 4. BUILD FEATURE VECTOR (aligned to osa_features order)
        # ═══════════════════════════════════════════════════════════════
        all_feats = {**calc_feats, **input_feats, **engineered}
        
        feature_vector = []
        for col in osa_predictor.osa_features:
            if col in all_feats:
                feature_vector.append(all_feats[col])
            else:
                feature_vector.append(osa_predictor.osa_medians.get(col, 0))
                
        X = pd.DataFrame([feature_vector], columns=osa_predictor.osa_features)
        
        # ═══════════════════════════════════════════════════════════════
        # 5. PREDICT + SHAP EXPLAIN
        # ═══════════════════════════════════════════════════════════════
        from osa_predictor import predict_osa_severity
        pred_label, proba_dict, feature_impacts = predict_osa_severity(X)
        
        # ═══════════════════════════════════════════════════════════════
        # 6. BUILD RESPONSE — full clinical report data
        # ═══════════════════════════════════════════════════════════════
        
        # AASM features table for the report
        aasm_features = {
            "timing": {
                "tst_min": round(tst_min, 1),
                "tib_min": round(tib_min, 1),
                "spt_min": round(spt_min, 1),
                "sol_min": round(sol_min, 1),
                "se_pct": round(se, 1),
                "waso_min": round(waso_min, 1),
            },
            "stages": {
                "N1_pct": round(n1_pct, 1),
                "N2_pct": round(n2_pct, 1),
                "N3_pct": round(n3_pct, 1),
                "REM_pct": round(rem_pct, 1),
            },
            "latencies": {
                "rem_latency_min": round(rem_latency_min, 1),
                "n3_latency_min": round(n3_latency_min, 1),
            },
            "fragmentation": {
                "frag_index": round(frag_index, 1),
                "n_wake_bouts": int(wake_bouts),
                "n_rem_cycles": int(rem_bouts),
                "nrem_rem_ratio": round(nrem_rem_ratio, 2),
            },
        }
        
        # Clinical interpretation
        interpretation = []
        if se < 85:
            interpretation.append({"type": "warning", "text": "Efficacité du sommeil réduite (<85%) — fragmentation significative"})
        if rem_pct < 15:
            interpretation.append({"type": "warning", "text": "Temps REM insuffisant (<15%) — perturbation du sommeil paradoxal"})
        if n3_pct < 10:
            interpretation.append({"type": "warning", "text": "Sommeil profond (N3) réduit (<10%) — régénération altérée"})
        if frag_index > 30:
            interpretation.append({"type": "danger", "text": "Index de fragmentation élevé — sommeil très instable"})
        if rem_latency_min > 0 and rem_latency_min < 60:
            interpretation.append({"type": "info", "text": "Latence REM courte — possible narcolepsie ou privation de sommeil"})
        if sol_min < 5:
            interpretation.append({"type": "info", "text": "Latence d'endormissement très courte — possible privation sévère"})
        if waso_min > 60:
            interpretation.append({"type": "warning", "text": "WASO élevé (>60 min) — réveils nocturnes fréquents"})
        
        # Model info
        model_used = "Stacking (XGB+LGBM+MLP→LR)" if osa_predictor.stacking_model else "XGBoost"
        
        return jsonify({
            "severity": pred_label,
            "probabilities": proba_dict,
            "model_used": model_used,
            "aasm_features": aasm_features,
            "interpretation": interpretation,
            "shap_explanations": feature_impacts[:10],  # Top 10 drivers
            "shap_all": feature_impacts,                 # All for detailed view
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
