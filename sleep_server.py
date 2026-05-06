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


@app.route("/extract_features", methods=["POST"])
def extract_features():
    """Compute and return hypnogram-derived features for display before OSA prediction."""
    try:
        data = request.json
        stages_int = data.get("stages_int", [])
        class_names = data.get("class_names", [])

        if not stages_int or not class_names:
            return jsonify({"error": "Missing stages data."}), 400

        hypno = np.array(stages_int)
        n = len(hypno)
        epoch_min = EPOCH_SEC / 60.0
        is_3cls = len(class_names) == 3

        label_map = {name: idx for idx, name in enumerate(class_names)}
        wake_idx = label_map.get("Wake", 0)
        rem_idx = label_map.get("REM", 2 if is_3cls else 4)

        sleep_epochs = np.where(hypno != wake_idx)[0]
        sleep_onset = int(sleep_epochs[0]) if len(sleep_epochs) > 0 else n
        last_sleep = int(sleep_epochs[-1]) if len(sleep_epochs) > 0 else n

        tib_min = n * epoch_min
        tst_min = len(sleep_epochs) * epoch_min
        sol_min = sleep_onset * epoch_min
        spt_min = (last_sleep - sleep_onset + 1) * epoch_min if len(sleep_epochs) > 0 else 0
        se = (tst_min / tib_min * 100) if tib_min > 0 else 0

        post_onset = hypno[sleep_onset:] if len(sleep_epochs) > 0 else np.array([], dtype=int)
        waso_min = float(np.sum(post_onset == wake_idx)) * epoch_min
        rem_min = float(np.sum(hypno == rem_idx)) * epoch_min

        if is_3cls:
            nrem_idx = label_map.get("NREM", 1)
            total_nrem_min = float(np.sum(hypno == nrem_idx)) * epoch_min
            n1_min = total_nrem_min * 0.073
            n2_min = total_nrem_min * 0.710
            n3_min = total_nrem_min * 0.217
            n1_idx_t = nrem_idx
            n2_idx_t = nrem_idx
            n3_idx_t = nrem_idx
        else:
            n1_idx_t = label_map.get("N1", 1)
            n2_idx_t = label_map.get("N2", 2)
            n3_idx_t = label_map.get("N3", 3)
            n1_min = float(np.sum(hypno == n1_idx_t)) * epoch_min
            n2_min = float(np.sum(hypno == n2_idx_t)) * epoch_min
            n3_min = float(np.sum(hypno == n3_idx_t)) * epoch_min

        rem_epochs = np.where(hypno == rem_idx)[0]
        rem_latency_min = float((rem_epochs[0] - sleep_onset) * epoch_min) if len(rem_epochs) > 0 else -1.0
        if is_3cls:
            n3_latency_min = -1.0
        else:
            n3_epochs = np.where(hypno == n3_idx_t)[0]
            n3_latency_min = float((n3_epochs[0] - sleep_onset) * epoch_min) if len(n3_epochs) > 0 else -1.0

        wake_bouts = 0
        rem_bouts = 0
        shifts = 0
        for i in range(1, len(post_onset)):
            prev, curr = int(post_onset[i-1]), int(post_onset[i])
            if prev != curr:
                shifts += 1
            if prev != wake_idx and curr == wake_idx: wake_bouts += 1
            if prev != rem_idx and curr == rem_idx: rem_bouts += 1

        mean_wake_bout_min = (waso_min / wake_bouts) if wake_bouts > 0 else 0
        mean_rem_bout_min = (rem_min / rem_bouts) if rem_bouts > 0 else 0
        frag_index = (shifts / (tst_min / 60)) if tst_min > 0 else 0

        nrem_min = n1_min + n2_min + n3_min
        nrem_rem_ratio = (nrem_min / rem_min) if rem_min > 0 else 0
        light_deep_ratio = ((n1_min + n2_min) / n3_min) if n3_min > 0 else 0

        n1_pct = (n1_min / tst_min * 100) if tst_min else 0
        n2_pct = (n2_min / tst_min * 100) if tst_min else 0
        n3_pct = (n3_min / tst_min * 100) if tst_min else 0
        rem_pct = (rem_min / tst_min * 100) if tst_min else 0

        # REM distribution across sleep thirds
        if len(rem_epochs) > 0 and spt_min > 0:
            spt_epochs = last_sleep - sleep_onset + 1
            third = spt_epochs / 3.0
            rem_in_spt = rem_epochs[(rem_epochs >= sleep_onset) & (rem_epochs <= last_sleep)]
            rem_first_third = int(np.sum(rem_in_spt < (sleep_onset + third)))
            rem_last_two = int(np.sum(rem_in_spt >= (sleep_onset + third)))
            total_rem = len(rem_in_spt)
            remt1p_val = (rem_first_third / total_rem * 100) if total_rem > 0 else 0
            remt34p_val = (rem_last_two / total_rem * 100) if total_rem > 0 else 0
        else:
            remt1p_val = 0
            remt34p_val = 0

        # Return features grouped for display
        return jsonify({
            "timing": [
                {"name": "Total Sleep Time (TST)", "key": "tst_min", "value": round(tst_min, 1), "unit": "min"},
                {"name": "Time in Bed (TIB)", "key": "tib_min", "value": round(tib_min, 1), "unit": "min"},
                {"name": "Sleep Period Time (SPT)", "key": "spt_min", "value": round(spt_min, 1), "unit": "min"},
                {"name": "Sleep Onset Latency (SOL)", "key": "sol_min", "value": round(sol_min, 1), "unit": "min"},
                {"name": "Sleep Efficiency (SE)", "key": "sleep_efficiency", "value": round(se, 1), "unit": "%"},
                {"name": "WASO", "key": "waso_min", "value": round(waso_min, 1), "unit": "min"},
            ],
            "stages": [
                {"name": "N1 %", "key": "N1_pct", "value": round(n1_pct, 1), "unit": "%", "note": "estimated" if is_3cls else ""},
                {"name": "N2 %", "key": "N2_pct", "value": round(n2_pct, 1), "unit": "%", "note": "estimated" if is_3cls else ""},
                {"name": "N3 (Deep) %", "key": "N3_pct", "value": round(n3_pct, 1), "unit": "%", "note": "estimated" if is_3cls else ""},
                {"name": "REM %", "key": "REM_pct", "value": round(rem_pct, 1), "unit": "%"},
            ],
            "latencies": [
                {"name": "REM Latency", "key": "rem_latency_min", "value": round(rem_latency_min, 1), "unit": "min"},
                {"name": "N3 Latency", "key": "n3_latency_min", "value": round(n3_latency_min, 1), "unit": "min", "note": "N/A" if is_3cls else ""},
            ],
            "fragmentation": [
                {"name": "Fragmentation Index", "key": "frag_index", "value": round(frag_index, 1), "unit": "shifts/h"},
                {"name": "Wake Bouts", "key": "n_wake_bouts", "value": int(wake_bouts), "unit": ""},
                {"name": "Mean Wake Bout", "key": "mean_wake_bout_min", "value": round(mean_wake_bout_min, 1), "unit": "min"},
                {"name": "REM Cycles", "key": "n_rem_cycles", "value": int(rem_bouts), "unit": ""},
                {"name": "Mean REM Bout", "key": "mean_rem_bout_min", "value": round(mean_rem_bout_min, 1), "unit": "min"},
                {"name": "NREM/REM Ratio", "key": "nrem_rem_ratio", "value": round(nrem_rem_ratio, 2), "unit": ""},
                {"name": "Light/Deep Ratio", "key": "light_deep_ratio", "value": round(light_deep_ratio, 2), "unit": ""},
            ],
            "rem_distribution": [
                {"name": "REM in 1st Third", "key": "remt1p", "value": round(remt1p_val, 1), "unit": "%"},
                {"name": "REM in Last 2/3", "key": "remt34p", "value": round(remt34p_val, 1), "unit": "%"},
            ],
            "metadata": {
                "n_epochs": n,
                "is_3class": is_3cls,
                "class_names": class_names,
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
        is_3cls = len(class_names) == 3  # Wake/NREM/REM
        
        # Identify stage indices
        label_map = {name: idx for idx, name in enumerate(class_names)}
        wake_idx = label_map.get("Wake", 0)
        rem_idx = label_map.get("REM", 2 if is_3cls else 4)
        
        sleep_epochs = np.where(hypno != wake_idx)[0]
        sleep_onset = int(sleep_epochs[0]) if len(sleep_epochs) > 0 else n
        last_sleep = int(sleep_epochs[-1]) if len(sleep_epochs) > 0 else n
        
        tib_min = n * epoch_min
        tst_min = len(sleep_epochs) * epoch_min
        sol_min = sleep_onset * epoch_min
        spt_min = (last_sleep - sleep_onset + 1) * epoch_min if len(sleep_epochs) > 0 else 0
        se = (tst_min / tib_min * 100) if tib_min > 0 else 0
        
        post_onset = hypno[sleep_onset:] if len(sleep_epochs) > 0 else np.array([], dtype=int)
        waso_min = float(np.sum(post_onset == wake_idx)) * epoch_min
        rem_min = float(np.sum(hypno == rem_idx)) * epoch_min
        
        if is_3cls:
            # 3-class: NREM is a single category — estimate N1/N2/N3 using
            # population ratios from SHHS2 (N1≈6%, N2≈60%, N3≈16% of NREM)
            nrem_idx = label_map.get("NREM", 1)
            total_nrem_min = float(np.sum(hypno == nrem_idx)) * epoch_min
            n1_min = total_nrem_min * 0.073  # ~7.3% of NREM is N1
            n2_min = total_nrem_min * 0.710  # ~71% of NREM is N2
            n3_min = total_nrem_min * 0.217  # ~21.7% of NREM is N3
            
            # For transitions, use the single NREM index
            n1_idx_t = nrem_idx
            n2_idx_t = nrem_idx
            n3_idx_t = nrem_idx
        else:
            # 5-class: use actual indices
            n1_idx_t = label_map.get("N1", 1)
            n2_idx_t = label_map.get("N2", 2)
            n3_idx_t = label_map.get("N3", 3)
            n1_min = float(np.sum(hypno == n1_idx_t)) * epoch_min
            n2_min = float(np.sum(hypno == n2_idx_t)) * epoch_min
            n3_min = float(np.sum(hypno == n3_idx_t)) * epoch_min
        
        # REM / N3 Latency
        rem_epochs = np.where(hypno == rem_idx)[0]
        rem_latency_min = float((rem_epochs[0] - sleep_onset) * epoch_min) if len(rem_epochs) > 0 else -1.0
        if is_3cls:
            n3_latency_min = -1.0  # Can't determine N3 latency from 3-class
        else:
            n3_epochs = np.where(hypno == n3_idx_t)[0]
            n3_latency_min = float((n3_epochs[0] - sleep_onset) * epoch_min) if len(n3_epochs) > 0 else -1.0
        
        # Bouts and Transitions
        wake_bouts = 0
        rem_bouts = 0
        shifts = 0
        transitions = {(wake_idx, n1_idx_t): 0, (rem_idx, wake_idx): 0, (n2_idx_t, wake_idx): 0, 
                       (n3_idx_t, wake_idx): 0, (n2_idx_t, rem_idx): 0, (n1_idx_t, wake_idx): 0}
        total_transitions_from = {wake_idx: 0, rem_idx: 0, n2_idx_t: 0, n3_idx_t: 0, n1_idx_t: 0}
        
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
            "p_W_N1": tp(wake_idx, n1_idx_t), "p_REM_W": tp(rem_idx, wake_idx),
            "p_N2_W": tp(n2_idx_t, wake_idx), "p_N3_W": tp(n3_idx_t, wake_idx),
            "p_N2_REM": tp(n2_idx_t, rem_idx), "p_N1_W": tp(n1_idx_t, wake_idx)
        }
        
        # Map to PSG-scored column aliases expected by the model.
        # These columns come from SHHS2 PSG scoring and have specific semantics:
        #   slpeffp  = sleep efficiency %
        #   slplatp  = sleep latency in minutes
        #   timest1p = % of sleep period in stage 1 (N1)
        #   timest2p = % of sleep period in stage 2 (N2)
        #   timest34 = time in stage 3+4 (N3) in MINUTES (not %)
        #   timeremp = % of sleep period in REM
        #   waso     = wake after sleep onset in minutes
        #   remt1p   = % of REM occurring in first third of sleep period
        #   remt34p  = % of REM occurring in last two thirds of sleep period
        
        # Compute REM distribution across sleep thirds
        if len(rem_epochs) > 0 and spt_min > 0:
            spt_epochs = last_sleep - sleep_onset + 1
            third = spt_epochs / 3.0
            rem_in_spt = rem_epochs[(rem_epochs >= sleep_onset) & (rem_epochs <= last_sleep)]
            rem_first_third = np.sum(rem_in_spt < (sleep_onset + third))
            rem_last_two = np.sum(rem_in_spt >= (sleep_onset + third))
            total_rem = len(rem_in_spt)
            remt1p_val = (rem_first_third / total_rem * 100) if total_rem > 0 else 0
            remt34p_val = (rem_last_two / total_rem * 100) if total_rem > 0 else 0
        else:
            remt1p_val = 0
            remt34p_val = 0
        
        calc_feats.update({
            "slpeffp": se, "slplatp": sol_min, 
            "timest1p": n1_pct, "timest2p": n2_pct,
            "timest34": n3_min,       # N3 time in MINUTES (SHHS convention)
            "timeremp": rem_pct,
            "waso": waso_min,
            "remt1p": remt1p_val,      # % of REM in first third
            "remt34p": remt34p_val     # % of REM in last two thirds
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
        used_features = {}
        for col in osa_predictor.osa_features:
            if col in all_feats:
                feature_vector.append(all_feats[col])
                used_features[col] = round(all_feats[col], 4)
            else:
                med = osa_predictor.osa_medians.get(col, 0)
                feature_vector.append(med)
                used_features[col] = round(float(med), 4)
                
        X = pd.DataFrame([feature_vector], columns=osa_predictor.osa_features)
        
        # Debug: verify features vary between different files
        print(f"\n[OSA DEBUG] {n} epochs | 3cls={is_3cls}")
        print(f"  TST={tst_min:.1f}min  WASO={waso_min:.1f}min  SE={se:.1f}%  SOL={sol_min:.1f}min")
        print(f"  N1%={n1_pct:.1f}  N2%={n2_pct:.1f}  N3%={n3_pct:.1f}  REM%={rem_pct:.1f}")
        print(f"  frag={frag_index:.1f}  wake_bouts={wake_bouts}  rem_cycles={rem_bouts}")
        print(f"  timest34={calc_feats['timest34']:.1f}min  remt1p={calc_feats['remt1p']:.1f}%  remt34p={calc_feats['remt34p']:.1f}%")
        
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
            "used_features": used_features,
            "interpretation": interpretation,
            "shap_explanations": feature_impacts[:10],  # Top 10 drivers
            "shap_all": feature_impacts,                 # All for detailed view
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/parse_features_file", methods=["POST"])
def parse_features_file():
    """Parse an uploaded CSV or XML file and extract feature values for OSA prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    f = request.files["file"]
    fname = f.filename.lower()
    
    try:
        import io
        content = f.read()
        
        if fname.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            if len(df) == 0:
                return jsonify({"error": "CSV file is empty."}), 400
                
            features = {}
            # Handle standard single-row CSV vs our exported 'Section,Key,Value' format
            if list(df.columns) == ["Section", "Key", "Value"]:
                for _, row in df.iterrows():
                    sec = str(row["Section"])
                    # Import only specific sections that contain actual numeric features
                    if sec.startswith("ModelFeature") or sec.startswith("AASM_"):
                        col = str(row["Key"]).strip()
                        val = row["Value"]
                        if pd.isna(val): continue
                        
                        # Strip percent signs and units if present
                        val_str = str(val).replace("%", "").replace(" min", "").strip()
                        try:
                            features[col] = float(val_str)
                        except ValueError:
                            features[col] = val_str
            else:
                # Take first row as feature values
                row = df.iloc[0]
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val): continue
                    try:
                        features[col.strip()] = float(val)
                    except (ValueError, TypeError):
                        if isinstance(val, str):
                            features[col.strip()] = val.strip()
            
            return jsonify({
                "features": features,
                "source": fname,
                "n_columns": len(features),
                "all_columns": list(df.columns) if list(df.columns) != ["Section", "Key", "Value"] else list(features.keys()),
                "n_rows": len(df),
            })
        
        elif fname.endswith(".xml"):
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content.decode("utf-8", errors="replace"))
            features = {}
            
            # Try multiple XML structures:
            # 1. Flat: <features><feature_name>value</feature_name>...</features>
            # 2. Key-value: <features><feature name="x" value="y"/>...</features>
            # 3. Nested: <ScoredEvent>/<EventConcept> style (NSRR XML)
            
            # Strategy: walk all leaf elements
            for elem in root.iter():
                if len(elem) == 0 and elem.text and elem.text.strip():
                    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                    try:
                        features[tag] = float(elem.text.strip())
                    except ValueError:
                        features[tag] = elem.text.strip()
                # Check for name/value attributes
                if elem.get("name") and elem.get("value"):
                    try:
                        features[elem.get("name")] = float(elem.get("value"))
                    except ValueError:
                        features[elem.get("name")] = elem.get("value")
            
            return jsonify({
                "features": features,
                "source": fname,
                "n_columns": len(features),
                "all_columns": list(features.keys()),
            })
        
        else:
            return jsonify({"error": f"Unsupported format: {fname}. Use .csv or .xml"}), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict_osa_custom", methods=["POST"])
def predict_osa_custom():
    """Run OSA severity prediction from manually provided feature values (no EDF needed)."""
    init_osa_predictor()
    if osa_predictor.osa_model is None:
        return jsonify({"error": "OSA models not initialized."}), 500
    
    try:
        data = request.json
        features = data.get("features", {})
        
        if not features:
            return jsonify({"error": "No features provided."}), 400
        
        def safe_float(val, fallback_key, default=0):
            if val is not None and val != "" and val != "null":
                try: return float(val)
                except: pass
            return float(osa_predictor.osa_medians.get(fallback_key, default))
        
        # Map user-friendly keys to model-expected keys
        key_aliases = {
            "age": "age_s2", "bmi": "bmi_s2",
            "sleep_efficiency": "slpeffp", "sol_min": "slplatp",
            "N1_pct": "timest1p", "N2_pct": "timest2p",
            "N3_min": "timest34", "REM_pct": "timeremp",
            "waso_min": "waso",
        }
        
        # Build flat feature dict with all possible names
        all_feats = {}
        for k, v in features.items():
            all_feats[k] = safe_float(v, k, 0)
            if k in key_aliases:
                all_feats[key_aliases[k]] = safe_float(v, key_aliases[k], 0)
        
        # Handle gender specially
        if "gender" in features:
            gval = str(features["gender"]).strip().lower()
            all_feats["gender"] = 1 if gval.startswith("m") or gval == "1" else 2
        
        # Compute engineered features if components are present
        ai_all = all_feats.get("ai_all", osa_predictor.osa_medians.get("ai_all", 0))
        ai_nrem = all_feats.get("ai_nrem", osa_predictor.osa_medians.get("ai_nrem", 0))
        ai_rem = all_feats.get("ai_rem", osa_predictor.osa_medians.get("ai_rem", 0))
        avgsat = all_feats.get("avgsat", osa_predictor.osa_medians.get("avgsat", 94))
        minsat = all_feats.get("minsat", osa_predictor.osa_medians.get("minsat", 85))
        pctsa90h = all_feats.get("pctsa90h", osa_predictor.osa_medians.get("pctsa90h", 0))
        pctsa85h = all_feats.get("pctsa85h", osa_predictor.osa_medians.get("pctsa85h", 0))
        pctsa95h = all_feats.get("pctsa95h", osa_predictor.osa_medians.get("pctsa95h", 0))
        bmi = all_feats.get("bmi_s2", osa_predictor.osa_medians.get("bmi_s2", 28))
        frag = all_feats.get("frag_index", osa_predictor.osa_medians.get("frag_index", 0))
        waso = all_feats.get("waso_min", all_feats.get("waso", osa_predictor.osa_medians.get("waso_min", 0)))
        n3_pct = all_feats.get("N3_pct", osa_predictor.osa_medians.get("N3_pct", 0))
        n_wake = all_feats.get("n_wake_bouts", osa_predictor.osa_medians.get("n_wake_bouts", 0))
        
        engineered = {}
        if "hypoxia_score" not in all_feats:
            engineered["hypoxia_score"] = pctsa95h * 1.0 + pctsa90h * 3.0 + pctsa85h * 9.0
        if "arousal_frag" not in all_feats:
            engineered["arousal_frag"] = ai_all * frag
        if "sat_drop" not in all_feats:
            engineered["sat_drop"] = avgsat - minsat
        if "arousal_per_bout" not in all_feats:
            engineered["arousal_per_bout"] = ai_all / (n_wake + 1)
        if "rem_nrem_arousal_ratio" not in all_feats:
            engineered["rem_nrem_arousal_ratio"] = ai_rem / (ai_nrem + 0.1)
        if "waso_arousal" not in all_feats:
            engineered["waso_arousal"] = waso * ai_all
        if "n3_suppression" not in all_feats:
            engineered["n3_suppression"] = n3_pct / (ai_all + 0.1)
        if "bmi_arousal" not in all_feats:
            engineered["bmi_arousal"] = bmi * ai_all
        
        all_feats.update(engineered)
        
        # Also add PSG aliases if not present
        if "slpeffp" not in all_feats and "sleep_efficiency" in all_feats:
            all_feats["slpeffp"] = all_feats["sleep_efficiency"]
        if "slplatp" not in all_feats and "sol_min" in all_feats:
            all_feats["slplatp"] = all_feats["sol_min"]
        if "timest1p" not in all_feats and "N1_pct" in all_feats:
            all_feats["timest1p"] = all_feats["N1_pct"]
        if "timest2p" not in all_feats and "N2_pct" in all_feats:
            all_feats["timest2p"] = all_feats["N2_pct"]
        if "timeremp" not in all_feats and "REM_pct" in all_feats:
            all_feats["timeremp"] = all_feats["REM_pct"]
        if "waso" not in all_feats and "waso_min" in all_feats:
            all_feats["waso"] = all_feats["waso_min"]
        
        # Build feature vector aligned to model columns
        feature_vector = []
        used_features = {}
        for col in osa_predictor.osa_features:
            if col in all_feats:
                feature_vector.append(all_feats[col])
                used_features[col] = round(all_feats[col], 4)
            else:
                med = osa_predictor.osa_medians.get(col, 0)
                feature_vector.append(med)
                used_features[col] = round(float(med), 4)
        
        X = pd.DataFrame([feature_vector], columns=osa_predictor.osa_features)
        
        print(f"\n[CUSTOM OSA] {len(features)} input features → {len(feature_vector)} model features")
        
        from osa_predictor import predict_osa_severity
        pred_label, proba_dict, feature_impacts = predict_osa_severity(X)
        
        model_used = "Stacking (XGB+LGBM+MLP→LR)" if osa_predictor.stacking_model else "XGBoost"
        
        return jsonify({
            "severity": pred_label,
            "probabilities": proba_dict,
            "model_used": model_used,
            "used_features": used_features,
            "shap_explanations": feature_impacts[:10],
            "shap_all": feature_impacts,
            "expected_features": list(osa_predictor.osa_features),
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
