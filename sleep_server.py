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

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EPOCH_SEC   = 30
SAMPLE_RATE = 100
MODELS_DIR  = "models"
STEP2_DIR   = "step2"

# ─────────────────────────────────────────────
# STEP 2: OSA PREDICTION INIT
# ─────────────────────────────────────────────
osa_model = None
osa_le = None
osa_features = None
osa_medians = None
explainer = None

def init_osa_predictor():
    global osa_model, osa_le, osa_features, osa_medians, explainer
    try:
        if osa_model is not None: return
        print("Loading Step 2 OSA prediction models...")
        osa_model = joblib.load(os.path.join(STEP2_DIR, "xgb_model.pkl"))
        osa_le = joblib.load(os.path.join(STEP2_DIR, "label_encoder.pkl"))
        osa_features = joblib.load(os.path.join(STEP2_DIR, "feature_columns.pkl"))
        df = pd.read_csv(os.path.join(STEP2_DIR, "sleep_features_shhs2.csv"))
        # Compute medians for all numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        osa_medians = numeric_df.median().to_dict()
        explainer = shap.TreeExplainer(osa_model)
        print("✓ Step 2 initialized.")
    except Exception as e:
        print(f"Warning: Could not initialize Step 2 predictor: {e}")

# ─────────────────────────────────────────────
# CHANNEL ALIASES
# Order: EEG, EEG(sec), EOG(L), EOG(R), EMG
# ─────────────────────────────────────────────
CHANNEL_ALIASES = {
    "eeg1": [
        "EEG", "EEG1", "EEG 1", "EEG_1",
        "EEG Fpz-Cz", "EEG FPZ-CZ", "Fpz-Cz",
        "C3-A2", "EEG C3-A2", "F3-A2", "Fp1-A2",
    ],
    "eeg2": [
        "EEG(sec)", "EEG(SEC)", "EEG2", "EEG 2", "EEG_2",
        "EEG Pz-Oz", "EEG PZ-OZ", "Pz-Oz",
        "C4-A1", "EEG C4-A1", "O1-A2", "Pz-A1",
    ],
    "eogl": [
        "EOG(L)", "EOG(l)", "EOG L", "EOG-L",
        "EOG left", "EOG Left", "EOGL", "E1",
        "LOC", "LOC-A2", "E1-M2",
    ],
    "eogr": [
        "EOG(R)", "EOG(r)", "EOG R", "EOG-R",
        "EOG right", "EOG Right", "EOGR", "E2",
        "ROC", "ROC-A1", "E2-M1",
    ],
    "emg": [
        "EMG", "Chin EMG", "CHIN EMG", "chin",
        "EMG1", "Chin", "CHIN", "Chinz",
        "EMG Chin", "EMGchin",
    ],
}
CH_ORDER = ["eeg1", "eeg2", "eogl", "eogr", "emg"]


def resolve_channels(available_channels):
    available_lower = {ch.lower(): ch for ch in available_channels}

    def find(aliases):
        for alias in aliases:
            if alias.lower() in available_lower:
                return available_lower[alias.lower()]
        return None

    resolved = {slot: find(CHANNEL_ALIASES[slot]) for slot in CH_ORDER}

    # Fallback: use any EEG-like channel for missing slots
    if resolved["eeg1"] is None:
        eeg_like = [ch for ch in available_channels
                    if "eeg" in ch.lower()]
        if not eeg_like:
            raise ValueError(
                f"No EEG channels found.\nAvailable: {available_channels}"
            )
        resolved["eeg1"] = eeg_like[0]

    fallback = resolved["eeg1"]
    for slot in ["eeg2", "eogl", "eogr", "emg"]:
        if resolved[slot] is None:
            resolved[slot] = fallback
            print(f"  WARNING: '{slot}' not found — using '{fallback}' as fallback")

    return resolved


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class SleepTransformer(nn.Module):
    def __init__(self, n_channels=5, patch_len=50,
                 d_model=128, nhead=8, num_layers=4, num_classes=3):
        super().__init__()
        self.patch_len   = patch_len
        self.n_patches   = 3000 // patch_len    # 60
        self.patch_embed = nn.Linear(n_channels * patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.n_patches+1)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        patches = x.unfold(-1, self.patch_len, self.patch_len)
        patches = patches.permute(0,2,1,3).reshape(B, self.n_patches, -1)
        x   = self.patch_embed(patches)
        cls = self.cls_token.expand(B,-1,-1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_encoder(x)
        x   = self.transformer(x)
        return self.classifier(x[:, 0])

class SleepCNN(nn.Module):
    def __init__(self, n_channels=5, num_classes=3):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=200, stride=25, padding=100),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*16*2, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64),      nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        b1 = self.branch1(x).flatten(1)
        b2 = self.branch2(x).flatten(1)
        return self.classifier(torch.cat([b1, b2], dim=1))


class SleepLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128,
                 num_layers=2, num_classes=3):
        super(SleepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        lstm_out = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, 1),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        B = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, B, self.hidden_size,
                         dtype=x.dtype, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, B, self.hidden_size,
                         dtype=x.dtype, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        attn   = self.attention(out)
        ctx    = torch.sum(attn * out, dim=1)
        return self.fc(ctx)


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
# AASM STATISTICS
# ─────────────────────────────────────────────
def compute_aasm_stats(predictions, class_names):
    label_map = {name: idx for idx, name in enumerate(class_names)}
    hypno     = np.array([label_map[s] for s in predictions])
    n         = len(hypno)
    epoch_min = EPOCH_SEC / 60.0

    sleep_idx   = np.where(hypno != 0)[0]
    sleep_onset = int(sleep_idx[0]) if len(sleep_idx) > 0 else n
    rem_idx     = np.where(hypno == 2)[0]
    rem_start   = int(rem_idx[0]) if len(rem_idx) > 0 else None

    tib_min = round(n * epoch_min, 2)
    tst_min = round(len(sleep_idx) * epoch_min, 2)
    se      = round(tst_min / tib_min * 100, 1) if tib_min > 0 else 0
    sol_min = round(sleep_onset * epoch_min, 2)

    rem_latency_min = None
    if rem_start is not None:
        rem_latency_min = round((rem_start - sleep_onset) * epoch_min, 2)

    post_onset = hypno[sleep_onset:]
    waso_min   = round(np.sum(post_onset == 0) * epoch_min, 2)

    stage_minutes, stage_pct = {}, {}
    for idx, name in enumerate(class_names):
        mins  = round(np.sum(hypno == idx) * epoch_min, 2)
        denom = tst_min if (tst_min > 0 and idx != 0) else tib_min
        pct   = round(mins / denom * 100, 1) if denom > 0 else 0.0
        stage_minutes[name] = mins
        stage_pct[name]     = pct
        
    nrem_pct = 0
    if len(class_names) == 5:
        nrem_pct = stage_pct.get("N1", 0) + stage_pct.get("N2", 0) + stage_pct.get("N3", 0)
    else:
        nrem_pct = stage_pct.get("NREM", 0)
        
    rem_pct = stage_pct.get("REM", 0)

    alerts = []
    if sol_min < 5:
        alerts.append("Latence très courte — Possible privation de sommeil")
    if rem_latency_min is not None and rem_latency_min < 60:
        alerts.append("Latence REM courte — Possible narcolepsie ou dépression")
    if se < 85:
        alerts.append("Efficacité du sommeil faible — Hygiène du sommeil à évaluer")
    if rem_pct < 15:
        alerts.append("REM réduit — Possible perturbation du sommeil paradoxal")
    if nrem_pct < 50:
        alerts.append("NREM insuffisant — Possible fragmentation du sommeil")

    return {
        "tib":           tib_min,
        "tst":           tst_min,
        "se":            se,
        "sol":           sol_min,
        "rem_latency":   rem_latency_min,
        "waso":          waso_min,
        "stage_minutes": stage_minutes,
        "stage_pct":     stage_pct,
        "alerts":        alerts,
        "class_names":   class_names,
    }


# ─────────────────────────────────────────────
# PREPROCESSING — per-epoch norm
# ─────────────────────────────────────────────
def preprocess_edf(filepath, channels_str="5"):
    import mne
    from scipy.signal import resample as sci_resample

    raw      = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    available = raw.ch_names
    native_sr = int(raw.info["sfreq"])
    print(f"  Channels ({len(available)}): {available}")
    print(f"  Sample rate: {native_sr}Hz  |  Duration: {raw.times[-1]/3600:.2f}h")

    ch_map  = resolve_channels(available)
    needed  = list(dict.fromkeys(ch_map.values()))
    raw.pick(needed)
    raw.load_data()
    data    = raw.get_data()
    ch_list = raw.ch_names
    del raw

    ch_order = CH_ORDER if channels_str == "5" else ["eeg1", "eeg2"]
    
    signals = {}
    for slot in ch_order:
        ch_name = ch_map[slot]
        idx     = ch_list.index(ch_name)
        sig     = data[idx].copy()
        if native_sr != SAMPLE_RATE:
            n_new = int(len(sig) * SAMPLE_RATE / native_sr)
            sig   = sci_resample(sig, n_new)
        signals[slot] = sig

    if native_sr != SAMPLE_RATE:
        print(f"  Resampled: {native_sr}Hz → {SAMPLE_RATE}Hz")

    epoch_len = EPOCH_SEC * SAMPLE_RATE
    n_epochs  = len(signals["eeg1"]) // epoch_len

    if n_epochs == 0:
        raise ValueError(f"EDF too short for a single {EPOCH_SEC}s epoch.")

    X = np.zeros((n_epochs, len(ch_order), epoch_len), dtype=np.float32)
    for ci, slot in enumerate(ch_order):
        sig = signals[slot]
        for ep in range(n_epochs):
            s   = ep * epoch_len
            seg = sig[s:s + epoch_len]
            if len(seg) < epoch_len:
                seg = np.pad(seg, (0, epoch_len - len(seg)))
            std       = seg.std()
            X[ep, ci] = (seg - seg.mean()) / (std + 1e-8)

    print(f"  Epochs: {n_epochs}  |  Shape: {X.shape}")
    return X


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
    if osa_model is None:
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
            "age_s2": float(clinical.get("age") or osa_medians.get("age_s2", 50)),
            "bmi_s2": float(clinical.get("bmi") or osa_medians.get("bmi_s2", 28)),
            "gender": gender_val,
            "avgsat": float(clinical.get("avgsat") or osa_medians.get("avgsat", 94)),
            "minsat": float(clinical.get("minsat") or osa_medians.get("minsat", 85)),
        }
        input_feats["hypoxia_score"] = (100 - input_feats["avgsat"]) * (100 - input_feats["minsat"]) / 100
        
        # 4. Build feature vector and impute
        feature_vector = []
        for col in osa_features:
            if col in calc_feats:
                feature_vector.append(calc_feats[col])
            elif col in input_feats:
                feature_vector.append(input_feats[col])
            else:
                feature_vector.append(osa_medians.get(col, 0))
                
        X = pd.DataFrame([feature_vector], columns=osa_features)
        
        # 5. Predict & SHAP
        pred_idx = int(osa_model.predict(X)[0])
        pred_label = str(osa_le.inverse_transform([pred_idx])[0]) if osa_le else str(pred_idx)
        
        shap_values = explainer(X)
        shap_vals = shap_values.values[0]
        # Depending on XGBoost objective, shap_values.values might have shape (n_features, n_classes)
        # We take the impacts for the predicted class if it's multi-class
        if len(shap_vals.shape) > 1:
            impacts = shap_vals[:, pred_idx]
        else:
            impacts = shap_vals
            
        feature_impacts = []
        for i, col in enumerate(osa_features):
            val = float(X.iloc[0, i])
            imp = float(impacts[i])
            if abs(imp) > 0.001:
                feature_impacts.append({"feature": col, "value": round(val, 2), "impact": round(imp, 4)})
                
        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_impacts = feature_impacts[:6] # Top 6 drivers
        
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
