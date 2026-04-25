import os
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EPOCH_SEC   = 30
SAMPLE_RATE = 100
MODELS_DIR  = "models"
STEP2_DIR   = "step2"

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
