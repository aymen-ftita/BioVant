"""
================================================================================
SHHS1 Data Preparation Script
================================================================================
Converts raw SHHS1 EDF + XML files into lightweight .pt files.
Run this ONCE. After that, training notebooks load instantly.

Usage:
    python prepare_shhs.py

Output:
    shhs1_processed/shhs1-200001.pt
    shhs1_processed/shhs1-200002.pt
    ...

Each .pt file contains:
    {
        "X": tensor(n_epochs, 5, 3000),  # float32, z-score normalized per epoch
        "y": tensor(n_epochs,),           # int8 stage labels (0-4)
        "subject_id": "200001",
        "n_epochs": int,
    }

Stage labels:
    0=Wake, 1=N1, 2=N2, 3=N3, 4=REM

Channels (5):
    EEG, EEG(sec), EOG(L), EOG(R), EMG
================================================================================
"""

import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import resample

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
EDF_DIR  = Path(r"E:\shhs\polysomnography\edfs\shhs1")
XML_DIR  = Path(r"E:\shhs\polysomnography\annotations-events-nsrr\shhs1")
OUT_DIR  = Path(r"E:\shhs1_processed")
OUT_DIR.mkdir(exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGET_SR  = 100
EPOCH_SEC  = 30
EPOCH_LEN  = TARGET_SR * EPOCH_SEC   # 3000 samples

CHANNELS = {
    "eeg1": "EEG",
    "eeg2": "EEG(sec)",
    "eogl": "EOG(L)",
    "eogr": "EOG(R)",
    "emg":  "EMG",
}
CH_ORDER = ["eeg1", "eeg2", "eogl", "eogr", "emg"]
N_CH     = len(CH_ORDER)

CONCEPT_MAP = {
    "wake|0"          : 0,
    "stage 1 sleep|1" : 1,
    "stage 2 sleep|2" : 2,
    "stage 3 sleep|3" : 3,
    "stage 4 sleep|4" : 3,
    "rem sleep|5"     : 4,
}


# ── XML PARSER ────────────────────────────────────────────────────────────────
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    events = []
    for event in root.findall(".//ScoredEvent"):
        etype   = (event.findtext("EventType") or "").lower()
        concept = (event.findtext("EventConcept") or "").lower().strip()
        if "stages" not in etype:
            continue
        start    = float(event.findtext("Start") or 0)
        duration = float(event.findtext("Duration") or 0)
        code     = CONCEPT_MAP.get(concept)
        if code is not None:
            events.append((start, duration, code))
    if not events:
        return np.array([], dtype=np.int8)

    total_sec = max(s + d for s, d, _ in events)
    n_epochs  = int(np.ceil(total_sec / EPOCH_SEC))
    hyp = np.full(n_epochs, -1, dtype=np.int8)
    for start, dur, stage in events:
        e0 = int(start / EPOCH_SEC)
        e1 = min(int(np.ceil((start + dur) / EPOCH_SEC)), n_epochs)
        hyp[e0:e1] = stage
    # Forward-fill unscored
    last = 0
    for i in range(len(hyp)):
        if hyp[i] == -1:
            hyp[i] = last
        else:
            last = hyp[i]
    return hyp


# ── SUBJECT PROCESSOR ────────────────────────────────────────────────────────
def process_subject(edf_path, xml_path, out_path):
    """Process one subject EDF+XML → save .pt file."""
    try:
        # Load EDF
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        ch_names = raw.ch_names

        # Check all channels present
        signals = {}
        for key, ch_name in CHANNELS.items():
            if ch_name not in ch_names:
                return False, f"missing channel {ch_name}"
            signals[key] = raw.get_data(picks=[ch_name])[0]

        # Resample to 100Hz
        sfreq = raw.info["sfreq"]
        if sfreq != TARGET_SR:
            n_new = int(len(signals["eeg1"]) * TARGET_SR / sfreq)
            signals = {k: resample(v, n_new) for k, v in signals.items()}

        # Parse hypnogram
        hyp = parse_xml(xml_path)
        if len(hyp) == 0:
            return False, "empty hypnogram"

        # Create epochs
        n_epochs = min(len(hyp), len(signals["eeg1"]) // EPOCH_LEN)
        if n_epochs == 0:
            return False, "zero epochs"

        X = np.zeros((n_epochs, N_CH, EPOCH_LEN), dtype=np.float32)
        for ep in range(n_epochs):
            s = ep * EPOCH_LEN
            e = s + EPOCH_LEN
            for ci, ck in enumerate(CH_ORDER):
                seg = signals[ck][s:e]
                if len(seg) < EPOCH_LEN:
                    seg = np.pad(seg, (0, EPOCH_LEN - len(seg)))
                # Per-epoch z-score normalization
                std = seg.std()
                X[ep, ci] = (seg - seg.mean()) / (std + 1e-8)

        y = hyp[:n_epochs]

        # Remove unscored epochs
        valid = y >= 0
        X = X[valid]
        y = y[valid]

        if len(y) == 0:
            return False, "all epochs unscored"

        # Save .pt file
        torch.save({
            "X"          : torch.tensor(X, dtype=torch.float32),
            "y"          : torch.tensor(y, dtype=torch.int8),
            "subject_id" : edf_path.stem.replace("shhs1-", ""),
            "n_epochs"   : len(y),
        }, out_path)

        return True, len(y)

    except Exception as ex:
        return False, str(ex)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    edf_files = sorted(EDF_DIR.glob("*.edf"))
    total     = len(edf_files)

    print("=" * 60)
    print(f"  SHHS1 Data Preparation")
    print(f"  EDF files found : {total}")
    print(f"  Output dir      : {OUT_DIR}")
    print("=" * 60)

    done, skipped, total_epochs = 0, 0, 0

    for i, edf_path in enumerate(edf_files):
        subj_id  = edf_path.stem.replace("shhs1-", "")
        xml_path = XML_DIR / f"shhs1-{subj_id}-nsrr.xml"
        out_path = OUT_DIR / f"shhs1-{subj_id}.pt"

        # Skip already processed
        if out_path.exists():
            done += 1
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{total}] Skipping already processed subjects...")
            continue

        if not xml_path.exists():
            skipped += 1
            continue

        success, result = process_subject(edf_path, xml_path, out_path)

        if success:
            done         += 1
            total_epochs += result
        else:
            skipped += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pt_files  = list(OUT_DIR.glob("*.pt"))
            disk_gb   = sum(f.stat().st_size for f in pt_files) / 1e9
            print(f"  [{i+1}/{total}] Done={done}  Skipped={skipped}  "
                  f"Epochs={total_epochs:,}  Disk={disk_gb:.1f}GB")

    print(f"\n[DONE] Processed {done} subjects → {OUT_DIR}")
    print(f"       Total epochs : {total_epochs:,}")
    print(f"       Skipped      : {skipped}")
    print(f"       Disk usage   : {sum(f.stat().st_size for f in OUT_DIR.glob('*.pt'))/1e9:.1f} GB")
    print(f"\nNow open the training notebooks and set:")
    print(f"  PT_DIR = Path(r'{OUT_DIR}')")


if __name__ == "__main__":
    main()
