import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

from preprocessing import STEP2_DIR

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
