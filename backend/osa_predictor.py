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
osa_model = None          # XGBoost — used for SHAP explanations
stacking_model = None     # Stacking ensemble — used for main prediction
osa_le = None
osa_features = None
osa_medians = None
explainer = None

def init_osa_predictor():
    global osa_model, stacking_model, osa_le, osa_features, osa_medians, explainer
    try:
        if osa_model is not None:
            return
        print("Loading Step 2 OSA prediction models...")

        # XGBoost — for SHAP explainability (TreeExplainer compatible)
        osa_model = joblib.load(os.path.join(STEP2_DIR, "xgb_model.pkl"))

        # Stacking ensemble — for main severity prediction (higher accuracy)
        stacking_path = os.path.join(STEP2_DIR, "stacking_model.pkl")
        if os.path.exists(stacking_path):
            stacking_model = joblib.load(stacking_path)
            print("  ✓ Stacking model loaded")
        else:
            stacking_model = None
            print("  ⚠ Stacking model not found, falling back to XGBoost")

        osa_le = joblib.load(os.path.join(STEP2_DIR, "label_encoder.pkl"))
        osa_features = joblib.load(os.path.join(STEP2_DIR, "feature_columns.pkl"))

        # Load feature matrix CSV for median imputation
        df = pd.read_csv(os.path.join(STEP2_DIR, "sleep_features_shhs2.csv"))
        numeric_df = df.select_dtypes(include=[np.number])
        osa_medians = numeric_df.median().to_dict()

        # SHAP explainer on XGBoost (TreeExplainer is fast + accurate for trees)
        explainer = shap.TreeExplainer(osa_model)
        print("✓ Step 2 initialized (XGBoost SHAP + Stacking prediction).")
    except Exception as e:
        print(f"Warning: Could not initialize Step 2 predictor: {e}")
        import traceback
        traceback.print_exc()


def predict_osa_severity(feature_df):
    """
    Run both models and return prediction + probabilities.
    feature_df: DataFrame with shape (1, n_features) aligned to osa_features.
    Returns: (severity_label, probabilities_dict, shap_explanations)
    """
    # 1. Stacking model prediction (main result)
    model = stacking_model if stacking_model is not None else osa_model
    pred_idx = int(model.predict(feature_df)[0])
    pred_label = str(osa_le.inverse_transform([pred_idx])[0])

    # 2. Probabilities — use stacking if available
    try:
        proba = model.predict_proba(feature_df)[0]
        proba_dict = {}
        for i, cls in enumerate(osa_le.classes_):
            proba_dict[cls] = round(float(proba[i]), 4)
    except Exception:
        proba_dict = {pred_label: 1.0}

    # 3. SHAP explanations (always from XGBoost — TreeExplainer)
    shap_values = explainer(feature_df)
    shap_vals = shap_values.values[0]

    # For multi-class: shape (n_features, n_classes) — get the predicted class
    if len(shap_vals.shape) > 1:
        impacts = shap_vals[:, pred_idx]
    else:
        impacts = shap_vals

    feature_impacts = []
    for i, col in enumerate(osa_features):
        val = float(feature_df.iloc[0, i])
        imp = float(impacts[i])
        if abs(imp) > 0.001:
            feature_impacts.append({
                "feature": col,
                "value": round(val, 3),
                "impact": round(imp, 4)
            })

    # Sort by absolute impact descending
    feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return pred_label, proba_dict, feature_impacts
