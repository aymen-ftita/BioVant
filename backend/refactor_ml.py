import re

with open('ml_routes.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Imports
code = code.replace('from flask import Flask, request, jsonify', 'from fastapi import APIRouter, UploadFile, File, Form, Body, HTTPException\nimport tempfile')
code = code.replace('from flask_cors import CORS\n', '')
code = code.replace('app = Flask(__name__)\nCORS(app)', 'router = APIRouter()')
code = code.replace('from models import ', 'from ml_models import ')

# analyze
code = code.replace('@app.route("/analyze", methods=["POST"])', '@router.post("/analyze")')
code = code.replace('def analyze():\n    if "file" not in request.files:', 'def analyze(file: UploadFile = File(...), models: str = Form("LSTM"), channels: str = Form("5"), classes: str = Form("3")):\n    if not file:')
code = code.replace('    f = request.files["file"]', '    f = file')
code = code.replace('    if not f.filename.lower().endswith(".edf"):', '    if not file.filename.lower().endswith(".edf"):')
code = code.replace('        return jsonify({"error": "Please upload an EDF file"}), 400', '        raise HTTPException(status_code=400, detail="Please upload an EDF file")')
code = code.replace('    import tempfile\n    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".edf")\n    os.close(tmp_fd)\n    f.save(tmp_path)', '    import tempfile\n    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".edf")\n    os.close(tmp_fd)\n    with open(tmp_path, "wb") as out_file:\n        out_file.write(file.file.read())')
code = code.replace('        model_types_str = request.form.get("models", "LSTM")', '        model_types_str = models')
code = code.replace('        channels   = request.form.get("channels", "5")', '')
code = code.replace('        classes    = request.form.get("classes", "3")', '')

# extract_features
code = code.replace('@app.route("/extract_features", methods=["POST"])', '@router.post("/extract_features")')
code = code.replace('def extract_features():', 'def extract_features(data: dict = Body(...)):')
code = code.replace('        data = request.json', '')

# predict_osa
code = code.replace('@app.route("/predict_osa", methods=["POST"])', '@router.post("/predict_osa")')
code = code.replace('def predict_osa():', 'def predict_osa(data: dict = Body(...)):')

# parse_features_file
code = code.replace('@app.route("/parse_features_file", methods=["POST"])', '@router.post("/parse_features_file")')
code = code.replace('def parse_features_file():', 'def parse_features_file(file: UploadFile = File(...)):')
code = code.replace('    if "file" not in request.files:\n        return jsonify({"error": "No file uploaded"}), 400\n    \n    f = request.files["file"]', '    if not file:\n        raise HTTPException(status_code=400, detail="No file uploaded")\n    f = file')
code = code.replace('        content = f.read()', '        content = f.file.read()')

# predict_osa_custom
code = code.replace('@app.route("/predict_osa_custom", methods=["POST"])', '@router.post("/predict_osa_custom")')
code = code.replace('def predict_osa_custom():', 'def predict_osa_custom(data: dict = Body(...)):')
code = code.replace('        data = request.json', '')

# Error handling
code = re.sub(r'return jsonify\((.*?)\), 500', r'raise HTTPException(status_code=500, detail=str(\1))', code)
code = re.sub(r'return jsonify\((.*?)\), 400', r'raise HTTPException(status_code=400, detail=str(\1))', code)
code = re.sub(r'return jsonify\((.*?)\)', r'return \1', code, flags=re.DOTALL)

with open('ml_routes.py', 'w', encoding='utf-8') as f:
    f.write(code)
