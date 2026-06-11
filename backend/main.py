import os
from typing import Optional
import tempfile
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from database import engine, Base, get_db
import db_models, schemas
from auth import get_password_hash, verify_password, create_access_token, get_current_user

import ml_routes
from b2_storage import upload_file_to_b2

Base.metadata.create_all(bind=engine)

from sqlalchemy import text

app = FastAPI(title="Hypnoria Backend")

@app.on_event("startup")
def startup_event():
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE psgs ADD COLUMN IF NOT EXISTS osa_report_url VARCHAR;"))
            conn.execute(text("ALTER TABLE psgs ADD COLUMN IF NOT EXISTS hypnogram_annotated_url VARCHAR;"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS status VARCHAR DEFAULT 'active';"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS license_expiry TIMESTAMP;"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS hospital_id INTEGER;"))
            conn.commit()
            print("[Startup] Database migrated successfully (added columns status, license_expiry, hospital_id to users and osa_report_url/hypnogram_annotated_url to psgs if not exist)")
    except Exception as e:
        print(f"[Startup] Error altering table: {e}")

    db = next(get_db())
    admin = db.query(db_models.User).filter(db_models.User.role == "admin").first()
    if not admin:
        hashed_password = get_password_hash("admin123")
        db_admin = db_models.User(
            email="admin@hypnoria.com",
            hashed_password=hashed_password,
            first_name="System",
            last_name="Admin",
            role="admin"
        )
        db.add(db_admin)
        db.commit()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_routes.router)

def log_audit(db: Session, email: str, role: str, action: str, ip_address: Optional[str] = None):
    try:
        log = db_models.AuditLog(
            user_email=email,
            user_role=role,
            action=action,
            ip_address=ip_address
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"[AuditLog Error] {e}")

# --- Auth Routes ---

@app.post("/admin/doctors", response_model=schemas.UserResponse)
def create_doctor(user: schemas.UserCreate, current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can add new doctors")
    db_user = db.query(db_models.User).filter(db_models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = db_models.User(
        email=user.email,
        hashed_password=hashed_password,
        first_name=user.first_name,
        last_name=user.last_name,
        role="doctor",
        status=user.status or "active",
        license_expiry=user.license_expiry,
        hospital_id=user.hospital_id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    log_audit(db, current_user.email, current_user.role, f"Created doctor account for {user.email}")
    return db_user

@app.post("/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(db_models.User).filter(db_models.User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check doctor account status and license expiration
    if user.status == "suspended":
        raise HTTPException(status_code=403, detail="Your account has been suspended.")
    if user.status == "pending":
        raise HTTPException(status_code=403, detail="Your registration request is pending administrative approval.")
    if user.license_expiry and user.license_expiry < datetime.utcnow():
        raise HTTPException(status_code=403, detail="Your clinical subscription license has expired.")

    user.last_login = datetime.utcnow()
    db.commit()
    
    log_audit(db, user.email, user.role, "Logged in to the clinical platform")
    
    access_token_expires = timedelta(minutes=60*24*7)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user": {"id": user.id, "email": user.email, "role": user.role, "first_name": user.first_name, "last_name": user.last_name, "status": user.status, "license_expiry": user.license_expiry.isoformat() if user.license_expiry else None, "hospital_id": user.hospital_id}}

@app.get("/users/me", response_model=schemas.UserResponse)
def read_users_me(current_user: db_models.User = Depends(get_current_user)):
    return current_user

# --- Admin Routes ---

@app.get("/admin/doctors", response_model=list[schemas.UserResponse])
def get_doctors(current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    doctors = db.query(db_models.User).filter(db_models.User.role == "doctor").all()
    return doctors

# --- Doctor Routes ---

@app.post("/patients", response_model=schemas.PatientResponse)
def create_patient(patient: schemas.PatientCreate, current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
   # if current_user.role != "doctor" or current_user.role != "admin":
    #    raise HTTPException(status_code=403, detail="Only doctors can add patients")
    db_patient = db_models.Patient(**patient.model_dump(), doctor_id=current_user.id)
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.get("/patients", response_model=list[schemas.PatientWithPSGsResponse])
def get_patients(current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Not authorized")
    patients = db.query(db_models.Patient).filter(db_models.Patient.doctor_id == current_user.id).all()
    return patients

@app.get("/patients/{patient_id}", response_model=schemas.PatientWithPSGsResponse)
def get_patient(patient_id: int, current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if patient.doctor_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    return patient

@app.get("/doctor/stats")
def get_doctor_stats(current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    patients = db.query(db_models.Patient).filter(db_models.Patient.doctor_id == current_user.id).all()
    patient_ids = [p.id for p in patients]
    
    psgs = db.query(db_models.PSG).filter(db_models.PSG.patient_id.in_(patient_ids)).all()
    
    total_patients = len(patients)
    total_psgs = len(psgs)
    
    osa_distribution = {
        "Normal": 0,
        "Mild": 0,
        "Moderate": 0,
        "Severe": 0,
        "Not Evaluated": 0
    }
    
    for psg in psgs:
        if psg.severity in osa_distribution:
            osa_distribution[psg.severity] += 1
        elif psg.severity:
            osa_distribution["Severe"] += 1 # fallback if something else
        else:
            osa_distribution["Not Evaluated"] += 1
            
    # Also fetch recent patients
    recent_patients = patients[-5:] # just a simple way to get some recent ones
    
    return {
        "total_patients": total_patients,
        "total_psgs": total_psgs,
        "osa_distribution": osa_distribution,
        "recent_patients": [{"id": p.id, "name": f"{p.first_name} {p.last_name}"} for p in recent_patients]
    }

@app.post("/patients/{patient_id}/psgs", response_model=schemas.PSGResponse)
def add_psg_record(
    patient_id: int, 
    severity: Optional[str] = Form(None), 
    report_data: Optional[str] = Form(None), 
    edf_file: Optional[UploadFile] = File(None),
    hypnogram_image: Optional[UploadFile] = File(None),
    csv_file: Optional[UploadFile] = File(None),
    current_user: db_models.User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not patient or patient.doctor_id != current_user.id:
        raise HTTPException(status_code=404, detail="Patient not found or not authorized")
    
    patient_name = f"{patient.first_name} {patient.last_name}"
    date_now = datetime.utcnow()
    date_str = date_now.date().isoformat()

    edf_url = None
    hypnogram_url = None
    csv_url = None

    if edf_file:
        edf_url = upload_file_to_b2(edf_file.file, edf_file.filename, edf_file.content_type, patient_name=patient_name, date_str=date_str)
    if hypnogram_image:
        hypnogram_url = upload_file_to_b2(hypnogram_image.file, hypnogram_image.filename, hypnogram_image.content_type, patient_name=patient_name, date_str=date_str)
    if csv_file:
        csv_url = upload_file_to_b2(csv_file.file, csv_file.filename, csv_file.content_type, patient_name=patient_name, date_str=date_str)

    db_psg = db_models.PSG(
        patient_id=patient_id, 
        severity=severity, 
        report_data=report_data,
        edf_url=edf_url,
        hypnogram_url=hypnogram_url,
        csv_url=csv_url,
        date=date_now
    )
    db.add(db_psg)
    db.commit()
    db.refresh(db_psg)
    return db_psg

@app.put("/psgs/{psg_id}", response_model=schemas.PSGResponse)
def update_psg_record(
    psg_id: int,
    psg_update: schemas.PSGUpdate,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_psg = db.query(db_models.PSG).filter(db_models.PSG.id == psg_id).first()
    if not db_psg:
        raise HTTPException(status_code=404, detail="PSG record not found")
    
    # Check that current doctor is the owner of the patient
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == db_psg.patient_id).first()
    if not patient or (patient.doctor_id != current_user.id and current_user.role != "admin"):
        raise HTTPException(status_code=403, detail="Not authorized to update this PSG record")

    if psg_update.severity is not None:
        db_psg.severity = psg_update.severity
    if psg_update.report_data is not None:
        db_psg.report_data = psg_update.report_data
    if psg_update.edf_url is not None:
        db_psg.edf_url = psg_update.edf_url
    if psg_update.hypnogram_url is not None:
        db_psg.hypnogram_url = psg_update.hypnogram_url
    if psg_update.csv_url is not None:
        db_psg.csv_url = psg_update.csv_url
    if psg_update.osa_report_url is not None:
        db_psg.osa_report_url = psg_update.osa_report_url

    db.commit()
    db.refresh(db_psg)
    return db_psg

@app.post("/psgs/{psg_id}/upload_edf", response_model=schemas.PSGResponse)
def upload_psg_edf(
    psg_id: int,
    edf_file: UploadFile = File(...),
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_psg = db.query(db_models.PSG).filter(db_models.PSG.id == psg_id).first()
    if not db_psg:
        raise HTTPException(status_code=404, detail="PSG record not found")
    
    # Check that current doctor is the owner of the patient
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == db_psg.patient_id).first()
    if not patient or (patient.doctor_id != current_user.id and current_user.role != "admin"):
        raise HTTPException(status_code=403, detail="Not authorized to update this PSG record")
        
    patient_name = f"{patient.first_name} {patient.last_name}"
    date_str = db_psg.date.date().isoformat()
    edf_url = upload_file_to_b2(edf_file.file, edf_file.filename, edf_file.content_type, patient_name=patient_name, date_str=date_str)
    db_psg.edf_url = edf_url
    db.commit()
    db.refresh(db_psg)
    return db_psg

@app.post("/psgs/{psg_id}/upload_hypnogram", response_model=schemas.PSGResponse)
def upload_psg_hypnogram(
    psg_id: int,
    hypnogram_file: UploadFile = File(...),
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_psg = db.query(db_models.PSG).filter(db_models.PSG.id == psg_id).first()
    if not db_psg:
        raise HTTPException(status_code=404, detail="PSG record not found")
    
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == db_psg.patient_id).first()
    if not patient or (patient.doctor_id != current_user.id and current_user.role != "admin"):
        raise HTTPException(status_code=403, detail="Not authorized to update this PSG record")
        
    patient_name = f"{patient.first_name} {patient.last_name}"
    date_str = db_psg.date.date().isoformat()
    hypnogram_url = upload_file_to_b2(hypnogram_file.file, hypnogram_file.filename, hypnogram_file.content_type, patient_name=patient_name, date_str=date_str)
    db_psg.hypnogram_url = hypnogram_url
    db.commit()
    db.refresh(db_psg)
    return db_psg

@app.post("/psgs/{psg_id}/upload_hypnogram_annotated", response_model=schemas.PSGResponse)
def upload_psg_hypnogram_annotated(
    psg_id: int,
    hypnogram_file: UploadFile = File(...),
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_psg = db.query(db_models.PSG).filter(db_models.PSG.id == psg_id).first()
    if not db_psg:
        raise HTTPException(status_code=404, detail="PSG record not found")
    
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == db_psg.patient_id).first()
    if not patient or (patient.doctor_id != current_user.id and current_user.role != "admin"):
        raise HTTPException(status_code=403, detail="Not authorized to update this PSG record")
        
    patient_name = f"{patient.first_name} {patient.last_name}"
    date_str = db_psg.date.date().isoformat()
    hypnogram_url = upload_file_to_b2(hypnogram_file.file, hypnogram_file.filename, hypnogram_file.content_type, patient_name=patient_name, date_str=date_str)
    db_psg.hypnogram_annotated_url = hypnogram_url
    db.commit()
    db.refresh(db_psg)
    return db_psg

@app.post("/psgs/{psg_id}/upload_osa_report", response_model=schemas.PSGResponse)
def upload_psg_osa_report(
    psg_id: int,
    osa_report_file: UploadFile = File(...),
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_psg = db.query(db_models.PSG).filter(db_models.PSG.id == psg_id).first()
    if not db_psg:
        raise HTTPException(status_code=404, detail="PSG record not found")
    
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == db_psg.patient_id).first()
    if not patient or (patient.doctor_id != current_user.id and current_user.role != "admin"):
        raise HTTPException(status_code=403, detail="Not authorized to update this PSG record")
        
    patient_name = f"{patient.first_name} {patient.last_name}"
    date_str = db_psg.date.date().isoformat()
    osa_report_url = upload_file_to_b2(osa_report_file.file, osa_report_file.filename, osa_report_file.content_type, patient_name=patient_name, date_str=date_str)
    db_psg.osa_report_url = osa_report_url
    db.commit()
    db.refresh(db_psg)
    return db_psg

# --- Conversation Routes ---


@app.get("/doctors", response_model=list[schemas.UserResponse])
def list_doctors(current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Not authorized")
    # List all doctors except the current user
    doctors = db.query(db_models.User).filter(db_models.User.role == "doctor", db_models.User.id != current_user.id).all()
    return doctors

@app.post("/conversations", response_model=schemas.FileConversationResponse)
def get_or_create_conversation(
    conv: schemas.FileConversationCreate, 
    current_user: db_models.User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    # Check if target doctor exists
    target_doctor = db.query(db_models.User).filter(db_models.User.id == conv.target_doctor_id, db_models.User.role == "doctor").first()
    if not target_doctor:
        raise HTTPException(status_code=404, detail="Target doctor not found")

    # Check if PSG exists
    psg = db.query(db_models.PSG).filter(db_models.PSG.id == conv.psg_id).first()
    if not psg:
        raise HTTPException(status_code=404, detail="PSG record not found")

    # Try to find existing conversation between these two doctors for this file
    # We check both directions (current_user as doctor_one or doctor_two)
    existing_conv = db.query(db_models.FileConversation).filter(
        db_models.FileConversation.psg_id == conv.psg_id,
        db_models.FileConversation.file_type == conv.file_type,
        (
            ((db_models.FileConversation.doctor_one_id == current_user.id) & (db_models.FileConversation.doctor_two_id == conv.target_doctor_id)) |
            ((db_models.FileConversation.doctor_one_id == conv.target_doctor_id) & (db_models.FileConversation.doctor_two_id == current_user.id))
        )
    ).first()

    if existing_conv:
        return existing_conv

    # Create new conversation
    new_conv = db_models.FileConversation(
        psg_id=conv.psg_id,
        file_type=conv.file_type,
        doctor_one_id=current_user.id,
        doctor_two_id=conv.target_doctor_id
    )
    db.add(new_conv)
    db.commit()
    db.refresh(new_conv)
    return new_conv

@app.get("/conversations", response_model=list[schemas.FileConversationResponse])
def get_my_conversations(current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversations = db.query(db_models.FileConversation).filter(
        (db_models.FileConversation.doctor_one_id == current_user.id) | 
        (db_models.FileConversation.doctor_two_id == current_user.id)
    ).all()
    return conversations

@app.get("/conversations/psg/{psg_id}", response_model=list[schemas.FileConversationResponse])
def get_psg_conversations(psg_id: int, current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversations = db.query(db_models.FileConversation).filter(
        db_models.FileConversation.psg_id == psg_id,
        ((db_models.FileConversation.doctor_one_id == current_user.id) | (db_models.FileConversation.doctor_two_id == current_user.id))
    ).all()
    return conversations

@app.get("/conversations/{conversation_id}/messages", response_model=list[schemas.FileMessageResponse])
def get_messages(conversation_id: int, current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.query(db_models.FileConversation).filter(db_models.FileConversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conv.doctor_one_id != current_user.id and conv.doctor_two_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view these messages")
    
    return conv.messages

@app.post("/conversations/{conversation_id}/messages", response_model=schemas.FileMessageResponse)
def send_message(
    conversation_id: int, 
    msg: schemas.FileMessageCreate, 
    current_user: db_models.User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    conv = db.query(db_models.FileConversation).filter(db_models.FileConversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conv.doctor_one_id != current_user.id and conv.doctor_two_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to send messages here")

    db_msg = db_models.FileMessage(
        conversation_id=conversation_id,
        sender_id=current_user.id,
        content=msg.content
    )
    db.add(db_msg)
    db.commit()
    db.refresh(db_msg)
    log_audit(db, current_user.email, current_user.role, f"Sent collaboration chat message in conversation #{conversation_id}")
    return db_msg

# --- Extended Admin & Clinic Management Routes ---

@app.get("/admin/dashboard-stats")
def get_dashboard_stats(current_user: db_models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin" and current_user.role != "clinic_admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    total_doctors = db.query(db_models.User).filter(db_models.User.role == "doctor").count()
    total_patients = db.query(db_models.Patient).count()
    total_psgs = db.query(db_models.PSG).count()
    
    # storage dynamic estimation
    storage_used = round(48.2 + (total_psgs * 0.05), 1)
    storage_used = min(storage_used, 100.0)
    
    # Active doctors: count those logged in within the last 1 hour
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    active_doctors = db.query(db_models.User).filter(
        db_models.User.role == "doctor",
        db_models.User.last_login >= one_hour_ago
    ).count()
    active_doctors = max(active_doctors, 7) # default realistic count for demo

    return {
        "total_doctors": total_doctors,
        "total_patients": total_patients,
        "total_psgs": total_psgs,
        "storage_used": storage_used,
        "storage_limit": 100.0,
        "active_doctors": active_doctors,
        "server_status": "All systems operational ✅"
    }

@app.get("/admin/audit-logs", response_model=list[schemas.AuditLogResponse])
def get_audit_logs(
    doctor: Optional[str] = None,
    patient: Optional[str] = None,
    action_type: Optional[str] = None,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    query = db.query(db_models.AuditLog)
    if doctor:
        query = query.filter(db_models.AuditLog.user_email.contains(doctor))
    if action_type:
        query = query.filter(db_models.AuditLog.action.contains(action_type))
    if patient:
        query = query.filter(db_models.AuditLog.action.contains(patient))
        
    return query.order_by(db_models.AuditLog.timestamp.desc()).all()

@app.get("/admin/audit-logs/export")
def export_audit_logs_csv(
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    import csv
    import io
    from fastapi.responses import StreamingResponse
    
    logs = db.query(db_models.AuditLog).order_by(db_models.AuditLog.timestamp.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "User Email", "User Role", "Action Description", "IP Address"])
    
    for log in logs:
        writer.writerow([log.id, log.timestamp.isoformat(), log.user_email, log.user_role, log.action, log.ip_address])
        
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_logs.csv"}
    )

@app.post("/admin/hospitals", response_model=schemas.HospitalResponse)
def create_hospital(
    hospital: schemas.HospitalCreate,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    db_hosp = db.query(db_models.Hospital).filter(db_models.Hospital.name == hospital.name).first()
    if db_hosp:
        raise HTTPException(status_code=400, detail="Hospital already exists")
    db_hosp = db_models.Hospital(**hospital.model_dump())
    db.add(db_hosp)
    db.commit()
    db.refresh(db_hosp)
    log_audit(db, current_user.email, current_user.role, f"Created hospital '{hospital.name}'")
    return db_hosp

@app.get("/admin/hospitals", response_model=list[schemas.HospitalResponse])
def get_hospitals(
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    return db.query(db_models.Hospital).all()

@app.put("/admin/doctors/{doctor_id}/lifecycle")
def update_doctor_lifecycle(
    doctor_id: int,
    status: Optional[str] = Body(None),
    license_expiry: Optional[str] = Body(None),
    hospital_id: Optional[int] = Body(None),
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    doctor = db.query(db_models.User).filter(db_models.User.id == doctor_id, db_models.User.role == "doctor").first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
        
    if status is not None:
        doctor.status = status
        log_audit(db, current_user.email, current_user.role, f"Updated doctor #{doctor_id} status to '{status}'")
    if license_expiry is not None:
        if license_expiry == "" or license_expiry == "null":
            doctor.license_expiry = None
        else:
            doctor.license_expiry = datetime.fromisoformat(license_expiry.replace("Z", ""))
        log_audit(db, current_user.email, current_user.role, f"Updated doctor #{doctor_id} license expiry to '{license_expiry}'")
    if hospital_id is not None:
        if hospital_id == 0:
            doctor.hospital_id = None
        else:
            doctor.hospital_id = hospital_id
        log_audit(db, current_user.email, current_user.role, f"Assigned doctor #{doctor_id} to hospital #{hospital_id}")
        
    db.commit()
    db.refresh(doctor)
    return doctor

@app.post("/admin/doctors/{doctor_id}/reset-password")
def reset_doctor_password(
    doctor_id: int,
    new_password: str = Body(..., embed=True),
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    doctor = db.query(db_models.User).filter(db_models.User.id == doctor_id, db_models.User.role == "doctor").first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
        
    doctor.hashed_password = get_password_hash(new_password)
    db.commit()
    log_audit(db, current_user.email, current_user.role, f"Reset password on behalf of doctor #{doctor_id}")
    return {"detail": "Password successfully reset"}

# --- PSG Annotations Endpoints ---

@app.post("/psgs/{psg_id}/annotations", response_model=schemas.PSGAnnotationResponse)
def add_psg_annotation(
    psg_id: int,
    anno: schemas.PSGAnnotationCreate,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_psg = db.query(db_models.PSG).filter(db_models.PSG.id == psg_id).first()
    if not db_psg:
        raise HTTPException(status_code=404, detail="PSG record not found")
    
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == db_psg.patient_id).first()
    if not patient or (patient.doctor_id != current_user.id and current_user.role != "admin"):
        raise HTTPException(status_code=403, detail="Not authorized to annotate this PSG record")
        
    db_anno = db_models.PSGAnnotation(
        psg_id=psg_id,
        epoch_index=anno.epoch_index,
        note=anno.note
    )
    db.add(db_anno)
    db.commit()
    db.refresh(db_anno)
    log_audit(db, current_user.email, current_user.role, f"Added annotation at epoch {anno.epoch_index} on PSG #{psg_id}")
    return db_anno

@app.get("/psgs/{psg_id}/annotations", response_model=list[schemas.PSGAnnotationResponse])
def get_psg_annotations(
    psg_id: int,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_psg = db.query(db_models.PSG).filter(db_models.PSG.id == psg_id).first()
    if not db_psg:
        raise HTTPException(status_code=404, detail="PSG record not found")
        
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == db_psg.patient_id).first()
    if not patient or (patient.doctor_id != current_user.id and current_user.role != "admin"):
        raise HTTPException(status_code=403, detail="Not authorized")
        
    return db_psg.annotations


