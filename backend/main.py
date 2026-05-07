import os
from typing import Optional
import tempfile
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request
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

app = FastAPI(title="Hypnoria Backend")

@app.on_event("startup")
def startup_event():
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
        role="doctor"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
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
    user.last_login = datetime.utcnow()
    db.commit()
    access_token_expires = timedelta(minutes=60*24*7)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user": {"id": user.id, "email": user.email, "role": user.role, "first_name": user.first_name, "last_name": user.last_name}}

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
    
    edf_url = None
    hypnogram_url = None
    csv_url = None

    if edf_file:
        edf_url = upload_file_to_b2(edf_file.file, edf_file.filename, edf_file.content_type)
    if hypnogram_image:
        hypnogram_url = upload_file_to_b2(hypnogram_image.file, hypnogram_image.filename, hypnogram_image.content_type)
    if csv_file:
        csv_url = upload_file_to_b2(csv_file.file, csv_file.filename, csv_file.content_type)

    db_psg = db_models.PSG(
        patient_id=patient_id, 
        severity=severity, 
        report_data=report_data,
        edf_url=edf_url,
        hypnogram_url=hypnogram_url,
        csv_url=csv_url
    )
    db.add(db_psg)
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
    return db_msg

