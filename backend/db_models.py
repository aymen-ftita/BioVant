from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
import datetime

class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    b2_bucket = Column(String, nullable=True)
    billing_tier = Column(String, default="Standard")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    users = relationship("User", back_populates="hospital")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="doctor") # "admin" or "doctor" or "clinic_admin"
    first_name = Column(String)
    last_name = Column(String)
    last_login = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="active") # "active", "suspended", "pending"
    license_expiry = Column(DateTime, nullable=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)

    hospital = relationship("Hospital", back_populates="users")
    patients = relationship("Patient", back_populates="doctor")

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    age = Column(Integer)
    imc = Column(Float)
    gender = Column(String)
    doctor_id = Column(Integer, ForeignKey("users.id"))

    doctor = relationship("User", back_populates="patients")
    psgs = relationship("PSG", back_populates="patient")

class PSG(Base):
    __tablename__ = "psgs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    date = Column(DateTime, default=datetime.datetime.utcnow)
    severity = Column(String, nullable=True) # OSA severity prediction
    report_data = Column(String, nullable=True) # Store JSON of features/results
    edf_url = Column(String, nullable=True)
    hypnogram_url = Column(String, nullable=True)
    hypnogram_annotated_url = Column(String, nullable=True)
    csv_url = Column(String, nullable=True)
    osa_report_url = Column(String, nullable=True)

    patient = relationship("Patient", back_populates="psgs")
    conversations = relationship("FileConversation", back_populates="psg")
    annotations = relationship("PSGAnnotation", back_populates="psg", cascade="all, delete-orphan")

class PSGAnnotation(Base):
    __tablename__ = "psg_annotations"

    id = Column(Integer, primary_key=True, index=True)
    psg_id = Column(Integer, ForeignKey("psgs.id"))
    epoch_index = Column(Integer)
    note = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    psg = relationship("PSG", back_populates="annotations")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_email = Column(String, index=True)
    user_role = Column(String, nullable=True)
    action = Column(String)
    ip_address = Column(String, nullable=True)

class FileConversation(Base):
    __tablename__ = "file_conversations"

    id = Column(Integer, primary_key=True, index=True)
    psg_id = Column(Integer, ForeignKey("psgs.id"))
    file_type = Column(String) # 'edf', 'hypnogram', 'csv', 'xml'
    doctor_one_id = Column(Integer, ForeignKey("users.id"))
    doctor_two_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    psg = relationship("PSG", back_populates="conversations")
    doctor_one = relationship("User", foreign_keys=[doctor_one_id])
    doctor_two = relationship("User", foreign_keys=[doctor_two_id])
    messages = relationship("FileMessage", back_populates="conversation", cascade="all, delete-orphan")

class FileMessage(Base):
    __tablename__ = "file_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("file_conversations.id"))
    sender_id = Column(Integer, ForeignKey("users.id"))
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("FileConversation", back_populates="messages")
    sender = relationship("User")

