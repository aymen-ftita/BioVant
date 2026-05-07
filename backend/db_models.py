from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="doctor") # "admin" or "doctor"
    first_name = Column(String)
    last_name = Column(String)
    last_login = Column(DateTime, default=datetime.datetime.utcnow)

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
    csv_url = Column(String, nullable=True)

    patient = relationship("Patient", back_populates="psgs")
    conversations = relationship("FileConversation", back_populates="psg")

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
