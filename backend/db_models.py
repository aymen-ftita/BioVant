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
