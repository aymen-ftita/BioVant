from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base
import db_models

engine = create_engine("postgresql://postgres:pgadmin@localhost:5432/hypnoriadb")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

print("--- USERS ---")
for u in db.query(db_models.User).all():
    print(f"ID: {u.id}, Email: {u.email}, Role: {u.role}, Name: {u.first_name} {u.last_name}")

print("\n--- PATIENTS ---")
for p in db.query(db_models.Patient).all():
    print(f"ID: {p.id}, Name: {p.first_name} {p.last_name}, Age: {p.age}, IMC: {p.imc}, Doctor ID: {p.doctor_id}")

print("\n--- PSGS ---")
for psg in db.query(db_models.PSG).all():
    print(f"ID: {psg.id}, Patient ID: {psg.patient_id}, Date: {psg.date}, Severity: {psg.severity}, Has Report Data: {psg.report_data is not None}")

db.close()

