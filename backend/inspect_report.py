from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
import sys

engine = create_engine("postgresql://postgres:pgadmin@localhost:5432/hypnoriadb")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

import db_models
psg = db.query(db_models.PSG).filter(db_models.PSG.id == 6).first()
if psg:
    print("Severity:", psg.severity)
    print("Report Data Type:", type(psg.report_data))
    if psg.report_data:
        try:
            data = json.loads(psg.report_data)
            print("Keys:", data.keys())
            if "results" in data:
                print("Results length:", len(data["results"]))
                first_res = data["results"][0]
                print("First result keys:", first_res.keys())
                print("Stats:", first_res.get("stats"))
        except Exception as e:
            print("Error parsing json:", e)
else:
    print("PSG 6 not found")

db.close()
