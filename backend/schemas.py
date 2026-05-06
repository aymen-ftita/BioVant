from pydantic import BaseModel
from typing import List, Optional
import datetime

class UserBase(BaseModel):
    email: str
    first_name: str
    last_name: str
    role: str = "doctor"

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(UserBase):
    id: int
    last_login: Optional[datetime.datetime]
    
    class Config:
        from_attributes = True

class PatientBase(BaseModel):
    first_name: str
    last_name: str
    age: int
    imc: float
    gender: str

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    id: int
    doctor_id: int

    class Config:
        from_attributes = True

class PSGResponse(BaseModel):
    id: int
    patient_id: int
    date: datetime.datetime
    severity: Optional[str] = None
    report_data: Optional[str] = None
    edf_url: Optional[str] = None
    hypnogram_url: Optional[str] = None
    csv_url: Optional[str] = None
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class PatientWithPSGsResponse(PatientResponse):
    psgs: List[PSGResponse] = []
    
    class Config:
        from_attributes = True
