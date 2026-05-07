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
    user: Optional[dict] = None

class PatientWithPSGsResponse(PatientResponse):
    psgs: List[PSGResponse] = []
    
    class Config:
        from_attributes = True

class FileMessageBase(BaseModel):
    content: str

class FileMessageCreate(FileMessageBase):
    pass

class FileMessageResponse(FileMessageBase):
    id: int
    conversation_id: int
    sender_id: int
    timestamp: datetime.datetime

    class Config:
        from_attributes = True

class FileConversationBase(BaseModel):
    psg_id: int
    file_type: str # 'edf', 'hypnogram', 'csv', 'xml'

class FileConversationCreate(FileConversationBase):
    target_doctor_id: int

class FileConversationResponse(FileConversationBase):
    id: int
    doctor_one_id: int
    doctor_two_id: int
    created_at: datetime.datetime
    doctor_one: UserResponse
    doctor_two: UserResponse
    messages: List[FileMessageResponse] = []

    class Config:
        from_attributes = True
