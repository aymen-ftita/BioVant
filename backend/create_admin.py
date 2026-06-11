import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base
import db_models
from auth import get_password_hash

# Create Database Connection
engine = create_engine("postgresql://postgres:pgadmin@localhost:5432/hypnoriadb")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

def create_or_reset_admin(email, password, first_name="System", last_name="Admin"):
    # Clean email
    email = email.strip().lower()
    
    # Check if user already exists
    user = db.query(db_models.User).filter(db_models.User.email == email).first()
    
    hashed_password = get_password_hash(password)
    
    if user:
        print(f"User with email '{email}' already exists. Updating role to 'admin' and resetting password...")
        user.role = "admin"
        user.hashed_password = hashed_password
        user.first_name = first_name
        user.last_name = last_name
        user.status = "active"
        db.commit()
        print(f"Successfully updated/reset Admin account: {email}")
    else:
        print(f"Creating new Admin account with email '{email}'...")
        db_admin = db_models.User(
            email=email,
            hashed_password=hashed_password,
            first_name=first_name,
            last_name=last_name,
            role="admin",
            status="active"
        )
        db.add(db_admin)
        db.commit()
        print(f"Successfully created new Admin account: {email}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        email = sys.argv[1]
        password = sys.argv[2]
        first_name = sys.argv[3] if len(sys.argv) >= 4 else "System"
        last_name = sys.argv[4] if len(sys.argv) >= 5 else "Admin"
        create_or_reset_admin(email, password, first_name, last_name)
    else:
        print("\n--- Hypnoria Admin Account Creator ---")
        print("Usage: python create_admin.py <email> <password> [first_name] [last_name]")
        print("\nCreating default/resetting system admin...")
        create_or_reset_admin("admin@hypnoria.com", "admin123", "System", "Admin")

    db.close()
