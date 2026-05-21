import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_flow():
    # 1. Login as Admin
    print("Logging in as admin...")
    login_data = {"username": "admin@hypnoria.com", "password": "admin123"}
    response = requests.post(f"{BASE_URL}/token", data=login_data)
    if response.status_code != 200:
        print(f"Failed to login as admin: {response.text}")
        return
    admin_token = response.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 2. Create Doctor 1
    print("Creating Doctor 1...")
    doc1_data = {
        "email": "doctor1@test.com",
        "password": "password123",
        "first_name": "John",
        "last_name": "Doe"
    }
    requests.post(f"{BASE_URL}/admin/doctors", json=doc1_data, headers=admin_headers)

    # 3. Create Doctor 2
    print("Creating Doctor 2...")
    doc2_data = {
        "email": "doctor2@test.com",
        "password": "password123",
        "first_name": "Jane",
        "last_name": "Smith"
    }
    requests.post(f"{BASE_URL}/admin/doctors", json=doc2_data, headers=admin_headers)

    # 4. Login as Doctor 1
    print("Logging in as Doctor 1...")
    login_data = {"username": "doctor1@test.com", "password": "password123"}
    response = requests.post(f"{BASE_URL}/token", data=login_data)
    if response.status_code != 200:
        print(f"Failed to login as Doctor 1: {response.text}")
        return
    
    data = response.json()
    if "user" not in data:
        print(f"Error: 'user' key missing from login response. Data: {data}")
        return
        
    doc1_token = data["access_token"]
    doc1_headers = {"Authorization": f"Bearer {doc1_token}"}
    doc1_id = data["user"]["id"]

    # 5. Get Doctor 2 ID
    print("Getting Doctor 2 info...")
    response = requests.get(f"{BASE_URL}/doctors", headers=doc1_headers)
    if response.status_code != 200:
        print(f"Failed to get doctors: {response.text}")
        return
    doctors = response.json()
    try:
        doc2_id = next(d["id"] for d in doctors if d["email"] == "doctor2@test.com")
    except StopIteration:
        print("Error: Doctor 2 not found in list")
        return

    # 6. Create Patient
    print("Creating Patient...")
    patient_data = {
        "first_name": "Test",
        "last_name": "Patient",
        "age": 45,
        "imc": 25.5,
        "gender": "Male"
    }
    response = requests.post(f"{BASE_URL}/patients", json=patient_data, headers=doc1_headers)
    if response.status_code != 200:
        print(f"Failed to create patient: {response.text}")
        return
    patient_id = response.json()["id"]

    # 7. Add PSG
    print("Adding PSG...")
    psg_data = {"severity": "Mild", "report_data": "{}"}
    response = requests.post(f"{BASE_URL}/patients/{patient_id}/psgs", data=psg_data, headers=doc1_headers)
    if response.status_code != 200:
        print(f"Failed to add PSG: {response.text}")
        return
    psg_id = response.json()["id"]

    # 8. Start Conversation
    print("Starting Conversation...")
    conv_data = {
        "psg_id": psg_id,
        "file_type": "hypnogram",
        "target_doctor_id": doc2_id
    }
    response = requests.post(f"{BASE_URL}/conversations", json=conv_data, headers=doc1_headers)
    if response.status_code != 200:
        print(f"Failed to start conversation: {response.text}")
        return
    conv_id = response.json()["id"]

    # 9. Send Message
    print("Sending Message...")
    msg_data = {"content": "Hello Doctor, what do you think of this hypnogram?"}
    response = requests.post(f"{BASE_URL}/conversations/{conv_id}/messages", json=msg_data, headers=doc1_headers)
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")
        return

    # 10. List Messages
    print("Listing Messages...")
    response = requests.get(f"{BASE_URL}/conversations/{conv_id}/messages", headers=doc1_headers)
    if response.status_code != 200:
        print(f"Failed to list messages: {response.text}")
        return
    messages = response.json()
    print(f"Found {len(messages)} messages.")
    for m in messages:
        print(f"- {m['content']}")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_flow()
