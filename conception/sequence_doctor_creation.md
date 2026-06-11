# Sequence Diagram: Doctor Creation (Admin)

```mermaid
sequenceDiagram
    participant Admin
    participant Frontend
    participant Backend
    participant DB

    Admin->>Frontend: Fill Doctor Details (Email, License Expiry, Hospital ID)
    Frontend->>Backend: POST /admin/users (Header: JWT)
    Backend->>Backend: Verify JWT (is admin?)
    Backend->>DB: Check if Email exists
    DB-->>Backend: Result (Not Found)
    Backend->>Backend: Hash Password
    Backend->>DB: Insert New User (role='doctor', status='active', license_expiry, hospital_id)
    DB-->>Backend: Success
    Backend-->>Frontend: 201 Created (User Data)
    Frontend-->>Admin: Show Success Notification & Refresh List
```
