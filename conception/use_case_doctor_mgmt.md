# Use Case: Doctor Management (Admin)

```mermaid
graph LR
    Admin((Admin))

    subgraph Admin_Module [Admin Module]
        UC1(Create Doctor Account)
        UC2(List All Doctors)
        UC3(Update Doctor Info)
        UC4(Delete Doctor)
    end

    Admin --- UC1
    Admin --- UC2
    Admin --- UC3
    Admin --- UC4
```
