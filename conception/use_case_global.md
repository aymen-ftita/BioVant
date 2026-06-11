# Global Use Case Diagram - BioVant

```mermaid
graph LR
    %% Actors
    Admin((Admin))
    Doctor((Doctor))
    User((User))

    %% Relationships
    User --- Admin
    User --- Doctor

    subgraph Hypnoria_System [Hypnoria System]
        UC1(Authenticate)
        UC2(Manage Doctors & Licenses)
        UC3(Manage Patients)
        UC4(Upload PSG Data)
        UC5(Analyze Sleep - AI Pipeline Builder)
        UC6(Custom OSA Prediction & Feature Export)
        UC7(Consult with Colleagues - Chat)
        UC8(View & Annotate Hypnograms)
    end

    %% Links
    User --- UC1
    Admin --- UC2
    Doctor --- UC3
    Doctor --- UC4
    Doctor --- UC5
    Doctor --- UC6
    Doctor --- UC7
    Doctor --- UC8
```
