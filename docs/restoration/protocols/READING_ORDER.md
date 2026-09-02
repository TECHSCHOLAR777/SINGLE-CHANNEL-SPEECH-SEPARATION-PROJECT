# Reading Order for Deep Project Understanding

Use this when you need to reconstruct the project quickly but accurately.

```mermaid
flowchart TD
    S[RESTORATION_STATE]
    P[PROJECT_STATUS]
    I[PROJECT_INVENTORY]
    A[ARCHITECTURE]
    E[APPROACH_EVOLUTION]
    D[DATA_AND_MODEL_INVENTORY]
    X[EXPERIMENT_REGISTRY]
    R[RESULTS]
    L[LEARNINGS]
    C[DECISIONS]
    V[VALIDATION_MATRIX]
    RP[REPRODUCTION]
    W[WORKLOG]
    T[ISSUE_LEDGER]

    S --> P --> I --> A
    A --> E
    A --> D
    D --> X --> R
    R --> L
    E --> C
    C --> V
    V --> RP
    T --> W
```

For operational work, read ISSUE_LEDGER, VALIDATION_MATRIX and WORKLOG before changing code.
