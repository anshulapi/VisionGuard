# Audited architecture diagram

```mermaid
flowchart LR
    I[RGB upload / synthetic image] --> Q[Classical quality analysis]
    I --> S[LAB border distance → Otsu both polarities → morphology → connected components → optional GrabCut]
    Q --> F[Quality flags, evidence, scores]
    S --> G[Mask confidence, bbox, coverage, centering]
    F --> C[Illustrative compliance checks]
    G --> C
    Q --> X[9 hand-crafted DIP features]
    X --> RF[Offline multi-output Random Forest]
    F --> P[Deterministic correction planner]
    P --> T[Noise → exposure → cast / contrast corrections]
    T --> V[Target metric + score + defect + SSIM + compliance guards]
    V -->|accepted| O[Final image + trace + JSON]
    V -->|rejected| O
    C --> O
    RF -. offline evaluation only .-> E[Held-out evaluation]
    O -. offline evaluation .-> E
    E --> R[CSV, metrics, charts, aligned PSNR / SSIM]
```

The Random Forest is currently not connected to the interactive app or correction planner; the dashed path indicates an offline experiment. Segmentation output is heuristic because the dataset has no pixel-level ground truth masks.
