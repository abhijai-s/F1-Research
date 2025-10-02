# F1 Track Simulation & Pit-Stop Prediction (Bi-LSTM) — Code-Only

Public repository with the **track replay/simulation** UI and **inference** path for a **pre-trained Bi-LSTM** model that predicts pit-stop laps from **FastF1** telemetry.  
**Training code, datasets, and full experiment artifacts are intentionally excluded.** This is a **code-only** demo for viewing and evaluation.

> **Ownership & Restrictions**
>
> **All rights reserved © 2025 Abhijai Sasikumar.**  
> This repository is **not open source**. **No copying, reproduction, redistribution, modification, hosting, sublicensing, derivative works, or commercial use** is permitted **without prior written permission** from the author.

---

## Contents
public/ # static assets for the track UI (e.g., circuit maps, icons)
src/ # UI/visualization code
models/ # place pre-trained weights here (e.g., bilstm_pitstop.h5) – not included
main.py # launcher / server glue for sim + inference
preprocessing.py # runtime feature prep for inference
requirements.txt # Python dependencies
package.json # optional UI deps (if you run the web interface)
package-lock.json # lockfile for npm (optional)

Notes

Scope: Inference + visualization only; training pipeline and datasets are excluded by design.

Model: Bi-LSTM for lap-level pit vs. no-pit prediction.

Data: Pulled via FastF1 at runtime (cached locally).

Contact

Author: Abhijai Sasikumar

License & Usage

All rights reserved. This repository is provided for viewing/evaluation only.
You may not copy, reproduce, distribute, modify, host, create derivative works from, or otherwise use any part of this repository without prior written permission from the author.
