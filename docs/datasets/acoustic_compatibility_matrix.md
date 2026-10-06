# Droneacharya Acoustic Dataset Compatibility Matrix

Same evidence vocabulary as the RF matrix: `author doc` (dataset authors' own description), `repo` (repository record), `n=…` (files checked), VERIFIED only for values we measured. Audit scripts: `scripts/inspect_acoustic.py`; per-file audit CSVs in `interim/<dataset>/`.

Datasets were selected on 2026-10-05 (selection record and audit conclusions: `acoustic_notes.md`) and downloaded 2026-10-06; every archive matched its published MD5.

| Parameter | DDL | UaVirBASE | Mięsikowska | DroneNoise | Svanström | ESC-50 (selected) |
| --- | --- | --- | --- | --- | --- | --- |
| **Primary purpose** | Drone detection + localization (author) | UAV sound-source localization (author) | UAV acoustic signals + speech in UAV presence (author) | sUAS noise / psychoacoustics (author) | Multi-sensor drone detection (author) | Environmental sound classification |
| **Licence** | CC BY 4.0 | CC BY 4.0 | CC BY 4.0 | CC BY 4.0 | "Free to download, use and edit" | CC BY-NC 3.0 (ESC-10: CC BY) |
| **Local artifact** | `MLSP_2022_Real_Data.zip` 12.6 GB, MD5 OK | `Microphone_array.zip` 13.8 GB, MD5 OK | 4 of 17 drone zips (D5, D11, D9, D17), MD5 OK | `22133411.zip` 0.74 GB | 90 WAVs from `Data/Audio` | `ESC-50-master.zip` |
| **Recording type** | Field, outdoor | Field, outdoor (urban) | Field, outdoor | Field, outdoor | Field, outdoor (airports) | Freesound clips (mixed sources) |
| **Positives** | DJI Mini 2, Phantom 4 Pro | DJI Mavic 3 Cine | 17 UAVs in release; local: 2× Mini 2, 2× Phantom 4 | DJI Mini 3 Pro (`3p`), `Fp`, Mavic 3 (`M3`), Yuneec (`Yn`) | Drone (30 clips) | None |
| **Negatives** | **None in release** — 0 `XXXX` clips (EMPIRICAL, all 62,092 parsed) | 4 ambient recordings, 416 s, same array | **None** | None (flyover edges = distant drone, not background) | Background (30) + helicopter (30), same equipment | Helicopter, airplane, engine, chainsaw, wind, rain, … (40 clips per class) |
| **Same-chain negatives?** | No | **Yes** | No | No | **Yes** | N/A (no positives) |
| **Sample rate / format** | **96 kHz, 32-bit integer PCM, 8 ch (EMPIRICAL n=62,103 headers)** | **96 kHz, 32-bit integer PCM, 8 ch (EMPIRICAL n=132)** | **44.1 kHz, 16-bit, mono (EMPIRICAL n=362)** | **50 kHz, 32-bit float, mono (EMPIRICAL n=174)**; M6–M9 in pascals, M1–M5 calibration ±5 dB | **44.1 kHz, 16-bit, stereo (EMPIRICAL n=90)** | 44.1 kHz, 16-bit, mono (EMPIRICAL n=2000) |
| **Clip / recording length** | 0.1 s clips (62,092) ≈ 103.5 min | 40 s (128 drone), 40–256 s ambient | 13–254 s; drone-only median ~70 s, speech ~19 s | 12–30 s per flyover/hover | 10 s | 5 s |
| **Microphones / channels** | 8-ch cube array, Zoom F8N | 8 × Rode NTG-2, Behringer UMC1820 | Olympus LS11 recorder (+ Norsonic 140 SLM metrics) | 9 ground microphones (M1–M9), calibrated | Stereo; 23/90 clips dual-mono | Unknown / various |
| **Calibration** | No | No | SLM metrics alongside (.NBF/.xlsx) | **9 calibration tones: M6–M9 consistent with pascals; M1–M5 tones unsteady (±5 dB)** | No | No |
| **Sessions / days** | **2 flight days** (29 & 31 Mar 2021), 36 recording segments | **1 day** (15 Nov 2024) | 1 day per drone; 4 sites, 5 days overall | 1 day (17 Aug 2022) | 3 airports (author) | Freesound source files |
| **Site** | 31 Mar: Phantom 17:14–18:20 then Mini 18:24–19:10 → **same site that day** (paper implied separate sites) | Warsaw, one site | Kielce, Gdańsk, Dębogórze, Łapalice | Edzell, Scotland | Halmstad, Gothenburg, Malmö airports | — |
| **Geometry metadata** | Range 0–249 m, bearing, altitude per clip (filename) | Distance 10/20 m × height 10/20 m × 8 azimuths × 4 orientations, hovering | Altitude 5/8/10 m × offset 0/5/8 m, hovering, 10 takes each | 10 m AGL; 15 m/s flyovers and hovers | Close / medium / distant (author) | — |
| **Weather** | Temperature only (1 value per day) | Per-recording weather station (wind 0–2.9 m/s) | Per-day (wind 18–25 km/h) | Not in files | — | — |
| **Speech / other confounds** | Suburban site: cars, people (author) | Urban ambient | **Speech commands in 50% of files** (`_sekwencja`) | — | — | — |
| **Known defects** | **3,945 empty 44-byte clips (6.4%)** in 10 sessions on 31 Mar; 11 garbled filenames; low-frequency comb (58–105 Hz, ~11.7 Hz spacing) in both drones and both days — likely site/equipment | — | — | Filename `dw` vs sheet `uw` for Mini 3 Pro; weight column shifted | — | Some clips clip heavily (up to 10.6%) |
| **Leakage-safe keys** | flight day + recording segment (filename) | recording folder (each 40 s file = one hover) | drone + position + take | flyover event | clip (event grouping unknown) | `src_file` (helicopter: 15 sources for 40 clips) |
| **Published split** | Random by sample (Train/Val/Test tables) — leaky | 30 s / 10 s within recording — leaky | None | None | None | 5 folds by `src_file` |
| **Level shortcut risk** | Broadband level differs 15–20 dB by day, but **only below 300 Hz**; in 300–8000 Hz days match (−67.1 / −67.9 dB) and level falls with range (r = −0.89 Phantom, −0.59 Mini on 31 Mar; 29 Mar Mini shows no trend) (n=720) | **Low — ambient −24…−14 dBFS vs drone −28…−13 dBFS** (n=28) | Unit/site levels differ 2–5 dB | Calibrated; flyover sweeps 48→71 dB SPL | **High — background −35.8, drone −24.8, helicopter −19.2 dBFS (median)** | Varies by source |
| **Provisional role** | **Demoted:** range-labelled positives (high-pass below ~300 Hz); future localization. Not a detection backbone (no negatives) | **Detection backbone (same chain)** + geometry robustness | Positive diversity, same-model cross-site units, speech confound | Independent calibrated test; detection vs distance | **Same-chain hard negatives** (helicopter) | Hard-negative layer (labelled classes) |

## Cross-dataset facts

| Parameter | Value |
| --- | --- |
| Sample rates | 96 kHz (DDL, UaVirBASE), 50 kHz (DroneNoise), 44.1 kHz (Mięsikowska, Svanström, ESC-50) → common band limited to 22.05 kHz unless 44.1 kHz sets are excluded |
| Same-chain negatives | Only UaVirBASE (4 ambient recordings, 416 s) and Svanström (30 background + 30 helicopter, 10 s each) |
| Main shortcut | Recording chain ↔ label: drones come from 5 chains, most negatives from others. Run a "predict the dataset" baseline and leave-one-dataset-out tests |
| Units | DroneNoise M6–M9 in pascals (M1–M5 ±5 dB); all others uncalibrated full-scale |
