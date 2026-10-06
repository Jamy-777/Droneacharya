# Acoustic Reconnaissance Notes

## Selection (2026-10-05)

Candidates from both review lists were checked against their repository records and papers.

| Kept | Why |
| --- | --- |
| DDL | Field recordings with range/bearing/session metadata |
| UaVirBASE | Drone and ambient on the same 8-mic array; geometry fully crossed |
| Mięsikowska | 17 UAVs outdoors, same-model units at different sites |
| DroneNoise (Salford) | Calibrated field overflights — independent test domain |
| Svanström audio + ESC-50 classes | Labelled hard negatives (helicopter, airplane, engine, …) |

| Excluded | Reason (primary source) |
| --- | --- |
| DADS (= "HF drone-audio-detection samples") | No per-clip source field; drone clips average 0.6 s vs 7.3 s for no-drone; contains Al-Emadi and drone-mounted ego-noise |
| Al-Emadi DroneAudioDataset | Licence forbids security, surveillance and defence use (repository, updated 2026-10-04) |
| Purdue 15/28/32-class line | Laptop built-in microphone; no no-drone class; public repo holds 1 sample per class |
| Casabianca | Training audio from online sources; own recordings are an iPhone in a quiet room + ~4.5 min real-world test; Dropbox-hosted |
| DREGON, DroneAudioSet | Microphones on the drone — opposite sensing geometry |
| RWDA | IEEE DataPort subscription only (optional if access is available) |
| NASA sUAS flyover | Optional external test; ZIP refuses scripted download |
| WSU 24-channel (2026) | Future localization work |

## Empirical audit (2026-10-06)

All archives matched their published MD5. Scripts: `scripts/inspect_acoustic.py`, per-file CSVs in `interim/<dataset>/`.

### What changed after inspection

- **DDL has no negatives.** All 62,092 decodable clips are Mini 2 or Phantom 4 Pro; the `XXXX` class and the two environment-only sessions in the paper are not in the release. 3,945 clips (6.4%) are empty 44-byte files.
- **DDL level follows the recording day, not range** (29 Mar −34…−42 dBFS, 31 Mar −49…−60 dBFS). A 58–105 Hz comb appears for both drones on both days. → DDL is demoted from detection backbone to range-labelled positives (per-session normalisation required).
- **DDL site correction:** both drones flew on 31 Mar at the same site (Phantom 17:14–18:20, Mini 18:24–19:10); model and site are not fully confounded.
- **UaVirBASE becomes the detection backbone.** Same 8-mic array for drone and ambient, and the urban ambient (−24…−14 dBFS) is as loud as the drone (−28…−13 dBFS): no loudness shortcut. Limits: one drone, one day, 4 ambient recordings.
- **Svanström has a strong loudness shortcut** (background −35.8, drone −24.8, helicopter −19.2 dBFS median). Use with level normalisation and always report an energy baseline.
- **DroneNoise is calibrated in pascals** (50 kHz float). Flyovers sweep ~48→71→49 dB SPL: a natural detection-vs-distance test. No negatives.
- **Mięsikowska** is a clean 9-position × 10-take grid per drone; half the files contain speech commands. No negatives. Same-model units differ 2–5 dB in level.

### The structural problem

Same-recording-chain negatives exist only in UaVirBASE (416 s) and Svanström (30 background + 30 helicopter clips). Every other negative comes from a different chain than the positives, so **recording chain ↔ label** is the acoustic version of RF's dataset-source shortcut.

Required controls for every acoustic result:

1. Energy-only baseline (level shortcut).
2. "Predict the dataset" baseline (chain shortcut).
3. Leave-one-dataset-out and leave-one-ambient-recording-out evaluation.
4. Level normalisation per recording, with and without, compared.

### Common format

Sample rates are 96 kHz (DDL, UaVirBASE), 50 kHz (DroneNoise) and 44.1 kHz (others). Keep native audio in storage; cross-dataset experiments resample to a common rate (44.1 kHz preserves the most shared band; 16 kHz as the deployment-like variant) — an experiment, not a global choice.

### Open actions

- Ask the DDL authors whether the no-drone sessions can be released.
- Download the remaining 13 Mięsikowska drones only when identification work starts.
- Group ESC-50 clips by `src_file`; helicopter has 15 sources for 40 clips.
