# Noisy Drone RF Signal Classification — Reconnaissance Notes

Status: desk recon from the Kaggle API, the ZHAW record and the authors' 2024 arXiv paper. Nothing downloaded.

## What it is

ZHAW (Zurich) benchmark for detecting and classifying drone RF links at low SNR. Version 1 on Kaggle, 2023-06-08, CC BY 4.0.

- 98,705 complex-IQ vectors of 16,384 samples (≈1.17 ms at 14 MS/s).
- 7 classes: 6 remote-controller links (DJI Phantom GL300F, Futaba T7C, Futaba T14SG, Graupner mx-16, FrSky Taranis, Turnigy 9X) + Noise.
- Every vector carries a synthetic SNR from −20 to +30 dB in 2 dB steps (~3,796 vectors per level).
- Files: `dataset.pt` (25.9 GB), `SNR_stats.csv`, `class_stats.csv`.

## How it was built

1. Controller signals recorded in an anechoic chamber with a USRP B210 at 56 MS/s, then downsampled to 14 MS/s (8th-order Chebyshev I). Centre 2.44175 GHz; some transmitters at 2.440 or 2.445 GHz.
2. "Labnoise" recorded separately in a busy university building: uncontrolled Bluetooth, Wi-Fi and amplifier noise.
3. Drone vectors normalised to unit power and mixed with Labnoise (50%) or Gaussian noise (50%) at a target SNR.
4. Noise class: Labnoise and Gaussian noise mixed in all combinations.

## Why it matters to Droneacharya

It adds a **new negative domain**: real, uncontrolled building traffic, as opposed to CardRF's single deliberate devices or DroneRF's quiet lab background. It also offers a calibrated SNR sweep.

## Problems

- **No recording IDs.** Only mixed vectors are released, without the source recording or noise snippet. Leakage-safe splits are impossible; any within-dataset score is optimistic. The authors' own split is stratified random 5-fold over vectors.
- **Site differs by class.** Positives come from a chamber, negatives from a building. Before mixing, a model could separate them by background character alone; mixing reduces this but doesn't remove it.
- **Centre-frequency offsets per transmitter** (2.440 / 2.44175 / 2.445 GHz) can identify the class without reading the signal.
- **Synthetic SNR**, not measured.
- **14 MHz window**: Wi-Fi channels are only partly visible, and controller hops outside the window are missed.
- **All-or-nothing download**: one 25.9 GB PyTorch file. Load it with `torch.load(path, weights_only=True, mmap=True)` — `weights_only` blocks code execution from the pickle, `mmap` avoids needing ~26 GB of RAM. `torch` is not yet in the venv.
- Kaggle subtitle says "6 consumer grade drones"; they are 6 remote-controller links.
- The 2024 arXiv paper's development set (16,648 vectors of 1,048,576 samples) is a different release from Kaggle v1. Don't mix their numbers.

## Proposed use

External low-SNR test set and a source of realistic building-noise negatives. Not for training claims unless a recording-level grouping can be recovered.

## Empirical plan (if downloaded)

1. Download `SNR_stats.csv` and `class_stats.csv` first (tiny).
2. Open `dataset.pt` memory-mapped: list keys, shapes and dtypes; look for any index or recording field.
3. Check for repeated noise snippets (hash vectors of the Noise class and of the noise component where SNR is lowest).
4. Measure each class's spectral centroid to confirm the centre-frequency offsets.
