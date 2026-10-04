# Drone Remote Controller RF Signal Dataset — Paper Read (backlog)

Decision: **paper read only, no download.** Outside the original 6-dataset plan; it adds no new axis of significant value (see verdict).

## Sources

- IEEE DataPort record, DOI 10.21227/ss99-8d56 (created 2020-11-25, updated 2026-01-05; open access with free login).
- Ezuma, Erden, Anjinappa, Ozdemir, Guvenc, "Detection and Classification of UAVs Using RF Fingerprints in the Presence of Wi-Fi and Bluetooth Interference," IEEE OJ-COMS 2020 (arXiv:1909.05429), read in full.

## What it is

NC State group — the same group as CardRF (Ezuma is a CardRF author), with the same acquisition style.

- 17 remote controllers from 8 makers: DJI Inspire 1 Pro, Matrice 100, Matrice 600 (×2), Phantom 4 Pro (×2), Phantom 3; Spektrum DX5e, DX6e, DX6i, JR X9303; Futaba T8FG; Graupner MC-32; HobbyKing HK-T6A; FlySky FS-T6; Turnigy 9X; Jeti Duplex DC-16.
- Two same-model pairs (Matrice 600, Phantom 4 Pro), labelled "Mpact" and "Ngat".
- Keysight MSOS604A oscilloscope (6 GHz bandwidth, 20 GSa/s), 2 dBi omni or 24 dBi grid antenna, LNA.
- ~1000 signals per controller in the release (the paper used 100), each 5M samples / 0.25 ms, `.mat` + MATLAB class scripts.
- One 124.13 GB zip.
- Paper reports indoor experiments only, though it shows indoor and outdoor setups.

## Paper contents relevant to us

- Detection: two-state Markov model on thresholded samples (δ = 3.5σ of noise), then Wi-Fi rejected by bandwidth (>20 MHz) and Bluetooth by GFSK modulation features.
- Wi-Fi routers and 6 Bluetooth phones were captured for the paper, but the DataPort release lists controller signals only.
- Classification: 15 statistical features of the transient (shape factor, kurtosis and variance rank highest), kNN 98.13% at 25 dB SNR; SNR studied from −10 dB up; safe above ~10 dB.

## Verdict against our portfolio

| What it offers | Already covered by |
| --- | --- |
| 20 GSa/s oscilloscope transients | CardRF (same group, same method) |
| Same-model controller units (2 pairs) | UAVSig (4 × DJI C1) |
| Hobby RC diversity | RMA, RFUAV, Noisy RF |
| Wi-Fi/Bluetooth | Not in the release; CardRF has them |

No new axis. Download not justified (124 GB, single zip). Revisit only if we need many more controller models for open-set tests.

## Useful ideas to carry forward

- Its interference rejection (bandwidth test for Wi-Fi, GFSK features for Bluetooth) is a cheap, explainable classical baseline for our hard-negative work.
- Its emphasis on the **turn-on transient** matches what Processed CardRF actually contains (slices from the trigger point).
