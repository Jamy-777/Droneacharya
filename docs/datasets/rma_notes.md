# RMA (Royal Military Academy / KU Leuven) Drone RF Dataset — Reconnaissance Notes

Status: desk recon complete from primary sources; 1 capture fetched to verify the file format. Papers not yet read.

## Sources used

1. KU Leuven RDR (Dataverse) API — file list, sizes, MD5, license, version (DOI 10.48804/HZRVNZ).
2. `README.txt`, `signal_visualize.m`, `get_spectrogram2.m` — shipped by the authors.
3. Remote listing of all 16 zip archives via HTTP Range requests — no download.
4. One capture, `SJRC_pro/sjrc__0.mat` (29.5 MB), pulled out of its zip with `scripts/fetch_zip_members.py`.

## What it is

Indoor recordings in the semi-anechoic chamber of the Royal Military Academy (Belgium), for a joint RMA/KU Leuven PhD. USRP X310, OmniLOG 70600 omni antenna, 100 MS/s complex IQ at 2.44 GHz, transmitters 7 m from the antenna. Version 1.0, released 2024-01-16, **CC BY-NC 4.0**.

README lists 9 systems: Spektrum DX4e, DJI Mini 2, DJI Inspire 2, DJI Matrice 300, Taranis Q X7, Nine Eagles, WLtoys, Q205, SJRC F11 Pro. DJI systems have separate **RC** (controller uplink) and **Vid** (video downlink) archives.

## Archive inventory (remote listing)

| Archive | Files | MB per file (median) |
| --- | --- | --- |
| Spektrum_DX4e, mini2RC, inspire2RC, matriceRC | 71 each | 30–33 |
| mini2vid / inspire2Vid / matricevid | 71 each | 63 / 78 / 41 |
| Frysky, NineEagles, wltoys, Q205, SJRC_pro | 51 each | 29–35 |
| MavicRC1 + MavicRC2 | 15 + 16 | 260 |
| MavicVid1 + MavicVid2 | 14 + 15 | 360 |

812 `.mat` files, 46.7 GB in total.

## File format (empirical, n=1)

- MAT v7.3 (HDF5), created 2022-05-05 on Linux.
- One variable, `uhd_samps`: shape (1, 10,000,000), complex double stored as HDF5 compound real/imag.
- 10M samples at 100 MS/s = 100 ms per file.
- Bursty: 1 ms window RMS median 4.66e-4, p90 3.06e-3 — most windows are noise-level.
- 12,514 exact-zero samples — probably quantization of low-level noise; check before treating as gaps.

## Documentation conflicts to resolve

- **Mavic is missing from the README**, yet its 60 files are ~19 GB (40% of the dataset). Model unknown.
- **Frysky vs "Taranis Q X7"**: FrSky makes the Taranis, so probably the same system — confirm.
- **SJRC F11 Pro** is listed as "drone + remote controller" but has one archive only.
- Mavic files are 8–12× larger than the others: longer captures, or video bandwidth compressing worse? Measure.

## Published preprocessing (author code)

`get_spectrogram2.m`: 1024-point Hann FFT, no overlap, log-amplitude, then **z-scored per spectrogram**. That normalisation removes absolute power, so the authors' published results don't rely on a power shortcut — worth replicating as one of our normalisation variants.

## Implications for Droneacharya

- **No negatives at all** (no background, Wi-Fi or Bluetooth). Cannot support detection on its own.
- Strongest use: identification of RC and video links across 10 systems, including hobby RC controllers that our other datasets barely cover, plus leave-one-system-out novelty tests.
- **Best cross-dataset partner for RFUAV**: same receiver model, sample rate and centre frequency, so differences come from site and devices rather than receiver settings.
- Chamber + fixed 7 m: no propagation or environment variation.
- Filenames repeat across RC and Vid archives (`Mavic_3.mat` exists in both): every key must include the archive.

## Proposed diagnostic subset (~850 MB, fetched from inside the zips)

| Member | Question |
| --- | --- |
| `mini2RC/mini2_0.mat`, `mini2RC/mini2_1.mat` | Are consecutive files one continuous recording? |
| `mini2vid/mini2_0.mat` | Same drone, video link vs RC |
| `MavicRC1/Mavic_1.mat`, `MavicVid1/Mavic_1.mat` | Why are Mavic files 8–12× larger? Which Mavic? |
| `Spektrum_DX4e/DX4e_0.mat`, `Frysky/Frysky_0.mat` | Hobby RC controllers; Frysky = Taranis? |
| `matricevid/matrice_0.mat` | Matrice 300 video link |

## Open questions

- Do the papers use negatives or synthetic interference? (Read TCCN 2022 and ICACT 2023.)
- Gain setting and whether it changed between recordings.
- Recording dates per system (MAT header gives creation time per file).
