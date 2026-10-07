<!-- Generated from dataset_cards/*.yaml, configs/dataset_roles.yaml and configs/matrices/rf.yaml by scripts/build_matrices.py. Edit those, not this file. -->

# Droneacharya RF Dataset Compatibility Matrix

Every cell is a graded claim in `dataset_cards/<dataset>.yaml`: kind (FACT or ASSESSMENT), evidence status, source, and for EMPIRICAL the sample size with its unit. Status definitions and grading rules: `src/droneacharya/cards/schema.py`. Per-dataset counts and every CONFLICTING or UNKNOWN claim: `evidence_audit.md`. Inline markers in cell text (`author doc`, `n=…`, INFERRED) are kept for reading; the graded status in the card is authoritative.

*Provisional Droneacharya role*, *Primary detection foundation?* and *Full-download justification* are Droneacharya decisions from `configs/dataset_roles.yaml`, not dataset properties.

## Schema frozen after DRFF-R2 + RFUAV reconnaissance. No new rows/columns unless a dataset reveals a genuinely new dimension.

Schema amendment (UAVSig): two genuinely new dimensions added — `Potential shortcut: position` and `Known acquisition defect`.

Schema amendment (agreed review, applied 2026-10-05): `Absolute frequency coverage` and `Independent sessions per class`.

| Parameter | DRFF-R2 | RFUAV | DroneRF | CardRF | UAVSig | RMA | Noisy RF | Remote Controller RF (paper read only) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Primary purpose** | RFF/model/state/robustness | Identification + detection benchmark | Detection + identification + flight-mode classification | UAV/controller vs Wi-Fi/Bluetooth discrimination + RF fingerprinting | Transmission detection + spectrum localization + RF fingerprinting | RC + video-link identification; novelty detection (per publication titles) | Low-SNR detection + RC-link classification benchmark | RC detection/classification with Wi-Fi/BT rejection (paper) |
| **Year** | 2026 | 2025 | 2019 | 2020 acquisition (Aug 2020, author doc) | 2024 | 2024 release (v1.0); captures created 2022 (MAT header, n=1) | 2023 (Kaggle v1, 2023-06-08) | 2020 (DataPort; OJ-COMS 2020) |
| **License** | CC BY 4.0 | Apache-2.0 (HF artifact); raw scope verify | CC BY 4.0 | UNKNOWN — check IEEE DataPort | CC0 1.0 | **CC BY-NC 4.0** (non-commercial) | CC BY 4.0 | Open access (login); license text UNKNOWN |
| **Full reported scale** | 26 physical units / 8 models | ~1.3 TB / 37 distinct UAVs reported | 227 logical segments / 454 CSV files | 71.1 GB zip; **9,600 captures** (6,090 LOS train / 2,610 LOS test / 900 NLOS); 6 UAS + 5 BT + 2 Wi-Fi devices | 8 transmitters: 4 DJI M100 UAVs + 4 DJI C1 controllers | 46.7 GB (= 43.5 GiB); 16 zips; 812 `.mat` captures; 10 systems | 98,705 vectors × 16,384 samples; 25.9 GB single `dataset.pt` | 17 RCs, 8 makers, ~1000 signals/RC; 124 GB single zip |
| **Public-release caveat** | Public dataset | **HF repo (2026-07-27): 37 class archives (109.2 GB) + `ValidationSet_5Drones` (5 DJI aircraft, 162.1 GB) + image set/weights (27.7 GB) ≈ 299 GB**; full 1.3 TB corpus not public | Full public dataset | Processed_CardRF (1 GB) = LOS UAV + controller only; Wi-Fi/Bluetooth/NLOS only in raw zip | **V4.0** (Dataverse API, 2026-01-28; cached web index shows 3.0); 726 files, 238.9 GB, MD5 per file | Full public release; README omits Mavic archives (~19 GB) | Only noise-mixed vectors released; all-or-nothing 25.9 GB file; 2024 arXiv dev set is a different release | Release lists RC signals only; paper's Wi-Fi/BT captures not listed |
| **Raw representation** | Complex IQ (RF0_I, RF0_Q) in all 731 files | Complex IQ | Signed real time-domain samples (likely the I component of USRP IQ — INFERRED); not an envelope | Real-valued direct-RF waveform (oscilloscope); no IQ | Complex IQ | Complex IQ | Complex IQ, already mixed with noise (derived) | Real-valued direct RF (oscilloscope) |
| **Scalar dtype** | **float32 — VERIFIED (all 731 files)** | **float32 — VERIFIED (XML DataType 'Complex Float' in all 53 packs)** | Text decimal values, all integers in −15,342 … 15,080 (all 454 files); stored losslessly as int16 in interim | int16 ADC codes — EMPIRICAL n=135; volts = Data×6.5841e-06 + YOrg | complex double holding int16 codes/32767 — EMPIRICAL n=8 (paper: float32) | complex double holding int16 codes/32768 — EMPIRICAL n=9 | **float32, I/Q as 2 real channels (98705, 2, 16384)** — VERIFIED | UNKNOWN (paper-read only) |
| **Raw format** | MAT v7.3 (HDF5) `.mat`, all 731 files | interleaved binary `.iq` + XML per pack (all 37 archives listed) | Single-row CSV (all 454 files) | MAT v7.3 (HDF5): structs `Channel_1` + `Frame`; waveform in `Channel_1.Data` | MAT v5 (compressed), variable `data` + label variables — EMPIRICAL n=8 | MAT v7.3 (HDF5), variable `uhd_samps` — EMPIRICAL n=9 | PyTorch zip `.pt`: x_iq, x_spec (128×128 spectrograms), y, snr, duty_cycle; read without executing the pickle | `.mat` + MATLAB class scripts |
| **I/Q layout** | Separate `RF0_I`, `RF0_Q`, each (140000000, 1) | `I0,Q0,I1,Q1...` | N/A — no complex IQ | N/A — real samples | Complex row vector (1, 50M) — EMPIRICAL n=8 | HDF5 compound real/imag, shape (1, N) — EMPIRICAL n=9 | I and Q as channel axis 0/1 of x_iq | N/A — real samples |
| **Samples/capture chunk** | **140M/channel — VERIFIED (all 731 files)** | **100M complex per full chunk — VERIFIED (sizes of all 358 chunks)** | **10M/band; 20M paired segment — VERIFIED (all 454 files)** | **5,000,000 — VERIFIED (NumPoints of all 9,600 captures)** | **50,000,000 complex — EMPIRICAL (n=360 files)** | **10M (11 archives), 20M (mini2vid, inspire2Vid), 100M (4 Mavic archives) — constant within each archive (VERIFIED, all 812)** | 16,384 per vector (repo metadata) | 5,000,000 (DataPort) |
| **Chunk duration** | **1.4 s — VERIFIED (all 731 files)** | **1.0 s per chunk; last chunk of 6 packs shorter — VERIFIED** | 0.25 s INFERRED (paper: ~5.25 s per mode over 21 segments) | 250 µs — VERIFIED (all 9,600 captures) | 1 s — EMPIRICAL (n=360 files: 50 M samples at 50 MS/s) | 100 ms / 200 ms / 1 s by archive (all 812 files) | ≈1.17 ms (INFERRED at 14 MS/s) | 0.25 ms |
| **Sample rate** | **100 MS/s — VERIFIED (Fs in all 731 files)** | **100 MS/s — VERIFIED (all 53 pack XMLs)** | **40 MS/s INFERRED** (paper: ~5.25 s per mode / 21 segments = 0.25 s; 10M samples per file) — not stated by authors | **20 GSa/s — VERIFIED (XInc = 5e-11 s in all 9,600 captures)** | **50 MS/s (author doc)** | 100 MS/s (author README + code) | 14 MS/s (downsampled from 56 MS/s, Chebyshev I) — repo + paper | 20 GSa/s (paper) |
| **Center frequency** | 5.745 GHz in Datasets 1–5 (incl. 'dual-frequency'); 2.437 GHz in Dataset 6; Dataset 7: one file each | Per pack: 5.76–5.80 GHz (Avata 2, FPV Combo, Mavic 3 Pro, Autel EVO Nano); 2.41–2.47 GHz (31 controllers, Mini 3, Mini 4 Pro) | 2.4-GHz acquisition region | N/A — direct RF sampling, no down-conversion; 2.4 GHz band of interest | **2.4435 GHz (author doc)** | 2.44 GHz (author README + code) | 2.44175 GHz; some transmitters 2.440 / 2.445 GHz (paper) | N/A — direct RF sampling, 2.4 GHz band |
| **Absolute frequency coverage** | 5.695–5.795 GHz (Datasets 1–5); 2.387–2.487 GHz (Dataset 6) | Fc ± 50 MHz per pack: 2.36–2.52 GHz and 5.71–5.85 GHz | ~2.400–2.480 GHz split into L/H halves (paper: 80 MHz); exact edges INFERRED | Direct RF, Nyquist 0–10 GHz; emitters at 2.4 GHz | 2.4185–2.4685 GHz | 2.39–2.49 GHz | 2.43475–2.44875 GHz (14 MHz) | Direct RF; 2.4 GHz |
| **IF/capture bandwidth** | UNKNOWN — not established | **100 MHz — VERIFIED (all 53 pack XMLs)** | Up to 40 MHz/receiver documented | Nyquist 10 GHz; analog front-end bandwidth UNKNOWN | 50 MHz (source) | 100 MHz (INFERRED from complex 100 MS/s) | 14 MHz (INFERRED) | Oscilloscope analog bandwidth 6 GHz (paper) |
| **Occupied signal bandwidth** | Not characterized (scenario/signal dependent) | class/condition dependent; distinct from IF BW | Signal/model/mode dependent; not yet characterized | Device-dependent; not characterized | UAV channels ~10 MHz; controllers frequency-hop (hop domain exceeds 50 MHz capture) | System and link dependent; RC vs video recorded separately | RC hopping links; Wi-Fi exceeds the 14 MHz window | RC links; paper rejects Wi-Fi by >20 MHz bandwidth |
| **Receiver** | Documentation conflict (USRP-2943 vs X310); empirical receiver IDs `u1/u2` in filenames | USRP X310 (DeviceType in all 53 pack XMLs) | 2× NI-USRP-2943R | Keysight MSOS604A, one serial (MY55510227) in all 9,600 captures | USRP B205mini-i (single receiver) | USRP X310 (single) + OmniLOG 70600 omni | USRP B210 + log-periodic antenna (paper) | Keysight MSOS604A + 2 dBi omni / 24 dBi grid + LNA |
| **Pack/unit identifier** | Physical UAV unit available | Pack = class folder + condition + pack number; XML SerialNumber is NOT unique (3 duplicated serials); physical unit unknown | BUI + segment; physical-unit identity unknown | Device per directory; one unit per model (INFERRED) → model = unit | Physical transmitter identity available | Archive (system × RC/Vid) + file index; one unit per system (INFERRED) | None — vectors carry no recording ID | Controller label incl. unit tags (Mpact/Ngat) |
| **Manufacturer coverage** | DJI only | Multiple apparent ecosystems; mapping incomplete | Parrot + DJI | DJI, Beebeerun, 3DR/FlySky + Apple, FitBit, Motorola, Cisco, TP-Link | DJI only | DJI + Spektrum, FrSky, Nine Eagles, WLtoys, Q205, SJRC | DJI, Futaba, Graupner, FrSky, Turnigy | DJI, Spektrum, Futaba, Graupner, HobbyKing, FlySky, Turnigy, Jeti |
| **Models/classes** | 8 models | 37 public image classes | 3 UAV models + background; hierarchical mode classes | 6 UAS (aircraft + controller), 5 Bluetooth, 2 Wi-Fi; flight-mode sub-labels | 1 UAV model + 1 controller model; 8 physical transmitters | 10 systems (9 in README + undocumented Mavic); RC/Vid split for DJI | 6 RC links + Noise | 15 models / 17 controllers |
| **Physical units** | **26** | UNKNOWN | UNKNOWN | 1 per model (INFERRED from Table I) | **4 UAV + 4 controller** | 1 per system (INFERRED) | 1 per class (INFERRED) | 17 |
| **Same-model multi-unit** | Yes, uneven | UNKNOWN | UNKNOWN | No | **YES — core strength** | No (as documented) | No | Yes — 2 pairs (Matrice 600, Phantom 4 Pro) |
| **Positive signal composition** | scenario-dependent; hopping/mixed/interference variants | control/hopping/video characteristics; class-dependent; VTSBW conditions | Drone/controller communications; mode-dependent; bursty | Aircraft (Non_Control) and controller (Control) captured separately; Flying/Hovering/Videoing | UAV fixed-channel transmissions + frequency-hopping controllers | RC uplink and video downlink recorded separately; hobby controllers RC only | RC uplinks only (DJI = GL300F remote) | RC uplinks only |
| **Day metadata** | **d1/d2/d3 in Dataset 3; d2 only in Datasets 1, 4, 5; none in 2, 6, 7** (all 731 files) | Not established | Not established | **Per-capture Frame.Date (all 9,600): Aug 25–29 2020; 3DR Iris Nov 1 2020** | Author doc: 5/9/24 (no-TX, controllers, one-drone) and 5/10/24 (two-drone); directories all say 2024-05-09 — CONFLICTING | MAT headers (all 812): Apr 1–6 2022 (most), May 5 2022 (SJRC), **Oct 30 (Mavic Vid) / Nov 2 2022 (Mavic RC)** | None | UNKNOWN |
| **Receiver variation** | **u1/u2 in Dataset 3 (24 / 155 files); u1 only in Dataset 1; u2 only in 4, 5; none in 2, 6, 7** | Across full corpus requires audit | Two simultaneous receivers represent L/H bands, not a robustness split | No — one capture system (all 9,600 captures) | No — one receiver | No — one receiver | None | Antenna variation (omni/grid); per-file UNKNOWN |
| **Gain/scale** | File documentation inconsistent; gain unresolved in inspected files | ScaleFactor by class: 90 (all DJI/Autel aircraft), 85/80 (SIYI, YUNZHUO), 60 (others); semantics not established — do not treat as gain | UNKNOWN | **Identical scope settings in all categories (EMPIRICAL n=135)** — no vertical-scale shortcut | 20 dB receiver gain (source) | Noise floor similar in the 9 inspected captures (min window RMS 4.5–4.6e-4) — gain likely constant (INFERRED) | Removed — unit-power normalization before mixing | UNKNOWN |
| **SNR metadata** | Not stored (no SNR field in any of the 731 files) | Per-pack ReferenceSNRLevel, 1–50 dB | Not provided per segment | Not provided | Not established per capture | Not provided; fixed 7 m in chamber | **Yes — synthetic, per vector, −20…30 dB in 2 dB steps** | Paper sweeps SNR; per-file not documented |
| **Environmental negatives** | **Two files (Dataset 7): outdoor 5.745 GHz; indoor 2.437 GHz carries drone metadata (TD=mini5PRO_1, hover) — CONFLICTING** | Raw corpus: unresolved | Yes — recorded RF background | No empty-band class (signal-triggered captures) | No-TX baseline: controller_p*_0000, 12 captures, 0 labelled transmissions (EMPIRICAL n=12) — no experimental TX on, ambient RF present (author doc) | **None in release** | Noise class = Labnoise + Gaussian mixtures | Not in release (as listed) |
| **Negative taxonomy** | ambient/no-emitter only | UNKNOWN | Ambient RF background | **Wi-Fi (2 routers) + Bluetooth (5 devices)** — raw only | No-TX baseline + unlabelled ambient RF; not a curated negative corpus | None | Wi-Fi + Bluetooth + amplifier (unlabelled mix) + Gaussian | Paper: Wi-Fi routers + 6 Bluetooth phones; not released |
| **Negative diversity** | Very low — one clean 1.4 s outdoor capture; the indoor file is in doubt | UNKNOWN — detection subset not inspected | Limited/unknown | Moderate: 7 devices, 2 technologies, one site; 500 captures per device (Motorola 350) | Weak | None | Moderate — real busy-building traffic, one site | N/A in release |
| **Negative provenance** | Same-site same-equipment recordings (Dataset-7) | UNKNOWN | Recorded with drones absent | Same site/system, deliberate single-emitter captures (author doc) | Uncontrolled ambient 2.4-GHz activity, spatially suppressed (directional antenna) | N/A | Public university building (site differs from positives) | N/A in release |
| **Hard-negative emitters** | None dedicated; Wi-Fi appears only mixed with UAV positives | UNKNOWN | None explicitly labeled | **YES — Wi-Fi (1,000) + Bluetooth (2,350) captures, LOS — EMPIRICAL** | None explicitly labeled | None | **YES — real Wi-Fi/BT traffic, unlabelled, mixed with Gaussian** | Not in release |
| **Recorded interference conditions** | UAV + Wi-Fi mixed positives (Dataset-6) | Not established | Uncontrolled ambient RF only | None — single-source captures | Ambient RF + real simultaneous UAVs | None (semi-anechoic chamber) | Synthetic: positives mixed with Labnoise (50%) or Gaussian (50%) | Paper only |
| **Positive/negative Fs matched** | **Yes, sampled** | UNKNOWN | Presumed same acquisition pipeline; not independently metadata-verified | **Yes — identical settings (EMPIRICAL n=135)** | N/A — no explicit negative corpus | N/A — no negatives | Yes — same pipeline (INFERRED) | N/A |
| **Positive/negative Fc matched** | **Yes, sampled** | UNKNOWN | Same paired acquisition bands; exact metadata not per file | N/A (direct RF); trigger and scale identical; **sessions NOT matched** | N/A | N/A | UNKNOWN — positives have per-transmitter Fc offsets | N/A |
| **Negative difficulty** | **Weak alone; major energy shortcut risk** | UNKNOWN | Better than simple silence, but narrow taxonomy | Real same-band emitters; clipping present in negatives too; session differs by class | Not suitable as standalone hard-negative benchmark | N/A — no negatives | High at low SNR by construction; site mismatch risk | N/A |
| **Indoor/outdoor** | Both | Real-world reported; class-level conditions need mapping | Laboratory | Outdoor (AERPAW Lake Wheeler, NC) | **Outdoor — neighbouring campus rooftops (author doc)** | Indoor semi-anechoic chamber | Indoor: chamber (positives), building (negatives) | Indoor reported; outdoor shown, not reported |
| **Distance metadata** | Distance field only in Dataset 1 | Not established in sampled XML | Not available per file | 8–12 m LOS (range, not per file) | Rooftop placement (drones upper roof, controllers lower roof); metres not given | Fixed 7 m (author README) | N/A | UNKNOWN |
| **Altitude/state metadata** | Height + State in all but the outdoor environment file; Dataset 1 adds FlightMode, Direction, V | Not established | Operational mode labels; no detailed altitude metadata | Flight-mode labels only | No flight-state focus | None | None | None |
| **Frequency hopping** | Explicit subset | **Major documented characteristic** | Not explicitly represented as dedicated metadata | Not labeled; 250 µs too short to observe hop sequences | **YES — controllers** | Likely in hobby RC links; not labeled (INFERRED) | Yes (RC links), not labelled | Yes (RC links), not labelled |
| **Video transmission signals** | Not central | **Explicit; VTSBW conditions** | Yes — video-recording operating mode | Inspire 'Videoing' mode only | No separate video label (label vocabulary) | **YES — separate Vid archives (Mini 2, Inspire 2, Matrice, Mavic)** | No | No |
| **Multi-UAV mixtures** | **Yes** | Not established | No | No | **YES — real OTA, two simultaneous drones** | No | No | No |
| **Controlled environment** | RF-absorbing setup | Not established | Laboratory | No — outdoor field | No — outdoor rooftops; RX antenna angled upward to limit interference | Yes — semi-anechoic chamber | Yes for positives (anechoic chamber) | Indoor lab (paper) |
| **Schema consistency** | **No — 5 metadata layouts** (Dataset 1 adds Direction/Distance/FlightMode; Dataset 2 NTD1/NTD2; Datasets 6/7 Near; outdoor environment only CenterFreq/Fs) | Same 11 XML fields in all 53 packs | Yes — one row of 10,000,000 integers in all 454 files | Raw and processed consistent (EMPIRICAL); 500 `__MACOSX` junk `.mat` entries to skip | Same 7 variables in all 360 local files (num_transmission, data, start, end, fc, bw, id) | One variable `uhd_samps` (1, N) in all 812 files | Single file; 5 tensors, counts match class_stats.csv and SNR_stats.csv | Not inspected |
| **Parent-recording identity** | Yes — unit (TD), config (C), receiver (U), day (D) in fields and file names | Pack identity (folder + pack number); chunks are consecutive 1 s cuts | Yes: paired L/H logical segment | One file = one capture; one recording block per device (Frame.Date of all 9,600) | Filename = transmitter set × channel × capture (decoded, EMPIRICAL n=5) | Archive = session; files are separate captures seconds apart (not continuous) | **Not provided — no ID field in the file** | Per-signal files; session UNKNOWN |
| **Window-safe splitting possible** | Yes — leakage-safe keys established | Pack-level; within-pack window independence requires design | **Segment grouping is NOT enough** — each mode is ~5.25 s cut into 21 segments (likely one recording); group by BUI/session | Only by device/session: **official Train/Test is same-session (interleaved seconds apart)** | YES, with scenario/capture grouping | Yes by archive/session; file-level splits are same-session | **No** — no recording or noise-snippet ID; authors use stratified random 5-fold | Not assessed |
| **Original/raw representation** | Complex IQ in MAT files | Complex IQ in binary `.iq` files | Amplitude CSV | Real int16 direct-RF waveform in MAT | Complex IQ in MAT files | Complex IQ (double) in MAT v7.3 | Not released (only mixed vectors) | Real direct-RF in MAT |
| **Published derived representations** | STFT spectrograms | Spectrograms + 37-class image dataset + FFT/STFT tooling | normalized time plots + FFT/spectral processing | Processed_CardRF: 1024-sample raw slices, 100/capture from the trigger point | 512×512 spectrograms + bounding labels; spectrogram generation code available | Author code: 1024-pt Hann spectrogram, log-amplitude, z-scored per spectrogram | IQ vectors + 2×128×128 spectrograms (x_spec), shipped in the same file | 15 statistical transient features (paper) |
| **Raw→derived reproducibility** | Yes — raw IQ available | Yes — raw IQ + published tooling available | Strong; public code available | Partial — shipped notebook (5 slices) ≠ released data (100 slices) | Strong; visualization/spectrogram/WHIRLS code available | Strong — raw IQ + author code | Low — clean recordings and noise snippets not released | Paper code not in release |
| **Train class support** | N/A for our raw audit | **37/37** | N/A until our split | Processed: 11/11 classes (350 captures each; Beebeerun ctrl 245) | Published same-scenario 5/6 capture training | N/A — no published split | N/A — no fixed split | N/A |
| **Validation class support** | N/A | **37/37** | N/A until our split | Processed: 11/11 classes (150 captures each; Beebeerun ctrl 105) | Published same-scenario 1/6 capture testing | N/A | N/A | N/A |
| **Class balance** | Unit/model imbalance | **63–499 train images/class (~7.9×)** | Imbalanced: 41 background / 84 Bebop / 81 AR / 21 Phantom segments | Processed: balanced by design; flight modes exactly 50/50 | Scenario-controlled; complete empirical audit pending | 51–71 files per archive; capture length differs by archive (100 ms–1 s) | Noise 53% (52,552); DJI 2,194 … Taranis 16,546; SNR levels balanced | ~1000 per controller |
| **Label taxonomy** | Model/unit/state are distinguishable | **AMBIGUOUS — UAV/RC-system-like classes** | BUI → detection/model/mode hierarchy | LOS/NLOS → category (UAV / Controller / Wi-Fi / Bluetooth) → device → flight mode | Per-transmission start/end/fc/bw/id (101–514 per UAS capture, 0 in the 12 baselines); id = drone, controller or active-controller set | System × signal type (RC / Vid) from archive name | Class + synthetic SNR | Controller make/model/unit |
| **Leakage-safe keys** | unit/day/receiver/config | pack (folder + pack number); stronger grouping unresolved | BUI = session (INFERRED); segment-level keys UNSAFE | device (= recording block); hold out whole devices; flight-mode results within a device are time-confounded | scenario + capture sequence + physical transmitter + day | archive (session) + file; names repeat across RC/Vid archives | **None available (VERIFIED: no ID field)** | Not assessed |
| **Independent sessions per class** | Dataset 3: up to 3 days × 2 receivers per unit; other subsets 1 day or no day field | UNKNOWN — 1–3 packs per class/condition; whether packs share a session is not established | Likely 1 per BUI (INFERRED) | **1 recording block per device (17–60 min), no two devices overlap**; flight modes are consecutive sub-blocks; NLOS Inspire/Phantom 4–6 h after LOS, NLOS M600 30 min after | 1 day per category; 6 consecutive captures per scenario | **1 per archive** (1–22 min, all 812 files); RC and Vid of a system in separate sessions | UNKNOWN — no recording IDs | Not assessed |
| **Potential shortcut: receiver** | High-priority audit | Requires full-corpus audit | Low as class shortcut; L/H receiver/band identity must be preserved | None within dataset; total in cross-dataset | Low within dataset; fixed receiver | None within; same receiver model as RFUAV (low cross-dataset receiver gap) | None within dataset | Not assessed |
| **Potential shortcut: center frequency** | **HIGH — subset ↔ band: Wi-Fi subset 2.437 GHz, all other drone subsets 5.745 GHz; each environment file in one band** | **HIGH — 5.76–5.80 GHz occurs only in 4 aircraft classes; all other classes 2.41–2.47 GHz** | Band-specific activity risk | N/A (direct RF) | **HIGH — assigned UAV channel must be controlled** | Low — fixed 2.44 GHz | **Yes — per-transmitter Fc offsets** | N/A (direct RF) |
| **Potential shortcut: SNR** | Possible | **High — metadata only (Fc, SNR, ScaleFactor) identifies the class of an unseen pack 52% of the time vs 8% chance (12 classes with ≥ 2 packs)**; per-pack SNR 1–50 dB | UNKNOWN | UNKNOWN; distance varies 8–12 m | UNKNOWN | UNKNOWN; fixed geometry | Low — balanced by design | Not assessed |
| **Potential shortcut: received power** | **HIGH — ~30 dB positive/negative power gap observed (n=1 vs 1); gain unknown, cause unresolved** | Not yet established | **YES — strongly band/time dependent** | **HIGH — measured on unseen devices: pre-trigger RMS alone AUC 0.80, clipping alone AUC up to 0.71, both + post-trigger RMS balanced accuracy 0.79**; NLOS ~13 dB weaker | **HIGH — drone 4 ≈ 2.7 dB above drone 1 on the same channel (n=1)** | Audit — RC vs video power; author code z-scores it away | Removed by normalization | Not assessed |
| **Potential shortcut: dataset source** | Future cross-dataset concern | Future cross-dataset concern | High concern in cross-dataset experiments | Total in cross-dataset; **within dataset: device ↔ recording block, flight mode ↔ sub-block in time** | Yes in cross-dataset; **two-drone = day 2 only (category ↔ day)** | Yes in cross-dataset; **within: capture length ↔ archive (perfect), Mavic sessions 7 months later** | Yes in cross-dataset experiments | Yes |
| **Potential shortcut: position** | Not established | Not established | Not established | Aircraft vs controller geometry; LOS/NLOS amplitude shift | **HIGH** — controllers in 2 positions; drones and controllers on different roofs | Low — fixed 7 m | Recording site differs by class (chamber vs building) | Not assessed |
| **Known acquisition defect** | Indoor environment file carries drone metadata; same file names reused across subsets with different content | 3 packs miss one 1 s chunk; MINI3 VTSBW=10 XML named pack1 for pack2 chunks; XML Drone field mislabels 5 packs | Not established | **ADC saturation in all categories incl. Wi-Fi/BT (EMPIRICAL n=135)** | **Dropped samples (lab warning); candidate marker: 600–730 runs of 10–13 zeros/s (EMPIRICAL n=8, causal link not proven)** | No zero-gap/dropout pattern in the 9 inspected captures (zeros 0.03–0.13% = quantization) | Synthetic mixing; possible noise-snippet reuse (UNKNOWN) | Not assessed |
| **Classical-ML opportunity** | Spectral/statistical features | **Very strong: FHSBW/FHSDT/FHSDC/FHSPP/VTSBW** | Strong: temporal + spectral + burst/activity features | Short-window spectral/transient features; clipping must be controlled | Strong spectral/fingerprint features | Strong: burst, hop and bandwidth features on clean signals | Spectral/hop features at mid-high SNR | Strong — paper's 15 transient features |
| **Phase-aware experiments** | Yes | Yes | No | No (real-valued) | Yes | Yes | Yes | No |
| **Magnitude-common experiments** | Yes | Yes | **No** — signed real samples ≠ IQ magnitude; spectral-common only, after frequency alignment | No directly; spectral-common only after digital down-conversion | Yes (spectral-common) | Yes (spectral-common) | Spectral-common within the 14 MHz window | Spectral-common after down-conversion |
| **Detection usefulness** | Useful but environmental negatives are weak alone | **Promising; negatives unresolved** | Good benchmark; negative diversity limited | Hard-negative rejection (UAV vs Wi-Fi/BT); no empty-band detection | Strong transmission-detector research; weak explicit negatives | Positive-only; needs negatives from another dataset | Low-SNR detection vs real building RF; synthetic | Paper-level only |
| **Identification usefulness** | Strong | **Strong** | Moderate; only 3 models | Strong among represented devices; model = unit | **Excellent** for same-model physical-unit fingerprinting | **Strong** — 10 systems incl. hobby RC; RC vs video | Moderate — 6 RC links | Strong in paper (98.13% kNN at 25 dB) |
| **Robustness usefulness** | **Excellent** | **Strong SNR/signal-condition potential** | Limited compared with DRFF-R2 | LOS→NLOS (3 UAVs) only | **Excellent** for single→multi-UAV scenario shift; weak environmental diversity | Low — single chamber, fixed distance | Synthetic SNR sweep −20…30 dB | Paper SNR study only |
| **Unique contribution** | controlled unit/day/receiver/state/interference variation | scale + RF-system diversity + hopping/video characteristics | Parrot + real background + legacy domain + simple detection + burst structure | Explicit Wi-Fi + Bluetooth hard negatives; aircraft vs controller separation; LOS/NLOS | same-model units + real multi-UAV + time-frequency labels | Clean anechoic RC + video references for 10 systems; closest acquisition match to RFUAV | Real uncontrolled-building Wi-Fi/BT/amplifier noise + calibrated SNR sweep | None new vs CardRF + UAVSig (see notes) |
| **Claim ceiling** | controlled robustness across represented DJI conditions | represented-class RF identification/robustness; not universal detection | DroneRF-domain benchmark detection/identification only | Short-window UAS vs Wi-Fi/BT discrimination for represented devices, one site, 8–12 m | controlled-domain same-model fingerprinting and transmission detection | Identification of represented RC/video emissions in a semi-anechoic chamber at 7 m | 6 RC links vs building/Gaussian noise under synthetic SNR, 14 MHz at 2.44 GHz | N/A — not adopted |
| **Provisional Droneacharya role** | Robustness laboratory + auxiliary detection | Positive-diversity/representation training + identification | baseline detector + classical ML + historical external domain | Hard-negative source + aircraft/controller discrimination | fingerprint/generalization diagnostic + activity-label reference + real multi-UAV | Clean-reference identification + leave-system-out novelty + RFUAV cross-dataset partner | External low-SNR test set + building-RF negative domain (evaluation only) | Backlog — paper read only |
| **Primary detection foundation?** | No, not alone | No, negatives unresolved | No, not alone | Hard-negative component; evaluate by holding out whole devices with clipping/session controls | No | No — no negatives | No — synthetic mixtures, no recording IDs | No |
| **Full-download justification** | Done — full release on D: (731 .mat, 405 GB) | Done — 37 class RARs (109.2 GB, SHA-256 checked); `ValidationSet_5Drones` (162 GB) not downloaded | Done — full release (Mendeley zip, 23 RARs) | Done — 71.1 GB, CRC of all 10,172 members verified | Subset — 366 of 726 files (360 `.mat`), MD5 verified; full 238.9 GB not needed yet | Done — all 16 archives (46.7 GB), MD5 verified | Done — `archive.zip` (23.5 GB), CRC verified; evaluation only | No — 124 GB, no new axis (paper read only) |
| **Window-label reliability** | **AUDIT NEEDED** — 10-ms window sampling insufficient to characterize temporal occupancy | UNKNOWN — pack-level inspection only | **HIGH PRIORITY** — bursty activity observed in BUI 10000 (Bebop *on & connected*, its least active mode); occupancy likely differs by mode | Burst classes: noise before trigger, ~5 µs transient after; **DJI aircraft: signal before trigger too** | **HIGH** — WHIRLS boxes align with energy (34–38 dB inside vs outside); gap markers every ~1.5 ms | RC bursty (12–65% of 1 ms windows active); video 40–93% — activity-aware labels needed | UNKNOWN — 1.17 ms vectors may miss RC bursts at low SNR | Triggered captures (INFERRED, as CardRF) |

## DRFF-R2 supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Dataset-3 metadata fields | TD, State, C, U, D, Height, V — VERIFIED (all 179 Dataset-3 files) |
| Center-frequency key naming | `CenterFrequence` in every file except outdoor_environment (`CenterFreq`) |
| Antenna | Dual-band omnidirectional, 5 dBi |
| Label granularity | File/capture |
| Label provenance | acquisition config + filename + MAT metadata |
| Published detection baseline | No |
| Published classification baseline | Yes (EfficientNetB0) |
| Positive/negative dtype matched | Yes — EMPIRICAL (n=1 positive vs 1 environment file) |
| Positive/negative length matched | Yes — EMPIRICAL (n=1 positive vs 1 environment file) |
| Positive/negative gain matched | Not stored in any file |
| Positive/negative receiver matched | UNKNOWN |
| Positive/negative day matched | UNKNOWN |

## RFUAV supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Label granularity | Pack / clip |
| Label provenance | per-pack XML metadata |
| Published detection baseline | Yes |
| Published classification baseline | Yes |
| Antenna documented | UNKNOWN |
| Controller ID available | UNKNOWN |
| Signal-condition structure | Class may contain multiple VTSBW conditions |
| Known defects | 37 vs 35 UAV conflict; partial release; taxonomy ambiguity; 5 XML Drone mislabels; 3 duplicated XML serials; 3 packs with a missing chunk |

## DroneRF supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Logical segments | 227 |
| Released CSV files | 454 |
| Background segments | 41 |
| Bebop segments | 84 |
| AR Drone segments | 81 |
| Phantom 3 segments | 21 |
| Samples per band file | **10,000,000 — VERIFIED (all 454 files)** |
| Samples per paired segment | **20,000,000 — VERIFIED (all 227 pairs)** |
| Published prose sample-count claim | CONFLICTING — states 1M/part |
| Recording length per mode | ~5.25 s per flight mode, ~10.25 s background (paper, verified via Europe PMC) |
| BUI digits | [drone present][drone type ×2][flight mode ×2]; 10000 = Bebop on & connected, 10011 = Bebop flying + video |
| Empirical CSV structure | One comma-separated row (all 454 files) |
| Filename structure | `<BUI><H/L>_<segment>.csv` |
| L/H pairing | **VERIFIED — all 227 segments have exactly one L and one H file** |
| Temporal positive behavior | **Bursty — empirically observed in BUI 10000 (Bebop on & connected); other modes not yet checked** |
| Bebop-L median RMS | 2.604 |
| Bebop-L RMS p90 | 1290.641 |
| Bebop-L RMS maximum | 1709.861 |
| Background-L median RMS | 2.378 |
| Window-label reliability | File-level positive does not guarantee continuously active RF |

## UAVSig supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Label granularity | Transmission time-frequency bounding box |
| Label provenance | WHIRLS DSP + known scenario/channel metadata |
| Identity assignment | Single transmitter: deterministic; two-UAV: frequency-assisted; multi-controller: active-set only (ambiguous) |
| Published detection baseline | Yes (MILCOM 2024) |
| Published fingerprinting baseline | Yes |
| Scenarios | 16 one-drone / 72 two-drone / 32 controller |
| Captures per scenario | 6 × 1 s |
| Published split | 5 train / 1 test per scenario; same-scenario, consecutive captures — weak for generalization |
| Known defects | Dropped-sample gaps; controller hops can fall outside 50 MHz capture; no explicit negative class |
| Droneacharya split policy | Do NOT use published split as primary; group by scenario + capture sequence + transmitter + day |

## CardRF supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Devices | UAS: DJI Phantom 4, Inspire, Matrice 600, Mavic Pro 1, Beebeerun FPV, 3DR Iris (FlySky FS-TH9x). BT: iPhone 6S, iPhone 7, iPad 3, FitBit Charge3, Motorola E5 Cruise. Wi-Fi: Cisco Linksys E3200, TP-Link TL-WR940N |
| NLOS coverage | UAV only: DJI Inspire, Matrice 600, Phantom (Flying) |
| Trigger | XOrg = −1.25e-4 s in every audited file → trigger at capture midpoint (EMPIRICAL n=135) |
| Processed slicing | 1024 samples × 100 consecutive slices per capture, start index 2,500,000, 1-sample overlap between slices |
| Processed split | 70/30 by capture (350/150 per class), inherited from the raw split — which is same-session |
| Clipping by class (processed, train) | UAV: Inspire 16%, Phantom 27%, M600 29%, Beebeerun 30%, Mavic Pro 30%. Controller: Beebeerun 0.001%, Inspire 7%, M600 8%, Phantom 14%, Mavic Pro 37%, 3DR Iris 40% |
| Documentation conflict | Scale factor 6.581e-06 (text) vs YInc 6.5841e-06 (struct) |
| Shipped code | `*_Resampling.m` slice only (no resampling); `Resampling_Dividers.mlx` uses 5 slices — not the production recipe |
| Recording timeline | Wi-Fi Aug 25 01:05–01:59; Bluetooth Aug 25 04:00 – Aug 26 05:39; aircraft Aug 25 22:06 – Aug 26 04:51 and Aug 28 22:52–23:52 (M600) and Aug 29 02:32–03:09 (Beebeerun); controllers Aug 28 23:56 – Aug 29 03:46; 3DR Iris Nov 1 (scope clock) |
| Scope clock | Hours run 22:00–07:59 across midnight every day; 12/24-hour convention and time zone not established |

## RMA supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Repository | KU Leuven RDR, DOI 10.48804/HZRVNZ, v1.0 (2024-01-16) |
| Archives | Spektrum_DX4e, mini2RC, mini2vid, inspire2RC, inspire2Vid, matriceRC, matricevid (71 files each); Frysky, NineEagles, wltoys, Q205, SJRC_pro (51 each); MavicRC1/2 (15+16), MavicVid1/2 (14+15) |
| Documentation conflicts | Mavic not in README; Frysky vs "Taranis Q X7"; SJRC F11 Pro "drone + RC" but one archive |
| Antenna / distance | OmniLOG 70600 omni; 7 m |
| Published preprocessing | `get_spectrogram2.m`: NFFT 1024, Hann, no overlap, log-amplitude, per-spectrogram z-score |
| Not to be confused with | Zenodo 10.5281/zenodo.21428081 (ZHAW 2026): different drones + Wi-Fi/BT background, spectrogram images only |
