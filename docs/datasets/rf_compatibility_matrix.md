# Droneacharya RF Dataset Compatibility Matrix

Status vocabulary:

- **VERIFIED** — confirmed by primary source or empirical inspection
- INFERRED — reasonable inference requiring confirmation
- CONFLICTING — sources disagree
- UNKNOWN — not established
- EMPIRICAL — requires inspection of actual data

CardRF and later columns also mark evidence level: `author doc` (the dataset authors' own description), `n=…` (number of files a value was checked on), and VERIFIED only for values we measured ourselves.

## Schema frozen after DRFF-R2 + RFUAV reconnaissance. No new rows/columns unless a dataset reveals a genuinely new dimension.

Schema amendment (UAVSig): two genuinely new dimensions added — `Potential shortcut: position` and `Known acquisition defect`.

| Parameter | DRFF-R2 | RFUAV | DroneRF | CardRF | UAVSig | RMA | Noisy RF | Remote Controller RF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Primary purpose** | RFF/model/state/robustness | Identification + detection benchmark | Detection + identification + flight-mode classification | UAV/controller vs Wi-Fi/Bluetooth discrimination + RF fingerprinting | Transmission detection + spectrum localization + RF fingerprinting | RC + video-link identification; novelty detection (per publication titles) | | |
| **Year** | 2026 | 2025 | 2019 | 2020 acquisition (Aug 2020, author doc) | 2024 | 2024 release (v1.0); captures created 2022 (MAT header, n=1) | | |
| **License** | CC BY 4.0 | Apache-2.0 (HF artifact); raw scope verify | CC BY 4.0 | UNKNOWN — check IEEE DataPort | CC0 1.0 | **CC BY-NC 4.0** (non-commercial) | | |
| **Full reported scale** | 26 physical units / 8 models | ~1.3 TB / 37 distinct UAVs reported | 227 logical segments / 454 CSV files | >65 GB raw (66.3 GB zip); 6 UAS + 5 Bluetooth + 2 Wi-Fi devices | 8 transmitters: 4 DJI M100 UAVs + 4 DJI C1 controllers | 46.7 GB; 16 zips; 812 `.mat` captures; 10 systems | | |
| **Public-release caveat** | Public dataset | Partial; repo says 37 raw clips / 35 types | Full public dataset | Processed_CardRF (1 GB) = LOS UAV + controller only; Wi-Fi/Bluetooth/NLOS only in raw zip | V3; 726 files; bulk browser ZIP discouraged | Full public release; README omits Mavic archives (~19 GB) | | |
| **Raw representation** | Complex IQ | Complex IQ | Time-domain amplitude | Real-valued direct-RF waveform (oscilloscope); no IQ | Complex IQ | Complex IQ | | |
| **Scalar dtype** | **float32 VERIFIED** | **float32 VERIFIED** | Text decimal values; numerical storage dtype N/A | int16 ADC codes (author doc, n=1 example); volts = Data×YInc+YOrg | float32 stored (source); 16-bit I/Q acquisition; empirical check pending | complex double (float64 real/imag) — EMPIRICAL n=1 | | |
| **Raw format** | MATLAB/HDF5 `.mat` | interleaved binary `.iq` + XML | Single-row CSV | `.mat`: structs `Channel_1` + `Frame`; waveform in `Channel_1.Data` | `.mat` | MAT v7.3 (HDF5), variable `uhd_samps` — EMPIRICAL n=1 | | |
| **I/Q layout** | Separate `RF0_I`, `RF0_Q` | `I0,Q0,I1,Q1...` | N/A — no complex IQ | N/A — real samples | **UNKNOWN — empirical check** | HDF5 compound real/imag, shape (1, N) — EMPIRICAL n=1 | | |
| **Samples/capture chunk** | **140M/channel VERIFIED** | **100M complex VERIFIED** | **10M/band; 20M paired segment VERIFIED** | 5,000,000 (author doc, n=1); raw audit pending | ~50M complex expected; **EMPIRICAL CHECK** | 10,000,000 (n=1); Mavic files 8–12× larger — UNKNOWN | | |
| **Chunk duration** | **1.4 s VERIFIED** | **1.0 s VERIFIED** | UNKNOWN pending verified Fs | 250 µs (INFERRED from XInc × NumPoints) | 1 s (source) | 100 ms (n=1) | | |
| **Sample rate** | **100 MS/s VERIFIED** | **100 MS/s sampled packs VERIFIED** | **UNKNOWN / VERIFY** | 20 GSa/s (author doc: XInc = 5e-11 s, n=1); raw audit pending | **50 MS/s SOURCE VERIFIED** | 100 MS/s (author README + code) | | |
| **Center frequency** | 5.745 GHz outdoor; 2.437 GHz Wi-Fi subset | Per-pack; observed 2.440/2.450 GHz | 2.4-GHz acquisition region | N/A — direct RF sampling, no down-conversion; 2.4 GHz band of interest | **2.4435 GHz SOURCE VERIFIED** | 2.44 GHz (author README + code) | | |
| **IF/capture bandwidth** | **VERIFY** | **100 MHz sampled packs VERIFIED** | Up to 40 MHz/receiver documented | Nyquist 10 GHz; analog front-end bandwidth UNKNOWN | 50 MHz (source) | 100 MHz (INFERRED from complex 100 MS/s) | | |
| **Occupied signal bandwidth** | scenario/signal dependent | class/condition dependent; distinct from IF BW | Signal/model/mode dependent; not yet characterized | Device-dependent; not characterized | UAV channels ~10 MHz; controllers frequency-hop (hop domain exceeds 50 MHz capture) | System and link dependent; RC vs video recorded separately | | |
| **Receiver** | Documentation conflict; empirical receiver IDs `u1/u2` | USRP X310 sampled packs | 2× NI-USRP-2943R | One oscilloscope-based capture system (RFSSCS) | USRP B205mini-i (single receiver) | USRP X310 (single) + OmniLOG 70600 omni | | |
| **Pack/unit identifier** | Physical UAV unit available | source-pack serial available; physical UAV unit unknown | BUI + segment; physical-unit identity unknown | Device per directory; one unit per model (INFERRED) → model = unit | Physical transmitter identity available | Archive (system × RC/Vid) + file index; one unit per system (INFERRED) | | |
| **Manufacturer coverage** | DJI only | Multiple apparent ecosystems; mapping incomplete | Parrot + DJI | DJI, Beebeerun, 3DR/FlySky + Apple, FitBit, Motorola, Cisco, TP-Link | DJI only | DJI + Spektrum, FrSky, Nine Eagles, WLtoys, Q205, SJRC | | |
| **Models/classes** | 8 models | 37 public image classes | 3 UAV models + background; hierarchical mode classes | 6 UAS (aircraft + controller), 5 Bluetooth, 2 Wi-Fi; flight-mode sub-labels | 1 UAV model + 1 controller model; 8 physical transmitters | 10 systems (9 in README + undocumented Mavic); RC/Vid split for DJI | | |
| **Physical units** | **26** | UNKNOWN | UNKNOWN | 1 per model (INFERRED from Table I) | **4 UAV + 4 controller** | 1 per system (INFERRED) | | |
| **Same-model multi-unit** | Yes, uneven | UNKNOWN | UNKNOWN | No | **YES — core strength** | No (as documented) | | |
| **Positive signal composition** | scenario-dependent; hopping/mixed/interference variants | control/hopping/video characteristics; class-dependent; VTSBW conditions | Drone/controller communications; mode-dependent; bursty | Aircraft (Non_Control) and controller (Control) captured separately; Flying/Hovering/Videoing | UAV fixed-channel transmissions + frequency-hopping controllers | RC uplink and video downlink recorded separately; hobby controllers RC only | | |
| **Day metadata** | **Yes; empirically verified** | Not established | Not established | Single campaign, Aug 2020; per-capture date UNKNOWN | Yes; two collection days (cross-day task mapping UNKNOWN) | Not provided; per-file MAT creation time available | | |
| **Receiver variation** | **Yes; empirically verified** | Across full corpus requires audit | Two simultaneous receivers represent L/H bands, not a robustness split | No — one capture system | No — one receiver | No — one receiver | | |
| **Gain/scale** | File documentation inconsistent; gain unresolved in inspected files | ScaleFactor per pack; sampled value 60 dB | UNKNOWN | Example YInc 6.5841e-6 V/code, YDispRange 0.4 V (n=1); per-class settings UNKNOWN | 20 dB receiver gain (source) | UNKNOWN | | |
| **SNR metadata** | Not consistently present in inspected MATs | Per-pack; sampled 15/29/31 dB | Not provided per segment | Not provided | Not established per capture | Not provided; fixed 7 m in chamber | | |
| **Environmental negatives** | **Yes** (Dataset-7 environmental baseline) | Raw corpus: unresolved | Yes — recorded RF background | No empty-band class (signal-triggered captures) | No explicit background class | **None in release** | | |
| **Negative taxonomy** | ambient/no-emitter only | UNKNOWN | Ambient RF background | **Wi-Fi (2 routers) + Bluetooth (5 devices)** — raw only | Weak/unlabeled ambient RF | None | | |
| **Negative diversity** | Low — single environmental baseline | UNKNOWN — detection subset not inspected | Limited/unknown | Moderate: 7 devices, 2 technologies, one site | Weak | None | | |
| **Negative provenance** | Same-site same-equipment recordings (Dataset-7) | UNKNOWN | Recorded with drones absent | Same site/system, deliberate single-emitter captures (author doc) | Uncontrolled ambient 2.4-GHz activity, spatially suppressed (directional antenna) | N/A | | |
| **Hard-negative emitters** | None dedicated; Wi-Fi appears only mixed with UAV positives | UNKNOWN | None explicitly labeled | **YES — Wi-Fi + Bluetooth (LOS, raw)**; not yet inspected | None explicitly labeled | None | | |
| **Recorded interference conditions** | UAV + Wi-Fi mixed positives (Dataset-6) | Not established | Uncontrolled ambient RF only | None — single-source captures | Ambient RF + real simultaneous UAVs | None (semi-anechoic chamber) | | |
| **Positive/negative Fs matched** | **Yes, sampled** | UNKNOWN | Presumed same acquisition pipeline; not independently metadata-verified | Presumed (same scope); EMPIRICAL CHECK per file | N/A — no explicit negative corpus | N/A — no negatives | | |
| **Positive/negative Fc matched** | **Yes, sampled** | UNKNOWN | Same paired acquisition bands; exact metadata not per file | N/A (direct RF); vertical scale/trigger matching EMPIRICAL CHECK | N/A | N/A | | |
| **Negative difficulty** | **Weak alone; major energy shortcut risk** | UNKNOWN | Better than simple silence, but narrow taxonomy | Potentially high (active same-band emitters); clipping/scale may make it trivial — audit | Not suitable as standalone hard-negative benchmark | N/A — no negatives | | |
| **Indoor/outdoor** | Both | Real-world reported; class-level conditions need mapping | Laboratory | Outdoor (AERPAW Lake Wheeler, NC) | Controlled lab-like OTA | Indoor semi-anechoic chamber | | |
| **Distance metadata** | Yes in dataset design; schema varies | Not established in sampled XML | Not available per file | 8–12 m LOS (range, not per file) | Fixed/set distance described; exact metadata to verify | Fixed 7 m (author README) | | |
| **Altitude/state metadata** | Yes | Not established | Operational mode labels; no detailed altitude metadata | Flight-mode labels only | No flight-state focus | None | | |
| **Frequency hopping** | Explicit subset | **Major documented characteristic** | Not explicitly represented as dedicated metadata | Not labeled; 250 µs too short to observe hop sequences | **YES — controllers** | Likely in hobby RC links; not labeled (INFERRED) | | |
| **Video transmission signals** | Not central | **Explicit; VTSBW conditions** | Yes — video-recording operating mode | Inspire 'Videoing' mode only | Not established as separate label | **YES — separate Vid archives (Mini 2, Inspire 2, Matrice, Mavic)** | | |
| **Multi-UAV mixtures** | **Yes** | Not established | No | No | **YES — real OTA, two simultaneous drones** | No | | |
| **Controlled environment** | RF-absorbing setup | Not established | Laboratory | No — outdoor field | Yes | Yes — semi-anechoic chamber | | |
| **Schema consistency** | **No — subset drift empirically observed** | XML schema consistent in sampled packs | Yes for sampled CSV structure | Processed: VERIFIED (1027 cols UAV, 1026 controller; all 22 files); raw EMPIRICAL | **EMPIRICAL** | EMPIRICAL (n=1) | | |
| **Parent-recording identity** | Yes — filename encodes unit/config/day/receiver | Pack identity available | Yes: paired L/H logical segment | Raw: one file = one capture; processed: **100-row chained runs = one capture (VERIFIED)** | Capture/scenario structure available; empirical decode required | Archive + file index; continuity of consecutive files UNKNOWN | | |
| **Window-safe splitting possible** | Yes — leakage-safe keys established | Pack-level; within-pack window independence requires design | Yes; group all windows by logical segment | Yes at capture level; session independence NOT established | YES, with scenario/capture grouping | Yes by file; session grouping UNKNOWN | | |
| **Original/raw representation** | Complex IQ in MAT files | Complex IQ in binary `.iq` files | Amplitude CSV | Real int16 direct-RF waveform in MAT | Complex IQ in MAT files | Complex IQ (double) in MAT v7.3 | | |
| **Published derived representations** | STFT spectrograms | Spectrograms + 37-class image dataset + FFT/STFT tooling | normalized time plots + FFT/spectral processing | Processed_CardRF: 1024-sample raw slices, 100/capture from the trigger point | 512×512 spectrograms + bounding labels; spectrogram generation code available | Author code: 1024-pt Hann spectrogram, log-amplitude, z-scored per spectrogram | | |
| **Raw→derived reproducibility** | Yes — raw IQ available | Yes — raw IQ + published tooling available | Strong; public code available | Partial — shipped notebook (5 slices) ≠ released data (100 slices) | Strong; visualization/spectrogram/WHIRLS code available | Strong — raw IQ + author code | | |
| **Train class support** | N/A for our raw audit | **37/37** | N/A until our split | Processed: 11/11 classes (350 captures each; Beebeerun ctrl 245) | Published same-scenario 5/6 capture training | N/A — no published split | | |
| **Validation class support** | N/A | **37/37** | N/A until our split | Processed: 11/11 classes (150 captures each; Beebeerun ctrl 105) | Published same-scenario 1/6 capture testing | N/A | | |
| **Class balance** | Unit/model imbalance | **63–499 train images/class (~7.9×)** | Imbalanced: 41 background / 84 Bebop / 81 AR / 21 Phantom segments | Processed: balanced by design; flight modes exactly 50/50 | Scenario-controlled; complete empirical audit pending | 51–71 files per archive; Mavic 29–31 files but much larger | | |
| **Label taxonomy** | Model/unit/state are distinguishable | **AMBIGUOUS — UAV/RC-system-like classes** | BUI → detection/model/mode hierarchy | LOS/NLOS → category (UAV / Controller / Wi-Fi / Bluetooth) → device → flight mode | Transmission bounding box + physical transmitter identity / active-controller set | System × signal type (RC / Vid) from archive name | | |
| **Leakage-safe keys** | unit/day/receiver/config | pack ID available; stronger grouping unresolved | BUI + logical segment index | capture file (processed: 100-row run); session UNKNOWN | scenario + capture sequence + physical transmitter + day | archive + file index (names repeat across RC/Vid archives); session UNKNOWN | | |
| **Potential shortcut: receiver** | High-priority audit | Requires full-corpus audit | Low as class shortcut; L/H receiver/band identity must be preserved | None within dataset; total in cross-dataset | Low within dataset; fixed receiver | None within; same receiver model as RFUAV (low cross-dataset receiver gap) | | |
| **Potential shortcut: center frequency** | Yes | **Yes** | Band-specific activity risk | N/A (direct RF) | **HIGH — assigned UAV channel must be controlled** | Low — fixed 2.44 GHz | | |
| **Potential shortcut: SNR** | Possible | **High-priority** | UNKNOWN | UNKNOWN; distance varies 8–12 m | UNKNOWN | UNKNOWN; fixed geometry | | |
| **Potential shortcut: received power** | **HIGH for Dataset-7 detection** | Not yet established | **YES — strongly band/time dependent** | **HIGH — clipping fraction 0.001%–40% by class (VERIFIED, processed)** | **HIGH-priority audit** | Audit — RC vs video power; author code z-scores it away | | |
| **Potential shortcut: dataset source** | Future cross-dataset concern | Future cross-dataset concern | High concern in cross-dataset experiments | Total in cross-dataset (unique 20 GSa/s real format) | Yes in cross-dataset experiments | Yes in cross-dataset experiments | | |
| **Potential shortcut: position** | Not established | Not established | Not established | Controller vs aircraft geometry differs; LOS/NLOS for 3 UAVs | **HIGH-priority audit** | Low — fixed 7 m | | |
| **Known acquisition defect** | Not established | Not established | Not established | **ADC saturation at codes −32736/30720, class-dependent (VERIFIED, all 22 processed files)** | **Random dropped-sample gaps (official lab warning)** | None documented; exact-zero samples (12,514 / 10M, n=1) to audit | | |
| **Classical-ML opportunity** | Spectral/statistical features | **Very strong: FHSBW/FHSDT/FHSDC/FHSPP/VTSBW** | Strong: temporal + spectral + burst/activity features | Short-window spectral/transient features; clipping must be controlled | Strong spectral/fingerprint features | Strong: burst, hop and bandwidth features on clean signals | | |
| **Phase-aware experiments** | Yes | Yes | No | No (real-valued) | Yes | Yes | | |
| **Magnitude-common experiments** | Yes | Yes | Yes | No directly; spectral-common only after digital down-conversion | Yes (spectral-common) | Yes (spectral-common) | | |
| **Detection usefulness** | Useful but environmental negatives are weak alone | **Promising; negatives unresolved** | Good benchmark; negative diversity limited | Hard-negative rejection (UAV vs Wi-Fi/BT); no empty-band detection | Strong transmission-detector research; weak explicit negatives | Positive-only; needs negatives from another dataset | | |
| **Identification usefulness** | Strong | **Strong** | Moderate; only 3 models | Strong among represented devices; model = unit | **Excellent** for same-model physical-unit fingerprinting | **Strong** — 10 systems incl. hobby RC; RC vs video | | |
| **Robustness usefulness** | **Excellent** | **Strong SNR/signal-condition potential** | Limited compared with DRFF-R2 | LOS→NLOS (3 UAVs) only | **Excellent** for single→multi-UAV scenario shift; weak environmental diversity | Low — single chamber, fixed distance | | |
| **Unique contribution** | controlled unit/day/receiver/state/interference variation | scale + RF-system diversity + hopping/video characteristics | Parrot + real background + legacy domain + simple detection + burst structure | Explicit Wi-Fi + Bluetooth hard negatives; aircraft vs controller separation; LOS/NLOS | same-model units + real multi-UAV + time-frequency labels | Clean anechoic RC + video references for 10 systems; closest acquisition match to RFUAV | | |
| **Claim ceiling** | controlled robustness across represented DJI conditions | represented-class RF identification/robustness; not universal detection | DroneRF-domain benchmark detection/identification only | Short-window UAS vs Wi-Fi/BT discrimination for represented devices, one site, 8–12 m | controlled-domain same-model fingerprinting and transmission detection | Identification of represented RC/video emissions in a semi-anechoic chamber at 7 m | | |
| **Provisional Droneacharya role** | Robustness laboratory + auxiliary detection | Positive-diversity/representation training + identification | baseline detector + classical ML + historical external domain | Hard-negative source (pending raw audit) + aircraft/controller discrimination | fingerprint/generalization diagnostic + activity-label reference + real multi-UAV | Clean-reference identification + leave-system-out novelty + RFUAV cross-dataset partner | | |
| **Primary detection foundation?** | No, not alone | No, negatives unresolved | No, not alone | Hard-negative component only, pending raw clipping/scale audit | No | No — no negatives | | |
| **Full-download justification** | Later/subset-dependent | Later/subset-dependent | Yes — only ~3.75 GB and useful benchmark | Yes — Wi-Fi/BT/NLOS exist only in raw (66.3 GB, downloading) | No — strategic subset sufficient initially | No — fetch individual captures from the remote zips | | |
| **Window-label reliability** | **AUDIT NEEDED** — 10-ms window sampling insufficient to characterize temporal occupancy | UNKNOWN — pack-level inspection only | **HIGH PRIORITY — file-level positive does not guarantee continuously active RF** | Signal present post-trigger by construction; transient/steady boundary per file UNKNOWN | UNKNOWN — WHIRLS time-frequency labels exist; label-to-window mapping and dropped-sample effects to audit | Bursty RC observed (n=1): window RMS p90 ≈ 6.6× median → activity-aware labels needed | | |

## DRFF-R2 supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Dataset-3 metadata fields | VERIFIED: TD, State, C, U, D, Height, V |
| Center-frequency key naming | DS 3/6: `CenterFrequence`; DS 7: `CenterFreq` |
| Antenna | Dual-band omnidirectional, 5 dBi |
| Label granularity | File/capture |
| Label provenance | acquisition config + filename + MAT metadata |
| Published detection baseline | No |
| Published classification baseline | Yes (EfficientNetB0) |
| Positive/negative dtype matched | **VERIFIED** |
| Positive/negative length matched | **VERIFIED** |
| Positive/negative gain matched | UNKNOWN |
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
| Known defects | 37 vs 35 UAV conflict; partial release; taxonomy ambiguity |

## DroneRF supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Logical segments | 227 |
| Released CSV files | 454 |
| Background segments | 41 |
| Bebop segments | 84 |
| AR Drone segments | 81 |
| Phantom 3 segments | 21 |
| Samples per band file | **10,000,000 VERIFIED** |
| Samples per paired segment | **20,000,000 VERIFIED** |
| Published prose sample-count claim | CONFLICTING — states 1M/part |
| Empirical CSV structure | One comma-separated row |
| Filename structure | `<BUI><H/L>_<segment>.csv` |
| L/H pairing | **VERIFIED** |
| Temporal positive behavior | **Bursty — empirically observed** |
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
| Droneacharya split policy | Do NOT use published split as primary; group by scenario + capture sequence + transmitter + day |
| Known defects | Dropped-sample gaps; controller hops can fall outside 50 MHz capture; no explicit negative class |
| Empirical checks outstanding | MAT schema, IQ layout, sample count, gap representation, label storage, filename decoding |

## CardRF supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Devices | UAS: DJI Phantom 4, Inspire, Matrice 600, Mavic Pro 1, Beebeerun FPV, 3DR Iris (FlySky FS-TH9x). BT: iPhone 6S, iPhone 7, iPad 3, FitBit Charge3, Motorola E5 Cruise. Wi-Fi: Cisco Linksys E3200, TP-Link TL-WR940N |
| NLOS coverage | UAV only: DJI Inspire, Matrice 600, Phantom (Flying) |
| Trigger | Example XOrg = −1.25e-4 s → trigger at capture midpoint (author doc) |
| Processed slicing | 1024 samples × 100 consecutive slices per capture, start index 2,500,000, 1-sample overlap between slices |
| Processed split | 70/30 by capture (350/150 per class); verified no capture straddles Train/Test |
| Clipping by class (processed, train) | UAV: Inspire 16%, Phantom 27%, M600 29%, Beebeerun 30%, Mavic Pro 30%. Controller: Beebeerun 0.001%, Inspire 7%, M600 8%, Phantom 14%, Mavic Pro 37%, 3DR Iris 40% |
| Documentation conflict | Scale factor 6.581e-06 (text) vs YInc 6.5841e-06 (struct) |
| Shipped code | `*_Resampling.m` slice only (no resampling); `Resampling_Dividers.mlx` uses 5 slices — not the production recipe |

## RMA supplementary detail (not duplicated into main matrix)

| Parameter | Value |
| --- | --- |
| Repository | KU Leuven RDR, DOI 10.48804/HZRVNZ, v1.0 (2024-01-16) |
| Archives | Spektrum_DX4e, mini2RC, mini2vid, inspire2RC, inspire2Vid, matriceRC, matricevid (71 files each); Frysky, NineEagles, wltoys, Q205, SJRC_pro (51 each); MavicRC1/2 (15+16), MavicVid1/2 (14+15) |
| Documentation conflicts | Mavic not in README; Frysky vs "Taranis Q X7"; SJRC F11 Pro "drone + RC" but one archive |
| Antenna / distance | OmniLOG 70600 omni; 7 m |
| Published preprocessing | `get_spectrogram2.m`: NFFT 1024, Hann, no overlap, log-amplitude, per-spectrogram z-score |
| Not to be confused with | Zenodo 10.5281/zenodo.21428081 (ZHAW 2026): different drones + Wi-Fi/BT background, spectrogram images only |
