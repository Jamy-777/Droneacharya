# Claims log

Hand-written. This is the only file in `docs/results/` that says which numbers may be quoted and with what limits.
Every other file here is raw generator output and is rewritten on each run.

**State on 2026-10-08, after the gate checks:** two results have passed the gate:

- **A1:** acoustic ranking on an unseen recording chain by the trained detector, and also by MFCC. It holds with or
  without averaging the microphone channels. The single 7 kHz band (A0) does **not**: it came from channel averaging
  and is withdrawn as a cue (A-SM, W6).
- **I-D:** RF identification of an unseen unit on an unseen day, at the same receiver.

Both are quotable only with the limits written beside them. No RF detection result has passed.

## Rules

**Gate.** A number is quotable only when all four are done and written next to it:

1. Broken down by drone type **and** by non-drone type.
2. Compared with the strongest single-feature baseline computed **inside the test set**.
3. One "what else could produce this number?" check, run.
4. The count of independent devices, recordings or units it rests on.

**Process, from 2026-10-08:**

- A test's design is committed before the test runs. In every script committed up to `0e188c0`, the
  "design fixed before results" docstring was committed together with its results, so git history cannot verify it.
  The gate checks below were designed in `e68d5dc` and run after it.
- CardRF is development data. It has been scored more than a dozen times, and tile width and steering were changed
  in response to its results. No CardRF number counts as evidence for a new claim.
- One revision per test set.
- A threshold or operating point is claimed only on the receiver or recording chain it was set on.
- Results text describes. No pitch language.

**Status words:**

| Status | Meaning |
| --- | --- |
| QUOTABLE | All four gate items are done; quote only with the limits beside it |
| PENDING GATE | The number stands, but some gate items are missing |
| HINT | Not a claim |
| NOT ESTABLISHED | Tested, and didn't hold |
| WITHDRAWN | Was claimed, is wrong |

**"Best single feature"** in this log means the feature, and the direction, that score best on the test set itself.
It is picked with hindsight, so it is optimistic for the baseline. "Power" means log RMS of the tile before
normalisation, with no training and the direction fixed in advance (more power = drone).

## Capabilities and acceptance criteria

Each test still gets its own design committed before it runs. This table says which question each capability
answers and what counts as passing.

| Capability | Question | Test | Passes if | Status |
| --- | --- | --- | --- | --- |
| Acoustic presence | Is a drone audible? | Train on one recording chain, test on another held out whole | Interval above 0.5; every non-drone class holds; beats in-test loudness and the best single feature | **A1:** the trained detector passes as a ranking (0.68 / 0.85; 0.65 / 0.85 on a single microphone); MFCC performs alike. The 7 kHz single band (A0) is withdrawn: it came from channel averaging. Thresholds do not transfer between chains. False alarms per hour are not measurable with this data. Both acoustic sets are now development data |
| RF activity | Is anything transmitting? | Energy above an estimated noise floor | Classical signal processing, not ML. Power inside UAVSig: 0.95–0.96 | Implementation only |
| RF presence (stage 2) | Is the transmitter a drone, not Wi-Fi/Bluetooth? | Both classes from one receiver, receiver unseen in training | Beats power and the best single feature inside the test set | **NOT ESTABLISHED.** Needs a set like DroneRFb-DIR (`docs/datasets/candidates.md`) or our own recordings |
| RF identification | Which known drone model is this? | Unseen unit on an unseen day; then another receiver | Interval above the permutation null's 95th percentile; beats bandwidth alone | **I-D:** passes at the same receiver, survives the SNR audit; behaviour features 0.62 (0.69 per capture). Across receivers: not established (behaviour features at chance; receiver fully predictable from them) |
| Robustness | How does it degrade with SNR, distance, receiver? | Curves, never one number | Reported as curves | Noisy RF SNR curve only (floor unresolved) |
| Fusion | Does combining RF and audio beat the best single channel? | RF and audio recorded of the same events | Beats the stronger channel on paired data | **Not testable:** no public set records both |

## What the system outputs

- **Two separate channels, each labelled with what it measures and how far it is validated:**
  - **Acoustic score:** a drone-presence ranking score. It is validated as a ranking across two recording chains.
    It is not a calibrated probability.
  - **RF score:** identification ("resembles known model X", with its score) plus activity (energy above the noise
    floor). It is not a drone-presence probability.
- **No merged probability.** A combined number may be displayed only as a "provisional decision score", defined by a
  fixed written rule. It stays that way until RF and audio recorded of the same events exist to validate it.

## Demonstration operating envelope (proposed; the project owner decides)

| Item | Proposed for the demo | Reason |
| --- | --- | --- |
| Sensors | Recorded files from public datasets; no live hardware | No project hardware yet |
| RF band | 2.4 GHz only | 5.8 GHz identification is untested |
| Latency | One decision per second per channel | Acoustic uses 1 s windows; RF tiles are aggregated per second |
| False-alarm budget | Target ≤ 1 per hour per channel | Currently unmeasurable: each public chain holds minutes, not hours, of continuous negatives (UaVirBASE ambient: 416 s). Zero alarms in 416 s still allows up to ~26 per hour at 95% confidence |
| Range | Not specified | Distance is not modelled yet |
| Unknown drones | RF identification must be able to answer "unknown" | Not implemented yet |
| Missing modality | Each channel works alone | The channels are separate by design |
| Compute | Real-time on a laptop CPU | To be measured |

## Acoustic

### A-SM. Single microphone and MFCC: A0 does not survive; A1 does

Source: `scripts/check_acoustic_single_mic_mfcc.py` (design committed in `ffbb935` before the run) →
`results/checks/acoustic_single_mic_mfcc.json`.

**Why this test.** Every earlier acoustic number averaged a recording's channels before extracting features.
UaVirBASE is 8 directional shotgun microphones (Rode NTG-2) metres apart, and Svanström is stereo. Averaging
spaced or directional microphones comb-filters the sum, most strongly at high frequency, which is where the 7 kHz
cue sat. Svanström and UaVirBASE are development data, so this test could only weaken a claim, never upgrade one.

**Primary: channel 0 only.**

| Train → test | Rule chosen on training | Rule | Detector (41 features) | MFCC gbm | MFCC logistic | Detector − rule | MFCC gbm − rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UaVirBASE → Svanström | 500–1000 Hz tonality (not 7 kHz) | 0.58 [0.47, 0.69], p = 0.06 | **0.65 [0.56, 0.73]**, p = 0.002 | 0.67 [0.60, 0.75] | 0.68 [0.57, 0.77] | +0.07 [+0.01, +0.13] | +0.09 [−0.04, +0.22] |
| Svanström → UaVirBASE | 3.5–3.75 kHz share (not 7 kHz) | 0.67 [0.45, 0.86], p = 0.11 | **0.85 [0.81, 0.88]**, p = 0.0005 | 0.81 [0.66, 0.89] | 0.87 [0.81, 0.92] | +0.18 [+0.01, +0.39] | +0.14 [+0.01, +0.29] |

The channel-averaged results were rule 0.84 / 0.89 and detector 0.68 / 0.85.

**Decisions under the committed rules:**

- **A0 does not survive without channel averaging.** On one microphone, neither training chain picks the 7 kHz band,
  and the chosen rule's interval reaches 0.5 in both directions. The 7 kHz cue was produced by averaging the
  channels. A0 is withdrawn (W6).
- **The detector survives:** 0.65 / 0.85 on one microphone against 0.68 / 0.85 averaged. It now beats the
  training-chosen single feature in both directions.
- **MFCC adds beyond the rule in Svanström → UaVirBASE only.** MFCC performs like the 41-feature detector, not
  better.

**Secondary: features computed per channel, then averaged.**

| Train → test | Rule | Detector | MFCC gbm | MFCC logistic |
| --- | --- | --- | --- | --- |
| UaVirBASE → Svanström | 0.60 [0.49, 0.71] | 0.61 | 0.66 | **0.85 [0.78, 0.90]** |
| Svanström → UaVirBASE | 0.89 [0.76, 0.97] (7.0–7.25 kHz) | 0.87 | 0.89 | **0.97 [0.93, 1.00]** |

MFCC with logistic regression on per-channel-averaged features gives the highest acoustic numbers so far. It is a
**HINT**: a secondary variant, on development data, after many looks at these two sets. It needs an untouched third
recording setup.

**Per recording:** window scores within a recording are highly correlated (ICC 0.11–0.88; 0.69–0.88 for the rule), so
averaging windows changes little. Per recording, the detector and MFCC gbm score 0.64 / 0.71 (90 Svanström
recordings) and 0.88 / 0.88 (132 UaVirBASE recordings). More recordings are the cure, not more windows.

### A-R. How the cues fade with range (DDL, drones only, channel 0)

Source: `scripts/check_acoustic_single_mic_mfcc.py` part 4 → `results/checks/ddl_cues_vs_range.json` and `.png`.

| Drone, day | Windows | Range | 7 kHz share vs range (Spearman) | Harmonic prominence vs range |
| --- | --- | --- | --- | --- |
| DJI Mini 2, 29 Mar | 300 | 22–149 m | −0.48 (p ≈ 1e−18) | +0.14 (p = 0.015) |
| DJI Mini 2, 31 Mar | 319 | 2–150 m | −0.52 (p ≈ 2e−23) | −0.36 (p ≈ 3e−11) |
| DJI Phantom 4 Pro, 31 Mar | 438 | 12–249 m | −0.63 (p ≈ 1e−50) | −0.83 (p ≈ 9e−111) |

- **The 7 kHz share falls with range for every drone and day,** as air absorption predicts.
- **The rotor-harmonic prominence also falls,** strongly for the Phantom and inconsistently for the Mini 2. The
  expectation that low-frequency harmonics survive distance better is not supported by this measure.
- Descriptive only: DDL has no negatives.

### A0. One band near 7 kHz, chosen on the training chain: WITHDRAWN as a cue (see A-SM)

Kept for the record. It was the best acoustic result on channel-averaged windows. On a single microphone it does not
hold (A-SM). Source:
`scripts/check_acoustic_single_feature_transfer.py` (design committed in `87f8417` before the run) →
`results/checks/acoustic_single_feature_transfer.json`.

**The rule.** Pick the single feature with the largest training-chain AUC among the 41 detector features and
loudness, then apply it unchanged to the other chain. Both training chains pick the share of window energy near
7 kHz:

| Train → test | Rule chosen on training | Test AUC [95% CI] | p | Detector on the same rows | Detector − rule |
| --- | --- | --- | --- | --- | --- |
| UaVirBASE → Svanström | 6.75–7.0 kHz share (training AUC 0.89) | **0.84 [0.76, 0.90]** | 0.0005 | 0.68 | −0.16 [−0.27, −0.05] |
| Svanström → UaVirBASE | 7.0–7.25 kHz share (training AUC 0.85) | **0.89 [0.82, 0.97]** | 0.001 | 0.85 | −0.04 [−0.23, +0.15] |

- **Gate 1, every non-drone part holds:**
  - Svanström: background 0.80 [0.70, 0.88], helicopter 0.88 [0.81, 0.94].
  - UaVirBASE ambients: 09:31 0.82, 10:39 0.86, 11:17 0.88, 12:47 1.00 (all intervals above 0.78).
- **Gate 2:** the rule *is* a single feature. Chosen without hindsight, it scores 0.84 / 0.89, against 0.85 / 0.89
  for the best feature picked on the test set. In-test loudness is inverted (0.43 / 0.42).
- **Gate 3, what else?**
  - Both chains point the same way, and both test classes come from one chain, so the chain itself can't produce
    this.
  - Untested alternative: energy near 7 kHz falls fastest with distance, so the rule may reflect how close the
    drone was to the microphone. No distance labels exist to check it.
- **Gate 4:** as A1 (1 drone model in UaVirBASE, 30 drone clips in Svanström, 2 chains).
- **Decisions under the committed rules:**
  - The rule transfers.
  - **The gbm detector adds nothing beyond it.**
  - **Thresholds do not transfer**, for the detector or the rule. A threshold set at 5% false positives on the
    training chain gives:

    | Train → test | Threshold from | Window false positives | Window recall |
    | --- | --- | --- | --- |
    | UaVirBASE → Svanström | Detector | 21% | 42% |
    | UaVirBASE → Svanström | Rule | 88% | 100% |
    | Svanström → UaVirBASE | Detector | 0% | 2% |
    | Svanström → UaVirBASE | Rule | 0% | 0% |

    The score scale shifts between chains, as it does for RF. Ranking transfers; absolute levels need calibration
    on site.
- **False alarms per hour (indicative, first ≤ 10 s of each recording):**
  - Detector on Svanström: 132/h (95% upper bound 188/h).
  - On UaVirBASE: 0 in 40 s, which still allows up to 270/h.
  - Not a deployment figure.

### A1. Detection on an unseen recording chain (gbm detector): QUOTABLE as a ranking

A detector trained on one chain ranks drones above non-drones on the other chain. It holds on a single microphone
(0.65 / 0.85, A-SM), where it beats the training-chosen single feature. The gate breakdowns below were computed on
channel-averaged windows.

| Train → test | AUC [95% CI] | p | Rests on |
| --- | --- | --- | --- |
| Svanström → UaVirBASE | 0.85 [0.74, 0.98] | 0.001 | 1 drone model (DJI Mavic 3 Cine, 128 recordings); 4 ambient recordings, one morning, one site |
| UaVirBASE → Svanström | 0.68 [0.59, 0.76] | 0.001 | 30 drone clips; 30 background + 30 helicopter clips |

Sources: `detectors.md`, and `scripts/check_acoustic_gate.py` → `results/checks/acoustic_gate.json`. The rerun
reproduced both AUCs exactly.

**1. Breakdown by non-drone type.** Every part holds (interval above 0.5):

| Test set | Non-drone part | AUC [95% CI] |
| --- | --- | --- |
| Svanström | Background | 0.65 [0.54, 0.75] |
| Svanström | Helicopter | 0.70 [0.61, 0.79] |
| UaVirBASE | Ambient 09:31 | 0.98 [0.97, 0.99] |
| UaVirBASE | Ambient 10:39 | 0.93 [0.91, 0.95] |
| UaVirBASE | Ambient 11:17 | 0.74 [0.70, 0.78] |
| UaVirBASE | Ambient 12:47 | 0.77 [0.73, 0.80] |

UaVirBASE has one drone model, so it can't be broken down by drone type. Each UaVirBASE interval resamples drone
recordings only, because each row holds one ambient recording.

**2. Single-feature baselines inside each test set:**

| Test set | Loudness | Detector − loudness | Best single feature | Detector − best |
| --- | --- | --- | --- | --- |
| Svanström | 0.43 [0.31, 0.55] (drones are not louder) | +0.25 [0.13, 0.36] | Energy share in 7.0–7.25 kHz, 0.85 | **−0.17 [−0.28, −0.06]** |
| UaVirBASE | 0.42 [0.22, 0.65] (drones are not louder) | +0.44 [0.31, 0.55] | Energy share in 6.75–7.0 kHz, 0.89 | −0.04 [−0.23, +0.15] |

So a single 250 Hz band near 7 kHz, chosen with hindsight, does as well as the detector on UaVirBASE and better on
Svanström. The direction is the same in both chains: more energy near 7 kHz means drone. That this band would hold
if chosen *without* hindsight (on the training chain) is untested; it needs its own committed design.

**3. What else could produce this?** Does one non-drone class or recording carry the result? No: every part in the
table above holds.

**4. What it rests on:** the counts in the first table.

**Limits:**

- Only two chains.
- It's a ranking only. No threshold has been tested across chains.
- The detector is not shown to add anything over one band near 7 kHz.
- UaVirBASE ambient recordings later in the morning are harder (0.74–0.77) than the early ones (0.93–0.98).
- A deployed microphone is a third chain, and chains are highly identifiable (dataset identification 0.81 balanced
  accuracy on drone windows alone, `shortcut_baselines.md`).

### A2. Detection within one chain: reference only

| Test | AUC [95% CI] | Rests on |
| --- | --- | --- |
| Svanström | 0.94 [0.90, 0.97] | 59 non-drone groups, 30 drone groups |
| UaVirBASE | 0.90 [0.81, 1.00] | 4 ambient recordings; development set, no significance claims |

Session effects within a chain are not excluded.

### A3. Widening acoustic training data: NOT ESTABLISHED

Adding ESC-50 negatives inverted UaVirBASE → Svanström (0.68 → 0.35). Leave-one-dataset-out showed no gain.

Variants B and D trained on identical sources, so three variants were tested, not four (`acoustic_widening.md`).

### A4. Acoustic one-class pool: NOT ESTABLISHED

- UaVirBASE: 0.68 / 0.71, intervals reach 0.46.
- Svanström: 0.54 / 0.43.

## RF detection

Source for power and best-single-feature numbers: `scripts/check_power_inside_tests.py` →
`results/checks/power_inside_tests.json`. Every detector was refitted and reproduced its saved AUC exactly.

### R1. Across receivers (UAVSig ↔ CardRF): NOT ESTABLISHED

**T2, CardRF → UAVSig (2 negative groups):**

| Version | Detector | Power inside UAVSig | Detector − power |
| --- | --- | --- | --- |
| v1.1, 25 MHz | 0.73 | 0.95 | −0.23 [−0.25, −0.19] |
| v1.2, 40 MHz | 0.75 | 0.96 | −0.21 [−0.24, −0.18] |

**T1, UAVSig → CardRF:**

| Version | Detector | Power inside CardRF | Best single feature |
| --- | --- | --- | --- |
| v1.1 | 0.38 | 0.61 | 0.82 |
| v1.2 | 0.32 | 0.61 | 0.87 |

The "power baseline" column in `rf_cross*.md` reads 0.50 because it was trained on the other receiver's scale.
Measured inside the test sets, power beats both directions.

### R2. CardRF → Noisy RF (stage 2): NOT ESTABLISHED; the cause of the floor is unresolved

| Model | AUC [95% CI] | Status |
| --- | --- | --- |
| gbm (primary) | 0.44 | Fails |
| CNN | 0.64 [0.55, 0.71] | Not a detection result |

The CNN by SNR: 0.60 at −20 dB, 0.54 at −4 dB, 0.74 at ≥ 20 dB.

The floor check (`scripts/check_noisy_rf_floor.py` → `results/checks/noisy_rf_floor.json`):

- **The pre-committed rule's outcome is "inconclusive".** A gbm trained inside Noisy RF on the quiet half of
  each tile reaches 0.57 at −20 dB. The rule was: ≥ 0.60 means the backgrounds explain the floor and Noisy RF is
  retired; < 0.55 means not explained by these features. **So Noisy RF is not retired.**
- **The CNN's score moves with the background statistics (descriptive).** Spearman correlation with the
  background-only score is 0.55 within −20 dB drone vectors and 0.33 within noise vectors, where there is no drone.
- **The background-only model is drone-free only at low SNR.** It rises to 0.80 at high SNR, because a strong
  drone enters the quiet half of the tile.
- **Power inside Noisy RF:** 0.51 at −20 dB, then 0.38–0.46 (drone tiles are slightly quieter).
- **Reproduction:** the CNN retrained to within 0.015 of the saved AUC at every SNR (GPU training is not
  bit-identical), 0.63 overall. The gbm reproduced exactly.
- **Rests on:** all non-drones come from one building, and the interval resamples the 6 RC transmitters only.

### R3. Merged drone-only pool → CardRF

Sources: `scripts/check_claim_breakdowns.py` and `scripts/check_power_inside_tests.py`.

| Score | All | vs Wi-Fi (2 routers) | vs Bluetooth (5 devices) | − power (0.61) | − best single (0.78) |
| --- | --- | --- | --- | --- | --- |
| kNN (declared primary) | 0.78 | 0.53 | 0.88 | +0.17 [−0.20, +0.47] | −0.00 [−0.16, +0.08] |
| GMM (declared secondary) | 0.86 | 0.79 | 0.89 | +0.25 [−0.04, +0.50] | +0.08 [−0.05, +0.23] |

- **kNN: NOT ESTABLISHED.** It separates drones from Bluetooth only and equals the best single feature.
- **GMM: HINT.** It doesn't beat power or the best single feature, the Wi-Fi side is 2 routers, and CardRF is
  development data.
- **Rests on:** 6 UAS systems (aircraft and controller), 2 Wi-Fi routers, 5 Bluetooth devices.
- **The same pool on other test sets is inverted:** UAVSig 0.19 / 0.44 and Noisy RF 0.33 / 0.37.
  - Two explanations fit: "task mismatch", and "the pool learned wide, continuous signals".
  - Neither has been tested.

### R4. Within one receiver: NOT ESTABLISHED beyond a single feature

| Test | Detector | Power | Detector − power | Best single feature | Detector − best |
| --- | --- | --- | --- | --- | --- |
| CardRF, v1.1 | 0.76 | 0.61 | +0.15 [−0.22, +0.38] | 0.82 | −0.06 [−0.33, +0.18] |
| CardRF, v1.2 | 0.82 | 0.61 | +0.21 [−0.21, +0.49] | 0.87 | −0.05 [−0.23, +0.04] |
| UAVSig, v1.1 | 0.81 | 0.95 | −0.14 [−0.17, −0.11] | | |
| UAVSig, v1.2 | 0.81 | 0.96 | −0.15 [−0.17, −0.12] | | |

The CardRF best single features are frequency-centroid movement within the tile: `centroid_jump_mean` (v1.1) and
`centroid_spread` (v1.2).

UAVSig asks "drone or empty band?", and power answers it at 0.95–0.96. Stage 1 (activity) is therefore energy
detection with a noise-floor estimate, not a trained model.

### R5. Fixed-threshold operating point: reference only

On CardRF, out of fold, the threshold catches 69.5% of drone tiles while flagging 14.5% of Wi-Fi/Bluetooth tiles.

That is development data, per 250 µs tile. No false-alarms-per-hour figure exists.

## RF identification (DRFF-R2 Dataset 3)

Source: `drff_identification.md`, plus `scripts/check_drff_bandwidth_and_test_d.py` →
`results/checks/drff_bandwidth_and_test_d.json`. The setup:

- 8 models, chance 0.125; gbm on all features.
- p comes from permuting model labels between units, and intervals resample units.
- All 26 units appear on day 1 (receiver u2).
- I1–I3 reproduced exactly.

"Bandwidth alone" is a gbm on occupied bandwidth only, on the same splits; DJI links switch between 10, 20 and
40 MHz.

### I-D. An unseen unit on an unseen day, same receiver: QUOTABLE, with limits

Each unit is held out of all training, the model trains on day 1, and the held-out unit is tested on days 2–3 at
receiver u2. The design and rule were committed in `e68d5dc` before the run.

| Measure | Value |
| --- | --- |
| Balanced accuracy [95% CI] | **0.44 [0.38, 0.61]** |
| Permutation null | median 0.21, 95th percentile 0.33 |
| p | 0.0015 |
| Rule (interval above the null's 95th percentile, p < 0.05) | **Met** |

- **Gate 1, per model:**

  | Model | Recall | Training units | Held-out units |
  | --- | --- | --- | --- |
  | mavicAir2 | 0.65 | 7 | 3 |
  | mavicAir2s | 0.59 | 6 | 3 |
  | mini4PRO | 0.43 | 4 | 2 |
  | mini3pro | 0.07 | 1 | 1 |

  It works where the model has seen several units, and fails with one.
- **Gate 2:** best single feature (amplitude kurtosis) 0.25; full − best +0.19 [+0.12, +0.37].
- **Gate 3:** bandwidth alone 0.18; full − bandwidth +0.26 [+0.18, +0.44]. This is not the link's bandwidth setting.
- **Gate 4:** 9 held-out units of 4 models, one receiver, one site.
- **Limits:**
  - Same receiver as training.
  - Only models with ≥ 2 units can be tested this way, and DJI models only.
  - Mavic 3, 3C and 3S (one unit each) are not covered.

### I-SNR. The SNR audit of I-D: it survives, and behaviour features do better at one receiver

Source: `scripts/check_drff_snr_audit.py` (design committed in `ffbb935` before the run) →
`results/checks/drff_snr_audit.json`. Motivation: on synthetic signals, the tile features change with SNR alone (the
shape quantiles move about 1 unit per 10 dB), while the emitter behaviour features did not move from 10 to 30 dB.

| Test | Tiles | Tiles, SNR-equalised | Emitter behaviour | SNR alone (tiles / emitters) |
| --- | --- | --- | --- | --- |
| D: unseen unit + unseen day (null 95th 0.33 / 0.29 / 0.39) | 0.44 [0.38, 0.61] | 0.36 [0.30, 0.51] | **0.62 [0.55, 0.83]** | 0.15 / 0.16 |
| D, scored per capture | 0.56 | 0.55 | **0.69** | |
| I3: unseen unit, same day | 0.51 | 0.41 | **0.61** (per capture **0.82**) | 0.20 / 0.18 |
| I2: same drones, other receiver | 0.24 | 0.19 (fails) | **0.125 = chance** | 0.20 / 0.13 |
| D-rx: unseen unit + day + receiver | 0.37 (fails) | 0.33 (fails) | 0.25 (fails) | |
| Receiver predictable from the features (AUC; same drones, day 2) | 0.65 [0.55, 0.75] | 0.75 | **1.00** | |

**Decisions under the committed rules:**

- **I-D survives the SNR audit.** SNR alone never meets the rule; it is at or below the null everywhere. The
  equalisation was severe: 90% of tiles were noised down to −2.3 dB, the committed 10th-percentile target, against
  a median of +21 dB. Even so, the tiles still meet the D rule.
- **The emitter behaviour features re-establish I-D, and do better: 0.62 against 0.44, and 0.69 per capture.**
  Per model (learning curve below), recall rises with training units.
- **Behaviour features do not transfer across receivers at all:**
  - I2 and D-rx are at chance, and the receiver is perfectly predictable from them (AUC 1.00).
  - The cause is measured: both receivers ran at 100 MS/s and 5.745 GHz, yet the dominant video link looks
    continuous on u1 (duty 1.0, 100 ms bursts) and like ~5 ms bursts at 42% duty on u2.
  - Burst segmentation depends on the receiver's dynamic range and placement.
  - The fix to test next: segment bursts relative to each emitter's own peak, not the noise floor.
- **Scoring per capture helps:** +0.07 to +0.21. The within-capture correlation of scores is moderate (ICC
  0.2–0.6), so tiles within a capture carry partly independent evidence.

**Learning curve** (D test, recall of held-out units with only k training units of their own model; 15 draws each):

| k training units | 1 | 2 | 4 | 6 |
| --- | --- | --- | --- | --- |
| mavicAir2, tiles | 0.33 | 0.45 | 0.59 | 0.61 |
| mavicAir2, emitters | 0.32 | 0.50 | 0.77 | **0.85** |
| mavicAir2s, tiles | 0.29 | 0.46 | 0.56 | 0.59 |
| mavicAir2s, emitters | 0.12 | 0.37 | 0.61 | **0.73** |

More physical units per model is the lever. Behaviour features keep improving through 6 units; tiles flatten after
about 4. This is a forecast for next semester's data collection: aim for ≥ 6 units per model.

**Updated I-D statement:** at one receiver, a model identifies the drone model of a physical unit and day it never
saw: 0.62 balanced accuracy with behaviour features (0.69 per capture), not explained by SNR or bandwidth. It does
not carry over to another receiver.

### I-other. The remaining identification tests

| Test | What it tests | Balanced accuracy | Full − bandwidth | Status |
| --- | --- | --- | --- | --- |
| I1a | Same drones, day 2 | 0.29 [0.24, 0.45] | +0.06 [−0.02, +0.26] | **Not beyond bandwidth** |
| I1b | Same drones, day 3 | 0.37 [0.34, 0.48] | +0.18 [+0.13, +0.37] | Beyond bandwidth; same drones, so not a generalisation claim |
| I2 | Same 8 drones, other receiver | 0.24 [0.19, 0.31] | +0.03 [−0.06, +0.14] | **Not beyond bandwidth.** A receiver change leaves only what bandwidth alone gives |
| I3 | Unseen unit, same day | 0.51 [0.45, 0.61] | +0.28 [+0.18, +0.46] | Superseded by I-D, which also changes the day |
| D-rx | Unseen unit + day + receiver (d2 u1) | 0.37 [0.12, 0.61], p = 0.17, 4 units | | NOT ESTABLISHED (declared descriptive) |

The CNN (secondary) is weaker, at 0.21–0.29.

### I-RFUAV and I-RMA. Identification within RFUAV and RMA

Source: `scripts/check_rfuav_rma_identification.py` (design committed in `9541847` before the run) →
`results/checks/rfuav_rma_identification.json`. Two representations, both gbm:

- **Tiles:** snapshot features of 250 µs steered 40 MHz tiles (as DRFF-R2).
- **Emitters:** behaviour of the 3 dominant emitters over 100 ms (`rf_emitters`).

**RFUAV: an unseen recording pack.** Each of the 28 packs of the 12 classes with ≥ 2 packs is held out in turn;
predictions are over all 37 classes.

| Representation | Balanced accuracy [95% CI] | Null median / 95th | Metadata alone (same rows) | Full − metadata | Bandwidth alone | Best single feature |
| --- | --- | --- | --- | --- | --- | --- |
| Tiles | 0.22 [0.17, 0.28] | 0.05 / 0.10 | 0.52 | **−0.30 [−0.45, −0.16]** | 0.10 | 0.18 |
| Emitters | **0.50 [0.41, 0.61]** | 0.07 / 0.15 | 0.37 | +0.13 [−0.03, +0.33] | 0.26 | 0.59 (dominant emitter's burst length) |

- **Both meet "works"** (interval above the null's 95th percentile, p = 0.0005), and both are beyond bandwidth.
- **Neither is shown beyond acquisition metadata.** Centre frequency, reference SNR and ScaleFactor alone identify
  the class on unseen packs, because packs of one class were recorded with the same settings. **Status: NOT
  ESTABLISHED beyond the metadata shortcut.**
- **Behaviour beats snapshots on the same 28 packs: 0.50 vs 0.22.** The rows differ, so this is not a paired
  comparison. A single behaviour feature, burst length (picked on the test rows), reaches 0.59.
- **Per class (emitters):**
  - Strong: FPV Combo 1.00, Mini 4 Pro 0.83, Herelink 0.83, SIYI MK32 0.83, Avata 2 0.73.
  - Weak or failing: Yunzhuo H12 0.00, H30 0.00, H16 0.25.

**RMA:** one session per system and link type.

| Test | Tiles | Emitters | Status |
| --- | --- | --- | --- |
| RMA-a: within session, 5-fold over files (session = class) | 0.66 | 0.99 | Upper bound only. This is the protocol most published results use |
| RMA-b: unseen Mavic archive, share predicted "Mavic" (RC1 / RC2 / Vid1 / Vid2) | 0.07 / 0.13 / 0.75 / 0.88 | 1.00 / 1.00 / 1.00 / 1.00 | Descriptive only |

**RMA-b is confounded with recording date.** Both Mavic RC archives were recorded on 2 Nov 2022 and both Vid archives
on 30 Oct 2022, while every other system was recorded in April–May 2022. A held-out Mavic archive therefore has a
same-day sibling in training, and the model may be recognising the late-2022 recording conditions.

## Withdrawn

| ID | Claim | Why |
| --- | --- | --- |
| W1 | **RF recall on unseen drones, 83–95%** (RFUAV 2.4 and 5.8 GHz, RMA, DRFF-R2). Advertised in commit `0e188c0` and `merged_one_class.md` | The same CNN threshold flags non-drones from other receivers just as often (Noisy RF noise 99.9%, UAVSig no-transmitter 100%, DRFF-R2 environment 100% / 97.5%; `scripts/check_recall_specificity.py`). It measured the change of receiver |
| W2 | "AUC is unaffected by a receiver change because a shift raises both classes equally" | An assumption. It holds only if the change moves both classes' scores equally, which has to be measured each time |
| W3 | "Site calibration" as a performance property | Calibration on local background places a threshold. It does not improve how well drones separate from background. It remains a deployment requirement, because thresholds don't transfer between receivers |
| W4 | "CardRF session leakage confirmed" (2026-10-07) | An artefact of fold-averaged AUC with test-fold model selection |
| W6 | **"One band near 7 kHz is the acoustic cue and beats the detector"** (A0, 2026-10-08) | On a single microphone the training chains no longer pick 7 kHz and the rule fails (0.58 / 0.67, intervals reaching 0.5). The cue was produced by averaging spaced or directional microphones (A-SM) |
| W5 | "Beats the power shortcut" / "beats loudness" margins in `rf_cross*.md` and `detectors.md` | Those baselines were trained on the other receiver or chain. Inside the test sets, power beats every RF detection result (R1, R4), and loudness is inverted (A1) |

## Corrections to reports and commit messages

- `rf_cross.md` said 20 MHz tiles. They were 25 MHz at 25 MS/s (corrected in the report).
- `rf_cross_v1.2.md` named the v1.1 results folder (corrected in the generator and the report).
- The docstrings of `run_rf_stage2.py`, `run_merged_one_class.py` and `run_drff_identification.py` dated their
  designs 2026-10-09. They were written on 2026-10-08 (corrected).
- Commit messages superseded by this log:
  - `0e188c0` "RF works on CardRF 0.78-0.86": see R3. "RF recall across datasets and bands": see W1.
  - `a33a2f9` "GPU CNN transfers modestly (0.64, 0.74 at high SNR)": see R2.
- `check_acoustic_single_feature_transfer.py`: two decisions were written under one key, so the threshold verdict
  overwrote the rule's AUC verdict (fixed by renaming the key). The per-class breakdown of the rule was added after
  the first run. Both changes are reporting only; every number reproduced.
- `check_drff_bandwidth_and_test_d.py` gained a per-model recall output after its first run. It is reporting only:
  the rerun reproduced every number.

## Known pipeline issues

- **The DC notch adds a receiver fingerprint.** Complex I/Q tiles get a 200 kHz zeroed gap at the receiver's centre
  frequency (`rf_common.py`). CardRF tiles (real input) never have it, and Noisy RF tiles always have it in the
  middle. The lowest spectral-shape quantile can see it.
- **No time integration yet.** RF numbers are per 250 µs tile (4,000 decisions per second), and acoustic numbers
  are per 1 s window. Nothing integrates over time or reports false alarms per hour.

## Open questions (each needs its own committed design before any run)

1. **Acoustic.** A-SM settled the 7 kHz question (an averaging artifact) and showed MFCC performs like the detector.
   Still open:
   - an untouched third recording setup with both classes, to confirm A1 and test the MFCC hint;
   - false alarms per hour over the full ambient recordings;
   - pretrained audio embeddings, needing a licence check and a download.
2. **RF identification:** test D at the other receiver with enough units. Dataset 3 has only 4 such units at u1.
   DroneRFb-DIR (3 units per model; Air 2S and Mini 4 Pro shared with DRFF-R2) could supply a second receiver.
3. **RF presence across receivers** can't be tested with the data we hold: every set we hold defines "drone vs not"
   differently, and power dominates within each set. DroneRFb-DIR may change that, if its background proves to be
   from the same receiver and site (`docs/datasets/candidates.md`). Otherwise it needs our own matched recordings of
   drones and background through one receiver.
