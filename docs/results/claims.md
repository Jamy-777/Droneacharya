# Claims log

Hand-written. This is the only file in `docs/results/` that says which numbers may be quoted and with what limits.
Every other file here is raw generator output and is rewritten on each run.

**State on 2026-10-08: no result has passed the full gate below.** Each entry lists which gate items are missing.

## Rules

**Gate.** A number is quotable only when all four are done and written next to it:

1. Broken down by drone type **and** by non-drone type.
2. Compared with the strongest single-feature baseline computed **inside the test set**.
3. One "what else could produce this number?" check, run.
4. The count of independent devices, recordings or units it rests on.

**Process, from 2026-10-08:**

- A test's design is committed before the test runs. In every script committed up to `0e188c0`, the
  "design fixed before results" docstring was committed together with its results, so git history cannot verify it.
- CardRF is development data. It has been scored more than a dozen times, and tile width and steering were changed
  in response to its results. No CardRF number counts as evidence for a new claim.
- One revision per test set.
- A threshold or operating point is claimed only on the receiver or recording chain it was set on.
- Results text describes. No pitch language.

**Status words:**

| Status | Meaning |
| --- | --- |
| PENDING GATE | The number stands, but some gate items are missing |
| HINT | Not a claim |
| NOT ESTABLISHED | Tested, and didn't hold |
| WITHDRAWN | Was claimed, is wrong |

## Acoustic

### A1. Detection on an unseen recording chain: PENDING GATE

| Train → test | AUC [95% CI] | p | Rests on |
| --- | --- | --- | --- |
| Svanström → UaVirBASE | 0.85 [0.74, 0.98] | 0.001 | 1 drone model (DJI Mavic 3 Cine, 128 recordings); 4 ambient recordings, one day, one site |
| UaVirBASE → Svanström | 0.68 [0.59, 0.76] | 0.001 | 30 drone clips; 30 background + 30 helicopter clips |

Source: `detectors.md`. The test sets were held out whole, and both test classes come from one chain.

- **Missing gate item 1:** the Svanström breakdown into background and helicopter (not saved).
- **Missing gate item 2:** the strongest single feature inside each test set. The loudness comparison in
  `detectors.md` was trained on the other chain and is inverted on the test chain (0.41). So the "+0.26 / +0.45
  over loudness" margins are not quotable; the margins over chance are +0.18 / +0.35.
- **Missing gate item 3:** a check for session effects inside each test chain. UaVirBASE's 4 ambient recordings
  span one morning.
- **Limits:**
  - Only two chains.
  - Only ranking has been tested across chains; no threshold has.
  - A deployed microphone is a third chain, and chains are highly identifiable (dataset identification 0.81
    balanced accuracy on drone windows alone, `shortcut_baselines.md`).

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

### R1. Across receivers (UAVSig ↔ CardRF): NOT ESTABLISHED

| Test | v1.1, 25 MHz tiles | v1.2, 40 MHz tiles | Rests on |
| --- | --- | --- | --- |
| T1, UAVSig → CardRF | 0.38 | 0.32 | |
| T2, CardRF → UAVSig | 0.73 | 0.75 | 2 negative groups |

The "power baseline" column in `rf_cross*.md` scores exactly 0.50 in every row. It was trained on one receiver's
amplitude scale and applied to the other's, so "detector − power" there means detector minus chance. Power
computed inside each test set has not been run.

### R2. CardRF → Noisy RF (stage 2): NOT ESTABLISHED

| Model | AUC [95% CI] | Status |
| --- | --- | --- |
| gbm (primary) | 0.44 | Fails |
| CNN | 0.64 [0.55, 0.71] | Not a detection result |

The CNN by SNR (`rf_stage2.md`): 0.60 at −20 dB, where the drone is 1% of the power; down to 0.54 at −4 dB; up to
0.74 at ≥ 20 dB. Separation where the drone is weakest points to a difference in how the two classes' backgrounds
were built. That cause is untested.

- All non-drones come from one building.
- The interval resamples the 6 RC transmitters only.

### R3. Merged drone-only pool → CardRF

Source: `scripts/check_claim_breakdowns.py` and `results/checks/claim_breakdowns.json`.

| Score | All non-drones | vs Wi-Fi (2 routers) | vs Bluetooth (5 devices) |
| --- | --- | --- | --- |
| kNN (declared primary) | 0.78 | 0.53 | 0.88 |
| GMM (declared secondary) | 0.86 | 0.79 | 0.89 |
| Best single feature (picked on CardRF, optimistic) | 0.78 | 0.47 | 0.91 |
| Occupied bandwidth alone | 0.64 | 0.18 | 0.83 |

- **kNN: NOT ESTABLISHED.** It separates drones from Bluetooth only and adds nothing over one feature.
- **GMM: HINT.** The Wi-Fi side is 2 routers, CardRF is development data, and the GMM was the declared secondary.
- **Rests on:** 6 UAS systems (aircraft and controller), 2 Wi-Fi routers, 5 Bluetooth devices.
- **The same pool on other test sets is inverted:** UAVSig 0.19 / 0.44 and Noisy RF 0.33 / 0.37.
  - Two explanations fit all three test sets: "task mismatch", and "the pool learned wide, continuous signals",
    which accepts Wi-Fi and rejects narrow emitters.
  - Neither has been tested.

### R4. Within one receiver: reference only

| Test | AUC [95% CI] | p | Note |
| --- | --- | --- | --- |
| CardRF, v1.2 | 0.82 [0.52, 0.99] | 0.03 | Development data; tile width was changed after v1.1 results on this set |
| UAVSig | 0.81 | | 2 negative groups |

### R5. Fixed-threshold operating point: reference only

On CardRF, out of fold, the threshold catches 69.5% of drone tiles while flagging 14.5% of Wi-Fi/Bluetooth tiles.

That is development data, per 250 µs tile. No false-alarms-per-hour figure exists.

## RF identification (DRFF-R2 Dataset 3)

### I. Which drone model this is: PENDING GATE

8 models, chance 0.125, gbm primary (`drff_identification.md`). All 26 units appear on day 1 (receiver u2), so every
unit tested on day 2, day 3 or receiver u1 was in day-1 training (`claim_breakdowns.json`).

| Test | What it actually tests | Balanced accuracy [95% CI] |
| --- | --- | --- |
| I1a, I1b | The same drones on a later day (was labelled "unseen day") | 0.29 [0.24, 0.45] / 0.37 [0.34, 0.48] |
| I2 | The same 8 drones, same day, other receiver (was labelled "unseen receiver") | 0.24 [0.19, 0.31] |
| I3 | An unseen unit of a known model, same day | 0.51 [0.45, 0.61] |

- I3 pools 22 held-out units of 4 models. The null median is 0.21, because only 4 classes are true.
- I2 shows that changing only the receiver erases most of the accuracy on drones the model already knows. The
  features largely encode the receiver.
- **Missing gate item 2:** identification from bandwidth alone. DJI links switch between 10, 20 and 40 MHz.
- **Missing gate item 3:** test D (an unseen unit on an unseen day) and the receiver-shortcut tests from
  `docs/datasets/drff_r2_experimental_design.md`, none of which were run.
- **Other observations:**
  - Mavic 3, 3C and 3S are confused with each other, so the model identifies radio family more than airframe.
  - The CNN (secondary) is weaker, at 0.21–0.29.

## Withdrawn

| ID | Claim | Why |
| --- | --- | --- |
| W1 | **RF recall on unseen drones, 83–95%** (RFUAV 2.4 and 5.8 GHz, RMA, DRFF-R2). Advertised in commit `0e188c0` and `merged_one_class.md` | The same CNN threshold flags non-drones from other receivers just as often (Noisy RF noise 99.9%, UAVSig no-transmitter 100%, DRFF-R2 environment 100% / 97.5%; `scripts/check_recall_specificity.py`). It measured the change of receiver |
| W2 | "AUC is unaffected by a receiver change because a shift raises both classes equally" | An assumption. It holds only if the change moves both classes' scores equally, which has to be measured each time |
| W3 | "Site calibration" as a performance property | Calibration on local background places a threshold. It does not improve how well drones separate from background. It remains a deployment requirement, because thresholds don't transfer between receivers |
| W4 | "CardRF session leakage confirmed" (2026-10-07) | An artefact of fold-averaged AUC with test-fold model selection |

## Corrections to reports and commit messages

- `rf_cross.md` said 20 MHz tiles. They were 25 MHz at 25 MS/s (corrected in the report).
- `rf_cross_v1.2.md` named the v1.1 results folder (corrected in the generator and the report).
- The docstrings of `run_rf_stage2.py`, `run_merged_one_class.py` and `run_drff_identification.py` dated their
  designs 2026-10-09. They were written on 2026-10-08 (corrected).
- Commit messages superseded by this log:
  - `0e188c0` "RF works on CardRF 0.78-0.86": see R3. "RF recall across datasets and bands": see W1.
  - `a33a2f9` "GPU CNN transfers modestly (0.64, 0.74 at high SNR)": see R2.

## Known pipeline issues

- **The DC notch adds a receiver fingerprint.** Complex I/Q tiles get a 200 kHz zeroed gap at the receiver's centre
  frequency (`rf_common.py`). CardRF tiles (real input) never have it, and Noisy RF tiles always have it in the
  middle. The lowest spectral-shape quantile can see it.
- **No time integration yet.** RF numbers are per 250 µs tile (4,000 decisions per second), and acoustic numbers
  are per 1 s window. Nothing integrates over time or reports false alarms per hour.

## Open checks

1. Noisy RF: the −20 dB floor, plus a baseline built from background only.
2. Power computed inside each RF test set.
3. DRFF-R2: identification from bandwidth alone, and test D.
4. Acoustic: the Svanström background/helicopter breakdown, and the best single feature inside each test set.
