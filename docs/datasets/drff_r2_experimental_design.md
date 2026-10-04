# DRFF-R2 Experimental Design Map

## Purpose

This document defines what each DRFF-R2 subset can legitimately
contribute to Droneacharya and which confounding variables must be
controlled before interpreting model performance.

---

## Dataset 1 — Single Drone States

Primary questions:

- Does detection survive changes in UAV operational state?
- Are learned RF features invariant to altitude and speed?
- Can flight state itself be classified?

Useful variables:

- physical UAV unit
- UAV model
- altitude
- speed
- flight maneuver
- lateral distance
- operational state

Potential confounders:

- physical unit
- distance
- altitude
- speed
- capture configuration

Required controls:

- compare states using the same physical UAV where possible
- avoid mixing unit identity with flight-state identity
- group related captures during train/test splitting

Droneacharya role:

- state robustness
- representation robustness
- secondary flight-state classification

---

## Dataset 2 — Drone Mixed

Primary questions:

- Can UAV RF evidence still be detected when multiple UAVs transmit?
- How does signal overlap affect classification?
- Can representations survive signal superposition?

Potential confounders:

- number of simultaneously active UAVs
- UAV models involved
- relative signal strength
- geometry/distance

Required controls:

- compare single-UAV and mixed-UAV cases using matched UAVs where possible
- do not treat mixed recordings as ordinary single-label identification data

Droneacharya role:

- multi-UAV stress testing
- overlapping-signal robustness
- future signal-separation research

---

## Dataset 3 — Single Drone Hover

Primary questions:

- Does a representation generalize across acquisition days?
- Does it generalize across receivers?
- Does it generalize across physical UAV units of the same model?

Key grouping variables:

- physical UAV unit
- acquisition day
- receiver ID

Required experiments:

A. Same receiver, unseen physical unit
B. Same physical unit, unseen day
C. Same physical unit, unseen receiver
D. Unseen physical unit + unseen day
E. Receiver-crossing experiment

Important:
Change one nuisance variable at a time for diagnosis.
Change multiple variables simultaneously only for stress testing.

Droneacharya role:

- shortcut-learning diagnosis
- cross-unit robustness
- cross-day robustness
- cross-receiver robustness

---

## Dataset 4 — Dual Frequency

Primary questions:

- Does detection survive frequency-hopping behavior?
- Does the model depend excessively on fixed spectral location?
- Can representations capture signal behavior rather than frequency alone?

Potential confounders:

- frequency itself becoming a class identifier
- receiver configuration
- UAV identity

Required controls:

- compare hopping and non-hopping observations of the same UAV where possible
- test frequency-normalized representations
- test whether center frequency predicts model labels

Droneacharya role:

- spectral robustness
- frequency-hopping analysis

---

## Dataset 5 — Absorbent Material

Primary questions:

- How much does propagation/environment affect learned RF features?
- Which features remain when external environmental effects are reduced?
- Does a model trained on clean signals transfer to realistic signals?

Required experiments:

A. Controlled → real environment
B. Real environment → controlled
C. Train on both → held-out environment

Important:
Domain-transfer performance should not be assumed symmetric.

Droneacharya role:

- propagation-domain analysis
- environmental robustness
- controlled signal characterization

---

## Dataset 6 — Wi-Fi Mixed

Primary questions:

- Can UAV RF activity be detected under Wi-Fi interference?
- Which features remain useful under interference?
- How quickly does detection performance degrade?

Potential confounders:

- center frequency
- Wi-Fi strength
- UAV distance
- receiver gain

Required controls:

- compare clean and Wi-Fi-interfered observations
- preserve interference status as metadata
- never treat synthetic and naturally recorded interference as equivalent

Droneacharya role:

- hard-negative/interference robustness
- real-world RF stress testing

---

## Dataset 7 — Environment

Primary questions:

- Can UAV activity be separated from the recorded environmental RF baseline?
- Does the negative class share acquisition conditions with positive recordings?

Required empirical audit:

- sample rate
- center frequency
- gain
- receiver ID
- acquisition day
- location
- antenna configuration

Critical warning:

Dataset 7 represents the DRFF-R2 acquisition environment.

It does NOT represent the complete non-UAV RF universe.

Droneacharya role:

- environmental negative data
- binary detection baseline
- confound auditing

---

# Cross-Subset Experiments

## Receiver Shortcut Test

Determine whether receiver identity predicts UAV/model labels.

If UAV model and receiver are strongly correlated, model-identification
performance may reflect receiver fingerprinting rather than UAV features.

## Day Shortcut Test

Determine whether acquisition day predicts model/unit labels.

## Frequency Shortcut Test

Determine whether center frequency predicts model labels.

## Gain Shortcut Test

Determine whether receiver gain predicts UAV-presence or UAV-model labels.

## Physical-Unit Leakage Test

Ensure windows from the same physical recording/unit do not leak between
training and test partitions when evaluating generalization.

## Dataset-Source Shortcut Test

When DRFF-R2 is eventually combined with other datasets, test whether
dataset source itself predicts the target label.

---

# Claim Ceiling

DRFF-R2 can support:

- controlled cross-unit experiments
- cross-day robustness experiments
- cross-receiver robustness experiments
- flight-state robustness
- Wi-Fi interference evaluation
- frequency-hopping evaluation
- multi-UAV signal-mixture evaluation
- detection against its recorded environmental baseline

DRFF-R2 alone cannot establish:

- manufacturer-independent UAV detection
- universal non-UAV rejection
- custom-UAV generalization
- battlefield detection range
- RF-silent UAV detection
- fiber-optic UAV detection
- universal cross-hardware generalization
