# UAVSig Reconnaissance Notes

## Why UAVSig matters to Droneacharya

UAVSig is not being considered primarily for UAV model diversity.

Its main contribution is controlled RF fingerprinting across multiple
physical transmitters of the same nominal hardware:

- 4 DJI Matrice 100 UAVs
- 4 DJI C1 controllers

This allows experiments separating:

- model identity
- physical-unit identity
- RF channel
- transmitter position
- single-transmitter vs multi-transmitter conditions

This makes UAVSig a useful dataset for determining whether a learned
representation captures transferable UAV RF structure or merely
individual hardware fingerprints.

## Acquisition

Primary-source reported configuration:

- USRP B205mini-i
- 20 dBi directional panel antenna
- 18 degree beamwidth
- 50 MS/s
- 50 MHz capture bandwidth
- 2.4435 GHz center frequency
- 20 dB receiver gain
- 2.4 GHz ISM band
- 1 second per capture
- six captures per scenario

Raw samples were acquired as 16-bit I/Q and stored as 32-bit floats
by the GNU Radio collection system.

Actual MAT schema and IQ layout remain to be empirically inspected.

## Scenario structure

### One-drone

4 physical UAVs × 4 selected RF channels = 16 scenarios.

Each scenario contains six consecutive one-second captures.

### Two-drone

72 scenarios containing two simultaneously transmitting UAVs.

These are real OTA simultaneous transmissions, not synthetic signal
addition.

### Controllers

4 physical DJI C1 controllers.

All on/off combinations were collected under two physical layouts,
giving 32 reported scenarios.

Controllers frequency-hop across the 2.4 GHz band.

## Labels

UAVSig contains time-frequency transmission labels.

Current UCLA documentation states that the dataset is labeled using
WHIRLS.

Labels include transmission timing and frequency information.

Identity provenance differs by scenario:

- one transmitter: scenario identity is deterministic;
- two drones: known assigned RF channels permit frequency-assisted
  transmitter identity assignment;
- multiple controllers: individual hopping transmissions cannot be
  deterministically assigned, so labels represent the active
  controller set rather than a precise transmitter for each burst.

Therefore label confidence/semantics must be preserved rather than
flattened into a single universal class field.

## Known defect: dropped samples

UCLA explicitly warns that the B205mini collection setup dropped
sample segments.

The posted raw IQ therefore contains random temporal gaps.

Consequences:

- sample index is not guaranteed to represent uninterrupted elapsed time;
- burst spacing and hopping-period measurements across gaps are unsafe;
- long temporal models must account for discontinuities;
- the empirical representation of gaps must be inspected before
  preprocessing.

The UCLA lab states that transmission labels remain valid despite
this defect.

## Controller bandwidth limitation

The receiver observes 50 MHz centered at 2.4435 GHz.

Controller signals frequency-hop over the broader 2.4 GHz ISM region.

Therefore some controller transmissions can be partially or entirely
outside the captured band.

Observed absence of a controller transmission in a particular window
does not necessarily mean the controller was inactive.

## Negative-class limitation

UAVSig does not provide an explicit hard-negative taxonomy comparable
to CardRF.

The collection environment contained uncontrolled 2.4 GHz activity,
but the directional antenna was used to reduce it.

Do not treat incidental ambient RF as a curated Wi-Fi/Bluetooth
negative corpus.

## Published split limitation

The paper trains on five captures and tests on the sixth capture from
the same scenario.

This is useful for reproducing the paper but is not a strong
generalization test.

Droneacharya should preserve scenario, physical transmitter, capture
sequence and day as grouping variables.

## Shortcut risks

High-priority audits:

1. transmitter identity vs assigned RF channel;
2. transmitter identity vs physical placement;
3. train/test capture adjacency;
4. received power vs transmitter;
5. dropped-sample pattern vs recording/scenario;
6. day vs capture type;
7. dataset identity in cross-dataset experiments.

## Provisional Droneacharya role

Primary:

- same-model physical-unit fingerprinting;
- activity-aware/time-frequency label reference.

Secondary:

- real simultaneous multi-UAV robustness;
- 2.4 GHz cross-dataset experiments;
- controller transmission detection.

Not suitable alone for:

- universal UAV detection;
- hard-negative rejection;
- cross-receiver generalization;
- broad cross-environment claims.

## Empirical questions

Before closing UAVSig:

1. What is the MAT schema?
2. How are I and Q stored?
3. What is the exact stored dtype?
4. Are there exactly ~50M complex samples per one-second file?
5. How are dropped-sample gaps represented?
6. Where are WHIRLS labels stored?
7. How are scenario IDs encoded in filenames?
8. How are physical UAV/controller identities encoded?
9. Can RF channel and transmitter identity be independently crossed?
10. How do directory dates map to capture types?
11. What constitutes an independent recording/session?
