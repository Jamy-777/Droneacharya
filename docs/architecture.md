# Droneacharya architecture: physics and protocols first, learning second

Adopted 2026-10-08, after the public-data phase. The reasons are measured, not assumed (`docs/results/claims.md`):

- receivers are 97–98% identifiable from the signal alone;
- single-class datasets teach the dataset, not the drone;
- acoustic ranking transfers between recording setups, and RF detection across receivers does not;
- inside every RF test set, received power or one simple feature does as well as any learned model;
- one band near 7 kHz beats the trained acoustic detector.

A 250 µs snapshot cannot show what makes a drone a drone. That shows up in behaviour over time: the control uplink
hops in a fixed rhythm, the video downlink is continuous and wide, and the rotors whine. So behaviour comes first,
and models come after it.

## Principles

1. **A detector belongs to its receiver.** The product is "this sensor, characterised on this hardware". Public
   datasets are references for designing features and stress tests, not training data for the deployed detector.
2. **Explain away the known emitters.** Wi-Fi and Bluetooth are published standards, recognised from their own
   numerology. Whatever is structured and unexplained becomes a candidate.
3. **Use cooperative signals where they exist:** Remote ID broadcasts and DJI DroneID.
4. **Judge by operations, not AUC.** A claim reads: "detects type X within R metres with at most N false alarms
   per hour at site type Y", evaluated over events and time. It is written down before the data is collected.
5. **Two evidence channels, RF and acoustic, each labelled with what it measures.** Fusion happens only on paired
   recordings (`claims.md`, "What the system outputs").

## Layers

| # | Layer | Method | State | Validated by |
| --- | --- | --- | --- | --- |
| 1 | RF front end | Receiver-specific: noise floor, gain, DC spur, I/Q balance, all logged per sensor | Partly: DC notch, median floor (`rf_emitters.noise_floor` takes a calibrated floor) | Own receiver |
| 2 | RF activity | CFAR on a Blackman-Harris spectrogram (~200 kHz × 5 µs), no ML. Gives time-frequency blobs with 99% occupied bandwidth | **Built:** `rf_emitters.cfar_mask`, `blobs` | Synthetic: false-alarm rate matches the design (`tests/test_rf_emitters.py`) |
| 3 | Emitter tracking | Blobs of one shape → an emitter with hop set, span, burst length, rhythm (interval and its CV), duty cycle | **Built (v1, shape grouping):** `rf_emitters.emitters` | Synthetic hopper and continuous link. On real controllers: FrSky 11 × 1 MHz channels, 6.7 ms dwell, 8.9 ms period; DJI Mini 2 11 × 1.8 MHz channels, 0.5 ms bursts, 4 ms period (smoke run, not a test) |
| 4a | Remote ID / DroneID decoding | Protocol decode | Planned | Own recordings: public sets have no labelled Remote ID |
| 4b | Known-standard recognisers | Wi-Fi from its short training field (0.8 µs periodicity, 20/40 MHz wide). Bluetooth from its 1 MHz grid and 625 µs slots | **Wi-Fi built:** `rf_emitters.is_wifi`. Bluetooth planned | Positive-only drone sets as negatives (must not fire); CardRF Wi-Fi/BT as positives |
| 4c | Known drone-link signatures | Emitter-level features (layer 3) per link family | Next | RFUAV, RMA, DRFF-R2, UAVSig long captures |
| 4d | ML on emitter features | Classifier over layer-3 features, with an explicit "unknown structured emitter" output | Later | Grouped splits; own receiver for deployment |
| 5 | Acoustic | Energy near 7 kHz (the measured cue) with on-site calibration; rotor-harmonic tracker as a candidate; array for direction | 7 kHz rule measured (A0); harmonic tracker planned | Cross-setup tests; own array |
| 6 | Fusion | Per-sensor likelihood ratios calibrated on site, sequential decisions over time, a false-alarm budget per hour | Planned | Paired RF + audio recordings only |

## Where this plan differs from the proposal that prompted it

1. **The acoustic core is not a harmonic comb, as far as our data shows.** Tonal peakiness features were in the
   detector, and one broadband band near 7 kHz beat it. A harmonic tracker must beat the 7 kHz rule on the same
   tests to earn its place.
2. **Explaining away needs protocol numerology, not modulation family.**
   - DJI's video links are OFDM, so "OFDM" is not "Wi-Fi". The Wi-Fi recogniser keys on 802.11's 0.8 µs short
     training field (`tests/test_rf_emitters.py`: an OFDM burst with LTE-like spacing is rejected).
   - Bluetooth's 1 MHz grid resembles RC hopping: FrSky uses 1 MHz channels too. A Bluetooth recogniser therefore
     has to use slot timing and modulation, and the drone controller captures are its negative test.
3. **"Unknown structured emitter" will fire in cities.** Zigbee, cordless phones, proprietary links and microwave
   ovens are all structured. The explain-away list grows per site, and the false-alarm budget decides when an
   unknown raises an alert.
4. **DroneNoise is level-calibrated (pascals) but has no distance labels in our card.** Detection-vs-distance curves
   need its flight trajectories first.

## Jobs for public data that need no cross-receiver training

- Layers 2–3: hop and frame signatures per link family on the long captures:
  - UAVSig: 1 s at 50 MS/s
  - RFUAV: 1 s chunks at 100 MS/s
  - DRFF-R2: 1.4 s at 100 MS/s
  - RMA: 0.1–1 s at 100 MS/s

  CardRF (250 µs snapshots) cannot show timing.
- Layer 4b: recogniser hit rate on CardRF Wi-Fi/Bluetooth, and false-fire rate on every drone capture.
- Layer 5: the 7 kHz rule's dependence on distance, once a distance source exists.

Each of these is a test, so each gets a committed design before it runs.

## Critical path

Our own recordings, from one SDR plus a microphone array at the same site, following
`docs/data_collection_protocol.md`. They remove the receiver confound by construction, give true negatives from the
same sensor, and are the first paired RF + audio data.
