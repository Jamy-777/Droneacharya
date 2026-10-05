# CardRF Reconnaissance Notes

Status: raw archive `CARDRF.zip` (66.3 GB) downloading. Everything about raw files below is from the authors' description (`CardRF.pdf`) until the raw audit is run.

## Sources used, in order of trust

1. Local processed data — `Processed_CardRF.zip`, all 22 CSVs streamed without extraction (2026-10-04).
2. Authors' dataset description — `CardRF.pdf` (4 pages, on disk).
3. Authors' code — `Code.zip`: `Telemetry_Resampling.m`, `Control_Resampling.m`, `Resampling_Dividers.mlx`, `SIGNAL_PLOT.mlx`.
4. Author repository `github.com/medosh09/Cardinal-RF` — access pointer only, no code.
5. Derivative papers — not relied on for any value in the card.

## What CardRF is

Outdoor 2.4 GHz captures at the AERPAW Lake Wheeler site (Raleigh, NC), August 2020.

- 6 UAS: DJI Phantom 4, Inspire, Matrice 600, Mavic Pro 1, Beebeerun FPV mini quadcopter, 3DR Iris (FlySky FS-TH9x transmitter).
- 5 Bluetooth devices: iPhone 6S, iPhone 7, iPad 3, FitBit Charge3, Motorola E5 Cruise.
- 2 Wi-Fi routers: Cisco Linksys E3200, TP-Link TL-WR940N.
- LOS at 8–12 m for all categories; NLOS (behind a building) for 3 UAVs only.

## Raw file format (author documentation, one example file)

- `.mat` with structs `Channel_1` and `Frame`.
- `Channel_1.Data`: `5000000×1 int16` ADC codes.
- `XInc = 5.0e-11 s` → 20 GSa/s; 5M samples → 250 µs.
- `XOrg = -1.25e-4 s` → trigger at the capture midpoint.
- Volts = `Data × YInc + YOrg`; example `YInc = 6.5841e-06`, `YOrg = 0.0066`, `YDispRange = 0.4 V`.
- Conflict: text says scale factor `6.581e-06`; struct says `6.5841e-06`.
- Real-valued direct RF sampling: no I/Q, no down-conversion, no centre frequency.

## Directory/label structure (author documentation)

```text
CardRF/
  LOS/Train/{WiFi, Bluetooth, UAV, UAV Controller}
  LOS/Test/{WiFi, Bluetooth, UAV, UAV Controller}
  NLOS/UAV/
```

LOS UAV flight modes: Phantom {Flying, Hovering}, Matrice {Flying}, Inspire {Flying, Videoing}, Mavic Pro {Hovering, Flying}, Beebeerun {Flying}.

## Processed artifact — empirical findings

- Classes: 5 UAV + 6 UAV-controller. **No Wi-Fi, no Bluetooth, no NLOS.**
- Rows: 1024 int16 codes + labels (1027 columns for UAV, 1026 for controller).
- **100 consecutive slices per capture.** Rows chain (last value of one row equals first value of the next) in runs of exactly 100 in all 22 files. The shipped notebook's `number_slice = 5` is not the production setting.
- 100 × 1024 samples ≈ 5.12 µs of each 250 µs capture, starting at sample 2,500,000 (≈ trigger point). This is most likely the **transient**, although the PDF says "steady".
- **Split is by capture:** 350 train / 150 test captures per class (Beebeerun controller 245 / 105). No capture's rows straddle Train and Test. Same-flight/session independence is not established.
- Flight modes are balanced exactly 50/50 where a device has two modes.

## Biggest risk found: class-correlated ADC saturation

Values pin at the same two codes (−32736, 30720) in every file. Fraction of samples at those rails:

| Class | UAV (train / test) | Controller (train / test) |
| --- | --- | --- |
| Beebeerun | 29.8% / 30.5% | 0.001% / 0.002% |
| DJI Inspire | 16.1% / 17.0% | 7.2% / 7.0% |
| DJI M600 | 29.0% / 29.4% | 8.4% / 8.1% |
| DJI Phantom | 27.0% / 27.4% | 14.2% / 13.5% |
| DJI Mavic Pro | 30.0% / 29.9% | 37.4% / 37.0% |
| 3DR Iris | — | 40.1% / 40.1% |

A model can separate several classes from the clipping fraction alone. The authors' own Fig. 6 shows the Phantom 4 controller flat-topped at about ±0.2 V. Before trusting any CardRF result: measure clipping per class in the raw files, and run a clipping-only baseline classifier.

## Implications for Droneacharya

- CardRF is the only dataset so far with explicit Wi-Fi and Bluetooth hard negatives, and they exist only in the raw archive.
- It has no empty-band class: every capture is triggered by a transmission.
- 250 µs captures cannot support burst timing, hopping or packet-sequence features.
- One physical unit per model: identification here is unit fingerprinting.
- Cross-dataset use needs digital down-conversion to complex baseband first; that derived representation must record its parameters.

## Raw audit plan (when the download finishes)

Inspect without extracting the whole archive: one file each from LOS UAV, LOS controller, LOS Wi-Fi, LOS Bluetooth and NLOS UAV. Then extend the metadata scan to all files (metadata only):

1. `Channel_1` field list and `Frame` contents.
2. `XInc`, `XOrg`, `YInc`, `YOrg`, `YDispRange` per file, summarised per class.
3. Clipping fraction and RMS per file, per class.
4. Capture counts per category, split and device.
5. Where the transient ends after the trigger.

## Open questions

- License.
- Whether vertical scale or trigger settings differ by category.
- Whether Wi-Fi/Bluetooth captures saturate like UAV captures.
- What `system_model.pdf` (176 KB, not yet downloaded) adds.

## Raw audit (2026-10-05)

Archive verified at 71,139,787,221 bytes (the size the server reported). `scripts/inspect_cardrf_raw.py`: inventory of every entry plus 135 captures read in full (3 per folder). Results in `interim/cardrf/inspection/`.

**Inventory.** 9,600 captures: 6,090 LOS train, 2,610 LOS test, 900 NLOS (300 each for Inspire, M600, Phantom). Plus 500 zero-byte `__MACOSX` entries that end in `.mat` but are not captures — adapters must skip them. Wi-Fi: 500 per router. Bluetooth: 500 per device (Motorola 350). Per-class counts are exactly the Processed_CardRF counts divided by 100, so **100 slices per raw capture is confirmed from both ends**.

**Acquisition settings are identical everywhere** (n=135): 20 GSa/s, 5M int16 samples, trigger at the midpoint, YInc 6.5841e-06 V/code, 0.4 V display range, quantization step 4. Instrument: Keysight MSOS604A, one serial. The vertical-scale shortcut is ruled out.

**Clipping is not a UAV marker.** Hard negatives clip too: Wi-Fi 7–18%, Bluetooth 0–22.5%, LOS aircraft 0–41%, controllers 0–42%, NLOS ~0%. Levels differ strongly by device (iPad 3 and iPhone 6S ~0%, Mavic Pro ~37%), so clipping is still a device-identity shortcut.

**Trigger content depends on class.** Burst emitters (Wi-Fi, Bluetooth, controllers, Beebeerun) show noise before the trigger (~1,180 codes RMS) and a ~5 µs turn-on transient after it. DJI aircraft usually transmit continuously, so their pre-trigger half already contains signal (75–100% of captures). Processed_CardRF slices cover 0–5.12 µs after the trigger: the transient for burst classes, steady signal for DJI aircraft.

**Sessions (Frame.Date).** Each device was recorded in one 10–40 minute block, and the official Train and Test captures are interleaved seconds apart inside that block. Wi-Fi was recorded on Aug 25, Bluetooth on Aug 25–26, LOS aircraft on Aug 25–29, controllers on Aug 28–29, and **3DR Iris on Nov 1 2020**. So:

- the official split is same-session (repeatability, not generalization);
- class and session are entangled — hold out whole devices;
- shortcut test: if a classifier can predict the class from pre-trigger noise of burst captures alone, the session is leaking.

**NLOS** captures are ~13 dB weaker than LOS (median post-trigger RMS 5,004 vs ~23,000 codes) and essentially unclipped. LOS→NLOS is partly an amplitude shift; compare with and without power normalisation.
