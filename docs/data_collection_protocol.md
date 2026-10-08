# Data collection protocol v0 (draft for the project owner)

Purpose: data in which **the only difference between "drone" and "not drone" is the drone**. Same receiver, same
microphones, same site, same session, gain fixed and logged. Every rule below exists because a public dataset broke
it and the result could not be trusted.

## Hardware

| Item | Requirement | Reason |
| --- | --- | --- |
| SDR | Covers 2.4 and 5.8 GHz. Instantaneous bandwidth ≥ 56 MHz; ≥ 84 MHz to see the whole 2.4 GHz band | We measured a DJI Mini 2 controller hopping across 76 MHz and a FrSky across 64 MHz. A 56 MHz receiver sees only part of the hop set |
| Gain | Fixed and logged per session; never automatic | AGC changes the noise floor and the signal shape mid-recording |
| Clock | GPS-disciplined, or a shared clock with the audio recorder | Pairs RF with audio and with flight logs |
| Microphones | An array (≥ 4) of identical microphones, flat to ≥ 10 kHz, with wind shields | The measured cue sits near 7 kHz; microphones differ most there, and wind hisses there |
| Logging | Drone flight logs (GPS track), a weather station (wind) | Ground truth for range; wind is the main acoustic confounder |

## Session structure (every session, every site)

Interleave the blocks within one session, never in separate sessions. Each block is 5–10 minutes:

1. **Background only.** No drone powered within line of sight; confirmed by a spotter and by Remote ID/DroneID
   scanning.
2. **Known non-drone emitters only.** Phone hotspot and Wi-Fi traffic, a Bluetooth speaker, a cordless device. Log
   which devices were on.
3. **Drone powered on the ground.** Controller and drone linked, not flying.
4. **Drone flying at logged ranges.** For example 50 / 100 / 200 / 400 m, hovering and transiting, with the GPS
   track logged.
5. **Background only again.** This checks that the background did not drift during the session.

Repeat across **at least 3 sites** (urban, suburban, open), **at least 2 days per site**, and as many drone models
and physical units as are available.

## Labels (per time segment, UTC)

`session, site, block, devices_on, drone_model, drone_unit, range_m (from GPS), altitude_m, wind_mps, sdr_gain_db, centre_hz, sample_rate`

## Splits and claims

- **Split unit:** whole sessions, and preferably whole sites or days. Never split within a session.
- **A valid negative:** background or known-emitter blocks from the same sensor, site and session as the positives.
- **Claims follow `docs/results/claims.md`:**
  - an operational form (type, range, false alarms per hour, site type);
  - the design committed before the data is analysed.

## Volume

- RF at 56 MS/s complex int16 is about 0.8 TB per hour. Record 1 s captures every 5 s (20% duty): about 160 GB per
  hour, and enough to see hop rhythms.
- Store all of the background blocks: false alarms per hour needs continuous negatives.
- Audio is negligible: 4 channels × 48 kHz × 24 bit is about 2 GB per hour.

## Minimum first campaign

- One site, two days, two drone models: about 1 hour of RF captures and 2 hours of audio per day.
- That is enough to run the first honest tests:
  - RF presence and false alarms per hour on one receiver;
  - acoustic detection vs range;
  - the first paired fusion test.
