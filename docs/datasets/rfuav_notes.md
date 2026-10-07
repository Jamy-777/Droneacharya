# RFUAV Reconnaissance Notes

Card: `dataset_cards/rfuav.yaml`. Detail of the original inspection: handover §11.

## Local data

- `raw/rfuav/original`: `DEVENTION DEVO.rar`, `DJI MINI4 PRO.rar` (publisher SHA-256 verified).
- `interim/rfuav/`: `devention_devo/pack1.xml`, `pack1_0-1s.iq`; `dji_mini4_pro/vtsbw10_pack1.xml`, `vtsbw20_pack2.xml`.

## Verified

- Each pack is a sequence of 1-second chunks (`pack1_0-1s.iq` … `pack1_5-6s.iq`), 800,000,000 bytes each = 100M interleaved float32 I/Q at 100 MS/s (n=1 IQ file; XML n=3).
- Per-pack XML: USRP X310, 100 MS/s, 100 MHz IF bandwidth, ScaleFactor 60, Fc 2.440 / 2.450 GHz, ReferenceSNRLevel 15 / 29 / 31 dB.
- The XML root element is `SignalHoundIQFile` while `DeviceType` says USRPX310: the file follows Signal Hound's format, so `ScaleFactor` semantics are not established — do not treat it as gain.
- `SerialNumber` is the source pack serial, not the receiver.
- DJI Mini 4 Pro has one pack per VTSBW condition (10 / 20): pack, serial and condition always change together.

## Conflicting

- Paper: 37 distinct UAVs; repository: 35 drone types and 37 raw clips. Public classes include controller product families (e.g. FUTABA T16IZ, RadioMaster TX16S): treat labels as RF-system classes.

## Open

- Packs (independent recordings) per class — decides whether pack-level splits are possible at all.
- Class-conditioned SNR and Fc distributions across all classes.
- Detection-subset negative construction.

## Public release (Hugging Face `kitofrank/RFUAV`, last modified 2026-07-27; full paginated listing 2026-10-07)

- **37 class archives, 109.2 GB** (all on disk, matched by exact size; SHA-256 from `manifests/rfuav_hf_listing.json`). Local filenames were URL-mangled by the browser; `manifests/rfuav_local_to_official.json` maps each to its official name.
- **`ValidationSet_5Drones`: 5 archives, 162.1 GB** — DJI Mini 4 Pro 38.8, FPV Combo 43.2, Avata 2 13.6, Mavic 3 Pro 35.7, Mini 3 30.9 GB. Not downloaded. Likely separate recordings of classes that are also in the main set — potentially a cross-session test; to be confirmed.
- Image set `ImageSet-AllDrones-MatlabPipeline` (~19,085 files) + model weights: 27.7 GB. Not needed (we build our own representations).
- **Correction:** a note written 2026-10-06 said "18 archives, 69.4 GB, 4-class image set". That listing read only the first page of a paginated API response and was wrong.
- RAR cannot be read in place by Python: extract only the `.iq` chunks we use, with WinRAR's command-line tool.
