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

## Public release as of 2026-10-06 (Hugging Face `kitofrank/RFUAV`, last modified 2026-07-27)

- 18 raw RAR archives, 69.4 GB total (Apache-2.0): DJI FPV COMBO 15.0, DJI AVATA2 11.4, DJI MINI3 6.5, DJI MAVIC3 PRO 5.2, DJI MINI4 PRO 5.0, DAUTEL EVO NANO 3.4, FUTABA T14SG 2.7, Herelink Hx4 2.3, FRSKY X20R 2.1, FRSKY X9DP2019 2.1, FRSKY X14 2.1, FLYSKY EL 18 1.8, FUTABA T18SZ 1.8, FUTABA T16IZ 1.8, FUTABA T10J 1.7, FLYSKY NV 14 1.6, FLYSKY FS I6X 1.6, DEVENTION DEVO 1.5 GB.
- Image set: `ImageSet-AllDrones-MatlabPipeline/train`, 895 files, 4 classes.
- The README still describes ~1.3 TB from 37 UAVs; the public raw release covers 18 classes. Earlier notes (37 classes, ~299 GB) describe a previous state of the repository.
- RAR cannot be read in place by Python: extract only the `.iq` chunks we use, with WinRAR's command-line tool.
