# DRFF-R2 Reconnaissance Notes

Card: `dataset_cards/drff_r2.yaml`. Subset design: `drff_r2_experimental_design.md`. Detail of the original inspection: handover §10.

## Local data

- 5 diagnostic files on D: (`raw/drff_r2/original`), publisher MD5s verified: `mavic3C_1_hover_c1_u1_d2`, `mavic3C_1_hover_c1_u2_d1`, `mavic3C_1_hover_c1_u2_d2` (Dataset-3), `mavic3C_1_c1` (Dataset-6, Wi-Fi mixed), `outdoor_environment` (Dataset-7).
- Full dataset reported on an external drive — **its location and contents are not yet recorded**. Record the path and file list before building the recording index.

## Verified (n = 5 files)

- MAT v7.3 / HDF5; `RF0_I`, `RF0_Q` float32, 140,000,000 samples each; 100 MS/s; 1.4 s.
- Outdoor Fc 5.745 GHz; Wi-Fi subset Fc 2.437 GHz — **the only DRFF data in the 2.4 GHz band shared with the other datasets**.
- Schema drift: Dataset-3/6 use `CenterFrequence`, Dataset-7 uses `CenterFreq`. Dataset-3 metadata fields: `TD, State, C, U, D, Height, V`. Paper fields `Gain, Distance, FlightMode` not found.
- Filenames encode unit, configuration, receiver (`u1/u2`) and day (`d1/d2`).

## Open / conflicting

- Receiver hardware: paper text says USRP-2943, a figure shows X310.
- Gain: documented as stored per file, not found in inspected files.
- Positive vs environment power gap ~30 dB (n=1 vs 1) — cause unresolved; gain could explain it.
- Signal-level comparison across receivers and days not yet done (only filenames confirmed the structure).
