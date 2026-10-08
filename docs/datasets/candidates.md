# Candidate datasets (not downloaded, not carded)

What we need most is a public RF set with **drones and realistic non-drone traffic recorded through the same
receiver at the same site**. Every set below is judged against that need. Facts come from the primary pages linked;
anything unverified is marked. A dataset gets a card only after its files are inspected.

## DroneRFb-DIR: priority 1 (download; verify the background before trusting it)

- **Source:** Ren, Yu, Zhou, Shi, Chen, *Journal of Electronics & Information Technology* 47(3):573–581, 2025,
  doi:10.11999/JEIT240804. Data on Science Data Bank:
  `scidb.cn/en/detail?dataSetId=84cf9101e739402784b1396783881202`. Login required; licence not stated on the
  pages read.
- **Content (from the paper page):**
  - SDR (model not named), 80 MS/s, 2.4–2.48 GHz, raw I/Q, urban site with Wi-Fi/Bluetooth.
  - 6 DJI models × 3 units each: Mavic 3 Pro, Mini 2 SE, Mini 4 Pro, Mini 3, Air 3, Air 2S.
  - One background class (B), urban.
  - 4,690 segments of over 4 M samples each.
  - Labels for unit and LoS/NLoS, plus a predefined train/test split (rule not stated).
- **The paper's own result (Table 3):** background recognised 100%; per-model individual identification 30–69%.
- **Why it matters:**
  - **Detection:** it is the only set found where a background class with Wi-Fi/BT sits beside drones in one
    acquisition. That gives a test of "drone vs crowded band" with both classes from one receiver, and so the
    cross-receiver test public data has lacked (train elsewhere → test here).
  - **Identification:** 3 units per model allows an unseen-unit test on a second receiver. Air 2S and Mini 4 Pro
    also appear in DRFF-R2, which allows an identification test across receivers for the same models.
- **Must verify from the files before use:**
  - Was the background recorded with the same SDR, site and time window as the drones?
  - How many background segments, and over how many separate recordings?
  - Licence terms.
  - Whether 4 M samples (50 ms at 80 MS/s) is enough per segment.

## DroneDetect (Swinney & Woods, University of Essex): priority 2 (useful, but no negatives)

- **Source:** IEEE DataPort, doi:10.21227/5jjj-1m32. Free IEEE account login; DroneDetect_V2.zip is 65.76 GB.
- **Content:**
  - Nuand BladeRF via GNU Radio, centre 2.4375 GHz, 28 MHz bandwidth, 2 s recordings (1.2 × 10⁸ samples).
  - 7 models: DJI Mavic 2 Air S, Mavic Pro, Mavic Pro 2, Inspire 2, Mavic Mini, Phantom 4, Parrot Disco.
  - States: switched on, hovering, flying.
  - Interference subsets: clean, Bluetooth, Wi-Fi, Bluetooth + Wi-Fi.
- **No drone-free recordings.** The interference subsets are drone signal plus Wi-Fi/BT, not Wi-Fi/BT alone.
  arXiv:2406.18624 states it lacks measurements without drones. A user comment on the DataPort page asks where the
  ambient class is, and has no answer. **It cannot test detection.**
- **What it can test:**
  - Does identification hold when Wi-Fi/BT overlap the drone signal (clean vs interference on one receiver)?
  - Identification across receivers for models shared with CardRF (Inspire, Phantom, Mavic Pro).
  - One unit per model only, so no unseen-unit test.

## DroneRFa (same group as DroneRFb-DIR): lead only

- Reported as 100 MS/s, 2.4 and 5.8 GHz (one source adds 915 MHz).
- 9 drone types outdoors and 15 indoors, plus one background class.
- Sources disagree on which label is the background, so nothing here is verified. Science Data Bank:
  `scidb.cn/en/detail?dataSetId=34f0a91e8a544904998b8fdc44477380`.
