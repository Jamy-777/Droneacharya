from pathlib import Path
import numpy as np

ROOT = Path(
    r"D:\Weeeeeeee\DroneacharyaData\interim\dronerf\inspection"
)

FILES = [
    "00000H_0.csv",
    "00000L_0.csv",
    "10000H_0.csv",
    "10000L_0.csv",
]

N_WINDOWS = 100


def inspect_file(path):
    # One CSV row containing 10M comma-separated amplitude samples.
    x = np.fromstring(
        path.read_text(),
        sep=",",
        dtype=np.float32,
    )

    # 10M / 100 = 100,000 samples per temporal window.
    windows = x.reshape(N_WINDOWS, -1)

    rms = np.sqrt(np.mean(windows * windows, axis=1))
    maxabs = np.max(np.abs(windows), axis=1)

    return {
        "samples": x.size,
        "window_samples": windows.shape[1],
        "rms_min": float(rms.min()),
        "rms_median": float(np.median(rms)),
        "rms_mean": float(rms.mean()),
        "rms_max": float(rms.max()),
        "rms_p90": float(np.percentile(rms, 90)),
        "maxabs_max": float(maxabs.max()),
        "lowest_window": int(np.argmin(rms)),
        "highest_window": int(np.argmax(rms)),
    }


for filename in FILES:
    r = inspect_file(ROOT / filename)

    print(
        f"{filename:14} | "
        f"RMS min/med/mean/p90/max = "
        f"{r['rms_min']:.3f} / "
        f"{r['rms_median']:.3f} / "
        f"{r['rms_mean']:.3f} / "
        f"{r['rms_p90']:.3f} / "
        f"{r['rms_max']:.3f} | "
        f"peak={r['maxabs_max']:.0f} | "
        f"lowW={r['lowest_window']:02d} "
        f"highW={r['highest_window']:02d}"
    )