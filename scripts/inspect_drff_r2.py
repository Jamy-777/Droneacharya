from pathlib import Path

import h5py
import numpy as np


DATA_ROOT = Path(
    r"D:\Weeeeeeee\DroneacharyaData\raw\drff_r2\original"
)


def decode_uint16_text(dataset):
    values = dataset[()].flatten()
    return "".join(chr(int(value)) for value in values)


def read_scalar(dataset):
    return dataset[()].item()


def inspect_file(file_path):
    with h5py.File(file_path, "r") as mat_file:

        row = {
            "file": file_path.name,
            "samples": mat_file["RF0_I"].shape[0],
            "iq_dtype": str(mat_file["RF0_I"].dtype),
            "fs_hz": read_scalar(mat_file["Fs"]),
            "center_hz": read_scalar(mat_file["CenterFrequence"]),
        }

        # Numeric configuration identifier
        if "C" in mat_file:
            row["C"] = read_scalar(mat_file["C"])

        # MATLAB character arrays stored as uint16 code points
        for field in ["TD", "State", "Height", "U", "D", "V"]:
            if field in mat_file:
                row[field] = decode_uint16_text(mat_file[field])

        # Fields documented by the paper but not necessarily
        # present in every actual MAT file.
        for field in ["Gain", "Distance", "FlightMode"]:
            if field in mat_file:
                dataset = mat_file[field]

                if dataset.dtype == np.uint16:
                    row[field] = decode_uint16_text(dataset)
                elif dataset.size == 1:
                    row[field] = read_scalar(dataset)
                else:
                    row[field] = dataset[()]
        
        window_samples = 1_000_000
        window_times_s = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.2,
        ]

        fs = row["fs_hz"]

        window_statistics = []

        for time_s in window_times_s:
            start = int(time_s * fs)
            end = start + window_samples

            i_window = mat_file["RF0_I"][start:end, 0]
            q_window = mat_file["RF0_Q"][start:end, 0]

            power = i_window**2 + q_window**2
            magnitude = np.sqrt(power)

            window_statistics.append(
                {
                    "time_s": time_s,
                    "magnitude_mean": float(magnitude.mean()),
                    "magnitude_std": float(magnitude.std()),
                    "mean_power": float(power.mean()),
                }
            )

        row["window_statistics"] = window_statistics

        return row

        


files = [
    "mavic3C_1_hover_c1_u1_d2.mat",
]


for filename in files:
    file_path = DATA_ROOT / filename

    print("=" * 80)
    print(filename)

    if not file_path.exists():
        print("MISSING")
        continue

    try:
        metadata = inspect_file(file_path)

        for key, value in metadata.items():
            print(f"{key:20}: {value}")

    except Exception as error:
        print(f"INSPECTION ERROR: {error}")