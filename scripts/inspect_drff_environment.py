from pathlib import Path

import h5py

FILE_PATH = Path(
    r"D:\Weeeeeeee\DroneacharyaData\raw\drff_r2\original\outdoor_environment.mat"
)

with h5py.File(FILE_PATH, "r") as mat_file:

    print(f"File: {FILE_PATH.name}")
    print(f"Size: {FILE_PATH.stat().st_size:,} bytes")
    print()

    print("Top Level Objects")
    print("-----------------")

    for name in mat_file.keys():

        obj = mat_file[name]

        if isinstance(obj, h5py.Dataset):
            print(
                f"{name:20} "
                f"shape={str(obj.shape):20} "
                f"dtype={obj.dtype}"
            )

        else:
            print(
                f"{name:20} "
                f"type={type(obj).__name__}"
            )

        print()
    print("Acquisition values:")
    print("-------------------")

    fs = mat_file["Fs"][0, 0]
    center_freq = mat_file["CenterFreq"][0, 0]

    print(f"Sample rate:      {fs:,.0f} samples/s")
    print(f"Center frequency: {center_freq:,.0f} Hz")
    print(f"Duration:         {mat_file['RF0_I'].shape[0] / fs:.6f} s")

    print()
    print("First 10 IQ samples:")
    print("--------------------")
    print("I:", mat_file["RF0_I"][:10, 0])
    print("Q:", mat_file["RF0_Q"][:10, 0])

    print()
    print("Temporal signal statistics:")
    print("---------------------------")

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

    for time_s in window_times_s:
        start = int(time_s * fs)
        end = start + window_samples

        i_window = mat_file["RF0_I"][start:end, 0]
        q_window = mat_file["RF0_Q"][start:end, 0]

        power = i_window**2 + q_window**2
        magnitude = power**0.5

        print(
            f"t={time_s:>3.1f}s | "
            f"mag_mean={magnitude.mean():.8f} | "
            f"mag_std={magnitude.std():.8f} | "
            f"power={power.mean():.10f}"
        )