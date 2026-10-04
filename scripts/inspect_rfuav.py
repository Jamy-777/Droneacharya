from pathlib import Path

import numpy as np


FILE_PATH = Path(
    r"D:\Weeeeeeee\DroneacharyaData\interim\rfuav\devention_devo\pack1_0-1s.iq"
)

file_size = FILE_PATH.stat().st_size

print(f"File: {FILE_PATH.name}")
print(f"Size: {file_size:,} bytes")

float_count = file_size // np.dtype(np.float32).itemsize
complex_sample_count = float_count // 2

print(f"float32 values: {float_count:,}")
print(f"possible complex samples: {complex_sample_count:,}")

print()
print("First 20 float32 values:")
print("------------------------")

values = np.fromfile(
    FILE_PATH,
    dtype=np.float32,
    count=20,
)

print(values)

sample_rate = 100_000_000

probe_times_s = [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
]

complex_samples_per_probe = 10


print()
print("IQ probes across recording:")
print("---------------------------")


with FILE_PATH.open("rb") as file:

    for time_s in probe_times_s:

        complex_index = int(time_s * sample_rate)

        byte_offset = complex_index * 2 * 4

        file.seek(byte_offset)

        values = np.fromfile(
            file,
            dtype=np.float32,
            count=complex_samples_per_probe * 2,
        )

        i_values = values[0::2]
        q_values = values[1::2]

        print()
        print(f"t = {time_s:.2f} s")
        print("I:", i_values)
        print("Q:", q_values)


print()
print("Temporal signal statistics:")
print("---------------------------")

window_samples = 1_000_000

window_times_s = [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
]

with FILE_PATH.open("rb") as file:

    for time_s in window_times_s:

        complex_index = int(time_s * sample_rate)
        byte_offset = complex_index * 8

        file.seek(byte_offset)

        values = np.fromfile(
            file,
            dtype=np.float32,
            count=window_samples * 2,
        )

        i_values = values[0::2]
        q_values = values[1::2]

        power = i_values**2 + q_values**2
        magnitude = np.sqrt(power)

        print(
            f"t={time_s:>4.2f}s | "
            f"mag_mean={magnitude.mean():.8f} | "
            f"mag_std={magnitude.std():.8f} | "
            f"power={power.mean():.10f}"
        )