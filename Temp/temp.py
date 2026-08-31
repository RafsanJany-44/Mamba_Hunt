# from pathlib import Path
# from scipy.io import whosmat

# root = Path("/media/data/rPPG/rPPG_Data/TokyoTechDataset")

# subjects = sorted(p for p in root.iterdir() if p.is_dir())

# print("Subjects found:", len(subjects))

# for subject in subjects:
#     videos = sorted((subject / "30fps").glob("*.avi"))
#     ppg_file = subject / "contactPPG.mat"

#     print("\nSubject:", subject.name)
#     print("30-fps videos:", len(videos))
#     print("Video names:", [video.name for video in videos])
#     print("PPG exists:", ppg_file.is_file())

#     if ppg_file.is_file():
#         print("MAT variables:", whosmat(ppg_file))



# from pathlib import Path
# from scipy.io import loadmat
# import numpy as np

# root = Path("/media/data/rPPG/rPPG_Data/TokyoTechDataset")
# ppg_fs = 2048

# for subject in sorted(p for p in root.iterdir() if p.is_dir()):
#     data = loadmat(subject / "contactPPG.mat")

#     ppg = np.asarray(data["dataA"]).squeeze()
#     locations = np.asarray(data["locsA"]).squeeze()
#     peaks = np.asarray(data["pksA"]).squeeze()

#     print("=" * 60)
#     print("Subject          :", subject.name)
#     print("PPG samples      :", len(ppg))
#     print("PPG duration     :", len(ppg) / ppg_fs)
#     print("Heartbeat times  :", len(locations))
#     print("Peak values      :", len(peaks))
#     print("First beat time  :", locations[0])
#     print("Last beat time   :", locations[-1])




# from pathlib import Path
# import cv2
# import numpy as np
# import pandas as pd

# root = Path("/media/data/rPPG/rPPG_Data/Pub_BH-rPPG_FULL")

# recordings = sorted(
#     [path for path in root.iterdir() if path.is_dir()],
#     key=lambda path: tuple(int(value) for value in path.name.split("_")),
# )

# rows = []
# problems = []

# for recording in recordings:
#     subject_id, condition_id = map(int, recording.name.split("_"))

#     frame_directory = recording / recording.name
#     frame_files = sorted(frame_directory.glob("Frame_*.png"))

#     timestamp_file = recording / "timestamps.csv"
#     wave_file = recording / "wave.csv"
#     sensor_file = recording / "sensor.csv"

#     if not frame_files or not timestamp_file.is_file():
#         problems.append(f"{recording.name}: missing frames or timestamps")
#         continue

#     timestamps = np.loadtxt(timestamp_file).reshape(-1)
#     wave = pd.read_csv(wave_file)["Wave"].to_numpy()
#     sensor = pd.read_csv(sensor_file)

#     duration = (timestamps[-1] - timestamps[0]) / 1000.0

#     effective_fps = (
#         (len(frame_files) - 1) / duration
#         if duration > 0 and len(frame_files) > 1
#         else np.nan
#     )

#     effective_wave_rate = (
#         (len(wave) - 1) / duration
#         if duration > 0 and len(wave) > 1
#         else np.nan
#     )

#     sample_indices = np.linspace(
#         0, len(frame_files) - 1, min(10, len(frame_files)), dtype=int
#     )

#     brightness_values = []
#     resolutions = set()

#     for index in sample_indices:
#         image = cv2.imread(str(frame_files[index]))

#         if image is None:
#             problems.append(
#                 f"{recording.name}: unreadable frame {frame_files[index].name}"
#             )
#             continue

#         resolutions.add((image.shape[1], image.shape[0]))
#         brightness_values.append(float(image.mean()))

#     if len(frame_files) != len(timestamps):
#         problems.append(
#             f"{recording.name}: frames={len(frame_files)}, "
#             f"timestamps={len(timestamps)}"
#         )

#     rows.append(
#         {
#             "recording": recording.name,
#             "subject": subject_id,
#             "condition": condition_id,
#             "frames": len(frame_files),
#             "timestamps": len(timestamps),
#             "duration": duration,
#             "fps": effective_fps,
#             "wave_samples": len(wave),
#             "wave_rate": effective_wave_rate,
#             "sensor_samples": len(sensor),
#             "brightness": np.mean(brightness_values),
#             "resolutions": str(sorted(resolutions)),
#         }
#     )

# results = pd.DataFrame(rows)

# print("=" * 70)
# print("BH-rPPG DATASET AUDIT")
# print("=" * 70)
# print("Recordings :", len(results))
# print("Subjects   :", results["subject"].nunique())
# print("Conditions :", sorted(results["condition"].unique()))
# print("Problems   :", len(problems))

# for condition in sorted(results["condition"].unique()):
#     group = results[results["condition"] == condition]

#     print("\n" + "-" * 70)
#     print("Condition:", condition)
#     print("-" * 70)
#     print("Recordings       :", len(group))
#     print("Frames min/max   :", group["frames"].min(), group["frames"].max())
#     print("Duration min/max :", group["duration"].min(), group["duration"].max())
#     print("FPS min/mean/max :", group["fps"].min(),
#           group["fps"].mean(), group["fps"].max())
#     print("Wave rate mean   :", group["wave_rate"].mean())
#     print("Brightness mean  :", group["brightness"].mean())
#     print("Resolutions      :", sorted(group["resolutions"].unique()))

# if problems:
#     print("\n" + "=" * 70)
#     print("PROBLEMS")
#     print("=" * 70)

#     for problem in problems:
#         print(problem)

# columns = [
#     "recording",
#     "condition",
#     "frames",
#     "duration",
#     "fps",
#     "wave_samples",
#     "wave_rate",
#     "sensor_samples",
# ]

# outliers = results[
#     (results["duration"] < 58)
#     | (results["duration"] > 63)
#     | (results["fps"] < 14)
#     | (results["fps"] > 16)
# ]

# print(outliers[columns].to_string(index=False))

# print(
#     results.groupby("condition")[["wave_samples", "sensor_samples"]]
#     .agg(["min", "median", "max"])
# )



# from pathlib import Path
# import cv2
# import numpy as np
# import pandas as pd

# root = Path("/media/data/rPPG/rPPG_Data/UBFC-PHYS")

# subjects = sorted(
#     [path for path in root.iterdir() if path.is_dir()],
#     key=lambda path: int(path.name[1:]),
# )

# rows = []
# problems = []

# for subject in subjects:
#     subject_number = int(subject.name[1:])
#     info_file = subject / f"info_s{subject_number}.txt"

#     scenario = "unknown"

#     if info_file.is_file():
#         info_lines = [
#             line.strip()
#             for line in info_file.read_text(errors="replace").splitlines()
#             if line.strip()
#         ]

#         if len(info_lines) >= 3:
#             scenario = info_lines[2]

#     for task_number in [1, 2, 3]:
#         task = f"T{task_number}"

#         video_file = subject / f"vid_s{subject_number}_{task}.avi"
#         bvp_file = subject / f"bvp_s{subject_number}_{task}.csv"
#         eda_file = subject / f"eda_s{subject_number}_{task}.csv"

#         if not video_file.is_file():
#             problems.append(f"Missing video: {video_file}")
#             continue

#         if not bvp_file.is_file():
#             problems.append(f"Missing BVP: {bvp_file}")
#             continue

#         if not eda_file.is_file():
#             problems.append(f"Missing EDA: {eda_file}")
#             continue

#         video = cv2.VideoCapture(str(video_file))

#         frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = float(video.get(cv2.CAP_PROP_FPS))
#         width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         video.release()

#         video_duration = frame_count / fps if fps > 0 else np.nan

#         bvp = np.loadtxt(bvp_file, delimiter=",").reshape(-1)
#         eda = np.loadtxt(eda_file, delimiter=",").reshape(-1)

#         bvp_duration = len(bvp) / 64.0
#         eda_duration = len(eda) / 4.0

#         if not np.isfinite(bvp).all():
#             problems.append(f"Non-finite BVP: {bvp_file}")

#         rows.append(
#             {
#                 "subject": subject_number,
#                 "task": task,
#                 "scenario": scenario,
#                 "frames": frame_count,
#                 "fps": fps,
#                 "video_duration": video_duration,
#                 "resolution": f"{width}x{height}",
#                 "bvp_samples": len(bvp),
#                 "bvp_duration": bvp_duration,
#                 "eda_samples": len(eda),
#                 "eda_duration": eda_duration,
#             }
#         )

# results = pd.DataFrame(rows)

# print("=" * 70)
# print("UBFC-PHYS DATASET AUDIT")
# print("=" * 70)
# print("Subjects found    :", results["subject"].nunique())
# print("Recordings found  :", len(results))
# print("Scenarios         :", results["scenario"].value_counts().to_dict())
# print("Problems          :", len(problems))

# for task in ["T1", "T2", "T3"]:
#     group = results[results["task"] == task]

#     print("\n" + "-" * 70)
#     print("Task:", task)
#     print("-" * 70)
#     print("Recordings              :", len(group))
#     print("Frames min/max          :", group["frames"].min(),
#           group["frames"].max())
#     print("FPS min/mean/max        :", group["fps"].min(),
#           group["fps"].mean(), group["fps"].max())
#     print("Video duration min/max  :", group["video_duration"].min(),
#           group["video_duration"].max())
#     print("BVP samples min/max     :", group["bvp_samples"].min(),
#           group["bvp_samples"].max())
#     print("BVP duration min/max    :", group["bvp_duration"].min(),
#           group["bvp_duration"].max())
#     print("EDA samples min/max     :", group["eda_samples"].min(),
#           group["eda_samples"].max())
#     print("Resolutions             :", sorted(group["resolution"].unique()))

# if problems:
#     print("\n" + "=" * 70)
#     print("PROBLEMS")
#     print("=" * 70)

#     for problem in problems:
#         print(problem)


from pathlib import Path
import cv2
import h5py
import numpy as np
import pandas as pd

root = Path("/media/data/rPPG/rPPG_Data/cohface_sorted")

recording_directories = sorted(
    [path for path in root.iterdir() if path.is_dir()],
    key=lambda path: (
        int(path.name.split("_")[1]),
        int(path.name.split("_")[2]),
    ),
)

expected = {
    (subject, session)
    for subject in range(1, 41)
    for session in range(4)
}

available = set()
rows = []
problems = []


def clean_attribute(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return clean_attribute(value.reshape(-1)[0])

        return str(value.tolist())

    return str(value)


for directory in recording_directories:
    parts = directory.name.split("_")

    subject = int(parts[1])
    session = int(parts[2])

    available.add((subject, session))

    video_file = directory / f"data_{subject}_{session}.avi"
    hdf5_file = directory / f"data_{subject}_{session}.hdf5"

    if not video_file.is_file():
        problems.append(f"Missing video: {video_file}")
        continue

    if not hdf5_file.is_file():
        problems.append(f"Missing HDF5: {hdf5_file}")
        continue

    video = cv2.VideoCapture(str(video_file))

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video.release()

    video_duration = frame_count / fps if fps > 0 else np.nan

    with h5py.File(hdf5_file, "r") as file:
        keys = list(file.keys())
        root_attributes = {
            key: clean_attribute(value)
            for key, value in file.attrs.items()
        }

        if "pulse" not in file:
            problems.append(f"Missing pulse signal: {hdf5_file}")
            continue

        pulse = np.asarray(file["pulse"]).reshape(-1)

        respiration = (
            np.asarray(file["respiration"]).reshape(-1)
            if "respiration" in file
            else np.array([])
        )

        pulse_attributes = {
            key: clean_attribute(value)
            for key, value in file["pulse"].attrs.items()
        }

        respiration_attributes = (
            {
                key: clean_attribute(value)
                for key, value in file["respiration"].attrs.items()
            }
            if "respiration" in file
            else {}
        )

    pulse_rate = float(pulse_attributes.get("sample-rate-hz", 256))
    respiration_rate = float(
        respiration_attributes.get("sample-rate-hz", 32)
    )

    pulse_duration = len(pulse) / pulse_rate
    respiration_duration = (
        len(respiration) / respiration_rate
        if len(respiration) > 0
        else np.nan
    )

    illumination = root_attributes.get("illumination", "unknown")

    if not np.isfinite(pulse).all():
        problems.append(f"Non-finite pulse signal: {hdf5_file}")

    rows.append(
        {
            "subject": subject,
            "session": session,
            "illumination": illumination,
            "frames": frame_count,
            "fps": fps,
            "video_duration": video_duration,
            "resolution": f"{width}x{height}",
            "pulse_samples": len(pulse),
            "pulse_rate": pulse_rate,
            "pulse_duration": pulse_duration,
            "respiration_samples": len(respiration),
            "respiration_rate": respiration_rate,
            "respiration_duration": respiration_duration,
            "hdf5_keys": str(keys),
        }
    )

results = pd.DataFrame(rows)

missing = sorted(expected - available)
unexpected = sorted(available - expected)

print("=" * 70)
print("COHFACE DATASET AUDIT")
print("=" * 70)
print("Subjects found       :", results["subject"].nunique())
print("Recordings found     :", len(results))
print("Missing recordings   :", missing)
print("Unexpected recordings:", unexpected)
print("Problems             :", len(problems))
print("HDF5 keys            :", sorted(results["hdf5_keys"].unique()))
print("Illuminations        :", results["illumination"].value_counts().to_dict())

for session in range(4):
    group = results[results["session"] == session]

    print("\n" + "-" * 70)
    print("Session:", session)
    print("-" * 70)
    print("Recordings             :", len(group))
    print("Illuminations          :", group["illumination"].value_counts().to_dict())
    print("Frames min/max         :", group["frames"].min(),
          group["frames"].max())
    print("FPS min/mean/max       :", group["fps"].min(),
          group["fps"].mean(), group["fps"].max())
    print("Video duration min/max :", group["video_duration"].min(),
          group["video_duration"].max())
    print("Pulse samples min/max  :", group["pulse_samples"].min(),
          group["pulse_samples"].max())
    print("Pulse rate values      :", sorted(group["pulse_rate"].unique()))
    print("Pulse duration min/max :", group["pulse_duration"].min(),
          group["pulse_duration"].max())
    print("Resolutions            :", sorted(group["resolution"].unique()))

if problems:
    print("\n" + "=" * 70)
    print("PROBLEMS")
    print("=" * 70)

    for problem in problems:
        print(problem)


results["duration_difference"] = abs(
    results["video_duration"] - results["pulse_duration"]
)

print("=" * 70)
print("LONG RECORDINGS")
print("=" * 70)

print(
    results[results["video_duration"] > 70][
        [
            "subject",
            "session",
            "illumination",
            "frames",
            "video_duration",
            "pulse_samples",
            "pulse_duration",
            "duration_difference",
        ]
    ].to_string(index=False)
)

print("\nMaximum video–BVP duration difference:")
print(results["duration_difference"].max())


import h5py
import numpy as np

path = (
    "/media/data/rPPG/rPPG_Data/cohface_sorted/"
    "Subj_1_0/data_1_0.hdf5"
)

with h5py.File(path, "r") as file:
    print("Root attributes:", dict(file.attrs))

    for key in file.keys():
        data = np.asarray(file[key])

        print("\nKey:", key)
        print("Shape:", data.shape)
        print("Dtype:", data.dtype)
        print("Attributes:", dict(file[key].attrs))
        print("First values:", data.reshape(-1)[:5])
        print("Last values:", data.reshape(-1)[-5:])