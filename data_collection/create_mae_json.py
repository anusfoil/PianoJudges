import os
import json
import csv

# Define your datasets directories
datasets = {
    "pianojudge": "/import/c4dm-datasets/PianoJudge/",
    "atepp": "/import/c4dm-datasets/ATEPP-audio/",
    "maestro": "/import/c4dm-datasets/maestro-v3.0.0/",
    "con_espressione": "/import/c4dm-datasets-ext/con_espressione/",
    "mazurkas": "/import/c4dm-datasets/Mazurkas/",
    "jazz800": "/import/c4dm-datasets-ext/JAZZ800/",
    "ycuppe": "/import/c4dm-datasets/YCU-PPE-III/"
}

# Prepare data structure for JSON
data_json = {"data": []}

for dataset_name, dataset_path in datasets.items():
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.wav')):
                file_path = os.path.join(root, file)
                # Adding dummy labels, replace or modify as needed
                data_json["data"].append({"wav": file_path, "labels": "dummy_label"})

# Write JSON file
with open('/homes/hz009/Research/PianoJudge/data_collection/audioset_train_wav_only.json', 'w') as json_file:
    json.dump(data_json, json_file, indent=4)

# Dummy labels for CSV
labels = [
    {"index": 0, "mid": "dummy_label", "display_name": "Generic Audio"}
]

# # Write CSV file
# with open('dummy_label.csv', 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(["index", "mid", "display_name"])
#     for label in labels:
#         writer.writerow([label["index"], label["mid"], label["display_name"]])

print("JSON and CSV files have been generated.")
