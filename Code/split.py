import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats


def process_directory(directory, label, data, center):
    """
    Process files in a directory and append metadata to the data list.

    Args:
        directory (str): Path to the directory containing files.
        label (str): Label for the files in the directory (e.g., "DRESS" or "MDE").
        data (list): List to store metadata for each file.
    """
    for filelists in os.listdir(directory):
        filename = os.path.splitext(filelists)[0]
        if label == "DRESS" and center == "MGH":
            if filename.split("_")[0] == "DRESS":
                case_id = filename.split("-")[0]
                slice_id = filename.split("_")[3]
            elif filename.split("-")[0] == "SR":
                case_id = filename.split("-")[0] + "-" + filename.split("-")[1]
                slice_id = filename.split("-")[2]
            # elif  filename.split                                 #OSU case
            else:
                case_id = filename.split("-")[0]
                slice_id = 1
        elif label == "MDE" and center == "MGH":
            parts = filename.split("__")
            slice_id = parts[0].split("_")[-1]
            if filename.split("_")[0] == "MDE":
                if filename == "MDE_missing1.svs":
                    case_id = "MDE_24"
                    slice_id = 1
                elif filename == "MDE_missing2.svs":
                    case_id = "MDE_25"
                    slice_id = 1
                else:
                    case_id = filename.split("-")[0]
                    slice_id = parts[0].split("_")[-1]
            else:
                case_id = filename.split("-")[0]
                slide_id = parts[0].split("_")[-1]
        else:
            parts = filename.split(" ")
            case_id = parts[0]
            if len(parts) == 3:
                slice_id = parts[2]
            else:
                slice_id = parts[1]
            # print(case_id, slice_id)

        path = os.path.join(directory, filename)
        data.append([case_id, slice_id, label, path])


def create_dataset_csv(output_csv, dataset_dirs):
    """
    Create a CSV file containing metadata for all files in the specified directories.

    Args:
        output_csv (str): Path to the output CSV file.
        dataset_dirs (list of tuples): List of tuples containing (directory, label).
    """
    data = []
    for directory, label, center in dataset_dirs:
        process_directory(directory, label, data, center)

    # Write data to CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "slice_id", "label", "path"])
        writer.writerows(data)


def create_split(
    df, input_csv, output_csv, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=7
):
    np.random.seed(seed)

    unique_cases = df["case_id"].unique()
    np.random.shuffle(unique_cases)

    train_size = int(train_ratio * len(unique_cases))
    val_size = int(val_ratio * len(unique_cases))

    train_cases = set(unique_cases[:train_size])
    val_cases = set(unique_cases[train_size : train_size + val_size])
    test_cases = set(unique_cases[train_size + val_size :])

    df["split"] = df["case_id"].apply(
        lambda x: "train" if x in train_cases else ("val" if x in val_cases else "test")
    )

    with open(input_csv) as f_in, open(output_csv, "w") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Write header
        header = next(reader)
        writer.writerow(header + ["split"])

        for row in reader:
            case_id = row[0]
            split = df[df["case_id"] == case_id]["split"].values[0]
            writer.writerow(row + [split])


# Patient voting in the case of multiple labels per patient WHICH IS NOT OUR CASE as of now
def patient_data_prep(df, patient_voting="max"):
    patients = np.unique(np.array(df["case_id"]))
    patient_labels = []

    for p in patients:
        locations = df[df["case_id"] == p].index.tolist()
        assert len(locations) > 0
        labels = df.iloc[locations]["label"].values
        if patient_voting == "max":
            label = label.max()
        elif patient_voting == "majority":
            label = stats.mode(labels)[0][0]
        else:
            raise ValueError("Invalid patient voting method")
        patient_labels.append(label)
    patient_df = pd.DataFrame({"case_id": patients, "label": np.array(patient_labels)})

    return patient_df


def center_split(output_dir="."):
    mgh_data = []
    process_directory("Dataset/DRESS", "DRESS", mgh_data, "MGH")
    process_directory("Dataset/MDE", "MDE", mgh_data, "MGH")

    osu_data = []
    process_directory("Dataset/OSU/DRESS", "DRESS", osu_data, "OSU")
    process_directory("Dataset/OSU/MDE", "MDE", osu_data, "OSU")

    with open(os.path.join(output_dir, "MGH_dataset.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "slice_id", "label", "path"])
        writer.writerows(mgh_data)

    with open(os.path.join(output_dir, "OSU_dataset.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "slice_id", "label", "path"])
        writer.writerows(osu_data)

    print(f"create MGH dataset: {len(mgh_data)}")
    print(f"create OSU dataset: {len(osu_data)}")


# Define directories and labels
dataset_dirs = [
    ("Dataset/DRESS", "DRESS", "MGH"),
    ("Dataset/MDE", "MDE", "MGH"),
    ("Dataset/OSU/DRESS", "DRESS", "OSU"),
    ("Dataset/OSU/MDE", "MDE", "OSU"),
]

# center_split()
df = pd.read_csv("Dataset_csv/OSU_dataset.csv")
create_split(df, "Dataset_csv/OSU_dataset.csv", "Dataset_csv/OSU_dataset_split.csv")
# Output CSV file
# output_csv = "dataset.csv"
# df = pd.read_csv(output_csv)

# Create the dataset CSV
# create_dataset_csv(output_csv, dataset_dirs)
# df = create_split(df)
# df = pd.read_csv("dataset_split.csv")
# patient_df = patient_data_prep(df)
# print(patient_df)
# print(df)
