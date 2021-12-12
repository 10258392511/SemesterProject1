import h5py
import argparse
import csv
import nibabel as nib
import numpy as np
import os


def read_patient_id(data_path_root):
    csv_files = {"csf": "mnms_csf.csv", "hvhd": "mnms_hvhd.csv", "uhe": "mnms_uhe.csv"}
    patient_id_dict = {"csf": [], "hvhd": [], "uhe": []}
    for key in csv_files:
        filename = os.path.join(data_path_root, csv_files[key])
        with open(filename, newline="") as rf:
            reader = csv.reader(rf, delimiter=",", quotechar="|")
            for row in reader:
                patient_id_dict[key].append(row[0])

    return patient_id_dict


def create_subtree(group: h5py.Group, patient_id_list: list, train_split_info: dict, data_dir):
    train_split = train_split_info["train"]
    eval_split = train_split_info["eval"]
    train_group = group.create_group("train")
    eval_group = group.create_group("eval")
    test_group = group.create_group("test")
    create_leaf(train_group, patient_id_list[:train_split], data_dir)
    create_leaf(eval_group, patient_id_list[train_split:eval_split], data_dir)
    create_leaf(test_group, patient_id_list[eval_split:], data_dir)


def create_leaf(group: h5py.Group, patient_id_list: list, data_dir):
    # imgs, masks = [], []
    counter = 0
    for patient_id in patient_id_list:
        print(f"current: {patient_id}")
        for time_str in ["ed", "es"]:
            img = nib.load(f"{data_dir}/{time_str}_{patient_id}_sa.nii.gz")
            mask = nib.load(f"{data_dir}/{time_str}_{patient_id}_sa_gt.nii.gz")
            img = img.get_fdata()
            mask = mask.get_fdata()
            assert img.shape == mask.shape, "shape doesn't match"
            for z in range(img.shape[-1]):
                group.create_dataset(f"img{counter}", data=img[..., z], compression="gzip", compression_opts=9)
                group.create_dataset(f"mask{counter}", data=mask[..., z], compression="gzip", compression_opts=9)
                counter += 1
            group.attrs["SIZE"] = counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)  # MnMs_extracted root
    parser.add_argument("--output_dir", required=True)  # can also be MnMs root
    args = parser.parse_args()

    scanners = ["csf", "hvhd", "uhe"]
    split_info = {"csf": {"train": 31, "eval": 41},
                  "hvhd": {"train": 56, "eval": 66},
                  "uhe": {"train": 11, "eval": 16}}

    patient_id_dict = read_patient_id(args.input_dir)
    data_dir = os.path.join(args.input_dir, "images_corrected")
    with h5py.File(f"{args.output_dir}/MnMs_extracted.h5", "w") as hdf:
        for scanner in scanners:
            group = hdf.create_group(scanner)
            create_subtree(group, patient_id_dict[scanner], split_info[scanner], data_dir)
