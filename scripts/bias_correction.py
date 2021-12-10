import numpy as np
import subprocess
import argparse
import os
import csv
import nibabel as nib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    new_dirs = ["images", "images_corrected"]
#    for new_dir in new_dirs:
#        new_dir_path = os.path.join(args.output_dir, new_dir)
#        print(f"{new_dir_path}")
#        if not os.path.isdir(new_dir_path):
#            os.makedirs(new_dir_path)
#
    csv_files = ["mnms_csf.csv", "mnms_hvhd.csv", "mnms_uhe.csv"]
    patient_info = {}
    for csv_file in csv_files:
        csv_file_path = os.path.join(args.input_dir, csv_file)
        with open(csv_file_path, newline="") as rf:
            reader = csv.reader(rf, delimiter=",", quotechar="|")
            for row in reader:
                row[1], row[2] = int(row[1]), int(row[2])
                patient_info[row[0]] = {"ED": row[1], "ES": row[2]}
    # print(len(patient_info))

    counter = 0
    for root, dirs, files in os.walk(os.path.join(args.input_dir, "Labeled")):
        for file in files:
            if ".nii.gz" in file:
                file_path = os.path.join(root, file)
                img = nib.load(file_path)
                img_data = img.get_fdata()
                ind = file.find("_sa")
                patient_num = file[:ind]
                ed = patient_info[patient_num]["ED"]
                es = patient_info[patient_num]["ES"]
                # img_ed_and_es = img_data[..., [ed, es]]  # (H, W, Z, 2)
                # img_ed_and_es = nib.Nifti1Image(img_ed_and_es, affine=np.eye(4))
                # img_ed_and_es_path = os.path.join(args.output_dir, "images", file)
                img_ed = nib.Nifti1Image(img_data[..., ed], affine=np.eye(4))
                img_es = nib.Nifti1Image(img_data[..., es], affine=np.eye(4))
                img_ed_path = os.path.join(args.output_dir, "images", f"ed_{file}")
                img_es_path = os.path.join(args.output_dir, "images", f"es_{file}")

                # print(f"current: {img_ed_and_es_path}")
                # print("-" * 40)
                nib.save(img_ed, img_ed_path)
                nib.save(img_es, img_es_path)
                img_ed_corrected_path = os.path.join(args.output_dir, "images_corrected", f"ed_{file}")
                img_es_corrected_path = os.path.join(args.output_dir, "images_corrected", f"es_{file}")
                if "_gt" in file:
                    nib.save(img_ed, img_ed_corrected_path)
                    nib.save(img_es, img_es_corrected_path)
                else:
                    counter += 1
                    print(f"current counter; {counter}: {img_ed_path}, {img_es_path}, "
                          f"{img_ed_corrected_path}, {img_es_corrected_path}")
                    subprocess.run(["/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4_th", img_ed_path,
                                    img_ed_corrected_path], capture_output=True)
                    subprocess.run(["/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4_th", img_es_path,
                                    img_es_corrected_path], capture_output=True)

    print("Done!")
