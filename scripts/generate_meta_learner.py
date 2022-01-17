import subprocess
import argparse


# The generated bash script should run in submission/


def make_bash_script(hyper_param_dict: dict):
    bash_script = f"""#!/bin/bash

#SBATCH --output=run_alt_norm_log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

eval "$(conda shell.bash hook)"
conda activate deep_learning
cd ../SemesterProject1

if [ ! -d params ]
then
    mkdir params
fi

python ./run_meta_learning.py --train_source_name "csf" --input_dir "/itet-stor/zhexwu/net_scratch/data/MnMs_extracted/MnMs_extracted.h5" --input_dir_3d "/itet-stor/zhexwu/net_scratch/data/MnMs_extracted/MnMs_extracted_3d.h5" --device "cuda" --batch_size 6 --num_workers 3 --lam_smooth {hyper_param_dict["lam_smooth"]} --num_batches_to_sample {hyper_param_dict["num_batches_to_sample"]} --num_learner_steps {hyper_param_dict["num_learner_steps"]}"""

    return bash_script


def create_filename(hyper_param_dict: dict):
    filename = ""
    for key, val in hyper_param_dict.items():
        filename += f"{key}_{val}_"

    filename = filename[:-1].replace(".", "_") + ".sh"

    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_num", type=int, choices=[1, 2], required=True)

    args = parser.parse_args()
    hyper_params = dict()
    # set 1
    hyper_params[1] = []
    num_batches = [1, 2, 5, 10]
    learner_steps = [1, 5, 10, 20]
    for num_batches_iter in num_batches:
        for learner_steps_iter in learner_steps:
            hyper_params[1].append({"lam_smooth": 0., "num_batches_to_sample": num_batches_iter,
                                    "num_learner_steps": learner_steps_iter})

    # set 2: extra
    hyper_params[2] = [{"lam_smooth": 0., "num_batches_to_sample": 5, "num_learner_steps": 5}]

    hyper_params_list = hyper_params[args.set_num]
    for hyper_param_dict_iter in hyper_params_list:
        filename = create_filename(hyper_param_dict_iter)
        bash_script = make_bash_script(hyper_param_dict_iter)
        subprocess.run(f"echo '{bash_script}' > {filename}", shell=True)
        # print(f"{filename}")
        # print(f"{bash_script}")
        # print("-" * 50)
