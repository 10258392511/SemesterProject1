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

python ./run_baseline_step_by_step.py --train_source_name "csf" --input_dir "/itet-stor/zhexwu/net_scratch/data/MnMs_extracted/MnMs_extracted.h5" --input_dir_3d "/itet-stor/zhexwu/net_scratch/data/MnMs_extracted/MnMs_extracted_3d.h5" --device "cuda" --batch_size 6 --num_workers 3 --epochs 30 --lam_ce {hyper_param_dict["lam_ce"]} --lam_dsc {hyper_param_dict["lam_dsc"]} --lam_smooth {hyper_param_dict["lam_smooth"]}{" --if_augment" if hyper_param_dict["if_augment"] else ""}{" --if_alt" if hyper_param_dict["if_alt"] else ""}"""

    return bash_script


def create_filename(hyper_param_dict: dict):
    filename = ""
    for key, val in hyper_param_dict.items():
        filename += f"{key}_{val}_"

    filename = filename[:-1].replace(".", "_") + ".sh"

    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_num", type=int, choices=[1, 2, 3, 4], required=True)

    args = parser.parse_args()
    hyper_params = dict()
    # set 1
    hyper_params[1] = [{"lam_ce": 1, "lam_dsc": 0, "lam_smooth": 0, "if_augment": False, "if_alt": False},
                       {"lam_ce": 0, "lam_dsc": 1, "lam_smooth": 0, "if_augment": False, "if_alt": False},
                       {"lam_ce": 0.5, "lam_dsc": 0.5, "lam_smooth": 0, "if_augment": False, "if_alt": False}]
   
    # set 2
    hyper_params[2] = [{"lam_ce": 1, "lam_dsc": 0, "lam_smooth": 0, "if_augment": True, "if_alt": True},
                       {"lam_ce": 0, "lam_dsc": 1, "lam_smooth": 0, "if_augment": True, "if_alt": True},
                       {"lam_ce": 0.5, "lam_dsc": 0.5, "lam_smooth": 0, "if_augment": True, "if_alt": True}]
    # set 3
    hyper_params[3] = [{"lam_ce": 0.5, "lam_dsc": 0.5, "lam_smooth": 1, "if_augment": True, "if_alt": True},
                       {"lam_ce": 0.5, "lam_dsc": 0.5, "lam_smooth": 0.1, "if_augment": True, "if_alt": True},
                       {"lam_ce": 0.5, "lam_dsc": 0.5, "lam_smooth": 0.01, "if_augment": True, "if_alt": True},
                       {"lam_ce": 0.5, "lam_dsc": 0.5, "lam_smooth": 0.001, "if_augment": True, "if_alt": True}]

    # set 4: extra
    hyper_params[4] = [{"lam_ce": 0, "lam_dsc": 1, "lam_smooth": 0, "if_augment": False, "if_alt": False}]

    hyper_params_list = hyper_params[args.set_num]
    for hyper_param_dict_iter in hyper_params_list:
        filename = create_filename(hyper_param_dict_iter)
        bash_script = make_bash_script(hyper_param_dict_iter)
        subprocess.run(f"echo '{bash_script}' > {filename}", shell=True)
        # print(f"{filename}")
        # print(f"{bash_script}")
        # print("-" * 50)
