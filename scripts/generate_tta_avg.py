import subprocess
import argparse
import os


# The generated bash script should run in submission/


def make_bash_script(hyper_param_dict: dict):
    bash_script = f"""#!/bin/bash

#SBATCH --output=run_alt_norm_log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

eval "$(conda shell.bash hook)"
conda activate deep_learning
cd ../SemesterProject1

python run_tta_avg.py --device "cuda" --batch_size 6 --input_dir "/itet-stor/zhexwu/net_scratch/data/MnMs_extracted/MnMs_extracted.h5" --input_dir_3d "/itet-stor/zhexwu/net_scratch/data/MnMs_extracted/MnMs_extracted_3d.h5" --u_net_path "{hyper_param_dict["u_net_path"]}" --norm_path "{hyper_param_dict["norm_path"]}" """
    return bash_script


def create_filename(hyper_param_dict: dict):
    u_net_path = hyper_param_dict["u_net_path"]  # test_time_adapt_results/timestamp_config/u_net_....pt
    filename = os.path.basename(os.path.dirname(u_net_path)) + ".sh"  # timestamp_config.sh

    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_num", type=int, choices=[1, 2], required=True)

    args = parser.parse_args()
    hyper_params = dict()
    # set 1
    dir_root = "test_time_adapt_params"
    hyper_params[1] = [{"u_net_path": os.path.join(dir_root,
                                                   "1642926298_7829247_batches_1_learner_10_s_0_0000_pre_train_5/u_net_epoch_9999_eval_loss_0_3086.pt"),
                        "norm_path": os.path.join(dir_root,
                                                  "1642926298_7829247_batches_1_learner_10_s_0_0000_pre_train_5/norm_epoch_9999_eval_loss_0_3086.pt")},
                       {"u_net_path": os.path.join(dir_root,
                                                   "1642926300_3914664_batches_1_learner_1_s_0_0000_pre_train_5/u_net_epoch_7600_eval_loss_0_2859.pt"),
                        "norm_path": os.path.join(dir_root,
                                                  "1642926300_3914664_batches_1_learner_1_s_0_0000_pre_train_5/norm_epoch_7600_eval_loss_0_2859.pt")},
                       {"u_net_path": os.path.join(dir_root,
                                                   "1643141863_9186742_batches_1_learner_20_s_0_0000_pre_train_5/u_net_epoch_9000_eval_loss_0_3041.pt"),
                        "norm_path": os.path.join(dir_root,
                                                  "1643141863_9186742_batches_1_learner_20_s_0_0000_pre_train_5/norm_epoch_9000_eval_loss_0_3041.pt")}
                       ]

    # set 2: extra
    hyper_params[2] = []

    hyper_params_list = hyper_params[args.set_num]
    for hyper_param_dict_iter in hyper_params_list:
        filename = create_filename(hyper_param_dict_iter)
        bash_script = make_bash_script(hyper_param_dict_iter)
        subprocess.run(f"echo '{bash_script}' > {filename}", shell=True)
        # print(f"{filename}")
        # print(f"{bash_script}")
        # print("-" * 50)
