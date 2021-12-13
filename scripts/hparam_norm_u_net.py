import subprocess


def make_bash_script(ce, dsc, smooth_weight):
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

if [ ! -d images ]
then
    mkdir images
fi

python ./run_alternating_opt.py --dataset_name "csf" --dataset_root "/itet-stor/zhexwu/net_scratch/data/MnMs_extracted/MnMs_extracted.h5" --device "cuda" --batch_size 6 --num_workers 2 --model_save_dir "./params/norm_u_net" --eval_plot_save_dir "./images/norm_u_net" --epochs 20 --ce_weight {ce} --dsc_weight {dsc} --smooth_weight {smooth_weight}"""

    return bash_script


if __name__ == '__main__':
    loss_weights = [{"ce": 1, "dsc": 0}, {"ce": 0.5, "dsc": 0.5}, {"ce": 0, "dsc": 1}]
    smooth_weights = [0.01]
    for loss_weight in loss_weights:
        for smooth_weight in smooth_weights:
            ce, dsc = loss_weight["ce"], loss_weight["dsc"]
            bash_script = make_bash_script(ce, dsc, smooth_weight)
            filename = f"ce_{ce}_dsc_{dsc}_s_{smooth_weight}".replace(".", "_")
            # subprocess.run(f"touch {filename}.sh", shell=True)
            subprocess.run(f"echo '{bash_script}' > {filename}.sh", shell=True)
