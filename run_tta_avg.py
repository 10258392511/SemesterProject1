import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import scripts.config as config

from torch.utils.data import DataLoader
from models.modules import Normalizer
from models.u_net import UNet
from helpers.datasets import MnMsHDF5SimplifiedDataset, MnMs3DDataset
from helpers.test_time_adaptation import evaluate_3D_adapt_avg


def save_results(save_dir, dice_losses_dict, figs_curve, figs_pred):
    # save_dir: absolute path
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, "dice_losses.pkl"), "wb") as wf:
        pickle.dump(dice_losses_dict, wf)

    for key in figs_curve:
        assert key in figs_pred
        assert len(figs_curve[key]) == len(figs_pred[key])
        for i in range(len(figs_curve[key])):
            curve_path = os.path.join(save_dir, f"{key}_{i}_curve.png")
            pred_path = os.path.join(save_dir, f"{key}_{i}_pred.png")
            figs_curve[key][i].savefig(curve_path)
            figs_pred[key][i].savefig(pred_path)


if __name__ == '__main__':
    """ 
    python run_tta_avg.py --batch_size 2 --num_agents 2 --max_iters 2 --input_dir "data/MnMs_extracted/MnMs_extracted.h5" --input_dir_3d "data/MnMs_extracted/MnMs_extracted_3d.h5" --u_net_path "test_time_adapt_params/1643141863_9186742_batches_1_learner_20_s_0_0000_pre_train_5/u_net_epoch_9000_eval_loss_0_3041.pt" --norm_path "test_time_adapt_params/1643141863_9186742_batches_1_learner_20_s_0_0000_pre_train_5/norm_epoch_9000_eval_loss_0_3041.pt"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--u_net_path", required=True)
    parser.add_argument("--norm_path", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--input_dir_3d", required=True)
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    save_root_dir_base = "test_time_adapt_results"
    save_root_dir = os.path.join(os.getcwd(), save_root_dir_base)
    if not os.path.isdir(save_root_dir):
        os.mkdir(save_root_dir)  # absolute path

    # load data
    source_names = ["hvhd", "csf", "uhe"]
    # source_names = ["csf"]
    data_path = args.input_dir
    data_path_test = args.input_dir_3d
    train_dataset_dict = {}
    for mode in ["train", "eval", "test"]:
        train_dataset_dict[mode] = MnMsHDF5SimplifiedDataset(data_path, "csf", mode, transforms=[], if_augment=True,
                                                             data_aug_dict=config.data_aug_defaults)
    test_dataset_dict = {}
    for source_name in source_names:
        test_dataset_dict[source_name] = MnMs3DDataset(data_path_test, source_name, "test")

    train_loader_dict = {}
    for mode in ["train", "eval", "test"]:
        if mode == "train":
            shuffle = True
        else:
            shuffle = False
        train_loader_dict[f"{mode}_loader"] = DataLoader(train_dataset_dict[mode], batch_size=args.batch_size,
                                                         shuffle=shuffle)

    # run 3d evaluation
    normalizer = Normalizer(**config.normalizer_params).to(DEVICE)
    normalizer_cp = Normalizer(**config.normalizer_params).to(DEVICE)
    normalizer_cp.load_state_dict(torch.load(args.norm_path, map_location=DEVICE))
    normalizer_cp.eval()
    u_net = UNet(**config.u_net_params).to(DEVICE)
    u_net.load_state_dict(torch.load(args.u_net_path, map_location=DEVICE))
    u_net.eval()

    kernel_sizes = [2 * i + 1 for i in range(10)]
    augmentors = [Normalizer(kernel_size=kernel_size).to(DEVICE) for kernel_size in kernel_sizes]

    normalizer_cps = [Normalizer(**config.normalizer_params).to(DEVICE) for _ in range(args.num_agents)]
    norm_opt_cps = []
    for normalizer_cp_iter in normalizer_cps:
        norm_opt_cps.append(torch.optim.SGD(normalizer_cp_iter.parameters(), lr=args.lr))

    losses_out, figs_out, figs_pred_out = evaluate_3D_adapt_avg(test_dataset_dict, normalizer, u_net, normalizer_cp,
                                                                normalizer_cps, norm_opt_cps, augmentors,
                                                                batch_size=args.batch_size, max_iters=args.max_iters,
                                                                device=DEVICE, if_notebook=False)

    directory_name = os.path.basename(os.path.dirname(args.u_net_path))  # timestamp_config
    save_dir = os.path.join(save_root_dir, directory_name)  # test_time_adapt_results/timestamp_config
    save_results(save_dir, losses_out, figs_out, figs_pred_out)
