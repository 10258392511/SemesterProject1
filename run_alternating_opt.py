import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import scripts.config as config_params

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from helpers.datasets import MnMsHDF5Dataset
from helpers.utils import get_transforms
from models.modules import Normalizer
from models.u_net import UNet
from helpers.train_eval import AlternatingTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run alternating training for normalizer")
    parser.add_argument("--dataset_name", choices=["csf", "hvhd", "uhe"], required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model_save_dir", required=True)
    parser.add_argument("--eval_plot_save_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    # parser.add_argument("--loss_type", default="CE", choices=["DSC", "CE"])
    parser.add_argument("--ce_weight", type=float, default=1.)
    parser.add_argument("--dsc_weight", type=float, default=0.)
    parser.add_argument("--smooth_weight", type=float, default=1.)

    args = parser.parse_args()

    DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device(args.device)

    # Dataset and DataLoader
    transforms = get_transforms()
    train_dataset = MnMsHDF5Dataset(args.dataset_root, args.dataset_name, "train", transforms)
    eval_dataset = MnMsHDF5Dataset(args.dataset_root, args.dataset_name, "eval", transforms)
    test_dataset = MnMsHDF5Dataset(args.dataset_root, args.dataset_name, "test", transforms)
    # print(f"{len(train_dataset)}, {len(eval_dataset)}, {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Done loading data!")

    # Models
    normalizer = Normalizer(**config_params.normalizer_params).to(DEVICE)
    u_net = UNet(**config_params.u_net_params).to(DEVICE)
    norm_opt = torch.optim.Adam(normalizer.parameters(), **config_params.normalizer_optimizer_params)
    u_net_opt = torch.optim.Adam(u_net.parameters(), **config_params.u_net_optimzer_params)

    # Training
    time_stamp = f"{time.time()}".replace(".", "_")
    log_dir = f"run/norm_u_net/{time_stamp}_ce_{args.ce_weight}_dsc_{args.dsc_weight}_" \
              f"s_{args.smooth_weight}".replace(".", "_")
    writer = SummaryWriter(log_dir)
    alt_trainer = AlternatingTrainer(normalizer, u_net, train_loader, eval_loader, test_loader, norm_opt, u_net_opt,
                                     ce_weight=args.ce_weight, dsc_weight=args.dsc_weight,
                                     smooth_weight=args.smooth_weight, device=DEVICE, epochs=args.epochs,
                                     writer=writer, time_stamp=time_stamp, notebook=False,
                                     img_save_dir=args.eval_plot_save_dir)
    train_losses, eval_losses = alt_trainer.train(args.model_save_dir)

    with open(f"{log_dir}/loss_curve.pkl", "wb") as wf:
        pickle.dump({"train_losses": train_losses, "eval_losses": eval_losses}, wf)
