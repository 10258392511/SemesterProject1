import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import pickle

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from helpers.datasets import MnMsHDF5Dataset
from models.u_net import UNet
from helpers.train_eval import UNetTrainer
from helpers.utils import get_transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ce_weight", type=float, default=1.)
    parser.add_argument("--dsc_weight", type=float, default=0.)
    parser.add_argument("--data_path_root", required=True)

    args = parser.parse_args()

    DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device(args.device)

    transforms = get_transforms()
    train_dataset = MnMsHDF5Dataset(args.data_path_root, "csf", "train", transforms)
    eval_dataset = MnMsHDF5Dataset(args.data_path_root, "csf", "eval", transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    u_net = UNet(num_down_blocks=4, target_channels=4, in_channels=1).to(DEVICE)
    u_net_opt = torch.optim.Adam(u_net.parameters(), lr=args.lr)

    time_stamp = f"{time.time()}_ce_{args.ce_weight}_dsc_{args.dsc_weight}".replace(".", "_")
    writer = SummaryWriter(f"run/u_net/{time_stamp}")

    u_net_trainer = UNetTrainer(train_loader, eval_loader, u_net, u_net_opt, ce_weight=args.ce_weight,
                                dsc_weight=args.dsc_weight, device=DEVICE, writer=writer)
    train_losses, eval_losses = u_net_trainer.train()

    with open(f"{time_stamp}/loss_curve.pkl", "wb") as wf:
        pickle.dump({"train": train_losses, "eval": eval_losses}, wf)
