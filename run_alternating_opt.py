import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scripts.config as config_params

from torch.utils.data import DataLoader
from helpers.datasets import MnMsDataset
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
    parser.add_argument("--loss_type", default="CE", choices=["DSC", "CE"])
    parser.add_argument("--smooth_weight", type=float, default=1.)

    args = parser.parse_args()

    DEVICE = args.device

    # Dataset and DataLoader
    transforms = get_transforms()
    train_dataset = MnMsDataset(args.dataset_name, args.dataset_root, transforms, "train")
    eval_dataset = MnMsDataset(args.dataset_name, args.dataset_root, transforms, "eval")
    test_dataset = MnMsDataset(args.dataset_name, args.dataset_root, transforms, "test")
    # print(f"{len(train_dataset)}, {len(eval_dataset)}, {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Done loading data!")

    # Models
    normalizer = Normalizer(**config_params.normalizer_params).to(args.device)
    u_net = UNet(**config_params.u_net_params).to(args.device)
    norm_opt = torch.optim.Adam(normalizer.parameters(), **config_params.normalizer_optimizer_params)
    u_net_opt = torch.optim.Adam(u_net.parameters(), **config_params.u_net_optimzer_params)

    # Training
    alt_trainer = AlternatingTrainer(normalizer, u_net, train_loader, eval_loader, test_loader, norm_opt, u_net_opt,
                                     epochs=args.epochs, loss_type=args.loss_type, device=args.device,
                                     img_save_dir=args.eval_plot_save_dir, notebook=False,
                                     smooth_weight=args.smooth_weight)
    train_losses, eval_losses = alt_trainer.train(args.model_save_dir)

    with open("./loss_curve.pkl", "wb") as wf:
        pickle.dump({"train_losses": train_losses, "eval_losses": eval_losses}, wf)
