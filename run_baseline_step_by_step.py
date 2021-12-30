import argparse
import torch
import time
import scripts.config as config

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.u_net import UNet
from models.modules import Normalizer
from helpers.datasets import MnMsHDF5SimplifiedDataset, MnMs3DDataset
from helpers.utils import get_separated_transforms
from helpers.baseline_step_by_step import OnePassTrainer, AltTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--if_augment", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input_dir", required=True)  # .../MnMs_extracted.h5
    parser.add_argument("--input_dir_3d", required=True)  # .../MnMs_extracted_3d.h5
    parser.add_argument("--train_source_name", default="csf", choices=["csf", "hvhd", "uhe"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lam_ce", type=float, required=True)
    parser.add_argument("--lam_dsc", type=float, required=True)
    parser.add_argument("--lam_smooth", type=float, required=True)
    parser.add_argument("--if_alt", action="store_true")
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--if_not_att", action="store_false")
    args = parser.parse_args()

    config.data_aug_defaults["data_aug_ratio"] = args.aug_prob
    config.u_net_params["attention"] = args.if_not_att
    print(f"{args}, {config.data_aug_defaults}, {config.u_net_params}")

    DEVICE = torch.device(args.device)

    if args.if_augment:
        transforms = get_separated_transforms()
    else:
        transforms = None

    source_names = ["csf", "hvhd", "uhe"]
    modes = ["train", "eval", "test"]
    train_dataset_dict = {}
    for mode in modes:
        train_dataset_dict[mode] = MnMsHDF5SimplifiedDataset(args.input_dir, args.train_source_name, mode,
                                                             transforms=transforms, if_augment=args.if_augment,
                                                             data_aug_dict=data_aug_defaults)
    test_dataset_dict = {}
    for source_name in source_names:
        test_dataset_dict[source_name] = MnMs3DDataset(args.input_dir_3d, source_name, "test")

    train_loader_dict = {}
    for mode in modes:
        if mode == "train":
            shuffle = True
        else:
            shuffle = False
        train_loader_dict[f"{mode}_loader"] = DataLoader(train_dataset_dict[mode], batch_size=args.batch_size,
                                                         shuffle=shuffle, num_workers=args.num_workers)

    u_net = UNet(**config.u_net_params).to(DEVICE)
    norm = Normalizer(**config.normalizer_params).to(DEVICE)
    u_net_opt = torch.optim.Adam(u_net.parameters(), **config.u_net_optimzer_params)
    norm_opt = torch.optim.Adam(norm.parameters(), **config.normalizer_optimizer_params)

    weights = dict(lam_ce=args.lam_ce, lam_dsc=args.lam_dsc, lam_smooth=args.lam_smooth)
    time_stamp = f"{time.time()}_ce_{weights['lam_ce']}_dsc_{weights['lam_dsc']}_s_{weights['lam_smooth']}_alt_" \
                 f"{args.if_alt}_aug_{args.if_augment}".replace(".", "_")
    writer = SummaryWriter(f"run/norm_u_net/{time_stamp}")
    trainer_args = dict(test_dataset_dict=test_dataset_dict, normalizer=norm, u_net=u_net,
                        norm_opt=norm_opt, u_net_opt=u_net_opt, epochs=args.epochs, num_classes=4,
                        weights=weights, notebook=False, writer=writer, param_save_dir="params/norm_u_net",
                        time_stamp=time_stamp, device=DEVICE, **train_loader_dict)
    if args.if_alt:
        trainer = AltTrainer(**trainer_args)
    else:
        trainer = OnePassTrainer(**trainer_args)

    trainer.train()
