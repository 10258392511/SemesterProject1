import argparse
import time
import torch
import scripts.config as config

from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from models.u_net import UNet
from models.modules import Normalizer
from helpers.datasets import MnMsHDF5SimplifiedDataset, MnMs3DDataset
from helpers.baseline_step_by_step import MetaLearner

if __name__ == '__main__':
    """
    python ./run_vis.py --train_source_name "csf" --input_dir "data/MnMs_extracted/MnMs_extracted.h5" --input_dir_3d "data/MnMs_extracted/MnMs_extracted_3d.h5" --device "cuda" --batch_size 2 --num_workers 0 --lam_smooth 0 --num_batches_to_sample 1 --num_learner_steps 20 --pre_train_epochs 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input_dir", required=True)  # .../MnMs_extracted.h5
    parser.add_argument("--input_dir_3d", required=True)  # .../MnMs_extracted_3d.h5
    parser.add_argument("--train_source_name", default="csf", choices=["csf", "hvhd", "uhe"])
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100)

    parser.add_argument("--lam_smooth", type=float, required=True)
    parser.add_argument("--num_batches_to_sample", type=int, required=True)
    parser.add_argument("--num_learner_steps", type=int, required=True)
    parser.add_argument("--pre_train_epochs", type=int, default=5)
    parser.add_argument("--loss_type", default="data-consistency", choices=["data-consistency", "avg-entropy"])
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    epochs = args.total_steps // args.num_batches_to_sample
    weights = {"lam_ce": 0.5, "lam_dsc": 0.5, "lam_smooth": args.lam_smooth}

    # load data
    train_src = args.train_source_name
    train_dataset_dict = dict()
    eval_dataset_3d_dict = dict()

    for key in ["train", "eval", "test"]:
        train_dataset_dict[key] = MnMsHDF5SimplifiedDataset(args.input_dir, train_src, mode=key, transforms=[],
                                                            if_augment=True, data_aug_dict=config.data_aug_defaults)

    for key in ["csf", "hvhd", "uhe"]:
    # for key in ["hvhd"]:
        eval_dataset_3d_dict[key] = MnMs3DDataset(args.input_dir_3d, key, mode="test")

    random_sampler = RandomSampler(train_dataset_dict["train"])
    train_loader_dict = dict()
    train_loader_dict[f"train_loader"] = DataLoader(train_dataset_dict["train"], batch_size=args.batch_size,
                                                    sampler=random_sampler, num_workers=args.num_workers)
    for key in ["eval", "test"]:
        train_loader_dict[f"{key}_loader"] = DataLoader(train_dataset_dict[key], batch_size=args.batch_size,
                                                        num_workers=args.num_workers)

    # models
    u_net = UNet(**config.u_net_params).to(DEVICE)
    norm = Normalizer(**config.normalizer_params).to(DEVICE)
    u_net_opt = torch.optim.Adam(u_net.parameters(), **config.u_net_optimzer_params)
    norm_opt = torch.optim.Adam(norm.parameters(), **config.normalizer_optimizer_params)
    norm_cp = Normalizer(**config.normalizer_params).to(DEVICE)

    lr_lambda = lambda epoch: 1 - epoch / epochs * (1 - config.scheduler_params["min_lr"] /
                                                    config.scheduler_params["max_lr"])
    u_net_scheduler = LambdaLR(u_net_opt, lr_lambda)

    # train
    time_stamp = f"{time.time()}_learner_{args.num_learner_steps}_pre_train_{args.pre_train_epochs}_" \
                 f"loss_{args.loss_type}".replace(".", "_")
    writer = SummaryWriter(f"run/norm_u_net/{time_stamp}")
    trainer_args = dict(test_dataset_dict=eval_dataset_3d_dict, normalizer_cp=norm_cp,
                        norm_opt_config=config.normalizer_optimizer_params,
                        num_batches_to_sample=args.num_batches_to_sample, num_learner_steps=args.num_learner_steps,
                        total_steps=args.total_steps, pre_train_epochs=args.pre_train_epochs,
                        eval_interval=args.eval_interval, normalizer=norm, u_net=u_net, norm_opt=norm_opt,
                        u_net_opt=u_net_opt,
                        epochs=epochs, num_classes=4, weights=weights, notebook=False, writer=writer,
                        param_save_dir="params/norm_u_net",
                        time_stamp=time_stamp, device=DEVICE, scheduler=[u_net_scheduler], **train_loader_dict)

    trainer = MetaLearner(**trainer_args)
    trainer.meta_train_vis(loss_type=args.loss_type)
