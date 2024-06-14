import copy
import logging
import os
from typing import Any, Dict, List
from collections import Counter

import torch
import torchvision.transforms
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import PIL
from itertools import cycle


from hps import Hparams
from utils import linear_warmup, write_images
import matplotlib.pyplot as plt
import wandb

def preprocess_batch(args: Hparams, batch: Dict[str, Tensor]):
    batch["x"] = (batch["x"].to(args.device).float() - 127.5) / 127.5
    batch["pa"] = {}
    for k, v in batch.items():
        if k != "x" and k!= "pa":
            v = v.to(args.device).float()
            batch["pa"][k] = v[..., None, None].repeat(1, 1, *(args.input_res,) * 2)
    parent_keys = [k for k in batch.keys() if k != "x" and k != "pa"]
    for key in parent_keys:
        del batch[key]
    return batch


def trainer(
    args: Hparams,
    model: nn.Module,
    ema: nn.Module,
    dataloaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    writer: SummaryWriter,
    logger: logging.Logger,
):
    if args.hps == "cmmnist":
        projname = "HVAE"
    elif args.hps == "mimic192":
        projname = "MIMIC"
    if args.wandb:
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=projname,
            name=args.exp_name,
            # track hyperparameters and run metadata
            config={
                "Architecture": "HVAE",
                "Learning Rate": args.lr,
                "Batch Size": args.bs,
                "Weight Decay": args.wd,
                "Beta": args.beta,
                "Class Weight": args.cw,
                "SCM Threshold": args.scm_thresh,
                "Regularisation Threshold": args.reg_thresh,
                "Regularisation Weight": args.rw,
                "Z Regularisation Weight": args.zrw,
                "Seed": args.seed,
            }
            # "tags"
        )

    for k in sorted(vars(args)):
        logger.info(f"--{k}={vars(args)[k]}")
    logger.info(f"total params: {sum(p.numel() for p in model.parameters()):,}")

    def run_epoch(dloaders: List[DataLoader], epoch: int, scm_thresh: List[int] = [0,1], reg_thresh: List[int] = [0,1], training: bool = True):
        assert reg_thresh[0] >= scm_thresh[0]
        scm_weight = np.clip((epoch+1-scm_thresh[0])/(scm_thresh[1]-scm_thresh[0]), 0, 1)
        reg_weight = np.clip((epoch+1-reg_thresh[0])/(reg_thresh[1]-reg_thresh[0]), 0, 1)
        model.train(training)
        model.zero_grad(set_to_none=True)
        stats = {k: 0 for k in ["elbo", "nll", "kl", "n", "ssl", "cls", "reg"]}
        if args.hps == "cmmnist":
            stats.update({k: 0 for k in ["fgcol", "bgcol", "thickness", "intensity", "digit"]})
            stats.update({"adv_accs": np.array([0., 0., 0., 0., 0.])})
        elif args.hps == "mimic192":
            stats.update({k: 0 for k in ["finding", "age", "sex", "race"]})
            stats.update({"adv_accs": np.array([0., 0., 0., 0.])})
        updates_skipped = 0

        mininterval = 300 if "SLURM_JOB_ID" in os.environ else 0.1

        if len(dloaders) > 1:
            loader = tqdm(
                enumerate(zip(cycle(dloaders[0]), dloaders[1])), total=sum([len(d) for d in dloaders]), mininterval=mininterval
            )
        else:
            loader = tqdm(
                enumerate(dloaders[0]), total=len(dloaders[0]), mininterval=mininterval
            )


        for i, batch in loader:
            if len(batch) == 2:
                batch1, batch2 = batch
                batch = {}
                for k,v in batch1.items():
                    batch[k] = torch.concat([v, batch2[k]], dim=0)
            batch = preprocess_batch(args, batch)
            bs = batch["x"].shape[0]
            update_stats = True

            if training:
                args.iter = i + 1 + (args.epoch - 1) * sum([len(d) for d in dloaders])
                if args.beta_warmup_steps > 0:
                    args.beta = args.beta_target * linear_warmup(
                        args.beta_warmup_steps
                    )(args.iter)

                writer.add_scalar("train/beta_kl", args.beta, args.iter)
                out = model(batch["x"], batch["pa"], use_scm=(epoch >= scm_thresh[0]), use_reg=(epoch >= reg_thresh[0]), beta=args.beta, cw=args.cw, scm_weight=scm_weight, reg_weight=reg_weight)
                loss = out["elbo"] / args.accu_steps
                cls_loss = out["cls"] / args.accu_steps

                loss.backward()

                if i % args.accu_steps == 0:  # gradient accumulation update
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                    writer.add_scalar("train/grad_norm", grad_norm, args.iter)
                    nll_nan = torch.isnan(out["nll"]).sum()
                    kl_nan = torch.isnan(out["kl"]).sum()

                    if grad_norm < args.grad_skip and nll_nan == 0 and kl_nan == 0:
                        optimizer.step()
                        scheduler.step()
                        ema.update()
                    else:
                        updates_skipped += 1
                        update_stats = False
                        logger.info(
                            f"Updates skipped: {updates_skipped}"
                            + f" - grad_norm: {grad_norm:.3f}"
                            + f" - nll_nan: {nll_nan.item()} - kl_nan: {kl_nan.item()}"
                        )


                    model.zero_grad(set_to_none=True)

                    if args.iter % args.viz_freq == 0 or (args.iter in early_evals):
                        with torch.no_grad():
                            image= write_images(args, ema.ema_model, viz_batch)
                            if args.wandb:
                                wandb.log({"Samples": wandb.Image(image)}, commit=False)

            else:
                with torch.no_grad():
                    out = ema.ema_model(batch["x"], batch["pa"], use_scm=(epoch >= scm_thresh[0]), use_reg=(epoch >= reg_thresh[0]), beta=args.beta)


            if update_stats:
                if training:
                    out["elbo"] *= args.accu_steps
                    if epoch >= reg_thresh[0]:
                        out["reg"] *= args.accu_steps
                stats["n"] += bs  # samples seen counter
                stats["elbo"] += out["elbo"].detach() * bs
                stats["nll"] += out["nll"].detach() * bs
                stats["kl"] += out["kl"].detach() * bs
                stats["ssl"] += out["ssl"].detach() * bs
                stats["cls"] += out["cls"].detach() * bs
                if args.hps == "cmmnist":
                    stats["fgcol"] += out["accs"][0].detach() * bs
                    stats["bgcol"] += out["accs"][1].detach() * bs
                    stats["thickness"] += out["accs"][2].detach() * bs
                    stats["intensity"] += out["accs"][3].detach() * bs
                    stats["digit"] += out["accs"][4].detach() * bs

                elif args.hps == "mimic192":
                    stats["finding"] += out["accs"][0].detach() * bs
                    stats["age"] += out["accs"][1].detach() * bs
                    stats["sex"] += out["accs"][2].detach() * bs
                    stats["race"] += out["accs"][3].detach() * bs
                if epoch >= reg_thresh[0]:
                    stats["reg"] += out["reg"].detach() * bs
                    stats["adv_accs"] += np.array(torch.tensor([x.detach() * bs for x in out["adv_accs"]]).cpu())


            split = "train" if training else "valid"
            if stats["n"] == 0:
                stats["n"] += 0.001

            loader.set_description(
                f' => {split} | nelbo: {stats["elbo"] / stats["n"]:.3f}'
                + f' - nll: {stats["nll"] / stats["n"]:.3f}'
                + f' - kl: {stats["kl"] / stats["n"]:.3f}'
                + f' - ssl: {stats["ssl"] / stats["n"]:.3f}'
                + f' - cls: {stats["cls"] / stats["n"]:.3f}'
                + f' - reg: {stats["reg"] / stats["n"]:.3f}'
                + f" - lr: {scheduler.get_last_lr()[0]:.6g}"
                + (f" - grad norm: {grad_norm:.2f}" if training else ""),
                refresh=False,
            )

        return {k: v / stats["n"] for k, v in stats.items() if k != "n"}

    if args.beta_warmup_steps > 0:
        args.beta_target = copy.deepcopy(args.beta)

    viz_batch = next(iter(dataloaders["valid"]))
    n = min(args.context_dim * 5, args.bs)
    viz_batch = {k: v[:n] for k, v in viz_batch.items()}
    # expand pa to input res, used for HVAE parent concatenation
    args.expand_pa = args.vae == "hierarchical"
    viz_batch = preprocess_batch(args, viz_batch)
    early_evals = set([args.iter + 1] + [args.iter + 2**n for n in range(3, 14)])

    # Start training loop
    for epoch in range(args.start_epoch, args.epochs):
        args.epoch = epoch + 1
        logger.info(f"Epoch {args.epoch}:")


        if "cmmnist" in args.hps:
            if epoch < args.scm_thresh[0]:
                stats = run_epoch([dataloaders["train_begin"]], epoch=epoch, scm_thresh=args.scm_thresh,reg_thresh=args.reg_thresh, training=True)

            else:
                stats = run_epoch([dataloaders["train"]], epoch=epoch, scm_thresh=args.scm_thresh,reg_thresh=args.reg_thresh, training=True)


        elif "mimic" in args.hps:
            if epoch < args.scm_thresh[0]:
                if args.random:
                    stats = run_epoch([dataloaders["train"]], epoch=epoch, scm_thresh=args.scm_thresh,reg_thresh=args.reg_thresh, training=True)
                else:
                    stats = run_epoch([dataloaders["train_lab"]], epoch=epoch, scm_thresh=args.scm_thresh,reg_thresh=args.reg_thresh, training=True)

            else:
                if args.random:
                    stats = run_epoch([dataloaders["train"]], epoch=epoch, scm_thresh=args.scm_thresh,reg_thresh=args.reg_thresh, training=True)
                else:
                    stats = run_epoch([dataloaders["train_lab"], dataloaders["train_unlab"]], epoch=epoch,scm_thresh=args.scm_thresh, reg_thresh=args.reg_thresh, training=True)


        writer.add_scalar(f"nelbo/train", stats["elbo"], args.epoch)
        writer.add_scalar(f"nll/train", stats["nll"], args.epoch)
        writer.add_scalar(f"kl/train", stats["kl"], args.epoch)
        writer.add_scalar(f"ssl/train", stats["ssl"], args.epoch)
        writer.add_scalar(f"cls/train", stats["cls"], args.epoch)
        writer.add_scalar(f"reg/train", stats["reg"], args.epoch)

        logger.info(
            f'=> train | nelbo: {stats["elbo"]:.4f}'
            + f' - nll: {stats["nll"]:.4f} - kl: {stats["kl"]:.4f}'
            + f' - ssl: {stats["ssl"]:.4f} - ssl: {stats["ssl"]:.4f}'
            + f' - cls: {stats["cls"]:.4f} - cls: {stats["cls"]:.4f}'
            + f' - reg: {stats["reg"]:.4f} - reg: {stats["reg"]:.4f}'
            + f" - steps: {args.iter}"
        )

        if (args.epoch - 1) % args.eval_freq == 0:
            valid_stats = run_epoch([dataloaders["valid"]], epoch=epoch, scm_thresh=args.scm_thresh, reg_thresh=args.reg_thresh, training=False)

            writer.add_scalar(f"nelbo/valid", valid_stats["elbo"], args.epoch)
            writer.add_scalar(f"nll/valid", valid_stats["nll"], args.epoch)
            writer.add_scalar(f"kl/valid", valid_stats["kl"], args.epoch)
            writer.add_scalar(f"ssl/valid", stats["ssl"], args.epoch)
            writer.add_scalar(f"cls/valid", stats["cls"], args.epoch)
            writer.add_scalar(f"reg/valid", stats["reg"], args.epoch)

            logger.info(
                f'=> valid | nelbo: {valid_stats["elbo"]:.4f}'
                + f' - nll: {valid_stats["nll"]:.4f} - kl: {valid_stats["kl"]:.4f}'
                + f' - ssl: {valid_stats["ssl"]:.4f} - ssl: {valid_stats["ssl"]:.4f}'
                + f' - cls: {valid_stats["cls"]:.4f} - cls: {valid_stats["cls"]:.4f}'
                + f' - reg: {valid_stats["reg"]:.4f} - cls: {valid_stats["reg"]:.4f}'
                + f" - steps: {args.iter}"
            )

        if epoch == args.scm_thresh[0]:
            args.best_loss = float("inf")

        if epoch == args.reg_thresh[0]:
            args.best_loss = float("inf")

        if valid_stats["elbo"] < args.best_loss:
            args.best_loss = valid_stats["elbo"]
            if args.random:
                save_dict = {
                    "epoch": args.epoch,
                    "step": args.epoch * len(dataloaders["train"]),
                    "best_loss": args.best_loss.item(),
                    "model_state_dict": model.state_dict(),
                    "ema_model_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "hparams": vars(args),
                }
            else:
                save_dict = {
                    "epoch": args.epoch,
                    "step": args.epoch * len(dataloaders["train_lab"]),
                    "best_loss": args.best_loss.item(),
                    "model_state_dict": model.state_dict(),
                    "ema_model_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "hparams": vars(args),
                }
            ckpt_path = os.path.join(args.save_dir, f'checkpoint.pt')
            torch.save(save_dict, ckpt_path)
            logger.info(f"Model saved: {ckpt_path}")

        if args.wandb:
            if args.hps == "cmmnist":
                wandb.log({'Train/nelbo': stats['elbo'],
                           'Train/nll': stats['nll'],
                           'Train/kl': stats['kl'],
                           'Train/ssl': stats['ssl'],
                           'Train/class': stats['cls'],
                           'Train/fgcol': stats['fgcol'],
                           'Train/bgcol': stats['bgcol'],
                           'Train/thickness': stats['thickness'],
                           'Train/intensity': stats['intensity'],
                           'Train/digit': stats['digit'],
                           'Validation/nelbo': valid_stats['elbo'],
                           'Validation/nll': valid_stats['nll'],
                           'Validation/kl': valid_stats['kl'],
                           'Validation/ssl': valid_stats['ssl'],
                           'Validation/class': valid_stats['cls'],
                           'Validation/fgcol': valid_stats['fgcol'],
                           'Validation/bgcol': valid_stats['bgcol'],
                           'Validation/thickness': valid_stats['thickness'],
                           'Validation/intensity': valid_stats['intensity'],
                           'Validation/digit': valid_stats['digit'],
                           }, commit=True)

            if args.hps == "mimic192":
                wandb.log({'Train/nelbo': stats['elbo'],
                           'Train/nll': stats['nll'],
                           'Train/kl': stats['kl'],
                           'Train/ssl': stats['ssl'],
                           'Train/class': stats['cls'],
                           'Train/finding': stats['finding'],
                           'Train/age': stats['age'],
                           'Train/sex': stats['sex'],
                           'Train/race': stats['race'],
                           'Validation/nelbo': valid_stats['elbo'],
                           'Validation/nll': valid_stats['nll'],
                           'Validation/kl': valid_stats['kl'],
                           'Validation/ssl': valid_stats['ssl'],
                           'Validation/class': valid_stats['cls'],
                           'Validation/finding': valid_stats['finding'],
                           'Validation/age': valid_stats['age'],
                           'Validation/sex': valid_stats['sex'],
                           'Validation/race': valid_stats['race'],
                           }, commit=True)

    wandb.finish()
    return
