# Note - from initial testing it seems that CPU training is ~10-20x slower than
# GPU training
import logging
import multiprocessing
import time

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import modules.commons as commons
import sovits_utils
from data_utils import TextAudioSpeakerLoader, TextAudioCollate
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from modules.losses import (
    kl_loss,
    generator_loss, discriminator_loss, feature_loss
)

from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"]=""

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

def validate(hps):
    # Errors that might occur:
    # dataset/44k has more than one character (generally not preferred)
    #assert len(os.listdir('dataset/44k')) == 1, "more than one character present!!!"
    ckpt = sovits_utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
    # no pretrained model present
    assert ckpt is not None, "no pretrained model present!!!"
    # pretrained model still present - no way to check for this?
    if hps.train.fp16_run:
        raise Exception("fp16 not supported on CPU training")
    pass

def main():
    hps = sovits_utils.get_hparams()

    n_gpus = 1
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    validate(hps)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = sovits_utils.get_logger(hps.model_dir)
        logger.info(hps)
        sovits_utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # for pytorch on win, backend use gloo    
    dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    collate_fn = TextAudioCollate()
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=1, pin_memory=False,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[])  # , find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[])

    skip_optimizer = False
    try:
        _, _, _, epoch_str = sovits_utils.load_checkpoint(sovits_utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = sovits_utils.load_checkpoint(sovits_utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
        if hps.reset:
            print("Resetting model steps!!!")
            epoch_str = 1
            global_step = 0
    except:
        raise Exception("No pretrained model found")
        print("Load old checkpoint failed!!!")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d],
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d],
                               [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv = items
        g = spk
        spec, y = spec, y
        c = c
        f0 = f0
        uv = uv
        lengths = lengths
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)

        y_hat, ids_slice, z_mask, \
        (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                            spec_lengths=lengths)

        y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

        optim_d.zero_grad()
        loss_disc_all.backward()
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        optim_d.step()

        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_lf0 = F.mse_loss(pred_lf0, lf0)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0

        optim_g.zero_grad()
        loss_gen_all.backward()
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        optim_g.step()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}")

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": sovits_utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": sovits_utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": sovits_utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/lf0": sovits_utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                          pred_lf0[0, 0, :].detach().cpu().numpy()),
                    "all/norm_lf0": sovits_utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                               norm_lf0[0, 0, :].detach().cpu().numpy())
                }

                sovits_utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                sovits_utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                sovits_utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    sovits_utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv = items
            g = spk[:1]
            spec, y = spec[:1], y[:1]
            c = c[:1]
            f0 = f0[:1]
            uv= uv[:1]
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_hat = generator.module.infer(c, f0, uv, g=g)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            f"gen/mel": sovits_utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": sovits_utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    sovits_utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()
