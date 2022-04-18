# coding=utf-8
"""
Training contrastive models
"""



import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pprint
import torch
import datetime
import os

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.models import build_model

# specific for contrastive learning
from slowfast.datasets import contrastive_loader as loader
from slowfast.utils.meters import EpochTimer, TrainMeter, ContrastiveValMeter

logger = logging.get_logger(__name__)

from tqdm import tqdm



def put_vars_to_cuda(frames, labels, tokens, meta):
    if isinstance(frames, (list,)):
        for i in range(len(frames)):
            frames[i] = frames[i].cuda(non_blocking=True)
    else:
        frames = frames.cuda(non_blocking=True)

    if isinstance(tokens, (list,)):
        for i in range(len(tokens)):
            tokens[i] = tokens[i].cuda(non_blocking=True)
    else:
        tokens = tokens.cuda(non_blocking=True)

    if isinstance(labels, dict):
        for k in labels:
            labels[k] = labels[k].cuda(non_blocking=True)
    else:
        labels = labels.cuda(non_blocking=True)

    for key, val in meta.items():
        if isinstance(val, (list,)):
            for i in range(len(val)):
                val[i] = val[i].cuda(non_blocking=True)
        elif isinstance(val, (dict,)):
            for k in val:
                val[k] = val[k].cuda(non_blocking=True)
        else:
            meta[key] = val.cuda(non_blocking=True)
    return frames, labels, tokens, meta



def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)


    # each frames is per_gpu_batch_size
    disable_tqdm = not cfg.USE_TQDM
    if not du.is_master_proc():
        disable_tqdm = True


    # for example, soft target CE loss, given x , y input
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(
        reduction="mean"
    )

    for cur_iter, (frames, labels, tokens, _, meta) in tqdm(enumerate(train_loader),
                                                    disable=disable_tqdm,
                                                    total=train_meter.overall_iters):

        # labels are unique id for entire text
        # tokens are BPE ids for the text_embedding
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            frames, labels, tokens, meta = put_vars_to_cuda(
                frames, labels, tokens, meta)

        # Update the learning rate.
        # data_size: number of batch per epoch
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        # this could reduce GPU memory usage but training time is slower on 1080 TI
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):

            # given [B, 3, T, H, W] and [B, 77]
            # got img features and text features [B, emb_size]
            # l2 normed [B, emb_dim]

            if cfg.TRAIN.USE_MOCO:
                video_features, text_features, logit_scale, \
                    video_features_moco, text_features_moco = model(
                        frames, tokens,
                        use_moco=True, moco_momentum=cfg.TRAIN.MOCO_MOMENTUM)
            else:
                video_features, text_features, logit_scale = model(frames, tokens)

            # scalar/ temperture
            logit_scale = logit_scale.mean()

            if cfg.NUM_GPUS > 1:
                # block till all good
                # [local_batch_size, D] -> [global_batch_size, D]
                # all_gather the labels as well
                # labels [global_batch_size]
                # junwei: don't use dist.all_gather, will not get any gradient
                # this function concat the input tensor as part of the global ones
                video_features, text_features, labels = du.all_gather_cat_self(
                    [video_features, text_features, labels])

                if cfg.TRAIN.USE_MOCO:
                    # we don't need gradient for moco encoder
                    # but we got errors
                    video_features_moco, text_features_moco = du.all_gather(
                       [video_features_moco, text_features_moco])


            # TODO(junwei): make the big computation distributed!
            # [global_batch_size, global_batch_size]
            # here each is between 0-1.0
            if cfg.TRAIN.USE_MOCO:
                logits_per_video = logit_scale * video_features @ text_features_moco.t()
                logits_per_text = logit_scale * text_features @ video_features_moco.t()
            else:
                logits_per_video = logit_scale * video_features @ text_features.t()
                logits_per_text = logits_per_video.t()

            # make a gt of [B, B]
            gt_per_video = metrics.make_contrastive_minibatch_gt(labels)

            gt_per_text = gt_per_video.t()

            # [B, B] -> [B, B] -> scalar
            loss_video = loss_fun(logits_per_video, gt_per_video)
            loss_text = loss_fun(logits_per_text, gt_per_text)

            loss = (loss_video + loss_text) / 2.

        # check Nan Loss.
        if misc.check_nan_losses(loss):
            raise RuntimeError(
                "ERROR: Got NaN losses. Try disable mixed precision training")
            #logger.info("nan loss encountered for process rank %s" % (du.get_rank()))

        # Perform the backward pass.
        # TODO: try set_to_none=True? lower GPU memory?
        # https://pytorch.org/docs/master/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad
        optimizer.zero_grad()
        # scaler is for Mixed precision training
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        # TODO: check this, this is not used in other code base like MoCo v3
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            # 1.0 for MViT
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()


        m = model.module if cfg.NUM_GPUS > 1 else model
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        # the loss should be global
        #if cfg.NUM_GPUS > 1:
        #    loss = du.all_reduce([loss])[0]
        loss = loss.item()

        # Update and log stats.
        train_meter.update_stats(
            None,
            None,
            loss,  # loss of this global mini-batch # will add to a moving average queue
            lr,
            frames[0].size(0)  # batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

        # write to tensorboard format if available.
        if writer is not None:
            write_scalars = {
                "Train/loss": loss,
                "Train/lr": lr,
            }
            # will check None in writer
            #write_scalars["Train/Top5_err"] = top5_err
            writer.add_scalars(
                write_scalars,
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)  # here the loss is averaged over all iterations
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None, train_size=None):
    """
    Evaluate the model on the val set, using
        **average recall@1/5 within each mini-batch***
        Full test should be on the whole dataset
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    disable_tqdm = not cfg.USE_TQDM
    if not du.is_master_proc():
        disable_tqdm = True

    # for Kinetics validation, one random clip and one crop is sampled from the each validation video
    for cur_iter, (frames, labels, tokens, _, meta) in tqdm(enumerate(val_loader),
                                                    disable=disable_tqdm,
                                                    total=val_meter.overall_iters):
        if cfg.NUM_GPUS:
            frames, labels, tokens, meta = put_vars_to_cuda(
                frames, labels, tokens, meta)

        val_meter.data_toc()

        # given [B, 3, T, H, W] and [B, 77]
        # got img features and text features [B, emb_size]
        # l2 normed
        video_features, text_features, logit_scale = model(frames, tokens)

        # scalar
        logit_scale = logit_scale.mean()

        if cfg.NUM_GPUS > 1:
            # block till all good
            # [local_batch_size, D] -> [global_batch_size, D]
            # all_gather the labels as well
            # labels [B]
            video_features, text_features, labels = du.all_gather(
                [video_features, text_features, labels])

        # TODO(junwei): make the big computation distributed!
        # [global_batch_size, global_batch_size]
        # here each is between 0-1.0
        logits_per_video = logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_video.t()

        # make a gt of [B, B]
        gt_per_video = metrics.make_contrastive_minibatch_gt(labels)

        gt_per_text = gt_per_video.t()

        # compute t2v, v2t recall@1/5 within this mini-batch
        v2t_recalls = metrics.compute_recall_at_rank(
            logits_per_video, gt_per_video, [1, 5])
        t2v_recalls = metrics.compute_recall_at_rank(
            logits_per_text, gt_per_text, [1, 5])

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            t2v_recalls[0],
            t2v_recalls[1],
            v2t_recalls[0],
            v2t_recalls[1],
        )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    # average recall@5 for t2v and v2t
    avg_recall_5 = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        writer.add_scalars(
            {"Recall@5": avg_recall_5},
            #global_step=cur_epoch,
            # set to global steps as in training, so we could see loss and val together?
            global_step=train_size * (cur_epoch + 1),  # end of an epoch
        )

    val_meter.reset()

    return avg_recall_5


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # the following will increase GPU usage for bs=1 from 8.5 GB to 9.7 GB
    # useful if input size do not change?
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True


    # Set up environment.
    # set _LOCAL_PROCESS_GROUP
    # get how many machine we are dealing with, and the rank for each
    # use torch.distributed.new_group() for sub group communication?
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config., only in the master process
    if cfg.LOG_CFG:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)


    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        # TODO: make this work
        #misc.log_model_info(model, cfg, use_train_input=True)
        print(model)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    # small float points in fp16 in loss will be underflow, so need to scale up
    # then scale down for the gradient
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    # every process will load from the checkpoint
    # return the epoch num for that checkpoint
    # the logging is suppressed for non-master process
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    # every process has one data loader?
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")


    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ContrastiveValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc():
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    if du.is_master_proc():
        cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
        logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()

    eval_r5s = []  # (epoch_idx, recall@5)
    if cfg.TRAIN.EVAL_FIRST:
        logger.info("Eval with previous checkpoint first ..")
        r5 = eval_epoch(
            val_loader, model, val_meter, -1, cfg, writer,
            train_size=len(train_loader))
        eval_r5s.append((-1, r5))
        print("Recall@5: %s" % r5)

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        # the dataloader could have a DistributedSampler,
        # in this function, sampler.set_epoch is called to properly shuffle the dataset
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            writer,
        )
        epoch_timer.epoch_toc()

        if du.is_master_proc():
            num_epoch_left = cfg.SOLVER.MAX_EPOCH - cur_epoch - 1
            time_left = str(datetime.timedelta(seconds=int(num_epoch_left * epoch_timer.avg_epoch_time())))
            logger.info(
                f"Epoch {cur_epoch + 1} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
                f"from {start_epoch + 1} to {cur_epoch + 1} take "
                f"{epoch_timer.avg_epoch_time():.2f}s in average and "
                f"{epoch_timer.median_epoch_time():.2f}s in median. "
                f"Estimated time left: {time_left}."
            )
            logger.info(
                f"For epoch {cur_epoch + 1}, each iteraction takes "
                f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
                f"From epoch {start_epoch + 1} to {cur_epoch + 1}, each iteraction takes "
                f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
            )

        # will check cur_epoch + 1 % checkpoint period == 0
        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None
        )


        # Save a checkpoint.
        if is_checkp_epoch:
            # only save if it is master process
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            # val loader is also a distributed sampler loader
            r5 = eval_epoch(
                val_loader, model, val_meter, cur_epoch, cfg, writer,
                train_size=len(train_loader))
            eval_r5s.append((cur_epoch + 1, r5))
            if du.is_master_proc():
                print("current recall@5: %s" % eval_r5s)

    if du.is_master_proc():
        print("Recall @ 5 of all evaluation:")
        print("  ".join([("%d, %.5f" % (e, ap)) for e, ap in eval_r5s]))
        print("model path: %s" % os.path.abspath(cfg.OUTPUT_DIR))
    if writer is not None:
        writer.close()
