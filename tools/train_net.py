# coding=utf-8

"""Train a video classification model."""


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pprint
import torch
import datetime
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)

from tqdm import tqdm

import os

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
    #print(data_size)
    #sys.exit()

    if cfg.MIXUP.ENABLE:

        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
            is_multi_head=cfg.MODEL.USE_MULTI_HEAD,
        )
    # overall_iters = total_sample / full_batch_size
    # each inputs is per_gpu_batch_size
    disable_tqdm = not cfg.USE_TQDM
    if not du.is_master_proc():
        disable_tqdm = True

    for cur_iter, (inputs, labels, _, meta) in tqdm(enumerate(train_loader),
                                                    disable=disable_tqdm,
                                                    total=train_meter.overall_iters):
        # inputs are video frames, and meta inlcudes bounding boxes for detection
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, dict):
                for k in labels:
                    labels[k] = labels[k].cuda(non_blocking=True)
            else:
                labels = labels.cuda(non_blocking=True)
            # TODO: some keys might not be useful
            #print(meta)
            #{'label_mask': {'kinetics': tensor([0., 0., 0.]), 'mmit': tensor([1., 1., 1.]), 'activitynet': tensor([0., 0., 0.])}}
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                elif isinstance(val, (dict,)):
                    for k in val:
                        val[k] = val[k].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        # Update the learning rate.
        # data_size: number of batch per epoch
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        # MULTI_PROJ_TRAIN_DIFF_LR: use a fix lr for cross_dataset_heads
        optim.set_lr(optimizer, lr, skip_last_group=cfg.MODEL.MULTI_PROJ_TRAIN_DIFF_LR)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            # if labels were [B], will be expanded to one-hot in mixup function
            #  [B, C, T, H, W]
            # mixing up or cut mix inputs within batch
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        # this could reduce GPU memory usage but training time is slower on 1080 TI
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            emb = None
            if cfg.DETECTION.ENABLE and not cfg.DETECTION.USE_CUBE_PROP:
                # N input clips, more boxes
                preds = model(inputs, meta["boxes"])
            else:
                # run_cross_proj for multi_head training
                preds = model(inputs,
                              run_cross_proj=cfg.MODEL.MULTI_ADD_CROSS_PROJ,
                              use_moco=cfg.MODEL.MULTI_USE_MOCO,
                              moco_momentum=cfg.MODEL.MULTI_MOCO_MOMENTUM)

                if cfg.MODEL.USE_VICREG_LOSS:
                    preds, emb = preds


            if cfg.TRAIN.GATHER_BEFORE_LOSS and cfg.NUM_GPUS > 1:
                # block till all good
                # [local_batch_size, D] -> [global_batch_size, D]
                # all_gather the labels as well
                # labels [global_batch_size]
                # junwei: don't use dist.all_gather, will not get any gradient
                # this function concat the input tensor as part of the global ones
                preds, emb, labels = du.all_gather_cat_self(
                    [preds, emb, labels])
                if cfg.MODEL.USE_MULTI_HEAD:
                    meta["label_masks"], = du.all_gather_cat_self([meta["label_masks"]])


            if cfg.MODEL.USE_VICREG_LOSS:
                assert cfg.TRAIN.GATHER_BEFORE_LOSS
                # scalar
                vicreg_loss = losses.compute_vicreg_loss(
                    emb, std_weight=1.0, cov_weight=1.0)  # vicreg paper is 25/1
                vicreg_loss *= cfg.MODEL.VICREG_LOSS_WEIGHT
            else:
                vicreg_loss = 0.0

            # junwei: added multi-dataset/head ,
            if cfg.MODEL.USE_MULTI_HEAD:
                # given the preds of {dataset_name: [B, num_classes]}
                # and labels of the same {dataset_name: [B] / [B, num_classes]}
                # label masks are also {dataset_name: [B], each is 1/0}
                loss_weight_dict = {
                    cfg.MODEL.MULTI_DATASETS[i]: cfg.MODEL.MULTI_LOSS_WEIGHTS[i]
                    for i in range(len(cfg.MODEL.MULTI_DATASETS))}
                loss = losses.compute_multi_dataset_loss(
                    preds, labels, meta["label_masks"],
                    cfg.MODEL.MULTI_DATASETS, cfg.MODEL.MULTI_LOSS_FUNCS,
                    loss_weight_dict,
                    add_cross_proj=cfg.MODEL.MULTI_ADD_CROSS_PROJ,
                    cross_proj_add_to_pred=cfg.MODEL.MULTI_CROSS_PROJ_ADD_TO_PRED,
                    proj_loss_func=cfg.MODEL.MULTI_PROJ_LOSS_FUNC,
                    proj_loss_weight=cfg.MODEL.MULTI_PROJ_LOSS_WEIGHT)

                # L2 norm is already apply to almost all params in optimizer through
                # weight decay
                # here we add L1/L0 norm to the multi-dataset projection layer,
                # to encourage sparsity
                if cfg.MODEL.MULTI_PROJ_SPARSITY_LOSS_TYPE:
                    m = model.module if cfg.NUM_GPUS > 1 else model
                    proj_reg_loss_type = cfg.MODEL.MULTI_PROJ_SPARSITY_LOSS_TYPE
                    assert proj_reg_loss_type in ["L1", "L0"]
                    reg_loss_weight = cfg.MODEL.MULTI_PROJ_SPARSITY_WEIGHT
                    all_proj_params = [p.view(-1)
                                       for proj_name in m.head.cross_dataset_heads.keys()
                                       for p in m.head.cross_dataset_heads[proj_name].parameters()]
                    all_proj_params = torch.cat(all_proj_params)
                    reg_loss = reg_loss_weight \
                        * torch.norm(all_proj_params, int(proj_reg_loss_type.lstrip("L")))
                    loss += reg_loss

            else:
                # single dataset forward
                # Explicitly declare reduction to mean.
                loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(
                    reduction="mean"
                )

                # Compute the loss.
                # torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.

                if cfg.TRAIN.MIXED_PRECISION and cfg.MODEL.LOSS_FUNC == "bce":
                    # this seems to make AVA training slower:
                    # on taiji, without mixed_precision is 15% faster than with it
                    # But Kinetics training without this chunk, it is 10% faster with it
                    with torch.cuda.amp.autocast(enabled=False):
                        # preds are float16, and labels are float32, so cast preds back to float32
                        loss = loss_fun(preds.float(), labels)
                else:
                    loss = loss_fun(preds, labels)

            loss += vicreg_loss  # zero if not used

        # check Nan Loss.
        if misc.check_nan_losses(loss):
            raise RuntimeError(
                "ERROR: Got NaN losses. Try disable mixed precision training")
            #logger.info("nan loss encountered for process rank %s" % (du.get_rank()))

        # Perform the backward pass.
        # TODO: try set_to_none=True? lower GPU memory?
        # https://pytorch.org/docs/master/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad
        optimizer.zero_grad()
        # TODO(junwei): check scaler use when the loss is already gathered
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
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

        if cfg.MIXUP.ENABLE and not cfg.MODEL.USE_MULTI_HEAD:
            # merge back the prediction and labels for stats only
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        top1_err, top5_err = None, None
        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)

        else:

            if cfg.DATA.MULTI_LABEL or cfg.MODEL.USE_MULTI_HEAD:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors. for this mini_batch
                # TODO (junwei): make this support multi-dataset training
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,  # loss of this global mini-batch # will add to a moving average queue
                lr,
                inputs[0].size(0)  # batch_size
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
            write_scalars["Train/Top1_err"] = top1_err
            write_scalars["Train/Top5_err"] = top5_err
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


def gather_multi_dataset(dict_tensors):
    # {dataset_name: tensor}
    dataset_names = sorted(list(dict_tensors.keys()))
    gathered = du.all_gather([dict_tensors[dataset_name] for dataset_name in dataset_names])
    output = {dataset_name: gathered[i] for i, dataset_name in enumerate(dataset_names)}
    return output


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None, train_size=None):
    """
    Evaluate the model on the val set.
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
    for cur_iter, (inputs, labels, _, meta) in tqdm(enumerate(val_loader),
                                                    disable=disable_tqdm,
                                                    total=val_meter.overall_iters):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, dict):
                for k in labels:
                    labels[k] = labels[k].cuda(non_blocking=True)
            else:
                labels = labels.cuda(non_blocking=True)
            # TODO: some keys might not be useful
            #print(meta)
            #{'label_mask': {'kinetics': tensor([0., 0., 0.]), 'mmit': tensor([1., 1., 1.]), 'activitynet': tensor([0., 0., 0.])}}
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                elif isinstance(val, (dict,)):
                    for k in val:
                        val[k] = val[k].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            # [N, T, H, W, 3] -> [M] predictions of M boxes in this N keyframes
            if cfg.DETECTION.USE_CUBE_PROP:
                # N sized mini-batch to N boxes
                preds = model(inputs)
            else:
                preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                # this will block till all GPU is done
                # all_gather the same thing in the nccl group, block until
                # all got, and run .cpu()
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:

            preds = model(inputs)
            if cfg.MODEL.USE_VICREG_LOSS:
                preds, emb = preds
            label_masks = None
            if cfg.DATA.MULTI_LABEL:

                if cfg.NUM_GPUS > 1:
                    if cfg.MODEL.USE_MULTI_HEAD:
                        # preds are {dataset_name: tensors}
                        preds = gather_multi_dataset(preds)
                        labels = gather_multi_dataset(labels)
                        meta["label_masks"] = gather_multi_dataset(meta["label_masks"])

                    else:
                        preds, labels = du.all_gather([preds, labels])

                if cfg.MODEL.USE_MULTI_HEAD:
                    label_masks = meta["label_masks"]

            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )

            val_meter.update_predictions(preds, labels, label_masks)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    # could be mAP for AVA, and top-5 err for Kinetics
    mAP = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map},
                #global_step=cur_epoch,
                # set to global steps as in training, so we could see loss and val together?
                global_step=train_size * (cur_epoch + 1),  # end of an epoch
            )
        else:
            """ # this is used to plot confusion matrix
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels,
                #global_step=cur_epoch,
                # set to global steps as in training, so we could see loss and val together?
                global_step=train_size * (cur_epoch + 1),  # end of an epoch
            )
            """
            if cfg.DATA.MULTI_LABEL:
                if cfg.MODEL.USE_MULTI_HEAD:
                    writer.add_scalars(
                        {
                            "Val/Top-1_Acc": val_meter.last_epoch_top1_acc,
                            "Val/Top-5_Acc": val_meter.last_epoch_top5_acc
                        },
                        #global_step=cur_epoch,
                        # set to global steps as in training, so we could see loss and val together?
                        global_step=train_size * (cur_epoch + 1),  # end of an epoch
                    )
                    # also collect the top1/5 for each dataset
                    for dataset in val_meter.last_epoch_per_dataset_acc:
                        writer.add_scalars(
                            {
                                ("Val/%s/Top-1_Acc" % dataset): val_meter.last_epoch_per_dataset_acc[dataset]["top1_acc"],
                                ("Val/%s/Top-5_Acc" % dataset): val_meter.last_epoch_per_dataset_acc[dataset]["top5_acc"],
                            },
                            #global_step=cur_epoch,
                            # set to global steps as in training, so we could see loss and val together?
                            global_step=train_size * (cur_epoch + 1),  # end of an epoch
                        )
                else:
                    writer.add_scalars(
                        {"Val/mAP": val_meter.last_epoch_map},
                        #global_step=cur_epoch,
                        # set to global steps as in training, so we could see loss and val together?
                        global_step=train_size * (cur_epoch + 1),  # end of an epoch
                    )
            else:
                writer.add_scalars(
                    {
                        "Val/Top-1_Acc": val_meter.last_epoch_top1_acc,
                        "Val/Top-5_Acc": val_meter.last_epoch_top5_acc
                    },
                    #global_step=cur_epoch,
                    # set to global steps as in training, so we could see loss and val together?
                    global_step=train_size * (cur_epoch + 1),  # end of an epoch
                )

    val_meter.reset()

    return mAP


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def load_cross_proj_weight(cfg, model, weight_file_dir, is_master_proc=False):

    # the weight file should be a npz dict of float types
    #print(model.head.cross_dataset_heads)
    """
    # the weight tensor should be of shape (400, 305)
    {'mit_kinetics': Linear(in_features=305, out_features=400, bias=False),
    'ssv2_kinetics': Linear(in_features=174, out_features=400, bias=False),
    'activitynet_kinetics': Linear(in_features=200, out_features=400, bias=False),
    'kinetics_mit': Linear(in_features=400, out_features=305, bias=False),
    'ssv2_mit': Linear(in_features=174, out_features=305, bias=False),
    'activitynet_mit': Linear(in_features=200, out_features=305, bias=False),
    'kinetics_ssv2': Linear(in_features=400, out_features=174, bias=False),
    'mit_ssv2': Linear(in_features=305, out_features=174, bias=False),
    'activitynet_ssv2': Linear(in_features=200, out_features=174, bias=False),
    'kinetics_activitynet': Linear(in_features=400, out_features=200, bias=False),
    'mit_activitynet': Linear(in_features=305, out_features=200, bias=False),
    'ssv2_activitynet': Linear(in_features=174, out_features=200, bias=False)}

    a = np.zeros((400, 305), dtype="float")
    model.head.cross_dataset_heads["mit_kinetics"].load_state_dict({
        "weight": torch.Tensor(a)
        })
    sys.exit()
    """
    datasets = cfg.MODEL.MULTI_DATASETS
    datasets_num_class = cfg.MODEL.MULTI_NUM_CLASSES
    if "kinetics" in datasets:
        kinetics_num_class = datasets_num_class[datasets.index("kinetics")]
    if cfg.NUM_GPUS > 1:
        model = model.module

    for proj_name in model.head.cross_dataset_heads.keys():
        # the projection file should be saved d1_d2.npy files in the path

        # kinetics -> kinetics400/kinetics/700
        d1_name, d2_name = proj_name.split("_")
        if "kinetics" == d1_name:
            d1_name = "kinetics%d" % kinetics_num_class
        if "kinetics" == d2_name:
            d2_name = "kinetics%d" % kinetics_num_class

        weight_file = os.path.join(weight_file_dir, "%s_%s.npy" % (d1_name, d2_name))

        weights = np.load(weight_file)
        if is_master_proc:
            print("loaded projection weight for %s, shape %s, summed to be %s" % (
                proj_name, weights.shape, np.sum(weights)))
        model.head.cross_dataset_heads[proj_name].load_state_dict({
            "weight": torch.Tensor(weights)
        })



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

    if cfg.MODEL.USE_MULTI_HEAD:
        assert cfg.TRAIN.DATASET in ["multi_dataset_seq"]
        assert cfg.DATA.MULTI_LABEL

    # Set up environment.
    # set _LOCAL_PROCESS_GROUP
    # get how many machine we are dealing with, and the rank for each
    # what is torch.distributed.new_group() use for?
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)  # same seed under the same script
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None

    # Print config., only in the master process
    if cfg.LOG_CFG:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    if cfg.AVA.USE_LABEL_SMOOTHING:
        logger.info("Train with label smoothing for AVA.")

    # Build the video model and print model statistics.
    model = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    # every process will load from the checkpoint
    # return the epoch num for that checkpoint
    # the logging is suppressed for non-master process
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    if cfg.MODEL.MULTI_ADD_CROSS_PROJ and cfg.MODEL.LOAD_MULTI_PROJ_INIT_FILE:
        # only use this when it is the first epoch
        if start_epoch == 0:
            # load the cross_dataset projection weights
            load_cross_proj_weight(
                cfg, model, cfg.MODEL.LOAD_MULTI_PROJ_INIT_FILE, du.is_master_proc())
            if du.is_master_proc():
                print("loaded cross-dataset projection weights.")

    if cfg.MODEL.MULTI_ADD_CROSS_PROJ and cfg.MODEL.MULTI_FIX_PROJ:
        # set the projection layer's weight to be fixed
        m = model.module if cfg.NUM_GPUS > 1 else model
        for proj_name in m.head.cross_dataset_heads.keys():
            m.head.cross_dataset_heads[proj_name].require_grad = False

        if du.is_master_proc():
            print("set cross-dataset project head to require_grad to be false")

    # Create the video train and val loaders.
    # every process has one data loader?
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

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
    eval_mAPs = []  # (epoch_idx, mAP/top-5-err)
    if cfg.TRAIN.EVAL_FIRST:
        logger.info("Eval with previous checkpoint first ..")
        mAP = eval_epoch(
            val_loader, model, val_meter, -1, cfg, writer,
            train_size=len(train_loader))
        eval_mAPs.append((-1, mAP))
        print("mAP/Top5_err: %s" % mAP)

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

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

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
            mAP = eval_epoch(
                val_loader, model, val_meter, cur_epoch, cfg, writer,
                train_size=len(train_loader))
            eval_mAPs.append((cur_epoch + 1, mAP))
            if du.is_master_proc():
                print("current mAPs(top-5 err): %s" % eval_mAPs)

    if du.is_master_proc():
        print("mAPs(top-5 err) of all evaluation:")
        print("  ".join([("%d, %.5f" % (e, ap)) for e, ap in eval_mAPs]))
        print("model path: %s" % os.path.abspath(cfg.OUTPUT_DIR))
    if writer is not None:
        writer.close()
