# coding=utf-8

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import time
import datetime

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter

from tqdm import tqdm

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    # junweil: compute total running time in the outer loop
    start_time = time.time()
    disable_tqdm = not cfg.USE_TQDM
    if not du.is_master_proc():
        disable_tqdm = True
    for cur_iter, (inputs, labels, video_idx, meta) in tqdm(
            enumerate(test_loader), total=test_meter.overall_iters,
            disable=disable_tqdm):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            if cfg.DETECTION.USE_CUBE_PROP:
                # N sized mini-batch to N boxes
                preds = model(inputs)
            else:
                preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                # block till all gather
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.

            if cfg.MODEL.USE_MULTI_HEAD:
                inf_dataset = cfg.TEST.DATASET
                if cfg.DATA.MODEL_DIFF_DATA:
                    inf_dataset = cfg.TRAIN.DATASET
                preds = model(inputs, dataset_name=inf_dataset)
                preds = preds[inf_dataset]
            else:
                preds = model(inputs)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather(
                    [preds, labels, video_idx]
                )
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            # gathered all the prediction for this batch

            test_meter.iter_toc()
            # Update and log stats.
            # will perform the multi clip summation in test_meter
            # labels: [N] or [N, num_class]
            # preds: [N, num_class]
            # collect the prediction output for each vid by sum or max
            # to test_meter.video_preds[vid] -> (num_class)
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        # (num_videos, num_model_cls) tensor
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        #if writer is not None:
        #    writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.ENABLE_SAVE:
            # we only save the predictions
            if du.is_master_proc():
                # whole dataset prediction in one file
                # (N, num_model_cls)
                # the order is same as the annotation csv
                np.save(cfg.TEST.SAVE_RESULTS_PATH, all_preds.numpy())

            logger.info("saved predicted results at %s" % cfg.TEST.SAVE_RESULTS_PATH)

    # if not testing on the same dataset, no need to compute metrics
    if not cfg.DATA.MODEL_DIFF_DATA:
        test_meter.finalize_metrics()
    logger.info("total run time: %s" % (
        str(datetime.timedelta(seconds=int(time.time() - start_time)))))
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    if cfg.MODEL.USE_MULTI_HEAD:
        assert cfg.TEST.DATASET not in ["multi_dataset_seq"], \
            "Please test each dataset separately"

    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    if not cfg.TEST.NO_LOG_CONFIG:
        logger.info("Test with config:")
        logger.info(cfg)

    if cfg.TEST.ENABLE_SAVE:
        assert cfg.TEST.SAVE_RESULTS_PATH, "Please provide result path"
        # save one prediction file to this

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        # no need, we use center crop testing so input would be the same
        #assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            # test the model on different trained dataset
            cfg.MODEL.NUM_CLASSES,
            cfg.DATA.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MODEL_DIFF_DATA,
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
