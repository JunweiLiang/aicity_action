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
from slowfast.utils.meters import ContrastiveTestMeter
import slowfast.utils.metrics as metrics


from tqdm import tqdm

logger = logging.get_logger(__name__)


def put_vars_to_cuda(frames, labels, tokens, video_idxs, meta):
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


    video_idxs = video_idxs.cuda(non_blocking=True)

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
    return frames, labels, tokens, video_idxs, meta


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, test_dataset):
    """

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

    # junweil: compute total running time in the outer loop
    start_time = time.time()
    disable_tqdm = not cfg.USE_TQDM
    if not du.is_master_proc():
        disable_tqdm = True


    # save all the vectors in RAM/GPU
    all_video_features, all_text_features = [], []
    all_labels = []

    for cur_iter, (frames, labels, tokens, video_idxs, meta) in tqdm(enumerate(test_loader),
                                                    disable=disable_tqdm,
                                                    total=len(test_loader)):

        # labels are unique id for entire text
        if cfg.NUM_GPUS:
            frames, labels, tokens, video_idxs, meta = put_vars_to_cuda(
                frames, labels, tokens, video_idxs, meta)

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
            video_features, text_features, labels, video_idxs = du.all_gather(
                [video_features, text_features, labels, video_idxs])

        all_video_features.append(video_features)


        all_text_features.append(text_features)
        all_labels.append(labels)

        test_meter.log_iter_stats(cur_iter)

    # matmul
    logger.info("matmul of (%d,%d) ^2 ..." % (
        sum([len(o) for o in all_video_features]), len(all_video_features[0][0])))

    # [test_num, 512]
    video_features = torch.cat(all_video_features, dim=0)

    if cfg.TEST.ENABLE_SAVE:
        # save the video features into a path
        if du.is_master_proc():

            np.save(
                cfg.TEST.SAVE_RESULTS_PATH,
                video_features.detach().cpu().numpy())

    text_features = torch.cat(all_text_features, dim=0)
    # [test_num]   # each is the unique text ID
    labels = torch.cat(all_labels)

    # [test_num, test_num]
    logits_per_video = video_features @ text_features.t()
    logits_per_video = logits_per_video.detach().cpu()
    logits_per_text = logits_per_video.t()

    # make a gt of [B, B]
    gt_per_video = metrics.make_contrastive_minibatch_gt(labels)

    gt_per_text = gt_per_video.t()

    # compute t2v, v2t recall@1/5 within this mini-batch
    v2t_recalls = metrics.compute_recall_at_rank(
        logits_per_video, gt_per_video, [1, 5, 10])
    t2v_recalls = metrics.compute_recall_at_rank(
        logits_per_text, gt_per_text, [1, 5, 10])

    logger.info("v2t recall@1/5/10: %s" % v2t_recalls)
    logger.info("t2v recall@1/5/10: %s" % t2v_recalls)

    logger.info("total run time: %s" % (
        str(datetime.timedelta(seconds=int(time.time() - start_time)))))
    return None


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

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

    # enable saving the results
    # save all the test video's features to npy files
    # video_file.npy (wear.1712.mp4.npy)
    if cfg.TEST.ENABLE_SAVE:
        assert cfg.TEST.SAVE_RESULTS_PATH, "Please provide result path"
        # save all the video features into one file
        # (10k x 512) would only be 27 MB
        #if du.is_master_proc():
        #    os.makedirs(cfg.TEST.SAVE_RESULTS_PATH, exist_ok=True)
        #    print("Enable saving prediction results to %s..." % cfg.TEST.SAVE_RESULTS_PATH)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        # TODO: make this work
        #misc.log_model_info(model, cfg, use_train_input=False)
        print(model)

    cu.load_test_checkpoint(cfg, model)

    assert cfg.TEST.NUM_ENSEMBLE_VIEWS == 1 == cfg.TEST.NUM_SPATIAL_CROPS, \
        "Currently only support single video per sample test!"

    # Create video testing loaders.
    test_loader, test_dataset = loader.construct_loader(
        cfg, "test", return_dataset=True)
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    test_meter = ContrastiveTestMeter(len(test_loader), cfg)

    assert (
        test_loader.dataset.num_videos
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, test_meter, cfg, test_dataset)
