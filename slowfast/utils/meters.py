#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import slowfast.datasets.ava_helper as ava_helper
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.utils.distributed as du
from slowfast.utils.ava_eval_helper import (
    evaluate_ava,
    read_csv,
    read_exclusions,
    read_labelmap,
)

logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):

        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret


class AVAMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
        self.mode = mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []
        self.overall_iters = overall_iters
        self.only_cat_ids = cfg.AVA.yewu_related
        self.excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
        )
        gt_filename = os.path.join(
            cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE
        )
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)
        self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)
        _, self.video_idx_to_name = ava_helper.load_image_lists(
            cfg, mode == "train"
        )
        if cfg.AVA.IS_TEST_ON_KINETICS:
            #assert cfg.AVA.LOAD_FROM_VIDEO
            """
            # load the video frame count file
            self._video_name_to_frame_count = {}
            with open(self.video_frame_count_file) as f:
                for line in f:
                    video_name, num = line.strip().split(",")
                    num = int(num)
                    self._video_name_to_frame_count[video_name] = num
            """
            boxes_and_labels = ava_helper.load_boxes_and_labels(
                cfg, mode=mode
            )
            # all video will be used
            video_names = list(boxes_and_labels.keys())
            self.video_idx_to_name = {i: video_names[i] for i in range(len(video_names))}

        self.output_dir = cfg.OUTPUT_DIR

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        # junwei: removed the eta/timer stuff since they are not moving averaged,

        #eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        #eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                #"eta": eta,
                #"dt": self.iter_timer.seconds(),
                #"dt_data": self.data_timer.seconds(),
                #"dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                "loss": self.loss.get_win_median(),

                "lr": self.lr,
                # junwei: (added to show better estimate)
                "overall_iters": self.overall_iters,
                "time_diff": self.iter_timer.seconds(),
            }
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                #"eta": eta,
                #"dt": self.iter_timer.seconds(),
                #"dt_data": self.data_timer.seconds(),
                #"dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                # junwei: (added to show better estimate)
                "overall_iters": self.overall_iters,
                "time_diff": self.iter_timer.seconds(),
            }
        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                #"eta": eta,
                #"dt": self.iter_timer.seconds(),
                #"dt_data": self.data_timer.seconds(),
                #"dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                # junwei: (added to show better estimate)
                "overall_iters": self.overall_iters,
                "time_diff": self.iter_timer.seconds(),
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        if du.is_master_proc():
            logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()

        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []

    def update_stats(self, preds, ori_boxes, metadata, loss=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            ori_boxes (tensor): original boxes (x1, y1, x2, y2).
            metadata (tensor): metadata of the AVA data.
            loss (float): loss value.
            lr (float): learning rate.
        """
        if self.mode in ["val", "test"]:
            # M box pred for N sized mini-batch
            # with prop, preds and ori_box will be 1-to-1 so N sized batch has N box
            self.all_preds.append(preds)
            self.all_ori_boxes.append(ori_boxes)
            self.all_metadata.append(metadata)
        if loss is not None:
            self.loss.add_value(loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, log=True):
        """
        Calculate and log the final AVA metrics.
        """
        # all the boxes in the dataset (boxes are unique)
        all_preds = torch.cat(self.all_preds, dim=0)
        all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
        all_metadata = torch.cat(self.all_metadata, dim=0)

        # ground truth is (video_id, keyframe) -> a list of boxes and labels
        # the boxes are not unique
        if self.mode == "test" or (self.full_ava_test and self.mode == "val"):
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth

        # this returns the category specified in defaults.py's label file
        # 80 or 60
        self.full_map = evaluate_ava(
            all_preds,
            all_ori_boxes,
            all_metadata.tolist(),
            self.excluded_keys,
            self.class_whitelist,
            self.categories, # 1 indexed
            groundtruth=groundtruth,
            video_idx_to_name=self.video_idx_to_name,
            only_cat_ids=self.only_cat_ids,
        )
        if log:
            stats = {"mode": self.mode, "map": self.full_map}
            logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            if du.is_master_proc():  # only main process run eval
                self.finalize_metrics(log=False)
                stats = {
                    "_type": "{}_epoch".format(self.mode),
                    "cur_epoch": "{}".format(cur_epoch + 1),
                    "mode": self.mode,
                    "map": self.full_map,
                    "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                    "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
                }

                logging.log_json_stats(stats)

                return self.full_map
        return None


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_model_cls,
        num_data_cls,
        overall_iters,
        model_diff_data=False,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method

        # we could test the model with different dataset
        if not model_diff_data:

            num_data_cls = num_model_cls

        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_model_cls))

        # junwei: no sure why the original PySlowFast use this
        # log(-1e10) ~= 0?
        #if multi_label:
        #    self.video_preds -= 1e10

        self.video_labels = (
            torch.zeros((num_videos, num_data_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        #if self.multi_label:
        #    self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                ), "for multi_head model, remember to turn off DATA.MULTI_LABEL if necessary"
            # save the label
            self.video_labels[vid_id] = labels[ind]
            # save the prediction
            # default to be sum for AVA/Kinetics/mmit/Activitynet/SSv2
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            # junwei: (added to show better estimate)
            "overall_iters": self.overall_iters,
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        # junwei: silenced this guy, use tqdm for better eta with moving average

        #logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} != num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                            if k != self.num_clips
                        ]
                    ),
                    self.num_clips,
                )
            )

        # self.video_labels and self.video_preds are [N, num_class]
        self.stats = {"split": "test_final"}
        if self.multi_label:
            # for each ks, accuracy, compute top-1/top-5 as well
            topks_correct = metrics.topks_correct_full_label(
                self.video_preds, self.video_labels, ks=ks, no_stacking=True)
            for k, topk in zip(ks, topks_correct):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk * 100., prec=2
                )
            # compute map
            map = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            self.stats["map"] = map

        else:
            # labels are [N], each is an int
            num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        if du.is_master_proc():
            logging.log_json_stats(self.stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.overall_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        # TODO: wrap the timer with moving average recorder
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # for moving averages
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size. (global)
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL and not self._cfg.CONTRA.ENABLE:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        # junwei: remove the non-moving average estimates, as they are not accurate

        #eta_sec = self.iter_timer.seconds() * (
        #    self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        #)
        #eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            # junwei: removed, not useful if not moving averaged
            #"dt": self.iter_timer.seconds(),
            #"dt_data": self.data_timer.seconds(),
            #"dt_net": self.net_timer.seconds(),
            #"eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            # junwei: (added to show better estimate)
            #"overall_iters": self.overall_iters,
            #"time_diff": self.iter_timer.seconds(),
        }
        if not self._cfg.DATA.MULTI_LABEL and not self._cfg.CONTRA.ENABLE:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        if du.is_master_proc():
            logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        #eta_sec = self.iter_timer.seconds() * (
        #    self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        #)
        #eta = str(datetime.timedelta(seconds=int(eta_sec)))
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            # junwei: removed, not useful if not moving averaged
            #"dt": self.iter_timer.seconds(),
            #"dt_data": self.data_timer.seconds(),
            #"dt_net": self.net_timer.seconds(),
            #"eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),

        }
        stats["loss"] = avg_loss

        if not self._cfg.DATA.MULTI_LABEL and not self._cfg.CONTRA.ENABLE:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err

        if du.is_master_proc():
            logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.overall_iters = max_iter
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        # for multi-dataset
        # {dataset_name: a list}
        self.all_preds_multi_datasets = defaultdict(list)
        self.all_labels_multi_datasets = defaultdict(list)

        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        # for multi-dataset
        # {dataset_name: a list}
        self.all_preds_multi_datasets = defaultdict(list)
        self.all_labels_multi_datasets = defaultdict(list)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def update_predictions(self, preds, labels, label_masks=None):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """

        if self._cfg.MODEL.USE_MULTI_HEAD:
            # given the preds of {dataset_name: [B, num_classes]}
            # and labels of the same {dataset_name: [B] / [B, num_classes]}
            # label masks are also {dataset_name: [B], each is 1/0}

            # gather only the sample for each dataset
            dataset_names = preds.keys()
            for dataset_name in dataset_names:
                this_preds = preds[dataset_name]
                this_labels = labels[dataset_name]
                this_mask = label_masks[dataset_name]
                for i in range(len(this_preds)):
                    if this_mask[i] == 1.:
                        self.all_preds_multi_datasets[dataset_name].append(this_preds[i])
                        self.all_labels_multi_datasets[dataset_name].append(this_labels[i])
                        """
                        if dataset_name == "kinetics":
                            print(torch.argsort(this_preds[i], descending=True)[:10])
                            print(torch.argmax(this_labels[i]))
                            sys.exit()
                        """

        else:
            # TODO: merge update_prediction with update_stats.
            self.all_preds.append(preds)
            self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        #eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        #eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            #"time_diff": self.iter_timer.seconds(),
            #"eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            # junwei: (added to show better estimate)
            #"overall_iters": self.overall_iters,
        }
        if not self._cfg.DATA.MULTI_LABEL and not self._cfg.CONTRA.ENABLE:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        if du.is_master_proc():
            logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            #"time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            if self._cfg.MODEL.USE_MULTI_HEAD:
                # compute the top-1/5 for each dataset
                # for multi-label dataset, top-5 might be useful
                ks = [1, 5]
                ks_list = [[] for k in ks]  # will average over all dataset
                self.last_epoch_per_dataset_acc = {}
                for dataset_name in self.all_preds_multi_datasets:
                    # a list of [num_class]
                    pred_list = self.all_preds_multi_datasets[dataset_name]
                    # a list of [num_class]
                    label_list = self.all_labels_multi_datasets[dataset_name]

                    """
                    if dataset_name == "kinetics":
                        for i in range(5):
                            print(torch.argsort(pred_list[i], descending=True)[:5])
                            #print(torch.argmax(label_list[i]))
                            print((label_list[i] == 1.0).nonzero(as_tuple=False))
                        sys.exit()
                    """

                    # for each ks, accuracy
                    topks_correct = metrics.topks_correct_full_label(
                        pred_list, label_list, ks=ks)

                    self.last_epoch_per_dataset_acc[dataset_name] = {
                        ("top%d_acc" % k): acc
                        for k, acc in zip(ks, topks_correct)}

                    for i in range(len(ks)):
                        ks_list[i].append(topks_correct[i])

                stats["top1_avg_acc"] = float(np.mean(ks_list[0]))
                stats["top5_avg_acc"] = float(np.mean(ks_list[1]))

                stats["dataset_topk"] = self.last_epoch_per_dataset_acc

                # record these for tensorboard to read
                self.last_epoch_top1_acc = stats["top1_avg_acc"]
                self.last_epoch_top5_acc = stats["top5_avg_acc"]

                eval_result = stats["top5_avg_acc"]
            else:
                stats["map"] = get_map(
                    torch.cat(self.all_preds).cpu().numpy(),
                    torch.cat(self.all_labels).cpu().numpy(),
                )
                self.last_epoch_map = stats["map"]
                eval_result = stats["map"]
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            # minimum error across all previous epochs
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)

            # record these for tensorboard to read
            # top1_err is err_rate * 100.0
            self.last_epoch_top1_acc = 100. - top1_err
            self.last_epoch_top5_acc = 100. - top5_err

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err
            eval_result = stats["top5_err"]

        if du.is_master_proc():
            logging.log_json_stats(stats)

        return eval_result

class ContrastiveValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.overall_iters = max_iter
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.t2v_mb_r1 = ScalarMeter(cfg.LOG_PERIOD)
        self.t2v_mb_r5= ScalarMeter(cfg.LOG_PERIOD)
        self.v2t_mb_r1 = ScalarMeter(cfg.LOG_PERIOD)
        self.v2t_mb_r5= ScalarMeter(cfg.LOG_PERIOD)
        self.all_t2v_r5 = []
        self.all_v2t_r5 = []

        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.t2v_mb_r1.reset()
        self.t2v_mb_r5.reset()
        self.v2t_mb_r1.reset()
        self.v2t_mb_r5.reset()
        self.all_t2v_r5 = []
        self.all_v2t_r5 = []


    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, t2v_r1, t2v_r5, v2t_r1, v2t_r5):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.t2v_mb_r1.add_value(t2v_r1)
        self.t2v_mb_r5.add_value(t2v_r5)
        self.v2t_mb_r1.add_value(v2t_r1)
        self.v2t_mb_r5.add_value(v2t_r5)
        self.all_v2t_r5.append(v2t_r5)
        self.all_t2v_r5.append(t2v_r5)


    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        #eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        #eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            #"time_diff": self.iter_timer.seconds(),
            #"eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            # junwei: (added to show better estimate)
            #"overall_iters": self.overall_iters,
        }
        stats["v2t_r1_moving"] = self.v2t_mb_r1.get_win_median()
        stats["v2t_r5_moving"] = self.v2t_mb_r5.get_win_median()
        stats["t2v_r1_moving"] = self.t2v_mb_r1.get_win_median()
        stats["t2v_r5_moving"] = self.t2v_mb_r5.get_win_median()
        if du.is_master_proc():
            logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            #"time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        v2t_r5_averaged = np.mean(self.all_v2t_r5)
        t2v_r5_averaged = np.mean(self.all_t2v_r5)


        stats["v2t_recall@5"] = v2t_r5_averaged
        stats["t2v_recall@5"] = t2v_r5_averaged
        eval_result = np.mean([v2t_r5_averaged, t2v_r5_averaged])

        if du.is_master_proc():
            logging.log_json_stats(stats)

        return eval_result

class ContrastiveTestMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.overall_iters = max_iter

        # currently this is just used for memory logging
        # TODO: put the computation in test_net_contrastive.py in here


    def log_iter_stats(self, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        #eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        #eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if du.is_master_proc():
            logging.log_json_stats(stats)


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    # every process will print this
    #logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)
