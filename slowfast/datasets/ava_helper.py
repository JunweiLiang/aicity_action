#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
from collections import defaultdict

from slowfast.utils.env import pathmgr

logger = logging.getLogger(__name__)

FPS = 30
AVA_VALID_FRAMES = range(902, 1799)


def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    # train.csv
    #original_vido_id video_id frame_id path labels
    #-5KQ66BBWC4 0 0 -5KQ66BBWC4/-5KQ66BBWC4_000001.jpg ""
    list_filenames = [
        os.path.join(cfg.AVA.FRAME_LIST_DIR, filename)
        for filename in (
            cfg.AVA.TRAIN_LISTS if is_train else cfg.AVA.TEST_LISTS
        )
    ]
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = [] # index -> video_name
    for list_filename in list_filenames:
        with pathmgr.open(list_filename, "r") as f:
            f.readline()
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                #logger.info(row)
                assert len(row) == 5
                video_name = row[0]

                # so assuming the framelist is not ordered
                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)  # 0-indexed
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                # 0 -
                data_key = video_name_to_idx[video_name]

                image_paths[data_key].append(
                    os.path.join(cfg.AVA.FRAME_DIR, row[3])
                )

    # frames might not be ordered
    image_paths = [image_paths[i] for i in range(len(image_paths))]

    logger.info(
        "Finished loading image paths from: %s" % ", ".join(list_filenames)
    )


    return image_paths, video_idx_to_name


def load_boxes_and_labels(cfg, mode, load_prop=False):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
        load_prop: load proposal file as well, should be one prop per person box
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    #  ["ava_train_v2.2.csv"]
    gt_lists = cfg.AVA.TRAIN_GT_BOX_LISTS if mode == "train" else []
    # train:
    #   ava_train_v2.2.csv
    #   person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv

    # -5KQ66BBWC4,0902,0.077,0.151,0.283,0.811,80,1
    # videoname,frame_idx, tlbr, label(1-80 action class), person_track_id
    # label could be -1, as a negatives
    # person_track_id is only in gt, in detection boxes, it is for box confidence

    pred_lists = (
        cfg.AVA.TRAIN_PREDICT_BOX_LISTS
        if mode == "train"
        else cfg.AVA.TEST_PREDICT_BOX_LISTS
    )
    ann_filenames = [
        os.path.join(cfg.AVA.ANNOTATION_DIR, filename)
        for filename in gt_lists + pred_lists
    ]
    prop_filenames = None
    if load_prop:
        prop_filenames = [
            os.path.join(cfg.AVA.ANNOTATION_DIR, filename + ".prop.csv")
            for filename in gt_lists + pred_lists
        ]

    # for train, there is one gt and two pred list
    # for val, there is one pred list
    ann_is_gt_box = [True] * len(gt_lists) + [False] * len(pred_lists)

    detect_thresh = cfg.AVA.DETECTION_SCORE_THRESH
    # Only select frame_sec % 4 = 0 samples for validation if not
    # set FULL_TEST_ON_VAL.
    boxes_sample_rate = (
        4 if mode == "val" and not cfg.AVA.FULL_TEST_ON_VAL else 1
        #8 # for debugging
    )
    all_boxes, count, unique_box_count = parse_bboxes_file(
        ann_filenames=ann_filenames,
        ann_is_gt_box=ann_is_gt_box,
        detect_thresh=detect_thresh,
        boxes_sample_rate=boxes_sample_rate,
        prop_filenames=prop_filenames,
    )
    logger.info(
        "Finished loading annotations from: %s" % ", ".join(ann_filenames)
    )
    logger.info("Detection threshold: {}".format(detect_thresh))
    # box filtering based on %.3f box coordinates
    logger.info("Number of unique boxes: %d" % unique_box_count)

    logger.info("Number of annotations: %d" % count)

    return all_boxes


def get_keyframe_data(boxes_and_labels, use_prop=False):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec, is_ava=True):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """
        # 900 since AVA takes movie clip starting from 15 minutes to 30 minutes
        # ori video has fps dist (9): ['24', '25', '29.97', '23.25', '29.94', '30', '23.98', '24.83', '23.97']
        if is_ava:
            return (sec - 900) * FPS
        else:
            return sec * FPS

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0

    for video_idx in range(len(boxes_and_labels)):
        sec_idx = 0  # more like the sample idx for each video
        keyframe_boxes_and_labels.append([])

        # frame_sec, from 900 -> 1700
        for sec in boxes_and_labels[video_idx].keys():
            if use_prop:
                # with prop, the data will be
                # box_and_labels[video_name][frame_sec][box_key] = [tlbr_i, box_i_labels, is_ava, prop_tlbr_i]

                for box_key in boxes_and_labels[video_idx][sec].keys():
                    is_ava = boxes_and_labels[video_idx][sec][box_key][2]
                    if is_ava and sec not in AVA_VALID_FRAMES:
                        continue

                    keyframe_indices.append(
                        # the last one is the actual frame_idx in image_path
                        (video_idx, sec_idx, sec, sec_to_frame(sec, is_ava))
                    )
                    # all the boxes
                    keyframe_boxes_and_labels[video_idx].append(
                        boxes_and_labels[video_idx][sec][box_key]
                    )
                    sec_idx += 1

            else:
                # each video each keyframe -> a list of box and action labels
                # check whether this video comes from AVA:
                is_ava = boxes_and_labels[video_idx][sec][0][2]
                if is_ava and sec not in AVA_VALID_FRAMES:
                    continue

                # a list of [tlbr, action_label_ids, is_ava, prop_i]
                if len(boxes_and_labels[video_idx][sec]) > 0:
                    keyframe_indices.append(
                        # the last one is the actual frame_idx in image_path
                        (video_idx, sec_idx, sec, sec_to_frame(sec, is_ava))
                    )
                    # all the boxes
                    keyframe_boxes_and_labels[video_idx].append(
                        boxes_and_labels[video_idx][sec]
                    )
                sec_idx += 1
            count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count


def parse_bboxes_file(
    ann_filenames, ann_is_gt_box, detect_thresh, boxes_sample_rate=1,
    prop_filenames=None
):
    """
    Parse AVA bounding boxes files.
    Args:
        ann_filenames (list of str(s)): a list of AVA bounding boxes annotation files.
        ann_is_gt_box (list of bools): a list of boolean to indicate whether the corresponding
            ann_file is ground-truth. `ann_is_gt_box[i]` correspond to `ann_filenames[i]`.
        detect_thresh (float): threshold for accepting predicted boxes, range [0, 1].
        boxes_sample_rate (int): sample rate for test bounding boxes. Get 1 every `boxes_sample_rate`.
    """

    # -5KQ66BBWC4,0902,0.077,0.151,0.283,0.811,80,1
    # videoname,frame_sec_idx, tlbr, label(1-80 action class), person_track_id
    # label could be -1, as a negatives
    # person_track_id is only in gt, in detection boxes, it is for box confidence


    all_boxes = {}
    count = 0
    unique_box_count = 0
    for file_i, (filename, is_gt_box) in enumerate(zip(ann_filenames, ann_is_gt_box)):
        with pathmgr.open(filename, "r") as f:
            if prop_filenames is not None:
                prop_lines = open(prop_filenames[file_i]).readlines()
            line_count = 0

            for line in f:
                row = line.strip().split(",")
                # When we use predicted boxes to train/eval, we need to
                # ignore the boxes whose scores are below the threshold.
                if not is_gt_box:
                    score = float(row[7]) # gt has person_track_id, which is >= 1
                    if score < detect_thresh:
                        continue



                # frame_sec is an int, since we only use one frame per second
                # ava frame_sec is 0902, AVA-kinetics is a single int
                is_ava = len(row[1]) == 4
                #assert len(row[1]) == 4 or len(row[1]) == 1
                video_name, frame_sec = row[0], int(row[1])
                # 01/12/2022, video_name could be a pathname/video_name (no mp4)

                if frame_sec % boxes_sample_rate != 0:  # for val, only 1 in 4
                    continue

                this_prop = None
                if prop_filenames is not None:
                    # prop Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
                    this_prop = [float(p) for p in prop_lines[line_count].strip().split(",")]

                # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.

                # box_key is the %.3f coordinates
                box_key = ",".join(row[2:6])
                box = list(map(float, row[2:6]))
                label = -1 if row[6] == "" else int(row[6])

                if video_name not in all_boxes:
                    all_boxes[video_name] = {}

                    # we may include AVA-Kinetics frames, which is not within 900-1700
                    #for sec in AVA_VALID_FRAMES:
                    #    all_boxes[video_name][sec] = {}
                if frame_sec not in all_boxes[video_name]:
                    all_boxes[video_name][frame_sec] = {}

                if box_key not in all_boxes[video_name][frame_sec]:
                    all_boxes[video_name][frame_sec][box_key] = [box, [], is_ava, this_prop]
                    unique_box_count += 1

                all_boxes[video_name][frame_sec][box_key][1].append(label)
                if label != -1:
                    count += 1

                line_count += 1

    if prop_filenames is not None:
        # total sample size == num_unique boxes
        pass
    else:
        # total sample size == num_keyframes
        for video_name in all_boxes.keys():
            for frame_sec in all_boxes[video_name].keys():
                # Save in format of a list of [box_i, box_i_labels, is_ava, [prop_i]].
                all_boxes[video_name][frame_sec] = list(
                    all_boxes[video_name][frame_sec].values()
                )

    return all_boxes, count, unique_box_count
