#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch
import os

from . import ava_helper as ava_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from . import decoder as decoder
from .build import DATASET_REGISTRY

from . import video_container as container

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Ava(torch.utils.data.Dataset):
    """
    AVA Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.AVA.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP

        # junwei: added label smoothing, to be used with bce loss, soft targets
        self.use_label_smoothing = cfg.AVA.USE_LABEL_SMOOTHING
        self.label_smoothing_eps = cfg.AVA.LABEL_SMOOTHING_EPS
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            # default false
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.DATA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.DATA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP

        # added by junwei
        self.is_load_from_video = cfg.AVA.LOAD_FROM_VIDEO
        if self.is_load_from_video:
            assert os.path.exists(cfg.AVA.VIDEO_PATH), "please provide video path"

        self.video_frame_count_file = None
        self._video_name_to_frame_count = None
        if cfg.AVA.ADD_KINETICS:
            self.video_frame_count_file = os.path.join(
                cfg.AVA.ANNOTATION_DIR, cfg.AVA.KINETICS_VIDEO_FRAME_COUNT)
            assert os.path.exists(self.video_frame_count_file), \
                "we need additional video frame count file: %s" % self.video_frame_count_file
        self._num_retries = 5


        # junwei: added support for cube proposal training method, removing ROI
        self.use_cube_prop = cfg.DETECTION.USE_CUBE_PROP

        self._load_data(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.


        # read from frame_lists/train.csv or (val/test uses) val.csv
        # image_path, each video -> a list of path to images
        # video_idx_to_name, 0-len(videos)-1, -> ori_vid_name, idx -> image_path
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = ava_helper.load_image_lists(cfg, is_train=(self._split == "train"))

        # for loading from videos each time
        # assume all video ended .mp4
        # and assume they are all 30fps
        self._video_paths = [os.path.join(cfg.AVA.VIDEO_PATH, self._video_idx_to_name[idx] + ".mp4")
                             for idx in range(len(self._video_idx_to_name))]

        # the above may include only AVA videos
        # TODO(junwei): fix this

        # Loading annotations for boxes and labels.
        # TRAIN_GT_BOX_LISTS
        # TRAIN_PREDICT_BOX_LISTS / TEST_PREDICT_BOX_LISTS
        # read from annotations/
        #   ava_train_v2.2.csv
        #   person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv
        # -5KQ66BBWC4,0902,0.077,0.151,0.283,0.811,80,1
        # videoname,frame_sec_idx, tlbr, label(1-80 action class), person_track_id
        # label could be -1, as a negatives
        # person_track_id is only in gt, in detection boxes, it is for box confidence

        # box_and_labels[video_name][frame_sec] = a list of [tlbr_i, box_i_labels, is_ava]
        # or with proposals:
        #   box_and_labels[video_name][frame_sec][box_key] = [tlbr_i, box_i_labels, is_ava, prop_tlbr_i]
        # frame_sec is an int indicating the seconds in the video
        # each frame_sec has all the boxes, each box has a list of action labels

        boxes_and_labels = ava_helper.load_boxes_and_labels(
            cfg, mode=self._split, load_prop=self.use_cube_prop,
        )

        # temporary fix
        # we can test with AVA val
        need_avakinetics_video = False
        if self._split == "test":
            if cfg.AVA.IS_TEST_ON_KINETICS and cfg.AVA.LOAD_FROM_VIDEO:
                need_avakinetics_video = True
        elif cfg.AVA.ADD_KINETICS and cfg.AVA.LOAD_FROM_VIDEO:  # train/val
            need_avakinetics_video = True

        if need_avakinetics_video:
            assert cfg.AVA.LOAD_FROM_VIDEO
            # load the video frame count file
            self._video_name_to_frame_count = {}
            with open(self.video_frame_count_file) as f:
                for line in f:
                    video_name, num = line.strip().split(",")
                    num = int(num)
                    self._video_name_to_frame_count[video_name] = num

            # all video will be used
            video_names = list(boxes_and_labels.keys())
            self._video_idx_to_name = {i: video_names[i] for i in range(len(video_names))}
            self._video_paths = [os.path.join(cfg.AVA.VIDEO_PATH, self._video_idx_to_name[idx] + ".mp4")
                             for idx in range(len(self._video_idx_to_name))]
            self._image_paths = self._video_paths
        elif cfg.AVA.ADD_KINETICS and not cfg.AVA.LOAD_FROM_VIDEO:
            # use ava-kinetics frames for training and testing
            # still need the frame count
            # load the video frame count file
            self._video_name_to_frame_count = {}
            with open(self.video_frame_count_file) as f:
                for line in f:
                    # video_name could include / , multiple levels
                    video_name, num = line.strip().split(",")
                    num = int(num)
                    self._video_name_to_frame_count[video_name] = num

            # all video will be used
            video_names = list(boxes_and_labels.keys())
            self._video_idx_to_name = {i: video_names[i] for i in range(len(video_names))}
            self._image_paths = []
            for idx in range(len(video_names)):
                this_video_frame_paths = []
                # this takes a lot of memory
                # TODO: change this for AVA as well and remove frame_list input
                """
                for frame_num in range(self._video_name_to_frame_count[self._video_idx_to_name[idx]]):
                    img_path = os.path.join(
                        cfg.AVA.FRAME_DIR,
                        self._video_idx_to_name[idx],
                        "%s_%06d.jpg" % (self._video_idx_to_name[idx], frame_num + 1))
                    this_video_frame_paths.append(img_path)
                """

                #self._image_paths.append(this_video_frame_paths)
                frame_num = self._video_name_to_frame_count[self._video_idx_to_name[idx]]
                video_name = self._video_idx_to_name[idx]
                self._image_paths.append((video_name, frame_num))

        # num_videos
        assert len(boxes_and_labels) == len(self._image_paths)

        # video_name -> {frame_sec: boxes}
        # with prop , each item would be {frame_sec:box_key -> boxes}
        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        # keyframe_indices, a list of
        # (video_idx, sec_idx, sec, actual_frameIdx_within_video)
        # keyframe_boxes_and_labels[video_idx][sec_idx] -> a list of [tlbr, labels] for this frame

        # whether to filter data based on  AVA valid frames
        #filter_ava = True
        #if self._split == "train":
        #    filter_ava = not cfg.AVA.ADD_KINETICS
        # now we filter automatically based on ava frame_sec is 4-digit int

        # the len of the dataset should be self._keyframe_indices,
        # each keyframe might have different number of boxes (avg ~3 box per frame)
        # for using prop, each like is video -> keyframe -> box_key
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels, use_prop=self.use_cube_prop)

        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        logger.info("=== AVA dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes=None):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape


        if boxes is not None:
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
            boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

            # `transform.py` is list of np.array. However, for AVA, we only have
            # one np.array.
            boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            # scale range: [256, 320]
            # the video's shorter edge will be scale to [256, 320], and the other
            # side is scaled accordingly
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            # random crop a 224x224 crop
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )
            # default true
            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            # this could be 224 x -1 or -1 x 224 size
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            if boxes is not None:
                boxes = [
                    cv2_transform.scale_boxes(
                        self._crop_size, boxes[0], height, width
                    )
                ]
            # crop a 224x224 from -1 x 224, etc.
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            # this could be 224 x -1 or -1 x 224 size
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            if boxes is not None:
                boxes = [
                    cv2_transform.scale_boxes(
                        self._crop_size, boxes[0], height, width
                    )
                ]

            # this is needed for MViT, originally only done for val
            # since MViT need fixed size inputs
            # crop a 224x224 from -1 x 224, etc.
            # 1 means center crop
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )
        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        if boxes is not None:
            boxes = cv2_transform.clip_boxes_to_image(
                boxes[0], imgs[0].shape[1], imgs[0].shape[2]
            )
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes=None):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        if boxes is not None:
            # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
            # range of [0, 1].
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
            boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # this is needed for MViT, originally only done for val
            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        if boxes is not None:
            boxes = transform.clip_boxes_to_image(
                boxes, self._crop_size, self._crop_size
            )

        return imgs, boxes

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        # here the sec_idx is more like sample_idx to get from
        # self._keyframe_boxes_and_labels
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        # Get the frame idxs for current clip.

        # given the center_frame_idx,
        # get the frame_idx around it, will check for out-of-bound
        # will make sure to have seq_len frame_idx
        # this is zero-indexed
        if self._video_name_to_frame_count is not None:
            frame_num = self._video_name_to_frame_count[self._video_idx_to_name[video_idx]]
        else:
            if self.cfg.AVA.ADD_KINETICS:
                _, frame_num = self._image_paths[video_idx]
            else:
                frame_num = len(self._image_paths[video_idx])
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=frame_num,  # all the frames for this video
        )

        # a list of [box_i, box_i_labels, is_ava, [prop_i]].
        # for prop, it'll be just one element
        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        if self.use_cube_prop:
            clip_label_list = [clip_label_list]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        props = []
        for box_labels in clip_label_list:
            boxes.append(box_labels[0])
            labels.append(box_labels[1])
            if self.use_cube_prop:
                props.append(box_labels[3])
        boxes = np.array(boxes)
        # Score is not used.
        boxes = boxes[:, :4].copy()
        ori_boxes = boxes.copy()
        if self.use_cube_prop:
            props = np.array(props)
            boxes = None  # not using this

        # Load images of current clip.
        if self.is_load_from_video:
            for i_try in range(self._num_retries):
                video_container = None
                try:
                    video_container = container.get_video_container(
                        self._video_paths[video_idx],
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                        self.cfg.DATA.DECODING_BACKEND_GPU_ENABLE,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video from {} with error {}".format(
                             self._video_paths[video_idx], e
                        )
                    )

                if video_container is None:
                    logger.warning(
                        "Failed to meta load video idx {} from {}; trial {}".format(
                            video_idx, self._path_to_videos[video_idx], i_try
                        )
                    )
                    continue

                #assert self.cfg.DATA.DECODING_BACKEND == "decord", "currently only support decord"
                if self.cfg.DATA.DECODING_BACKEND == "decord":
                    imgs = video_container.get_batch(seq).asnumpy()
                elif self.cfg.DATA.DECODING_BACKEND == "torchvision":
                    # junwei (12/2021): this has not been tested
                    imgs = decoder.torchvision_get_frames(video_container, seq)

                if imgs is None:
                    logger.warning(
                        "Failed to decode video idx {} from {}; trial {}".format(
                            video_idx, self._video_paths[video_idx], i_try
                        )
                    )
                    continue
                if self.cfg.AVA.IMG_PROC_BACKEND == "pytorch":
                    imgs = torch.as_tensor(imgs)


                # to be consistent with cv2.imdecode used in utils.retry_load_images,
                # which gets BGR images
                # [T, H, W, C]
                imgs = imgs[:, :, :, [2, 1, 0]]  # RGB to BGR
                # the imgs will be convert back to RGB in the end
        else:
            if self.cfg.AVA.ADD_KINETICS:
                video_name, _ = self._image_paths[video_idx]
                image_paths = [os.path.join(
                        self.cfg.AVA.FRAME_DIR,
                        video_name,
                        "%s_%06d.jpg" % (
                            os.path.basename(video_name), frame_num + 1)) for frame_num in seq]
            else:
                image_paths = [self._image_paths[video_idx][frame] for frame in seq]
            # read video so that we save space, would it ever be faster than image
            # reading on clusters/network disks?
            imgs = utils.retry_load_images(
                image_paths, backend=self.cfg.AVA.IMG_PROC_BACKEND  # both backend uses cv2 to decode image
            )

        # now we have imgs of [T, H, W, C]
        # for prop exp, we directly crop the cube based on the proposal box,
        # then do the data augmentation stuff without the boxes
        if self.use_cube_prop:

            # the images list to be a tensor
            imgs = np.array(imgs) # [T, H, W, C]
            height, width = imgs.shape[1], imgs.shape[2]

            prop = props[0]  # in 0-1.0
            x1, y1, x2, y2 = [o for o in prop]
            x1, y1, x2, y2 = int(x1*width), int(y1*height), int(x2*width), int(y2*height)

            cropped_imgs = imgs[:, y1:y2+1, x1:x2+1, :]
            imgs = cropped_imgs

        if self.cfg.AVA.IMG_PROC_BACKEND == "pytorch":  # by default this is not used
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            imgs, boxes = self._images_and_boxes_preprocessing(
                imgs, boxes=boxes
            )
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                imgs, boxes=boxes
            )
            for img in imgs:
                max_val = torch.max(img)
                assert torch.isfinite(max_val)

        # Construct label arrays.
        # one video clip has multiple multiple boxes
        #label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.float32)
        if self.use_label_smoothing:
            label_arrs += self.label_smoothing_eps

        for i, box_labels in enumerate(labels):
            # AVA label index starts from 1.
            for label in box_labels:
                if label == -1:
                    continue
                assert label >= 1 and label <= 80
                label_arrs[i][label - 1] = 1 if not self.use_label_smoothing else 1 - self.label_smoothing_eps

        imgs = utils.pack_pathway_output(self.cfg, imgs)
        metadata = [[video_idx, sec]] * len(ori_boxes)

        extra_data = {
            "ori_boxes": ori_boxes,
            "metadata": metadata,
        }
        # could be None for using prop
        if boxes is not None:
            extra_data["boxes"] = boxes
        else:
            extra_data["props"] = props

        return imgs, label_arrs, idx, extra_data
