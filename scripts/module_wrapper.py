# coding=utf-8
"""Wrapper function for all sorts of action classification models that takes multiple frame inputs
"""

import os
import sys
import torch
import av
import decord
import cv2
from PIL import Image
import numpy as np
# remove all the annoying warnings from tf v1.10 to v1.13
import logging
logging.getLogger("tensorflow").disabled = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import tensorflow as tf
    from semantic_features.img2feat_utils import get_sf_feat_forward_graph
except Exception:
    pass

try:
    import onnxruntime
except Exception:
    pass


from torchvision import transforms
from torch.utils.data._utils.collate import default_collate

from utils import load_prop_file
from utils import short_edge_resize
from utils import spatial_shift_crop_list

from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model as build_pyslowfast_model
import slowfast.utils.checkpoint as pyslowfast_checkpoint



def pixel_norm(frames, mean, std, channel_first=True):
    """ Given np frames of [C, T, H, W] or [T, H, W, C], do pixel norm
        mean and std are shape=[3] array
    """
    if channel_first:
        C, T, H, W = frames.shape
        tiled_mean = np.tile(np.expand_dims(mean, axis=[1, 2, 3]), [1, T, H, W])
        tiled_std = np.tile(np.expand_dims(std, axis=[1, 2, 3]), [1, T, H, W])
    else:
        T, H, W, C = frames.shape
        tiled_mean = np.tile(np.expand_dims(mean, axis=[0, 1, 2]), [T, H, W, 1])
        tiled_std = np.tile(np.expand_dims(std, axis=[0, 1, 2]), [T, H, W, 1])

    return (frames - tiled_mean) / tiled_std

def crop_and_resize(np_frames, size_scale, crop_size, crop_tlbr=None, boxes=None,
                    keep_scale=True,
                    spatial_sample_index=1):
    """ Given the numpy frame, crop the tlbr out, and resize to size_scale and keeping ratio
        tlbr means the box to crop from np_frames before resizing, boxes are the
        ones within the image
        Args:
            size_scale: short_edge size
            crop_size: crop size
            crop_tlbr: used to crop np_frames at first
            boxes: the box associated with np_frames, need to change as frames change
            spatial_sample_index, 0: left (top), 1: center, 2: right (bottom) crop
    """
    if crop_tlbr is not None:
        # avoid negative numbers, possible bugs
        left, top, right, bottom = [max(int(o), 0) for o in crop_tlbr]
        # [T, H, W, C]
        # here the out-of-frame errors will be ignored and crop the largest possible
        # but we need to check for zero width and height

        cropped_frames = np_frames[:, top:bottom+1, left:right+1,  :]
        height, width = cropped_frames.shape[1:3]
        if height == 0 or width == 0:
            raise Exception("got zero size crop: %s, crop_tlbr: %s" % (
                cropped_frames.shape, crop_tlbr))
        if boxes is not None:
            # make the box local
            new_boxes = []
            for l, t, r, b in boxes:
                new_boxes.append([l - left, t - top, r - left, b - top])
            boxes = np.array(new_boxes)
    else:
        cropped_frames = np_frames

    #print(cropped_frames.shape)
    # (32, 631, 269, 3) -> 32, (600, 256, 3)  # (resizing)
    # (32, 304, 134, 3) -> 32, (580, 256, 3)

    cropped_resized_frames, _ = short_edge_resize(
        cropped_frames,
        size=size_scale,
        boxes=boxes,
        keep_scale=keep_scale)

    #print([c.shape for c in cropped_resized_frames], len(cropped_resized_frames))
    # 32, (600, 256, 3) -> 32, (256, 256, 3) # (center cropping)
    cropped_resized_frames, _ = spatial_shift_crop_list(
          crop_size, cropped_resized_frames, spatial_sample_index, boxes=boxes)
    #print([c.shape for c in cropped_resized_frames], len(cropped_resized_frames))
    #sys.exit()
    return cropped_resized_frames, boxes


def collate_rois(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    framess, boxess, prop_idss = zip(*batch)
    framess = default_collate(framess)
    #print(len(framess), len(prop_idss))  # 2, 10
    #print(framess[0].shape) # [N, C, T, H, W]
    #print(prop_idss[0].shape) # [1, 4]
    # put all prop_idss into one dim [M]
    prop_idss = [
        one_idx
        for i in range(len(prop_idss))
        for one_idx in prop_idss[i][0]  # [N][1, 4]
    ]
    prop_idss = np.array(prop_idss, dtype="int32")  # no need to be torch tensor

    # Append idx info to the bboxes before concatenating them.
    # Given a list of [K, 4], get a list of [K, 5] with batch_idx
    # [K, 4] concat [K, 1]
    boxess = [
        np.concatenate(
            [np.full((boxess[i].shape[0], 1), float(i)), boxess[i]], axis=1
        )
        for i in range(len(boxess))
    ]
    # [M, 5]
    boxess = np.concatenate(boxess, axis=0)
    boxess = torch.tensor(boxess).float()
    return framess, boxess, prop_idss

def collate_rois_prop(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    framess, boxess, track_idss = zip(*batch)
    framess = default_collate(framess)
    track_idss = default_collate(track_idss)
    #print(len(framess), len(prop_idss))  # 2, 10
    #print(framess[0].shape) # [N, C, T, H, W]
    #print(prop_idss[0].shape) # [1, 4]
    # put all prop_idss into one dim [M]

    # Append idx info to the bboxes before concatenating them.
    # Given a list of [K, 4], get a list of [K, 5] with batch_idx
    # [K, 4] concat [K, 1]
    boxess = [
        np.concatenate(
            [np.full((boxess[i].shape[0], 1), float(i)), boxess[i]], axis=1
        )
        for i in range(len(boxess))
    ]
    # [M, 5]
    boxess = np.concatenate(boxess, axis=0)
    boxess = torch.tensor(boxess).float()
    return framess, boxess, track_idss

class ActionProposalFromVideoTemporalDataset(torch.utils.data.Dataset):
    """ A dataset object to serve the proposals directly given the video file
    """
    def __init__(self, video_path, args):
        self.video_file = video_path

        self.video_num_frame = None  # number of frames for the opened video
        self.video_fps = None

        self.frame_format = args.frame_format
        self.video_reader_name = args.video_decoder

        assert args.video_decoder in ["decord"], "only decord is supported"


        self.frame_length = args.frame_length
        self.frame_stride = args.frame_stride
        self.frame_size = args.frame_size


        self.model_type = args.model_type

        self.video_cap = self.open()  # pointer of the opened video file

        # get a list of [tlbr, start_frame_idx, end_frame_idx]
        # relative
        frame_height, frame_width = self.video_cap[0].shape[:2]
        self.roi_tlbr = [
            frame_width * args.roi_x1,
            frame_height * args.roi_y1,
            frame_width * args.roi_x2,
            frame_height * args.roi_y2]

        # assuming the proposal length in arguments is based on args.target_fps
        # we convert the proposal length to target_fps based on video_fps
        proposal_length = self.frame_length * self.frame_stride
        proposal_stride = args.proposal_stride
        if abs(args.video_fps - args.target_fps) > 2.0:  # so 29.97 is fine to consider as 30
            if args.video_fps <= 0.0:
                video_fps = self.video_cap.get_avg_fps()
                if video_fps is None:
                    video_fps = 30.0
            else:
                video_fps = args.video_fps

            convert_frame_length_rate = video_fps / args.target_fps
            proposal_length = int(convert_frame_length_rate * proposal_length)
            proposal_stride = int(convert_frame_length_rate * proposal_stride)

            print("warning, video %s converted proposal length from %s to %s, fps %.1f (video) -> %.1f" % (
                os.path.basename(video_path),
                args.proposal_length, proposal_length,
                video_fps, args.target_fps))

        self.prop_data = self._get_proposals(
            self.roi_tlbr,
            proposal_length,
            proposal_stride,
            len(self.video_cap))

        # in RGB
        self.color_mean = np.array([0.45, 0.45, 0.45], dtype="float32")
        self.color_std = np.array([0.225, 0.225, 0.225], dtype="float32")

        self.video_cap = None  # we need to open it everytime, for pyTorch dataloader

    def _get_proposals(self, roi_tlbr, prop_length, prop_stride, num_frames):
        props = []
        for i in range(0, num_frames, prop_stride):
            t0 = i
            t1 = i + prop_length
            prop = [roi_tlbr, t0, t1]
            props.append(prop)
        return props


    def open(self):
        """Open the video file and extract the metadatas."""
        if self.video_reader_name == "opencv":

            video_cap = cv2.VideoCapture(self.video_file)
            if not video_cap.isOpened():
                raise Exception("Opencv cannot open %s" % self.video_file)

            # opencv 3/4
            self.video_num_frame = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = float(video_cap.get(cv2.CAP_PROP_FPS))

        elif self.video_reader_name == "pyav":
            # pyav will throw exception if file cannot be read
            video_cap = av.open(self.video_file)

            self.video_num_frame = int(video_cap.streams.video[0].frames)
            self.video_fps = float(video_cap.streams.video[0].average_rate)

        elif self.video_reader_name == "decord":

            # num_threads=0 means auto
            video_cap = decord.VideoReader(
                self.video_file, num_threads=0)#, ctx=decord.cpu(0))

            self.video_num_frame = len(video_cap)
            self.video_fps = float(video_cap.get_avg_fps())
        return video_cap

    def _read_frames_by_idxs(self, frame_idxs, video_cap):

        np_frames = []
        if self.video_reader_name == "decord":
            """
            for frame_idx in frame_idxs:
                # in rgb
                np_frame = video_cap[frame_idx].asnumpy()
            """
            np_frames = video_cap.get_batch(frame_idxs).asnumpy()
            if self.frame_format == "bgr":
                new_frames = []
                for np_frame in np_frames:
                    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
                    new_frames.append(np_frame)
                np_frames = np.array(new_frames)

        return np_frames

    def __getitem__(self, idx):
        """Get the idx's proposal, crop all video frames and return
        """
        if self.video_cap is None:
            video_cap = self.open()
        else:
            video_cap = self.video_cap

        tlbr, start_frame_idx, end_frame_idx = self.prop_data[idx]


        # junwei: 02/23/2022, proposal could takes variable fps videos, so
        # the start_frame -> end_frame could have less than the frames classification
        # model required. We will get num_frame from [start_frame_idx, end_frame_idx]
        # by uniform duplicating indices
        frame_idxs_to_get = self._get_frame_idxs_uniform(
            start_frame_idx, end_frame_idx, self.frame_length)

        # [T, H, W, C]
        np_frames = self._read_frames_by_idxs(frame_idxs_to_get, video_cap)
        # get the proposal cubes
        # this will became float32
        np_frames, _ = crop_and_resize(
            np_frames, self.frame_size, self.frame_size, crop_tlbr=tlbr,
            boxes=None,
            keep_scale=False)


        # pyslowfast way

        # [0, 255] -> [0, 1]
        np_frames = np.array(np_frames, dtype="float32")
        np_frames /= 255.

        # [T, H, W, C] -> [C, T, H, W]
        # pyslowfast uses channel first for efficiency
        np_frames = np_frames.transpose([3, 0, 1, 2])
        np_frames = pixel_norm(np_frames, self.color_mean, self.color_std, channel_first=True)



        # [C, T, H, W] / [T, H, W, C]
        transformed_frames = np.ascontiguousarray(np_frames)

        if self.model_type == "slowfast":
            fast_pathway = transformed_frames
            # Perform temporal sampling from the fast pathway.
            slowfast_alpha = 4
            """
            slow_pathway = torch.index_select(
                np_frames,
                1,
                torch.linspace(
                    0, np_frames.shape[1] - 1, np_frames.shape[1] // slowfast_alpha
                ).long(),
            )
            """
            T = transformed_frames.shape[1]
            frame_indexs = torch.linspace(0, T-1, T//slowfast_alpha).long().cpu().numpy().tolist()
            slow_pathway = fast_pathway[:, frame_indexs, :, :]
            transformed_frames = [slow_pathway, fast_pathway]
        else:
            transformed_frames = [transformed_frames]

        # mid_tlbrs is  (1, 4)
        #return transformed_frames, np.expand_dims(tlbr, axis=0), mid_tlbrs, track_ids
        return transformed_frames, start_frame_idx, end_frame_idx

    def _get_frame_idxs(self, start_fidx):
        """Will ignore out-of-video by repeating last frame
        """
        frame_idxs = []
        end_fidx = start_fidx + self.frame_length * self.frame_stride
        for fidx in range(start_fidx, end_fidx, self.frame_stride):
            if fidx >= self.video_num_frame:
                # repeat the last frame, so should have at least 1 valid frame
                fidx = frame_idxs[-1]
            frame_idxs.append(fidx)
        return frame_idxs

    def _get_frame_idxs_uniform(self, start_fidx, end_fidx, num_frames):
        """ sames as pyslowfast/datasets/decoder.py -> temporal_sampling
        """
        frame_idxs = []

        # get k float points uniformly within [s, e]
        index = torch.linspace(start_fidx, end_fidx, num_frames)
        # make sure frame_idxs all within range
        # long() is same as running int() on the float points
        index = torch.clamp(index, 0, self.video_num_frame - 1).long()

        frame_idxs = index.numpy().tolist()

        return frame_idxs

    def __len__(self):
        return len(self.prop_data)


class VideoActionClassifier(object):
    def __init__(self, model_path, args, config_overwrites=None, direct_input=False):
        """
        """
        self.args = args
        self.gpu_id = args.gpu_id
        self.model_dataset = args.model_dataset
        self.model_type = args.model_type
        self.use_onnx = args.use_onnx
        self.direct_input = direct_input  # frames or [slow_frames, fast_frames] input
        if args.model_type in ["slowfast", "mvit"]:
            if args.use_onnx:
                self.ort_session = onnxruntime.InferenceSession(
                    args.onnx_model_path,
                    providers=['TensorrtExecutionProvider',
                               'CUDAExecutionProvider',
                               'CPUExecutionProvider']
                )
                # TODO: bind input and output to GPU
                # since output still need to be GPU to do ROI Align
                # https://github.com/microsoft/onnxruntime/issues/4390#issuecomment-652266231
                #self.io_binding = self.ort_session.io_binding()

                self.model = self.onnx_run_once
            elif hasattr(args, "use_jit") and args.use_jit:
                self.model = torch.jit.load(
                    args.jit_model_path,
                    map_location=torch.device("cuda:%d" % args.gpu_id))
            elif hasattr(args, "use_trt") and args.use_trt:
                self.model = torch.jit.load(
                    args.trt_model_path,
                    map_location=torch.device("cuda:%d" % args.gpu_id))
            else:
                # load the slowfast configs
                # TODO: make this more elegant
                sys.argv = [sys.argv[0]]
                slowfast_args = parse_args()
                slowfast_args.cfg_file = args.pyslowfast_cfg
                slowfast_cfg = load_config(slowfast_args)
                # whether ROI align head is used
                self.detection_enabled = slowfast_cfg.DETECTION.ENABLE
                slowfast_cfg.TRAIN.ENABLE = False
                slowfast_cfg.TEST.ENABLE = True
                slowfast_cfg.TEST.CHECKPOINT_FILE_PATH = args.model_path
                slowfast_cfg.NUM_GPUS = 1
                slowfast_cfg.BN.NUM_SYNC_DEVICES = 1
                slowfast_cfg.TEST.BATCH_SIZE = args.batch_size
                slowfast_cfg.TEST.CROP_SIZE = args.frame_size
                #slowfast_cfg.TRAIN.BATCH_SIZE = args.batch_size
                #slowfast_cfg.TEST.NUM_ENSEMBLE_VIEWS = 5
                #slowfast_cfg.TEST.NUM_SPATIAL_CROPS = 1
                #slowfast_cfg.DATA.DECODING_BACKEND = "pyav"
                if args.model_dataset == "kinetics600":
                    slowfast_cfg.MODEL.NUM_CLASSES = 600
                elif args.model_dataset == "kinetics400":
                    slowfast_cfg.MODEL.NUM_CLASSES = 400
                elif args.model_dataset == "charades":
                    slowfast_cfg.MODEL.NUM_CLASSES = 157
                #logging.get_logger(__name__).info(slowfast_cfg)
                #logging.info(slowfast_cfg)
                if config_overwrites is not None:
                    slowfast_cfg.merge_from_list(config_overwrites)

                self.model = build_pyslowfast_model(
                    slowfast_cfg, gpu_id=args.gpu_id)
                #misc.log_model_info(self.model, slowfast_cfg, use_train_input=False)

                pyslowfast_checkpoint.load_test_checkpoint(slowfast_cfg, self.model)

                self.model.eval()
        elif args.model_type in ["img_based"]:
            # from the tensorflow trained models

            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth = True
            tfconfig.gpu_options.visible_device_list = "%s" % args.gpu_id
            self.graph1 = tf.Graph()
            self.sess1 = tf.Session(config=tfconfig, graph=self.graph1)
            self.graph2 = tf.Graph()
            self.sess2 = tf.Session(config=tfconfig, graph=self.graph2)
            with self.graph1.as_default():
                modelpath = tf.train.get_checkpoint_state(args.model_path).model_checkpoint_path

                saver = tf.train.import_meta_graph(modelpath+".meta")

                saver.restore(self.sess1, modelpath)
                # input place holders
                self.input = tf.get_collection("input")[0]
                self.is_train = tf.get_collection("is_train")[0] # TODO: remove this

                self.output = tf.get_collection("output")[0]
                #print [n.name for n in tf.get_default_graph().as_graph_def().node]

                # now get the frame feature extraction model
            with self.graph2.as_default():

                self.input_place_holders, self.sf_logits, self.sf_feats = get_sf_feat_forward_graph()
                saver2 = tf.train.Saver()
                saver2.restore(self.sess2, args.sf_img_model_path)



    def onnx_run_once(self, imgs, boxes=None):
        if self.args.model_type in ["slowfast"]:
            inputs = {"slow_frames": imgs[0], "fast_frames": imgs[1]}
        else:
            inputs = {"frames": imgs[0]}

        if boxes is not None:
            inputs.update({"bboxes": boxes})
        # None so will return all outputs
        return self.ort_session.run(None, inputs)

    def _l2norm(self, features):
        """ Given [B, nDim], l2 norm it along last dim
        """
        l2_norm = np.linalg.norm(features, 2, axis=-1, keepdims=True)
        return features / l2_norm

    def inference(self, frames, bboxes=None):
        # frames is [N, C, T, H, W] or [N, T, H, W, C]
        # pyslowfast uses [slowframes, fast frames] inputs
        if self.model_type in ["mit"]:
            # [N, num_class]
            sigmoid_outputs = self.model(frames)
            return sigmoid_outputs
        elif self.model_type in ["img_based"]:
            # given the frames, extract features first
            # the frames are channel last [N, T, H, W, C] and numpy image
            N, T, H, W, C = frames.shape
            reshaped_frames = np.reshape(frames, [N*T, H, W, C])
            # [N*T, 1001] and [N*T, 1536]
            feats, logits = self.sess2.run(
                [self.sf_feats, self.sf_logits],
                feed_dict={self.input_place_holders: reshaped_frames})
            logits = self._l2norm(logits)
            feats = self._l2norm(feats)
            # [N*T, 2537]
            final_frame_features = np.concatenate(
                [feats, logits], axis=1)

            final_frame_features = np.reshape(final_frame_features, [N, T, -1])
            features_avg = np.mean(final_frame_features, axis=1)
            features_avg = self._l2norm(features_avg)

            # feature_input: [B, nDim]
            predictions = self.sess1.run(
                self.output,
                feed_dict={self.input: features_avg, self.is_train: False})
            return predictions

        else:

            # pyslowfast model
            #if self.model_dataset in ["ava"]:
            if hasattr(self, "detection_enabled") and self.detection_enabled:
                if bboxes is None:
                    # use the whole frame as the box

                    # need bbox, generate full frame size boxes
                    N, C, T, H, W = list(frames[0].size())
                    if self.use_onnx:
                        N = self.args.batch_size
                    bboxes = np.zeros((N, 5), dtype="float32")
                    # scaling_factor = 1/32
                    # the following coordinates is the same as mean pooling the
                    # whole WH dimension
                    bboxes[:, [1, 2]] = 0
                    bboxes[:, [3, 4]] = [W, H]

                    for i in range(N):  # box's batch idx
                        bboxes[i, 0] = i
                    bboxes = torch.tensor(bboxes).float().cuda(
                        "cuda:%s" % self.gpu_id, non_blocking=True)


            keep_num = len(frames[0])
            if self.use_onnx:
                frames = [f.cpu().numpy() for f in frames]
                # onnx model needs fixed batch_size input
                keep_num = len(frames[0])
                if len(frames[0]) < self.args.batch_size:
                    repeat_times = self.args.batch_size - len(frames[0])
                    new_frames = []
                    for f in frames:
                        new_f = f[:]
                        new_f = np.concatenate(
                            (new_f, [new_f[-1] for i in range(repeat_times)]),
                            axis=0)
                        new_frames.append(new_f)
                    frames = new_frames

                if bboxes is not None:
                    bboxes = bboxes.cpu().numpy()


            #print(frames[0].shape) # [B, 3, T, H, W]

            # not using slowfast type two frames tensor input
            if hasattr(self, "direct_input") and self.direct_input:
                frames = frames[0]

            if bboxes is not None:
                preds = self.model(frames, bboxes)
            else:
                preds = self.model(frames)
            if self.use_onnx:
                preds = preds[0]
            return preds[:keep_num]
