# coding=utf-8
"""Given the video list, action classification model, do temporal localization
should be the same as running full inf PPL with notrack proposals
"""
import argparse
import os
import logging
logging.basicConfig(level=logging.INFO)
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from module_wrapper import VideoActionClassifier
from module_wrapper import ActionProposalFromVideoTemporalDataset

parser = argparse.ArgumentParser()
parser.add_argument("video_lst",
                        help="a list of absolute paths to proposal files")
parser.add_argument("video_dir")
parser.add_argument("model_path", )
parser.add_argument("out_dir",
                    help="output will be out_dir/$video_file.pkl")
# save the full last layer classification scores, ignore classname for now
parser.add_argument("--model_dataset", default="meva",
                    choices=[
                        "meva", "kinetics600", "ava", "kinetics400", "kinetics700",
                        "multi-moments-in-time", "moments-in-time",
                        "charades", "fcvid", "imgnet",
                        "sin346", "yfcc609", "aicity"])

parser.add_argument("--model_type", default="mvit",
                    choices=["mvit", "slowfast"])

parser.add_argument("--video_fps", default=-1., type=float,
                    help="if -1, will get the fps from the video")
parser.add_argument("--target_fps", default=30.0, type=float,
                    help="the fps the proposal/cls model is under, we train usually with 30fps")

# used to compute proposal length
parser.add_argument("--frame_length", default=16, type=int)
parser.add_argument("--frame_stride", default=4, type=int)
parser.add_argument("--proposal_length", type=int, default=64, help="in frames")
parser.add_argument("--proposal_stride", type=int, default=16, help="in frames")

parser.add_argument("--frame_size", default=224, type=int)

# the Default ROI for proposal, relative box to the video
parser.add_argument("--roi_x1", type=float, default=0.0)
parser.add_argument("--roi_y1", type=float, default=0.0)
parser.add_argument("--roi_x2", type=float, default=1.0)
parser.add_argument("--roi_y2", type=float, default=1.0)


parser.add_argument("--pyslowfast_cfg", default=None)
parser.add_argument("--pyslowfast_config_overwrites", default=None, nargs="*")

parser.add_argument("--gpu_id", default=0, type=int)
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--num_cpu_workers", default=5, type=int)

# video decoder option
parser.add_argument("--video_decoder", default="decord",
                    choices=["decord"])
parser.add_argument("--frame_format", default="rgb",
                    choices=["rgb"],
                    help="frame format to be fed into NN. ")


# ---for onnx
parser.add_argument("--use_onnx", action="store_true")
parser.add_argument("--onnx_model_path")

@torch.no_grad()
def main(args):
    assert args.proposal_length == args.frame_length * args.frame_stride
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. load classification model
    print("loading model...")
    action_model = VideoActionClassifier(
        args.model_path, args,
        config_overwrites=args.pyslowfast_config_overwrites)
    print("done")

    # 2. go through each video and each proposal
    with open(args.video_lst) as f:
        video_files = [os.path.join(args.video_dir, line.strip())
                       for line in f]

    for video_file in tqdm(video_files):
        video_name = os.path.basename(video_file)

        # get the proposals

        # construct the proposal dataset and data loader
        proposal_dataset = ActionProposalFromVideoTemporalDataset(
            video_file, args)
        proposal_loader = torch.utils.data.DataLoader(
            proposal_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_cpu_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=default_collate,
            worker_init_fn=None,
        )

        pred_list = []
        for cur_iter, (frames, t0s, t1s) in tqdm(
                enumerate(proposal_loader), total=len(proposal_loader)):
            # tensor(0) tensor(96) torch.Size([2, 3, 32, 448, 448])
            #print(t0s[0], t1s[0], frames[0].shape)
            # for slowfast, there will be [slow_frames, fast_frames]
            frames = [f.cuda("cuda:%s" % args.gpu_id, non_blocking=True) for f in frames]
            # [N, 600/num_class]
            preds = action_model.inference(frames)
            preds = preds.cpu().numpy()
            for b in range(frames[0].shape[0]):
                t0, t1 = int(t0s[b].numpy()), int(t1s[b].numpy())
                pred = preds[b]  # [num_classes]
                pred_list.append((t0, t1, pred))

        pred_list.sort(key=lambda x: x[0])
        # get aggregated scores for each frame

        target_file = os.path.join(args.out_dir, "%s.pkl" % video_name)
        with open(target_file, "wb") as f:
            pickle.dump(pred_list, f)


if __name__ == "__main__":
    main(parser.parse_args())
