# coding=utf-8
"""
    Given the path of precomputed video feature path, video list and text, do retrieval
"""

import argparse
import os
import json
import sys
import time

import numpy as np

import torch

from utils.simple_tokenizer import SimpleTokenizer

from slowfast.models import build_model
from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model as build_pyslowfast_model
import slowfast.utils.checkpoint as pyslowfast_checkpoint
import slowfast.utils.logging as logging


parser = argparse.ArgumentParser()
parser.add_argument("query_json")
parser.add_argument("video_feat_file")
parser.add_argument("test_set_anno")  # needed to get video_id
parser.add_argument("bpe_dict")
parser.add_argument("cfg")
parser.add_argument("model_path")
parser.add_argument("output_file")
parser.add_argument("--context_length", type=int, default=77)
parser.add_argument("--gpu_id", default=0, type=int)
parser.add_argument("--config_overwrites", default=None, nargs="*")


def tokenize(text, bpe_dict, context_length):

    # TODO: make this a function from scripts/prepro_tokenize.py
    # some string that is absolutely not useful from shutterstock
    # should be all lower case
    word_blacklist = [
        "4k", "4 k", "in 4k",
        "HD",
        "1920x1080", "1280x720",
        "view from above",
        "more options in my portfolio",
        "top view",
        "side view",
        "medium shot",
        #"slomo",
        "High quality FullHD footage",
        "CLOSE UP",
        "close up pan down",
    ]
    word_blacklist = [w.lower() for w in word_blacklist]
    # rank the longer ones in front so later replacement dont affect previous
    word_blacklist.sort(key=len, reverse=True)

    def remove_words_from_string(string):

        # this is faster than regex
        for word in word_blacklist:
            string = string.replace(word, '')
        return string

    tokenizer = SimpleTokenizer(bpe_dict)
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]


    text_removed_words = remove_words_from_string(text.lower())
    token_ids = tokenizer.encode(text_removed_words)[:context_length]

    all_tokens = [sot_token] + token_ids + [eot_token]

    return all_tokens

@torch.no_grad()
def main(args):
    with open(args.query_json, "r") as f:
        query = json.load(f)

    text = query["query_text"]

    # 1. tokenize the text and load the text model part to get text vector
    # this part takes 3-4 seconds
    print("getting query embedding...")
    tokens = tokenize(text, args.bpe_dict, args.context_length)  # list of ints
    token_tensor = torch.zeros((1, args.context_length), dtype=torch.long)
    for i, token in enumerate(tokens):
        token_tensor[0, i] = token
    # (1, 77)
    token_tensor = token_tensor.cuda(non_blocking=True)

    # load the full model
    sys.argv = [sys.argv[0]]
    slowfast_args = parse_args()
    slowfast_args.cfg_file = args.cfg
    slowfast_cfg = load_config(slowfast_args)
    slowfast_cfg.NUM_GPUS = 1
    slowfast_cfg.TRAIN.ENABLE = False
    slowfast_cfg.TEST.ENABLE = True
    slowfast_cfg.TEST.CHECKPOINT_FILE_PATH = args.model_path
    # not load the visual part
    slowfast_cfg.MODEL.LOAD_VISUAL = False

    if args.config_overwrites is not None:
        slowfast_cfg.merge_from_list(args.config_overwrites)

    model = build_pyslowfast_model(slowfast_cfg, gpu_id=args.gpu_id)
    pyslowfast_checkpoint.load_test_checkpoint(slowfast_cfg, model)

    model.eval()

    # (1, 512)
    text_features = model.text_encoder.encode_text(token_tensor)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp().mean()

    # list of video_id (basename of video file)
    video_list = query["video_list"]

    # 2. gather the video features first, assuming all can fit in GPU
    print("getting video features...")
    # load the big file and the videoname to video_idx mapping
    full_video_feature = np.load(args.video_feat_file)
    full_video_name_to_idx = {}
    for i, line in enumerate(open(args.test_set_anno, "r").readlines()):
        video_name = os.path.basename(line.strip().split(" ", 1)[0])
        full_video_name_to_idx[video_name] = i
    assert len(full_video_name_to_idx) == len(full_video_feature), \
        (len(full_video_name_to_idx), len(full_video_feature))

    video_features = []
    for video_name in video_list:
        video_name = os.path.basename(video_name)

        video_feature = full_video_feature[full_video_name_to_idx[video_name]]

        video_feature = torch.Tensor(video_feature)

        video_features.append(video_feature)

    # [N, 512]
    video_features = torch.stack(video_features, dim=0).cuda(non_blocking=True)

    # 3. matrix mul and return the ranklist and similarities
    logits_per_video = logit_scale * video_features @ text_features.t()
    # (20/num_video)
    #probs = logits_per_video.squeeze(1).softmax(dim=0).detach().cpu().numpy()
    # show original logits
    probs = logits_per_video.squeeze(1).detach().cpu().numpy()

    video_probs = zip(video_list, probs)

    with open(args.output_file, "w") as f:
        for video_name, prob in video_probs:
            f.writelines("%s %s\n" % (video_name, prob))

if __name__ == "__main__":
    start = time.time()
    main(parser.parse_args())
    ran_time = time.time() - start
    # 2 minutes for 10k videos
    print("total run time %.2f seconds" % ran_time)
