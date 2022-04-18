# coding=utf-8
"""
Given the classification scores, get final aicity results
"""
import argparse
import os
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from aicity_inf_graph import aggregate_predictions
from aicity_inf_graph import get_chunks

parser = argparse.ArgumentParser()
parser.add_argument("pred_pickle_path")
parser.add_argument("thres_file", default=None,
                    help="threshold for 1-17 classes")
parser.add_argument("vid_csv")
parser.add_argument("output_file")

# hyper-parameters for aggregating results
parser.add_argument("--num_class", default=18, type=int)
parser.add_argument("--agg_method", default="avg", choices=["avg", "max"])
# take max among all synchronized views, then
#parser.add_argument("--thres_take", default="min", choices=["min", "max", "mean"],
#                    help="given a list of score thres hold for all vid")
parser.add_argument("--chunk_sort_base_single_vid", default="score", choices=["score", "length"],
                    help="given action chunk, select top-1 based on")
parser.add_argument("--chunk_sort_base_multi_vid", default="length", choices=["score", "length"],
                    help="given action chunk among all video views, select top-1 based on")
parser.add_argument("--use_num_chunk", default=1, type=int,
                    help="how many action chunks per action class for eval")

def main(args):


    video_fps = 30.0

    if args.agg_method == "avg":
        aggregate_func = np.mean
    elif args.agg_method == "max":
        aggregate_func = np.max


    action_id_to_thres = {}
    for line in open(args.thres_file).readlines():
        action_id, thres = line.strip().split()
        action_id_to_thres[int(action_id)] = float(thres)

    test_vids = {}  # vid -> list of videos
    all_videos = []
    for line in open(args.vid_csv, "r").readlines()[1:]:
        vid, file1, file2, file3 = line.strip().split(",")
        # 1 -> Dashboard_user_id_42271_NoAudio_3.MP4,Rear_view_user_id_42271_NoAudio_3.MP4
        test_vids[vid] = [file1, file2, file3]
        all_videos += [file1, file2, file3]


    # load all the pickle files and aggregate the score first
    pickle_data = {}  # file_id -> a list of scores per frame
    for file_id in all_videos:
        pred_file = os.path.join(args.pred_pickle_path, "%s.pkl" % (file_id))
        with open(pred_file, "rb") as f:
            pred = pickle.load(f)

        # a list of (t0, t1, pred) ->
        # [num_frame, num_class]
        pickle_data[file_id] = aggregate_predictions(pred, aggregate_func, args.num_class)


    # 1. get the action chunks per video first
    action_chunks = {}  # file_id -> one action chunk per action
    # file_id: 'Rightside_user_id_24491_1'
    for file_id in pickle_data:
        preds = pickle_data[file_id]  # [num_frame, num_classes]

        action_instances = defaultdict(list)  # action_id -> one action chunk (start_frame, end_frame, length, mean_score, score_list)
        for action_id in action_id_to_thres:
            this_preds = preds[:, action_id]  # [num_frame]
            chunks = get_chunks(this_preds, action_id_to_thres[action_id])
            if not chunks:
                print("warning, %s %s got no action chunks" % (file_id, action_id))
                continue

            if args.chunk_sort_base_single_vid == "length":
                chunks.sort(key=lambda x: x[2], reverse=True)
            else:
                # based on average score
                chunks.sort(key=lambda x: x[3], reverse=True)

            for i in range(args.use_num_chunk):
                if i >= len(chunks):
                    break
                start_frame, end_frame, num_frame, mean_score, score_list = chunks[i]

                action_instances[action_id].append((
                    start_frame/video_fps, end_frame/video_fps, num_frame, mean_score))

        action_chunks[file_id] = action_instances

    # aggregate the action chunk in each video to synchronized vid views
    outputs = []  # vid, act_id, start_time, end_time
    for vid in test_vids:
        for action_id in action_id_to_thres:
            all_vid_action_chunks = [
                one
                for file_id in test_vids[vid] for one in action_chunks[file_id][action_id]
                if action_id in action_chunks[file_id]]
            if not all_vid_action_chunks:
                print("warning, %s %s has no action chunks" % (vid, action_id))
                continue

            if args.chunk_sort_base_multi_vid == "length":
                all_vid_action_chunks.sort(key=lambda x: x[2], reverse=True)
            else:
                all_vid_action_chunks.sort(key=lambda x: x[3], reverse=True)

            all_vid_action_chunks = all_vid_action_chunks[:args.use_num_chunk]
            for action_chunk in all_vid_action_chunks:
                #pred_start_sec, pred_end_sec = action_chunk[0], action_chunk[1]
                pred_start_sec, pred_end_sec = round(action_chunk[0]) + 1.0, round(action_chunk[1]) - 1.0
                outputs.append((vid, action_id, pred_start_sec, pred_end_sec))

    print("total pred %s" % len(outputs))
    with open(args.output_file, "w") as f:
        for vid, action_id, start, end in outputs:
            f.writelines("%s %s %.6f %.6f\n" % (vid, action_id, start, end))


if __name__ == "__main__":
    main(parser.parse_args())
