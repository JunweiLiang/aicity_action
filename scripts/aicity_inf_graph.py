# coding=utf-8
"""
Given the classification scores and gt, generate graphs
"""
import argparse
import os
import math
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("pred_pickle_path")
parser.add_argument("anno_csv")

# hyper-parameters for aggregating results
parser.add_argument("--num_class", default=18, type=int)
parser.add_argument("--agg_method", default="avg", choices=["avg", "max"])
# take max among all synchronized views, then
parser.add_argument("--thres_take", default="min", choices=["min", "max", "mean"],
                    help="given a list of score thres hold for all vid")
parser.add_argument("--chunk_sort_base_single_vid", default="score", choices=["score", "length"],
                    help="given action chunk, select top-1 based on")
parser.add_argument("--chunk_sort_base_multi_vid", default="length", choices=["score", "length"],
                    help="given action chunk among all video views, select top-1 based on")
parser.add_argument("--use_num_chunk", default=1, type=int,
                    help="how many action chunks per action class for eval")


parser.add_argument("--replace_thres_file", default=None, help="use a existing thres file instead")

# generate visualization or the threshold file for testing
parser.add_argument("--graph_path", default=None,
                    help="generate aggregated prediction graph along with gt")
parser.add_argument("--thres_file", default=None,
                    help="save threshold for 1-17 classes")


parser.add_argument("--appendix", default="MP4")

parser.add_argument("--use_tight_times", action="store_true")
parser.add_argument("--use_ori_times", action="store_true")

def main(args):

    video_fps = 30.0

    if args.agg_method == "avg":
        aggregate_func = np.mean
    elif args.agg_method == "max":
        aggregate_func = np.max



    classes = range(1, args.num_class)
    anno_data = defaultdict(list)  # vid -> list of classes, vid is user_id + 0/1
    all_pickle = {}
    for line in open(args.anno_csv, "r").readlines():
        video_file, action_class = line.strip().split()
        file_id, user_id, t0, t1, _ = video_file.split(".")
        #assert file_id[-1] in ["0", "1"], file_id  # could be 3/4
        vid = "%s_%s" % (user_id, file_id[-1])
        # {'24491_1': [('Rightside_user_id_24491_1', '24491', '0', '17', '0'),
        #('Rightside_user_id_24491_1', '24491', '18', '45', '3'),
        #('Rightside_user_id_24491_1', '24491', '45', '54', '14'),
        #('Rightside_user_id_24491_1', '24491', '74', '105', '2')
        anno_data[vid].append((file_id, user_id, int(t0), int(t1), int(action_class)))

        all_pickle[file_id] = 1



    # load all the pickle files and aggregate the score first
    pickle_data = {}  # file_id -> a list of scores per frame
    for file_id in all_pickle:
        pred_file = os.path.join(args.pred_pickle_path, "%s.%s.pkl" % (file_id, args.appendix))
        with open(pred_file, "rb") as f:
            pred = pickle.load(f)

        # a list of (t0, t1, pred) ->
        # [num_frame, num_class]
        pickle_data[file_id] = aggregate_predictions(pred, aggregate_func, args.num_class)

    if args.graph_path is not None:
        os.makedirs(args.graph_path, exist_ok=True)
        print("generating %s graphs, each has 3 videos from diff view" % len(anno_data))

    action_id_to_thres = defaultdict(list)
    for vid in tqdm(anno_data):
        for action_id in classes:
            anno = [o for o in anno_data[vid] if o[-1] == action_id]
            # [('Rightside_user_id_24491_1', '24491', 386, 399, 1),
            #('Rearview_user_id_24491_1', '24491', 386, 399, 1),
            #('Dashboard_user_id_24491_1', '24491', 386, 399, 1)]
            #assert len(anno) == 3, anno
            #if len(anno) != 3:
                #print("warning, %s class %s does not has 3 videos but have %s" % (
                #    vid, action_id, len(anno)))

            # save some threshold in this validation set
            score_thres = [0.0]  # junwei: bug for thres_take == mean?
            #score_thres = []

            # draw the three graphs together
            fig, axes = plt.subplots(1, 3, figsize=(20, 9))

            for i, (file_id, user_id, t0, t1, action_id) in enumerate(anno[:3]):
                scores = pickle_data[file_id][:, action_id]
                # seconds to frame_idx
                anno_t0t1 = [int(t0*video_fps), int(t1*video_fps)]

                # compute the average scores within the gt segment
                gt_scores = []
                for j in range(anno_t0t1[0], min(len(scores), anno_t0t1[1])):
                    gt_scores.append(scores[j])
                # get the averaged scores among the gt frames
                mean_gt_score = float(np.mean(gt_scores))
                score_thres.append(mean_gt_score)

                if args.graph_path is None:
                    continue

                x = range(len(scores))
                y = scores
                axes[i].plot(x, y)
                axes[i].axvline(x=anno_t0t1[0], color="orange", linestyle="dashed")
                axes[i].axvline(x=anno_t0t1[1], color="orange", linestyle="dashed")


                axes[i].set_title("%s (%.3f-%d)" % (file_id, mean_gt_score, anno_t0t1[1] - anno_t0t1[0]))

            # so for this video with 3 views, save the maximum mean score among the gt segments
            # will later get a global score thres
            #action_id_to_thres[action_id].append(max(score_thres))
            action_id_to_thres[action_id] += score_thres

            if args.graph_path is not None:
                target_graph = os.path.join(args.graph_path, "%s.action_%d.png" % (vid, action_id))
                fig.savefig(target_graph)

            plt.clf()
            plt.cla()
            plt.close()

    # each action_id have multiple threshold,
    for action_id in action_id_to_thres:
        scores = action_id_to_thres[action_id]
        if args.thres_take == "min":
            action_id_to_thres[action_id] = min(scores)
        elif args.thres_take == "max":
            action_id_to_thres[action_id] = max(scores)
        else:
            action_id_to_thres[action_id] = np.mean(scores)

    if args.replace_thres_file is not None:
        print("using existing thresholds...")
        for line in open(args.replace_thres_file).readlines():
            action_id, score = line.strip().split()

            action_id_to_thres[int(action_id)] = float(score)

    # save the score threshold
    if args.thres_file is not None:
        with open(args.thres_file, "w") as f:
            for i in range(1, args.num_class):
                f.writelines("%d %.5f\n" % (i, action_id_to_thres[i]))

    # use this score threshold to compute F1 score

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
                #print("warning, %s %s got no action chunks" % (file_id, action_id))
                continue
            # only using the longest one

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

    # 2. get one action chunk per 3-view and evaluate
    # consider gt +- k seconds to be correct
    f1_1sec, precision, recall = compute_f1(anno_data, classes, action_chunks,
                         sec_thres=1.0, chunk_sort_base=args.chunk_sort_base_multi_vid,
                         use_num_chunk=args.use_num_chunk, return_pr=True,
                         use_tight_times=args.use_tight_times,
                         use_ori_times=args.use_ori_times)
    #f1_2sec = compute_f1(anno_data, classes, action_chunks,
    #                     sec_thres=2.0, chunk_sort_base=args.chunk_sort_base_multi_vid,
    #                     use_num_chunk=args.use_num_chunk)
    #f1_3sec = compute_f1(anno_data, classes, action_chunks,
    #                     sec_thres=3.0, chunk_sort_base=args.chunk_sort_base_multi_vid,
    #                     use_num_chunk=args.use_num_chunk)
    #print("precision/recall +-1 sec: %.6f / %.6f" % (precision, recall))
    #print("F1 (+-1/2/3 sec): %.6f %.6f %.6f" % (f1_1sec, f1_2sec, f1_3sec))
    print("F1, precision, recall: %.6f %.6f %.6f" % (f1_1sec, precision, recall))


def compute_f1(anno_data, classes, action_chunks, use_num_chunk=1, sec_thres=1.0,
               chunk_sort_base="length", return_pr=False,
               use_tight_times=False,
               use_ori_times=False):
    TP, FP, FN = 0, 0, 0
    for vid in anno_data:
        for action_id in classes:
            anno = [o for o in anno_data[vid] if o[-1] == action_id]
            # [('Rightside_user_id_24491_1', '24491', 386, 399, 1),
            #('Rearview_user_id_24491_1', '24491', 386, 399, 1),
            #('Dashboard_user_id_24491_1', '24491', 386, 399, 1)]
            #assert len(anno) == 3, anno
            if len(anno) != 3:
                #print("warning, %s class %s does not has 3 videos but have %s, skipped" % (
                #    vid, action_id, len(anno)))
                continue

            all_vid_action_chunks = [
                one
                for o in anno for one in action_chunks[o[0]][action_id]
                if action_id in action_chunks[o[0]]]

            #assert all_vid_action_chunks, (vid, action_id, "all 3 videos got no chunk")
            if not all_vid_action_chunks:
                print("for computing F1, %s %s has no action pred " % (vid, action_id))
                FN += 1
                continue

            if chunk_sort_base == "length":
                all_vid_action_chunks.sort(key=lambda x: x[2], reverse=True)
            else:
                all_vid_action_chunks.sort(key=lambda x: x[3], reverse=True)

            all_vid_action_chunks = all_vid_action_chunks[:use_num_chunk]
            # assuming there is only one gt
            match_gt = 0
            for action_chunk in all_vid_action_chunks:
                #pred_start_sec, pred_end_sec = action_chunk[0], action_chunk[1]
                #pred_start_sec, pred_end_sec = math.ceil(action_chunk[0]), math.ceil(action_chunk[1])
                if use_tight_times:
                    pred_start_sec, pred_end_sec = round(action_chunk[0]) + 1., round(action_chunk[1]) - 1.
                else:
                    pred_start_sec, pred_end_sec = round(action_chunk[0]), round(action_chunk[1])
                if use_ori_times:
                    pred_start_sec, pred_end_sec = action_chunk[0], action_chunk[1]
                gt_start_sec = anno[0][2]
                gt_end_sec = anno[0][3]
                if (gt_start_sec - sec_thres <= pred_start_sec and pred_start_sec <= gt_start_sec + sec_thres) and \
                        (gt_end_sec - sec_thres <= pred_end_sec and pred_end_sec <= gt_end_sec + sec_thres):

                    # assuming there is only one gt
                    # so other pred is consider FP
                    if match_gt == 1:
                        FP += 1
                    else:
                        TP += 1
                        match_gt += 1
                    #print(action_chunk, anno[0])
                else:
                    FP += 1
            if not match_gt:
                FN += 1
    if return_pr:
        return TP/(TP + 0.5 * (FP + FN)), (TP / (TP + FP)), (TP / (TP + FN))
    else:
        return TP/(TP + 0.5 * (FP + FN))

def get_chunks(score_list, threshold):
    # return continuous chunks
    chunks = []
    start = None
    for fidx in range(len(score_list)):
        score = score_list[fidx]

        if score >= threshold:
            if start is None:
                start = fidx
            elif start is not None and fidx == len(score_list) - 1:
                chunks.append(
                    (start, fidx, fidx - start + 1,
                     np.mean(score_list[start:fidx+1]), score_list[start:fidx+1]))
                start = None
        else:
            if start is not None:
                chunks.append(
                    (start, fidx, fidx - start + 1,
                     np.mean(score_list[start:fidx+1]), score_list[start:fidx+1]))
                start = None
    return chunks



def aggregate_predictions(pred_list, aggregate_func, num_class):

    # frame_idx are 0-indexed
    frame_idxs = [t[0] for t in pred_list]
    frame_idxs += [t[1] for t in pred_list]
    min_frame_idx = min(frame_idxs)
    max_frame_idx = max(frame_idxs)
    frame_num = max_frame_idx - min_frame_idx

    # construct a list per frame_idx for all scores
    # assume scores are between 0.0 to 1.
    # len == num_frame, each is a list of predictions of all classes
    score_list_per_frame = [
        [np.zeros((num_class), dtype="float32")]
        for i in range(frame_num)]

    # t1- t0 == 64
    for t0, t1, cls_data in pred_list:
        for t in range(t0, t1):
            save_idx = t - min_frame_idx
            score = cls_data  # num_class
            assert len(score) == num_class
            score_list_per_frame[save_idx].append(score)

    # aggregate the scores at each frame idx
    # get the chunks in (t0, t1) with scores >= thres
    agg_score_per_frame = []
    for i in range(len(score_list_per_frame)):
        # stack all the scores first
        if len(score_list_per_frame[i]) > 1:
            score_list_per_frame[i].pop(0)  # remove the zero padding
        # [K, num_class]
        stacked_scores = np.vstack(score_list_per_frame[i])
        # [num_class]
        this_frame_scores = aggregate_func(stacked_scores, axis=0)
        this_frame_idx = min_frame_idx + i
        agg_score_per_frame.append(this_frame_scores)

    return np.vstack(agg_score_per_frame)  # [num_frame, num_class]


if __name__ == "__main__":
    main(parser.parse_args())
