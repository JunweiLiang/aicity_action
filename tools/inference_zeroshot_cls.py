# coding=utf-8
"""
    Given the path to a list of computed dataset-specific predictions and labels,
    and the dataset-text-embeddings, 
    and the query,
    get the query-to-dataset concept weights, compute the final score 
"""

import argparse
import os
import json
import re
import numpy as np
import pickle

from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("query")
parser.add_argument("dataset_config")
parser.add_argument("pred_path")
parser.add_argument("cls_emb_path")
parser.add_argument("word_emb_file")
parser.add_argument("vocab_path")
parser.add_argument("anno_path")
parser.add_argument("output_file")
parser.add_argument("--emb_dim", default=300, type=int)
parser.add_argument("--min_simi", default=0.6, type=float)
parser.add_argument("--top_k_cls", default=10, type=int,
                    help="take at most top 5 class")

def load_word_embeddings(file):
    vocab = {}
    with open(file, "r") as f:
        for line in f:
            values = line.strip().split()
            vocab[values[0]] = np.array(values[1:], "float32")
    return vocab

def query_processing(text):
    # return a list of words
    lower_case_text = text.strip().lower()
    words = re.split(r"\W+", lower_case_text)
    words = [w for w in words if w]  # remove empty ones
    return words

def get_dataset_cls_scores_and_weights(
        dataset_config, query_embedding, vocab_path, cls_emb_path, min_simi, top_k_cls):
    dataset_scores = []   # dataset_cls_id, score

    dataset_to_classnames = defaultdict(dict)
    weight_matrices = {}  # dataset_name -> [C]
    for dataset_name in dataset_config["dataset_class_embeddings"]:

        vocab_file = os.path.join(
            vocab_path, dataset_config["dataset_vocab_files"][dataset_name])

        for i, line in enumerate(open(vocab_file).readlines()):
            dataset_to_classnames[dataset_name][i] = line.strip()


        class_emb_file = os.path.join(
            cls_emb_path, dataset_config["dataset_class_embeddings"][dataset_name])

        # [C, 300]
        class_embs = np.load(class_emb_file)

        # [C]
        class_simi = np.matmul(class_embs, query_embedding)

        for i in range(len(class_simi)):
            dataset_cls_id = "%s_%d" % (dataset_name, i+1)
            dataset_scores.append((
                dataset_cls_id, class_simi[i],
                dataset_to_classnames[dataset_name][i],
                dataset_name, i))

        C = len(class_simi)
        weight_matrices[dataset_name] = np.zeros((C), dtype="float32")


    dataset_scores.sort(key=lambda x: x[1], reverse=True)

    # take at most top_k class with simi at least min_simi
    dataset_scores = [o for o in dataset_scores if o[1] >= min_simi][:top_k_cls]

    # set the weight matrices
    for _, class_simi, _, dataset_name, cls_id in dataset_scores:
        weight_matrices[dataset_name][cls_id] = class_simi #1.

    return dataset_scores, weight_matrices


def get_predictions(dataset_config, weight_matrices, pred_path):
    preds_all = []
    for dataset_name in weight_matrices:
        pred_file = os.path.join(
            pred_path, dataset_config["dataset_pred_files"][dataset_name])
        # [N, C]
        preds = np.load(pred_file)
        # [N]
        preds = np.matmul(preds, weight_matrices[dataset_name])
        preds_all.append(preds)

    # [N, 3] -> [N]
    preds_all = np.stack(preds_all, axis=1).sum(axis=1)
    return preds_all


def get_embeddings(words, word_embeddings, emb_dim=300):

    embedding = np.zeros((emb_dim), dtype="float32")
    got_word_num = 0
    for word in words:
        if word in word_embeddings:
            got_word_num += 1
            emb = word_embeddings[word]
            embedding += emb

    if got_word_num == 0:
        return None

    # mean pooled
    embedding /= got_word_num

    # l2 normed [300 dim]
    embedding /= np.linalg.norm(embedding)
    return embedding


def main(args):

    print("querying %s" % args.query)

    query_words = query_processing(args.query)

    # load the word embedding
    word_embeddings = load_word_embeddings(args.word_emb_file)

    # get query embedding
    query_embedding = get_embeddings(query_words, word_embeddings)

    if query_embedding is None:
        print("sorry, no word in word embeddings matched query")
        return

    with open(args.dataset_config, "r") as f:
        dataset_config = json.load(f)

    dataset_scores, weight_matrices = get_dataset_cls_scores_and_weights(
        dataset_config, query_embedding, args.vocab_path, args.cls_emb_path,
        args.min_simi, args.top_k_cls)
    print("prediction using %s" % dataset_scores)
    #print(weight_matrices)
    # go over the dataset predictions again
    preds_all = get_predictions(dataset_config, weight_matrices, args.pred_path)

    # get video_name list
    dataset_annotation_file = os.path.join(args.anno_path, dataset_config["annotation"])
    all_video_list = []
    with open(dataset_annotation_file) as f:
        for line in f:
            video_name = os.path.basename(line.strip().split(" ", 1)[0])
            all_video_list.append(video_name)

    video_probs = list(zip(all_video_list, preds_all.tolist()))

    video_probs.sort(key=lambda x: x[1], reverse=True)
    print("top 10 prediction: %s" % video_probs[:10])

    # save this as a pickle
    output = {
        "pred": preds_all,
        "dataset_scores": dataset_scores,
    }
    with open(args.output_file, "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    main(parser.parse_args())
