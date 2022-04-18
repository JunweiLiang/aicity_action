# coding=utf-8
"""
Given zhuhe processed ai city annotations, convert to a clip-based pyslowfast anno
# each clip could be 0-17, and -1 for NA, -2 for empty
"""

import argparse
import os
import decord
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("anno_file")
parser.add_argument("video_path")  # get the total seconds of the video
parser.add_argument("out_anno_file")
parser.add_argument("clip_cmds")  # ffmpeg bash file for cutting the videos into clips
parser.add_argument("target_path")
parser.add_argument("--resolution", default="-2:540", help="ffmpeg -vf scale=")

def time2int(time_str):
    # 00:18 to integer seconds
    minutes, seconds = time_str.split(":")
    minutes = int(minutes)
    seconds = int(seconds)
    return minutes*60 + seconds

def int2time(secs):
    # seconds to 00:00
    m, s = divmod(secs, 60)
    if s >= 10.0:
        return "%02d:%.3f" % (m, s)
    else:
        return "%02d:0%.3f" % (m, s)

def process_file_name(file_name, user_id, view):
    # Rightside_user_id_24491_1, Rightside_window -> Rightside_window_user_id_24491_NoAudio_1
    perform_id = file_name[-1]
    # Dashboard_User_id_24026_NoAudio_3.MP4
    if user_id == "38508":  # junwei: wow
        user_id = "38058"
    if user_id in ["24026", "38058"]:
        if view == "Rightside_window":
            view = "Right_side_window"  # junwei: srsly?
        if view == "Rearview":
            view = "Rear_view"
        if view == "Rightside window":
            view = "Right_side_window"

        return "%s_User_id_%s_NoAudio_%s" % (view, user_id, perform_id)
    if user_id in ["35133"]:
        if view == "Rearview":
            view = "Rear_view"
        if view == "Rightside window":
            view = "Rightside_window"
    if user_id in ["49381"]:
        if view == "Rear_view":
            view = "Rearview_mirror"
        if view == "Rightside_window":
            view = "Right_window"

    return "%s_user_id_%s_NoAudio_%s" % (view, user_id, perform_id)

def main(args):
    data = defaultdict(list)  # video_file to segments
    users = {}
    action_lengths = []
    action_id_to_count = defaultdict(int)
    vid_to_seg = defaultdict(dict)  # video_file to segment, make sure no overlap
    # compute some stats
    # 1. the action id num, the length stats
    for line in open(args.anno_file, "r").readlines()[1:]:
        user_id, video_file_name, view, _, start, end, action_id, block = line.strip().split(",")
        users[user_id] = 1
        #original video has "NoAudio" but annotation does not
        video_file_name = "%s.MP4" % process_file_name(video_file_name.strip(), user_id.strip(), view.strip())

        start = time2int(start)
        end = time2int(end)
        # action_id could be 0-17, and "NA"

        #action_id = int(action_id)
        #assert action_id in range(18), line

        action_id = action_id.strip()
        action_id_to_count[action_id] += 1

        # assert no overlap
        assert (start, end) not in vid_to_seg[video_file_name], line
        vid_to_seg[video_file_name][(start, end)] = 1

        action_lengths.append(end - start)

        data[video_file_name].append((user_id, video_file_name, start, end, action_id))

    print(action_id_to_count)
    # user num: 5, action length min/max/median: 3, 38, 20.0
    print("user num: %s, action length min/max/median: %s, %s, %s" % (
        len(users),
        min(action_lengths), max(action_lengths), np.median(action_lengths)))

    # get the max length of each video, and check non-annotated segment length
    total_empty, total_length = 0, 0
    data_empty = {}  # video_file -> empty segments
    for video_file in data:
        video = os.path.join(args.video_path, video_file)
        vcap = decord.VideoReader(video)
        num_frame = len(vcap)
        max_length = int(num_frame / 30.0)
        anno_max_length = data[video_file][-1][3]
        user_id = data[video_file][0][0]

        anno_segments = [(None, None, 0, 0, 0)] + data[video_file]

        if max_length > anno_max_length:
            print("%s anno ends on %s, has %s total" % (video_file, anno_max_length, max_length))
            anno_segments += [(None, None, max_length, 0, 0)]
        elif max_length < anno_max_length:
            print("warning for %s, %s, %s" % (video_file, anno_segments[-1], max_length))
            # some annotation might be longer than the video

        empty_segments = []
        for s1, s2 in zip(anno_segments[0:-1], anno_segments[1:]):
            last_end = s1[3]
            next_start = s2[2]

            gap = next_start - last_end
            if gap > 0:
                empty_segments.append((user_id, video_file, last_end, next_start, "empty"))
                total_empty += gap
            elif gap < 0:
                print(s1, s2)
                sys.exit()

        data_empty[video_file] = empty_segments
        total_length += max_length
    print("total length %s, empty %s" % (total_length, total_empty))

    # write the annotation file
    video_clips = []  # video_file_name.user_id.start.end.mp4
    with open(args.out_anno_file, "w") as f:

        for video_file in data:

            anno_segs = data[video_file]
            empty_segs = data_empty[video_file]
            for user_id, _, start, end, action_id in anno_segs + empty_segs:
                video_id = "%s.%s.%d.%d.MP4" % (
                    os.path.splitext(video_file)[0],
                    user_id, start, end)
                if action_id == "NA":
                    action_id = -1
                elif action_id == "empty":
                    action_id = -2
                action_id = int(action_id)
                video_clips.append((video_file, int2time(start), int2time(end), video_id))

                f.writelines("%s %d\n" % (video_id, action_id))

    # write the cutting command

    with open(args.clip_cmds, "w") as f:
        for ori_video, start, end, target_clip in video_clips:
            f.writelines("ffmpeg -nostdin -y -i %s -vf scale=%s -c:v libx264 -ss %s -to %s %s\n" % (
                os.path.join(args.video_path, ori_video),
                args.resolution,
                start, end,
                os.path.join(args.target_path, target_clip)))


if __name__ == "__main__":
    main(parser.parse_args())
