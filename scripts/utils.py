# coding=utf-8

import os
import cv2
import av
import decord
import numpy as np
import math

def load_prop_file(prop_file, include_middle_frame_tlbr=False):
    prop_data = []
    with open(prop_file, "r") as f:
        for line in f:
            track_ids = line.strip().split()[-1]
            tlbr = [float(o) for o in line.strip().split()[:4]]
            start_frame_idx, end_frame_idx = [int(float(o)) for o in line.strip().split()[4:6]]
            middle_frame_tlbr = [float(o) for o in line.strip().split()[6:10]]
            if include_middle_frame_tlbr:
                prop_data.append([tlbr, start_frame_idx, end_frame_idx, middle_frame_tlbr, track_ids])
            else:
                prop_data.append([tlbr, start_frame_idx, end_frame_idx, track_ids])
    return prop_data


def get_video_name_and_appendix(file_path_like_prop):
    """Given file path like
        # 2018-03-05.13-20-01.13-25-01.bus.G331.r13.avi.txt
        Return 2018-03-05.13-20-01.13-25-01.bus.G331.r13 and avi
    """
    ori_video_name = os.path.splitext(os.path.basename(file_path_like_prop))[0]
    video_name, appendix = os.path.splitext(ori_video_name)
    return video_name, appendix.strip(".")

def load_track_file(file_path, cat_names):
    """Given the MOT track output files, filename/Person[Vehicle]/ etc.
    """
    track_data = {}  # cat_name -> numpy array of [N, 8], with tlbr boxes
    video_name = os.path.splitext(os.path.basename(file_path))[0]
    for cat_name in cat_names:
        track_file_path = os.path.join(file_path, cat_name, video_name + ".txt")
        assert os.path.exists(track_file_path), track_file_path
        data = []
        with open(track_file_path, "r") as f:
            for line in f:
                frame_idx, track_id, left, top, width, height, conf, gid, _, _ = line.strip().split(",")
                top, left, width, height = float(top), float(left), float(width), float(height)
                data.append([frame_idx, track_id, left, top, left + width, top + height, conf, gid])

        data = np.array(data, dtype="float32")  # [N, 8]
        #frame_ids = np.unique(data[:, 0]).tolist()

        track_data[cat_name] = data
    return track_data

def read_ava_label_map(label_map_path):
    """Parse ava's annoying ava_action_list_v2.2.pbtxt,
        TODO: label_type is not used yet
    """
    label_id = None
    label_name = None
    label_type = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line = line.strip()
            if line == "label {":
                pass
            elif line == "}":
                pass
            elif "label_id" in line:
                label_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                label_name = line.split(":")[-1].replace("\"", "").strip()
            elif "label_type" in line:
                label_type = line.split(":")[-1].strip()
            if label_id is not None and label_name is not None and label_type is not None:
                items[label_id - 1] = label_name  # 1-indexed to 0-indexed
                label_id = None
                label_name = None
                label_type = None

    return items

def parse_scene_virat(video_name):
    """Scene parsing of the video_name for virat, assume no appendix

    Returns:
        a string like 0000, 0002
    """

    s = video_name.split("_S_")[-1]
    s = s.split("_")[0]
    return s[:4]


def parse_date_time_meva(video_name):
    """File name parsing of the video_name for meva, assume no appendix

    Returns:
        a tuple of strings like 2018-03-05, 13, G339
    """

    date, start_time, end_time, location, camera = video_name.split(".")
    return date, end_time.split("-")[0], camera


def get_video_file(video_name, video_path, dataset, appendix="avi"):
    """given the top path, return the video files path"""
    assert dataset in ["meva", "virat", "other"]
    video_file = ""
    if dataset == "meva":

        if video_name.endswith("r13"):
            video_file_name = "%s.%s" % (video_name, appendix)
            date, end_time, _ = parse_date_time_meva(os.path.splitext(video_name)[0])
        else:
            video_file_name = "%s.r13.%s" % (video_name, appendix)
            date, end_time, _ = parse_date_time_meva(video_name)
        video_file = os.path.join(video_path, date, end_time,
                                  video_file_name)
    elif dataset == "virat":
        scene = parse_scene_virat(video_name)
        video_file = os.path.join(video_path, scene, "%s.%s" % (video_name, appendix))
    else:
        video_file = os.path.join(video_path, "%s.%s" % (video_name, appendix))
    assert os.path.exists(video_file), video_file
    return video_file

# from pySlowFast
def random_short_edge_jitter(images, min_size, max_size, boxes=None):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
            images (list): list of images to perform scale jitter. Dimension is
                    `height` x `width` x `channel`.
            min_size (int): the minimal size to scale the frames.
            max_size (int): the maximal size to scale the frames.
            boxes (list): optional. Corresponding boxes to images. Dimension is
                    `num boxes` x 4.
    Returns:
            (list): the list of scaled images with dimension of
                    `new height` x `new width` x `channel`.
            (ndarray or None): the scaled boxes with dimension of
                    `num boxes` x 4.
    """
    size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))

    height = images[0].shape[0]
    width = images[0].shape[1]
    if (width <= height and width == size) or \
            (height <= width and height == size):
        return images, boxes
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = [
                    proposal * float(new_height) / height for proposal in boxes]
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = [proposal * float(new_width) / width for proposal in boxes]
    return [
            cv2.resize(
                    image, (new_width, new_height),
                    interpolation=cv2.INTER_LINEAR).astype(np.float32)
            for image in images], boxes

def short_edge_resize(images, size, boxes=None, keep_scale=True):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
            images (list): list of images to perform scale jitter. Dimension is
                    `height` x `width` x `channel`.
            size (int): short edge will be resized to this
            boxes (numpy array):
            keep_scale: if False, width and height will be resize to size, otherwise only short edge
    Returns:
            (list): the list of scaled images with dimension of
                    `new height` x `new width` x `channel`.
            (ndarray or None): the scaled boxes with dimension of
                    `num boxes` x 4.
    """
    # size=256
    # (32, 631, 269, 3) -> 32, (600, 256, 3),
    height = images[0].shape[0]
    width = images[0].shape[1]
    if (width <= height and width == size) or \
            (height <= width and height == size):
        return images, boxes
    new_width = size
    new_height = size

    if keep_scale:
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

    if boxes is not None:
        boxes[:, [1, 3]] *= (float(new_height) / height)
        boxes[:, [0, 2]] *= (float(new_width) / width)
    return [
            cv2.resize(
                    image, (new_width, new_height),
                    interpolation=cv2.INTER_LINEAR).astype(np.float32)
            for image in images], boxes

def spatial_shift_crop_list(size, images, spatial_shift_pos, boxes=None):
    """
    Perform left, center, or right crop of the given list of images.
    Args:
            size (int): size to crop.
            image (list): ilist of images to perform short side scale. Dimension is
                    `height` x `width` x `channel` or `channel` x `height` x `width`.
            spatial_shift_pos (int): option includes 0 (left), 1 (middle), and
                    2 (right) crop.
            boxes (list): optional. Corresponding boxes to images.
                    Dimension is `num boxes` x 4.
    Returns:
            cropped (ndarray): the cropped list of images with dimension of
                    `height` x `width` x `channel`.
            boxes (list): optional. Corresponding boxes to images. Dimension is
                    `num boxes` x 4.
    """
    # size=256
    # 32, (600, 256, 3) -> 32, (256, 256, 3)
    assert spatial_shift_pos in [0, 1, 2]

    height = images[0].shape[0]
    width = images[0].shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_shift_pos == 0:
            y_offset = 0
        elif spatial_shift_pos == 2:
            y_offset = height - size
    else:
        if spatial_shift_pos == 0:
            x_offset = 0
        elif spatial_shift_pos == 2:
            x_offset = width - size

    cropped = [
            image[y_offset : y_offset + size, x_offset : x_offset + size, :]
            for image in images
    ]
    assert cropped[0].shape[0] == size, "Image height not cropped properly"
    assert cropped[0].shape[1] == size, "Image width not cropped properly"

    if boxes is not None:
        boxes[:, [0, 2]] -= x_offset
        boxes[:, [1, 3]] -= y_offset
    return cropped, boxes



class SeqVideoReader(object):
    def __init__(self, video_file, video_reader_name, open_now=False,
                 frame_format="rgb"):
        """A wrapper for all sorts of sequential video reader. open/close,
        then a iterator for frame read.

        Currently supports opencv, pyav, decord
        """
        assert video_reader_name in ["opencv", "pyav", "decord"]
        assert frame_format in ["bgr", "rgb"]
        self.frame_format = frame_format
        self.video_reader_name = video_reader_name
        self.video_file = video_file  # path to the video file
        self.video_cap = None  # pointer of the opened file
        self.video_num_frame = None  # number of frames for the opened video
        self.video_fps = None

        if open_now:
            self.open()

    def open(self):
        """Open the video file and extract the metadatas."""
        if self.video_reader_name == "opencv":

            self.video_cap = cv2.VideoCapture(self.video_file)
            if not self.video_cap.isOpened():
                raise Exception("Opencv cannot open %s" % self.video_file)

            # opencv 3/4
            self.video_num_frame = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = float(self.video_cap.get(cv2.CAP_PROP_FPS))

        elif self.video_reader_name == "pyav":
            # pyav will throw exception if file cannot be read
            self.video_cap = av.open(self.video_file)

            self.video_num_frame = int(self.video_cap.streams.video[0].frames)
            self.video_fps = float(self.video_cap.streams.video[0].average_rate)

        elif self.video_reader_name == "decord":

            self.video_cap = decord.VideoReader(self.video_file)

            self.video_num_frame = len(self.video_cap)
            self.video_fps = float(self.video_cap.get_avg_fps())

    def iter_frames(self):
        """Return frame in numpy array of [H, W, RGB]
        """
        if self.video_cap is None:
            raise Exception("Video %s is not open yet!" % self.video_file)

        if self.video_reader_name == "opencv":
            frame_count = 0
            while frame_count < self.video_num_frame:
                suc, frame = self.video_cap.read()
                # TODO: add handler for suc==False
                frame_count += 1
                # convert the frame to RGB format
                if self.frame_format == "rgb":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
        elif self.video_reader_name == "pyav":
            for frame in self.video_cap.decode(video=0):
                frame_format = "rgb24" \
                               if self.frame_format == "rgb" else "bgr24"
                yield frame.to_ndarray(format=frame_format)
        elif self.video_reader_name == "decord":
            for i in range(self.video_num_frame):
                # originally it is a decord.ndarray.NDArray type
                # in rgb
                np_frame = self.video_cap[i].asnumpy()
                if self.frame_format == "bgr":
                    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
                yield np_frame

    def close(self):
        if self.video_reader_name == "opencv":
            self.video_cap.release()
            cv2.destroyAllWindows()


def draw_boxes(img, boxes,
               font_scale=0.6, font_thick=1, box_thick=1, bottom_text=False):
    """Boxes are (bbox, text, color, [offset])
    """
    if not boxes:
        return img

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        box_data = boxes[i]
        assert len(box_data) in [3, 4]
        box = box_data[0]
        label = box_data[1]
        color = box_data[2]
        offset = 0
        if len(box_data) == 4:
            offset = box_data[3]

        # expand the box on y axis for overlapping results
        if offset != 0:
            box[0] -= box_thick * offset + 1
            box[2] += box_thick * offset + 1
            if bottom_text:
                box[1] -= box_thick * offset + 1
                box[3] += offset
            else:
                box[3] += box_thick * offset + 1
                box[1] -= offset

        box = [int(o) for o in box]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                      color=color, thickness=box_thick)

        # for box enlarging, replace with text height if there is label
        lineh = 2

        # find the best placement for the text
        ((linew, lineh), _) = cv2.getTextSize(label, FONT, font_scale, font_thick)
        bottom_left = [box[0] + 1, box[1] - 0.3 * lineh]
        top_left = [box[0] + 1, box[1] - 1.3 * lineh]
        if top_left[1] < 0:   # out of image
            top_left[1] = box[3] - 1.3 * lineh
            bottom_left[1] = box[3] - 0.3 * lineh

        textbox = [int(top_left[0]), int(top_left[1]),
                   int(top_left[0] + linew), int(top_left[1] + lineh)]

        text_offset = lineh * offset

        if bottom_text:
            cv2.putText(img, label, (box[0] + 2, box[3] - 4 + text_offset),
                        FONT, font_scale, color=color)
        else:
            cv2.putText(img, label, (textbox[0], textbox[3] - text_offset),
                        FONT, font_scale, color=color)


    return img
# --------- generate a list of colors
PALETTE_HEX = [
  "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
  "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
  "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
  "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
  "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
  "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
  "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
  "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
  "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
  "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
  "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
  "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
  "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
  "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
  "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
  "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
  "#7ED379", "#012C58"]


def _parse_hex_color(s):
    r = int(s[1:3], 16)
    g = int(s[3:5], 16)
    b = int(s[5:7], 16)
    return (r, g, b)
COLORS = list(map(_parse_hex_color, PALETTE_HEX))
# -----------------------
