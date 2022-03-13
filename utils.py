from pathlib import Path
import glob
import re

import torch
import numpy as np
import cv2
import math


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, shape[1])  # x1
    boxes[:, 1].clamp_(0, shape[0])  # y1
    boxes[:, 2].clamp_(0, shape[1])  # x2
    boxes[:, 3].clamp_(0, shape[0])  # y2
    return boxes


def inter(box1, box2):
    return (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)


def box_ioa(box1, box2):
    area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    return inter(box1, box2) / area.view(-1, 1)  # iou = inter / (area1 + area2 - inter)


def plot_one_box(box, im, color=(128, 128, 128), txt_color=(255, 255, 255), label=None, line_width=3):
    # Plots one xyxy box on image im with label
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    lw = line_width or max(int(min(im.size) / 200), 2)  # line width
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        txt_width, txt_height = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        c2 = c1[0] + txt_width, c1[1] - txt_height - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    return im


def resize(img, new_size, stride=32):
    h, w, _ = img.shape
    if h != 640 or w % 32 != 0:
        ratio = h / new_size
        w = round(w / ratio / stride) * stride
        img = cv2.resize(img, (w, new_size), interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
        img = img.astype(np.float32) / 255.0
    return img


def read_img(path):
    img = cv2.imread(path)
    # img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)# Chinese in Path
    assert img is not None, 'Image Not Found ' + path
    return img


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 1  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    directory = path if path.suffix == '' else path.parent  # directory
    if not directory.exists() and mkdir:
        directory.mkdir(parents=True, exist_ok=True)  # make directory
    return path
