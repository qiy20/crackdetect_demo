from pathlib import Path
import logging
import time
import math
from multiprocessing import Pool

import numpy as np
import torch
import torchvision
import cv2
import onnxruntime

from utils import box_ioa, clip_coords, xywh2xyxy, plot_one_box, resize, read_img, increment_path

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CrackDetector:
    names = ['TransverseCrack', 'LongitudinalCrack', 'AlligatorCrack', 'StripRepair']
    color = [(0, 0, 255), (255, 0, 255), (3, 97, 255), (255, 0, 0)]

    def __init__(self, model,  # onnx model
                 device,  # device: cpu or cuda
                 max_bs=48,  # max batch_size(consider cuda memory)
                 infer_size=640,
                 stride=32,  # max down sampling stride
                 # preprocess
                 num_workers=0,
                 # box params
                 conf_threshold_box=0.1, iou_threshold=0.45, classes_box=None, max_det=100,
                 ras_threshold=0.6, ras_crack_agnostic=True, max_w=4096,
                 # grid params
                 conf_threshold_grid=0.1, match=True, classes_grid=None,):
        if Path(model).suffix == '.onnx':
            model = onnxruntime.InferenceSession(model, providers=[f"{device.upper()}ExecutionProvider"])
        else:
            raise TypeError(f"{model} is not an onnx model")
        self.device = device
        self.model = model
        self.max_bs = max_bs
        new_size = math.ceil(infer_size / stride) * stride
        if new_size != infer_size:
            logger.warning(f'infer-size {infer_size} must be multiple of max stride {stride}, updating to {new_size}')
        self.infer_size = new_size
        self.stride = stride
        assert 0 <= conf_threshold_box <= 1, f'Invalid Confidence threshold {conf_threshold_box}, ' \
                                             f'valid values are between 0.0 and 1.0'
        assert 0 <= conf_threshold_grid <= 1, f'Invalid Confidence threshold {conf_threshold_grid},' \
                                              f' valid values are between 0.0 and 1.0'
        assert 0 <= iou_threshold <= 1, f'Invalid IoU {iou_threshold}, valid values are between 0.0 and 1.0'
        assert 0 <= ras_threshold <= 1, f'Invalid IoU {ras_threshold}, valid values are between 0.0 and 1.0'
        self.conf_threshold_box = conf_threshold_box
        self.iou_threshold = iou_threshold
        self.conf_threshold_grid = conf_threshold_grid
        self.max_det = max_det
        self.match = match
        if classes_box is None:
            classes_box = [0, 1, 2, 3]
        if classes_grid is None:
            classes_grid = [0, 1]
        self.classes_box = classes_box
        self.classes_grid = classes_grid
        self.ras_threshold = ras_threshold
        self.ras_crack_agnostic = ras_crack_agnostic
        self.max_w = max_w
        self.num_workers = num_workers

    def run(self, paths,  # path or List[paths]
            save_dir, view_img=False, save_img=False, save_txt=True, grid=True, ):
        # directory
        save_dir = increment_path(save_dir, mkdir=True)
        for i in ['box_image', 'box_txt', 'grid_image', 'grid_txt']:
            (save_dir / i).mkdir(exist_ok=True)
        if isinstance(paths, (str, Path)):  # path-->[path]
            paths = [paths]

        # inferences
        start = time.time()
        paths, images, raw_images, ratio = self.preprocess(paths)
        batch_size = len(paths)
        pred = self.model.run(None, {self.model.get_inputs()[0].name: images})
        pred = [torch.tensor(pre, device=self.device) for pre in pred]
        if len(pred) == 2:
            box_pred = pred[0]
            grid_pred = pred[1]
        elif len(pred) == 6:
            grid_pred = pred[0].permute(0, 3, 1, 2).contiguous()
            box_pred = pred[1]
        box_pred, grid_pred, paths = self.postprocess(box_pred, grid_pred, paths, ratio)
        logger.info(f'batch_size is {batch_size},total inference time is {time.time() - start}s')
        assert len(box_pred) == len(grid_pred) == len(paths) == len(raw_images)

        # save result
        self.export('box', save_dir, box_pred, paths, raw_images, view_img, save_img, save_txt)
        if grid:
            self.export('grid', save_dir, grid_pred, paths, raw_images, view_img, save_img, save_txt)

    def preprocess(self, paths):
        paths = [str(Path(path).absolute()) for path in paths]
        image_ids, images, raw_images, ratio = [], [], [], []
        # split wide(little aspect ratio) img (2048*4096 --> 2*2048*2048) and resize to infer_size
        if self.num_workers > 0:
            pool = Pool(self.num_workers)
            for path in paths:
                raw_images.append(pool.apply_async(func=read_img, args=(path,)))
            pool.close()
            pool.join()
            raw_images = [p.get() for p in raw_images]
        else:
            for path in paths:
                raw_images.append(read_img(path))
        for path, img0 in zip(paths, raw_images):
            h, w, _ = img0.shape
            new_img = resize(img0, self.infer_size, self.stride)
            _, new_h, new_w = new_img.shape
            images.append(new_img)
            image_ids.append(path)
            ratio.append((w / new_w, h / new_h))
        self.new_size = (new_h, new_w)
        assert len(image_ids) <= self.max_bs, 'There are too many pictures to process'
        return image_ids, np.stack(images), raw_images, ratio

    def postprocess(self, box_pred, grid_pred, paths, ratio):
        box_pred = self.ras(self.nms(box_pred))
        grid_pred = self.grid2box(grid_pred)
        box_pred = [clip_coords(d, self.new_size) for d in box_pred]
        assert len(box_pred) == len(grid_pred)
        # match gird to box
        for i, (d, g, r) in enumerate(zip(box_pred, grid_pred, ratio)):
            d[:, :4:2] *= r[0]
            d[:, 1:4:2] *= r[1]
            g[:, :4:2] *= r[0]
            g[:, 1:4:2] *= r[1]
            g_class = g[:, 5:]
            g_class[g_class == 1] = 3
            # copy crack 2 times
            g_crack_1 = g[g_class.flatten() == 0].clone()
            g_crack_1[:, 5:] = 1
            g_crack_2 = g[g_class.flatten() == 0].clone()
            g_crack_2[:, 5:] = 2
            g = torch.cat([g, g_crack_1, g_crack_2])
            if g.numel() > 0 and self.match:
                d_class = d[:, 5:]
                d_boxes = d[:, :4] + d_class * self.max_w
                g_boxes = g[:, :4] + g[:, 5:] * self.max_w
                iou = box_ioa(g_boxes, d_boxes)
                index = iou.max(1)[0] > 0 if d.numel() > 0 else torch.zeros(g.shape[0], device=self.device).bool()
                g = g[index]
            box_pred[i] = d
            grid_pred[i] = g

        return box_pred, grid_pred, paths

    def grid2box(self, prediction):  # [bs,gy,gx,3]-->[bs,nc,6]
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):
            # keep only the qualified pred
            conf, j = x.max(0)
            top, left = (conf > self.conf_threshold_grid).nonzero(as_tuple=False).T
            # generate boxes
            x = torch.stack([left, top, left + 1, top + 1, conf[top, left], j[top, left]], dim=1)
            if self.classes_grid is not None:  # filter by class
                x = x[(x[:, 5:6] == torch.tensor(self.classes_grid, device=x.device)).any(1)]
            x[:, :4] *= self.stride
            output[xi] = x
        return output

    def nms(self, prediction):  # [bs,nc,11]-->[bs,new_nc,6]
        xc = prediction[..., 4] > self.conf_threshold_box  # candidates
        # Settings
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = xywh2xyxy(x[:, :4])
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.conf_threshold_box]
            if self.classes_box is not None:  # filter by class
                x = x[(x[:, 5:6] == torch.tensor(self.classes_box, device=x.device)).any(1)]
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            bias = x[:, 5:6] * self.infer_size  # nms on each class separately
            boxes, scores = x[:, :4] + bias, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, self.iou_threshold)  # NMS
            if i.shape[0] > self.max_det:  # limit nums of boxes
                i = i[:self.max_det]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                logger.warning(f'NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded
        return output

    def ras(self, prediction):
        output = [torch.zeros((0, 6), device=prediction[0].device)] * len(prediction)
        for xi, x in enumerate(prediction):
            n = x.shape[0]
            _, order = torch.sort(x[:, 4])
            x = x[order]
            # if group the three classes of crack together
            bias = x[:, 5:6] * self.max_w if not self.ras_crack_agnostic else (x[:, 4:5] < 4) * self.max_w
            boxes = x[:, :4] + bias
            ioa = box_ioa(boxes, boxes)
            mask = torch.arange(n).repeat(n, 1) <= torch.arange(n).view(-1, 1)  # mask low conf boxes
            ioa[mask] = 0
            x = x[ioa.sum(1) < self.ras_threshold]
            output[xi] = x
        return output

    def export(self, stage, save_dir, pred, paths, raw_images, view_img, save_img, save_txt):
        for p, dp, img in zip(paths, pred, raw_images):
            p = Path(p)
            img_path = save_dir / f'{stage}_image' / p.name
            txt_path = save_dir / f'{stage}_txt' / (p.stem + '.txt')
            img0 = img.copy()
            for *xyxy, conf, cls in dp:
                c = int(cls)
                label = f'{conf:.2f}' if stage == 'grid' else f'{self.names[c]} {conf:.2f}'
                if save_txt:  # Write to file
                    line = (self.names[c], *xyxy, conf)
                    with open(txt_path, 'a') as f:
                        f.write(('%s ' + '%g ' * (len(line) - 1)).rstrip() % line + '\n')
                if save_img or view_img:
                    img0 = plot_one_box(xyxy, img0, label=label, color=self.color[c])
            if save_img:
                cv2.imwrite(str(img_path), img0)
            if view_img:
                cv2.imshow(str(p), img0)
                cv2.waitKey(1)


if __name__ == '__main__':
    my_model = CrackDetector(model='models/x_version2.onnx', device='cuda')
    for project in ['中公高科', '武大', '大车']:
        root = Path('data/' + project)
        data_paths = [path for path in root.iterdir()]
        # first run will be slower
        my_model.run(data_paths, save_dir='res_version2/' + project, save_img=True, save_txt=True, view_img=False)
        # test real time consuming
        # my_model.run(data_paths, save_dir='res', save_img=False, save_txt=False, view_img=False)
