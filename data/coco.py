# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import copy
import random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from . import transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    fields = ["labels", "area", "iscrowd", "boxes", "tags"]
    def __init__(self,  img_folder, ann_file, transforms, norm_transforms, body = True, head = True,
                 overflow_boxes=False, remove_no_obj_imgs=True):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._norm_transforms = norm_transforms
        self.prepare = ConvertCocoPolysToMask(overflow_boxes, body, head)

        annos_image_ids = [
            ann['image_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())]
        if remove_no_obj_imgs:
            self.ids = sorted(list(set(annos_image_ids)))

    def _getitem_from_id(self, image_id, random_state=None):
        # if random state is given we do the data augmentation with the state
        # and then apply the random jitter. this ensures that (simulated) adjacent
        # frames have independent jitter.
        if random_state is not None:
            curr_random_state = {
                'random': random.getstate(),
                'torch': torch.random.get_rng_state()}
            random.setstate(random_state['random'])
            torch.random.set_rng_state(random_state['torch'])

        img, target = super(CocoDetection, self).__getitem__(image_id)

        image_id = self.ids[image_id]
        target = {'image_id': image_id,
                  'annotations': target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # ignore
        ignore = target.pop("ignore").bool()
        for field in self.fields :
            if field in target :
                target[f"{field}_ignore"] = target[field][ignore]
                target[field] = target[field][~ignore]

        if random_state is not None:
            random.setstate(curr_random_state['random'])
            torch.random.set_rng_state(curr_random_state['torch'])

        target['ori_image'] = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()/255.0
        target['file_name'] = self.coco.loadImgs(image_id)[0]["file_name"]
        img, target = self._norm_transforms(img, target)

        return img, target


    def __getitem__(self, idx):
        random_state = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()}
        img, target = self._getitem_from_id(idx, random_state)
        return img, target

    def write_result_files(self, *args):
        pass


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if isinstance(polygons, dict):
            rles = {'size': polygons['size'],
                    'counts': polygons['counts'].encode(encoding='UTF-8')}
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, overflow_boxes=False, body = True, head = True):
        self.overflow_boxes = overflow_boxes
        self.body = body
        self.head = head
    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        annos = target["annotations"]
        annos = [obj for obj in annos if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes, labels, tags, area, ignore, iscrowd, numid = [], [], [], [], [], [], 0
        for anno in annos:
            if self.body and self.head:
                if 'bbox' in anno:
                    body_bbox = anno['bbox']
                    area.append(body_bbox[2] * body_bbox[3])
                    boxes.append(body_bbox)
                    labels.append(1)
                    tags.append(numid)
                    if 'ignore' in anno:
                        ignore.append(anno['ignore'])
                    else:
                        ignore.append(0)
                    if 'iscrowd' in anno:
                        iscrowd.append(anno['iscrowd'])
                    else:
                        iscrowd.append(0)

                if 'h_bbox' in anno :
                    head_bbox = anno['h_bbox']
                    area.append(head_bbox[2] * head_bbox[3])
                    boxes.append(head_bbox)
                    labels.append(2)
                    tags.append(numid)
                    if 'ignore' in anno :
                        ignore.append(anno['ignore'])
                    else :
                        ignore.append(0)
                    if 'iscrowd' in anno :
                        iscrowd.append(anno['iscrowd'])
                    else :
                        iscrowd.append(0)

            if self.body and not(self.head):
                if 'bbox' in anno:
                    body_bbox = anno['bbox']
                    area.append(body_bbox[2] * body_bbox[3])
                    boxes.append(body_bbox)
                    labels.append(0)
                    tags.append(numid)
                    if 'ignore' in anno:
                        ignore.append(anno['ignore'])
                    else:
                        ignore.append(0)
                    if 'iscrowd' in anno:
                        iscrowd.append(anno['iscrowd'])
                    else:
                        iscrowd.append(0)

            if not(self.body) and self.head:
                head_bbox = anno['h_bbox']
                area.append(head_bbox[2] * head_bbox[3])
                boxes.append(head_bbox)
                labels.append(0)
                tags.append(numid)
                if 'ignore' in anno :
                    ignore.append(anno['ignore'])
                else :
                    ignore.append(0)
                if 'iscrowd' in anno :
                    iscrowd.append(anno['iscrowd'])
                else :
                    iscrowd.append(0)

            numid = numid + 1

        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        boxes[:, 2 :] += boxes[:, :2]   # x,y,w,h --> x,y,x,y
        # boxes[:, 0 : :2].clamp_(min = 0, max = w)
        # boxes[:, 1 : :2].clamp_(min = 0, max = h)
        labels = torch.as_tensor(labels, dtype = torch.long)
        area = torch.as_tensor(area, dtype = torch.float32)
        tags = torch.as_tensor(tags, dtype = torch.int32)
        ignore = torch.as_tensor(ignore, dtype = torch.int32)
        iscrowd = torch.as_tensor(iscrowd, dtype = torch.int32)

        keep = ((boxes[:, 3] - boxes[:, 1]) > 1) & ((boxes[:, 2] - boxes[:, 0]) > 1)

        target = {}
        target["boxes"] = boxes[keep]
        target['labels'] = labels[keep]
        target['tags'] = tags[keep]
        target["area"] = area[keep]
        target["ignore"] = ignore[keep]
        target["iscrowd"] = iscrowd[keep]
        target["image_id"] = image_id
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target['annotations'] = annos
        return image, target

def make_coco_transforms(image_set, img_transform=None, overflow_boxes=False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # default
    max_size = 1333
    val_width = 800
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    random_resizes = [400, 500, 600]
    random_size_crop = (384, 600)

    if img_transform is not None:
        scale = img_transform.max_size / max_size
        max_size = img_transform.max_size
        val_width = img_transform.val_width

        # scale all with respect to custom max_size
        scales = [int(scale * s) for s in scales]
        random_resizes = [int(scale * s) for s in random_resizes]
        random_size_crop = [int(scale * s) for s in random_size_crop]

    if image_set == 'train':
        transforms = [
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(random_resizes),
                    T.RandomSizeCrop(*random_size_crop, overflow_boxes=overflow_boxes),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
        ]
    elif image_set == 'val':
        transforms = [
            T.RandomResize([val_width], max_size=max_size),
        ]
    else:
        ValueError(f'unknown {image_set}')

    # transforms.append(normalize)
    return T.Compose(transforms), normalize


def build(image_set, args, mode='instances'):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    # image_set is 'train' or 'val'
    split = getattr(args, f"{image_set}_split")

    splits = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    if image_set == 'train':
        prev_frame_rnd_augs = args.coco_and_crowdhuman_prev_frame_rnd_augs
    elif image_set == 'val':
        prev_frame_rnd_augs = 0.0

    transforms, norm_transforms = make_coco_transforms(image_set, args.img_transform, args.overflow_boxes)
    img_folder, ann_file = splits[split]
    dataset = CocoDetection(
        img_folder, ann_file, transforms, norm_transforms,
        return_masks=args.masks,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame,
        min_num_objects=args.coco_min_num_objects)

    return dataset
