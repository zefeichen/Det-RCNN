import os
from argparse import Namespace
import numpy as np
import sacred
import torch
from models import build_model
import shutil
from PIL import Image, ImageDraw, ImageFont
import cv2
from lib.util.misc import nested_dict_to_namespace
from lib.util.misc import body_match
from data import transforms as T
from PIL import Image
import torchvision.transforms.functional as F
from tqdm import tqdm
import copy
import warnings
warnings.filterwarnings('ignore')

ex = sacred.Experiment('train')
ex.add_config(os.path.join(os.getcwd(), 'cfgs/cfg.yaml'))
COLOR = [(255, 255, 0), (0, 255, 0)]
COLORS = [(0, 203, 204), (209, 225, 233), (241, 64, 64), (242, 203, 5), (242, 212, 201)]

def get_data(img_path):
    ori_image = Image.open(img_path).convert("RGB")
    image, target = T.RandomResize([800], max_size = 1333)(ori_image)
    image = F.to_tensor(image)
    image = F.normalize(image, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    w, h = ori_image.size
    ori_image = torch.from_numpy(np.array(ori_image)).permute(2, 0, 1).float()/255.0
    return ori_image, image, torch.tensor([h, w])

def resize_img(image, short_size, max_size):
    height = image.shape[0]
    width = image.shape[1]
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    resized_image = cv2.resize(
            image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale


def plot_boxes(out, target, image_path, save_path, font_path):
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    box_width = 3

    body_boxes = out['body_boxes']
    body_scores = out['body_scores']

    head_boxes = out['head_boxes']
    head_scores = out['head_scores']

    file_name = target['file_name']
    img = Image.open(os.path.join(image_path, file_name))
    draw = ImageDraw.Draw(img)
    for i, (body_box, body_score) in enumerate(zip(body_boxes, body_scores)):
        x1, y1, x2, y2 = body_box
        draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = box_width)

    for i, (head_box, head_score) in enumerate(zip(head_boxes, head_scores)):
        x1, y1, x2, y2 = head_box
        draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = box_width)
    img.save(os.path.join(save_path, file_name))

def main(args: Namespace, image_path, checkpoint_path) -> None:
    font_path = os.path.join(args.root, args.font_path)
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.deterministic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    model.eval()

    save_path = os.path.join(image_path, 'output')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    image_names = sorted(os.listdir(image_path))
    image_names = [image_name for image_name in image_names if ('jpg' in image_name or 'png' in image_name)]
    pbar = tqdm(image_names)
    for image_name in pbar :
        pbar.set_description(f"{image_name}")
        ori_image, resized_img, size = get_data(os.path.join(image_path, image_name))
        resized_img = resized_img.unsqueeze(0).to(device)
        size = size.to(device)
        targets = [{'file_name' : image_name, 'ori_image' : ori_image, 'size' : size}]
        orig_target_sizes = torch.stack([t["size"] for t in targets], dim = 0)

        with torch.no_grad() :
            outputs, outputs_head_decoder = model(resized_img)
        outputs = postprocessors['bbox'](outputs, orig_target_sizes)
        outputs_head_decoder = postprocessors['bbox'](outputs_head_decoder, orig_target_sizes)
        outs = body_match(outputs, outputs_head_decoder, thr = args.model_cfg.inference_thr, match_iou_thr = args.model_cfg.match_iou_thr)
        for out, target in zip(outs, targets) :
            plot_boxes(out, target, image_path, save_path, font_path)


@ex.main
def load_config(_config, _run):
    sacred.commands.print_config(_run)

if __name__ == '__main__':
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    args.root = os.getcwd()
    image_path = os.path.join(args.root, 'imgs')
    args.checkpoint_dir = os.path.join(args.root, 'weights/CrowdHuman.pth')
    main(args, image_path, args.checkpoint_dir)
