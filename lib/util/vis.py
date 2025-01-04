import copy
import logging
import os.path
import random
import matplotlib.patches as mpatches
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib import colors
from matplotlib import pyplot as plt
from torchvision.ops.boxes import clip_boxes_to_image
from visdom import Visdom
import numpy as np
from PIL import Image, ImageDraw, ImageFont
logging.getLogger('visdom').setLevel(logging.CRITICAL)


class BaseVis(object):

    def __init__(self, viz_opts, update_mode='append', env=None, win=None,
                 resume=False, port=8097, server='http://localhost'):
        self.viz_opts = viz_opts
        self.update_mode = update_mode
        self.win = win
        if env is None:
            env = 'main'
        self.viz = Visdom(env=env, port=port, server=server)
        # if resume first plot should not update with replace
        self.removed = not resume

    def win_exists(self):
        return self.viz.win_exists(self.win)

    def close(self):
        if self.win is not None:
            self.viz.close(win=self.win)
            self.win = None

    def register_event_handler(self, handler):
        self.viz.register_event_handler(handler, self.win)


class LineVis(BaseVis):
    """Visdom Line Visualization Helper Class."""

    def plot(self, y_data, x_label):
        """Plot given data.

        Appends new data to exisiting line visualization.
        """
        update = self.update_mode
        # update mode must be None the first time or after plot data was removed
        if self.removed:
            update = None
            self.removed = False

        if isinstance(x_label, list):
            Y = torch.Tensor(y_data)
            X = torch.Tensor(x_label)
        else:
            y_data = [d.cpu() if torch.is_tensor(d)
                      else torch.tensor(d)
                      for d in y_data]

            Y = torch.Tensor(y_data).unsqueeze(dim=0)
            X = torch.Tensor([x_label])

        win = self.viz.line(X=X, Y=Y, opts=self.viz_opts, win=self.win, update=update)

        if self.win is None:
            self.win = win
        self.viz.save([self.viz.env])

    def reset(self):
        #TODO: currently reset does not empty directly only on the next plot.
        # update='remove' is not working as expected.
        if self.win is not None:
            # self.viz.line(X=None, Y=None, win=self.win, update='remove')
            self.removed = True


class ImgVis(BaseVis):
    """Visdom Image Visualization Helper Class."""

    def plot(self, images):
        """Plot given images."""

        # images = [img.data if isinstance(img, torch.autograd.Variable)
        #           else img for img in images]
        # images = [img.squeeze(dim=0) if len(img.size()) == 4
        #           else img for img in images]

        self.win = self.viz.images(
            images,
            nrow=1,
            opts=self.viz_opts,
            win=self.win, )
        self.viz.save([self.viz.env])


def vis_results(visualizer, img, result, target, tracking):
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    imgs = [inv_normalize(img).cpu()]
    img_ids = [target['image_id'].item()]
    for key in ['prev', 'prev_prev']:
        if f'{key}_image' in target:
            imgs.append(inv_normalize(target[f'{key}_image']).cpu())
            img_ids.append(target[f'{key}_target'][f'image_id'].item())

    # img.shape=[3, H, W]
    dpi = 96
    figure, axarr = plt.subplots(len(imgs))
    figure.tight_layout()
    figure.set_dpi(dpi)
    figure.set_size_inches(
        imgs[0].shape[2] / dpi,
        imgs[0].shape[1] * len(imgs) / dpi)

    if len(imgs) == 1:
        axarr = [axarr]

    for ax, img, img_id in zip(axarr, imgs, img_ids):
        ax.set_axis_off()
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1))

        ax.text(
            0, 0, f'IMG_ID={img_id}',
            fontsize=20, bbox=dict(facecolor='white', alpha=0.5))

    num_track_queries = num_track_queries_with_id = 0
    if tracking:
        num_track_queries = len(target['track_query_boxes'])
        num_track_queries_with_id = len(target['track_query_match_ids'])
        track_ids = target['track_ids'][target['track_query_match_ids']]

    keep = result['scores'].cpu() > result['scores_no_object'].cpu()

    cmap = plt.cm.get_cmap('hsv', len(keep))

    prop_i = 0
    for box_id in range(len(keep)):
        rect_color = 'green'
        offset = 0
        text = f"{result['scores'][box_id]:0.2f}"

        if tracking:
            if target['track_queries_fal_pos_mask'][box_id]:
                rect_color = 'red'
            elif target['track_queries_mask'][box_id]:
                offset = 50
                rect_color = 'blue'
                text = (
                    f"{track_ids[prop_i]}\n"
                    f"{text}\n"
                    f"{result['track_queries_with_id_iou'][prop_i]:0.2f}")
                prop_i += 1

        if not keep[box_id]:
            continue

        # x1, y1, x2, y2 = result['boxes'][box_id]
        result_boxes = clip_boxes_to_image(result['boxes'], target['size'])
        x1, y1, x2, y2 = result_boxes[box_id]

        axarr[0].add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color=rect_color, linewidth=2))

        axarr[0].text(
            x1, y1 + offset, text,
            fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        if 'masks' in result:
            mask = result['masks'][box_id][0].numpy()
            mask = np.ma.masked_where(mask == 0.0, mask)

            axarr[0].imshow(
                mask, alpha=0.5, cmap=colors.ListedColormap([cmap(box_id)]))

    query_keep = keep
    if tracking:
        query_keep = keep[target['track_queries_mask'] == 0]

    legend_handles = [mpatches.Patch(
        color='green',
        label=f"object queries ({query_keep.sum()}/{len(target['boxes']) - num_track_queries_with_id})\n- cls_score")]

    if num_track_queries:
        track_queries_label = (
            f"track queries ({keep[target['track_queries_mask']].sum() - keep[target['track_queries_fal_pos_mask']].sum()}"
            f"/{num_track_queries_with_id})\n- track_id\n- cls_score\n- iou")

        legend_handles.append(mpatches.Patch(
            color='blue',
            label=track_queries_label))

    if num_track_queries_with_id != num_track_queries:
        track_queries_fal_pos_label = (
            f"false track queries ({keep[target['track_queries_fal_pos_mask']].sum()}"
            f"/{num_track_queries - num_track_queries_with_id})")

        legend_handles.append(mpatches.Patch(
            color='red',
            label=track_queries_fal_pos_label))

    axarr[0].legend(handles=legend_handles)

    i = 1
    for frame_prefix in ['prev', 'prev_prev']:
        # if f'{frame_prefix}_image_id' not in target or f'{frame_prefix}_boxes' not in target:
        if f'{frame_prefix}_target' not in target:
            continue

        frame_target = target[f'{frame_prefix}_target']
        cmap = plt.cm.get_cmap('hsv', len(frame_target['track_ids']))

        for j, track_id in enumerate(frame_target['track_ids']):
            x1, y1, x2, y2 = frame_target['boxes'][j]
            axarr[i].text(
                x1, y1, f"track_id={track_id}",
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            axarr[i].add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color='green', linewidth=2))

            if 'masks' in frame_target:
                mask = frame_target['masks'][j].cpu().numpy()
                mask = np.ma.masked_where(mask == 0.0, mask)

                axarr[i].imshow(
                    mask, alpha=0.5, cmap=colors.ListedColormap([cmap(j)]))
        i += 1

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.axis('off')

    img = fig_to_numpy(figure).transpose(2, 0, 1)
    plt.close()

    visualizer.plot(img)


def build_visualizers(args: dict, train_loss_names: list):
    visualizers = {}
    visualizers['train'] = {}
    visualizers['val'] = {}

    if args.eval_only or args.no_vis or not args.vis_server:
        return visualizers

    env_name = str(args.output_dir).split('/')[-1]

    vis_kwargs = {
        'env': env_name,
        'resume': args.resume and args.resume_vis,
        'port': args.vis_port,
        'server': args.vis_server}

    #
    # METRICS
    #

    legend = ['loss']
    legend.extend(train_loss_names)
    # for i in range(len(train_loss_names)):
    #     legend.append(f"{train_loss_names[i]}_unscaled")

    legend.extend([
        'class_error',
        # 'loss',
        # 'loss_bbox',
        # 'loss_ce',
        # 'loss_giou',
        # 'loss_mask',
        # 'loss_dice',
        # 'cardinality_error_unscaled',
        # 'loss_bbox_unscaled',
        # 'loss_ce_unscaled',
        # 'loss_giou_unscaled',
        # 'loss_mask_unscaled',
        # 'loss_dice_unscaled',
        'lr',
        'lr_backbone',
        'iter_time'
    ])

    # if not args.masks:
    #     legend.remove('loss_mask')
    #     legend.remove('loss_mask_unscaled')
    #     legend.remove('loss_dice')
    #     legend.remove('loss_dice_unscaled')

    opts = dict(
        title="TRAIN METRICS ITERS",
        xlabel='ITERS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend)

    # TRAIN
    visualizers['train']['iter_metrics'] = LineVis(opts, **vis_kwargs)

    opts = copy.deepcopy(opts)
    opts['title'] = "TRAIN METRICS EPOCHS"
    opts['xlabel'] = "EPOCHS"
    opts['legend'].remove('lr')
    opts['legend'].remove('lr_backbone')
    opts['legend'].remove('iter_time')
    visualizers['train']['epoch_metrics'] = LineVis(opts, **vis_kwargs)

    # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = "VAL METRICS EPOCHS"
    opts['xlabel'] = "EPOCHS"
    visualizers['val']['epoch_metrics'] = LineVis(opts, **vis_kwargs)

    #
    # EVAL COCO
    #

    legend = [
        'BBOX AP IoU=0.50:0.95',
        'BBOX AP IoU=0.50',
        'BBOX AP IoU=0.75',
    ]

    if args.masks:
        legend.extend([
            'MASK AP IoU=0.50:0.95',
            'MASK AP IoU=0.50',
            'MASK AP IoU=0.75'])

    if args.tracking and args.tracking_eval:
        legend.extend(['MOTA', 'IDF1'])

    opts = dict(
        title='TRAIN EVAL EPOCHS',
        xlabel='EPOCHS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend)

    # TRAIN
    visualizers['train']['epoch_eval'] = LineVis(opts, **vis_kwargs)

    # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = 'VAL EVAL EPOCHS'
    visualizers['val']['epoch_eval'] = LineVis(opts, **vis_kwargs)

    #
    # EXAMPLE RESULTS
    #

    opts = dict(
        title="TRAIN EXAMPLE RESULTS",
        width=2500,
        height=2500)

    # TRAIN
    visualizers['train']['example_results'] = ImgVis(opts, **vis_kwargs)

    # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = 'VAL EXAMPLE RESULTS'
    visualizers['val']['example_results'] = ImgVis(opts, **vis_kwargs)

    return visualizers


def vis_dataload(samples, targets, path, font_path = None):
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    font = ImageFont.truetype(font = font_path, size = 20)
    for target in targets :
        img = target['ori_image'].permute(1, 2, 0).numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_w, img_h = img.size
        boxes = target['boxes'].cpu().numpy()
        tags = target['tags'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        file_name = target['file_name']
        draw = ImageDraw.Draw(img)

        for box, label, tag in zip(boxes, labels, tags) :

            xc = round(max(0, box[0]) * img_w)
            yc = round(max(0, box[1]) * img_h)
            w = round(max(0, box[2]) * img_w)
            h = round(max(0, box[3]) * img_h)

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[int(tag)]), width = 2)
            draw.text((x1, y1), f'{str(label)}', font = font, fill = tuple(COLORS[int(tag)]))
        img.save(f'{path}/{file_name}.png')

def vis_body_and_head_match(targets, path, font_path = None):
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    font = ImageFont.truetype(font = font_path, size = 20)
    for target in targets :
        img = target['ori_image'].cpu().permute(1, 2, 0).numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_w, img_h = img.size
        body_boxes = target['target_body_boxes'].cpu().numpy()
        pred_head_boxes = target['output_head_boxes'].cpu().detach().numpy()
        targ_head_boxes = target['target_head_boxes'].cpu().detach().numpy()
        file_name = target['file_name']
        draw = ImageDraw.Draw(img)

        for i, (body_box, pred_head_box, targ_head_box) in enumerate(zip(body_boxes, pred_head_boxes, targ_head_boxes)) :

            xc = round(max(0, body_box[0]) * img_w)
            yc = round(max(0, body_box[1]) * img_h)
            w = round(max(0, body_box[2]) * img_w)
            h = round(max(0, body_box[3]) * img_h)

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = 2)

            xc = round(max(0, pred_head_box[0]) * img_w)
            yc = round(max(0, pred_head_box[1]) * img_h)
            w = round(max(0, pred_head_box[2]) * img_w)
            h = round(max(0, pred_head_box[3]) * img_h)

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = 2)

            xc = round(max(0, targ_head_box[0]) * img_w)
            yc = round(max(0, targ_head_box[1]) * img_h)
            w = round(max(0, targ_head_box[2]) * img_w)
            h = round(max(0, targ_head_box[3]) * img_h)

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = 2)

        img.save(f'{path}/{file_name}.png')


def vis_body_and_head_match_label(targets, path, font_path = None):
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    font = ImageFont.truetype(font = font_path, size = 20)
    for target in targets :
        img = target['ori_image'].cpu().permute(1, 2, 0).numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_w, img_h = img.size
        target_body_boxes = target['target_body_boxes'].cpu().numpy()
        pred_head_boxes = target['out_head_boxes'].cpu().detach().numpy()
        target_head_boxes = target['target_head_boxes'].cpu().detach().numpy()
        file_name = target['file_name']

        draw = ImageDraw.Draw(img)
        for k, (pred_head_box, target_head_box, target_body_box) in enumerate(zip(pred_head_boxes, target_head_boxes, target_body_boxes)):
            xc = round(max(0, pred_head_box[0]) * img_w)
            yc = round(max(0, pred_head_box[1]) * img_h)
            w = round(max(0, pred_head_box[2]) * img_w)
            h = round(max(0, pred_head_box[3]) * img_h)

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[k]), width = 1)

            xc = round(max(0, target_head_box[0]) * img_w)
            yc = round(max(0, target_head_box[1]) * img_h)
            w = round(max(0, target_head_box[2]) * img_w)
            h = round(max(0, target_head_box[3]) * img_h)

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[k]), width = 1)

            xc = round(max(0, target_body_box[0]) * img_w)
            yc = round(max(0, target_body_box[1]) * img_h)
            w = round(max(0, target_body_box[2]) * img_w)
            h = round(max(0, target_body_box[3]) * img_h)

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[k]), width = 1)

        img.save(f'{path}/{file_name}.png')


def vis_results(samples, targets, results, save_path, font_path, plot_boxes = True):

    COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    font = ImageFont.truetype(font = font_path, size = 20)
    text_shift = 20
    box_width = 3
    outs = []
    for result, target in zip(results, targets):
        file_name = target['file_name']
        img = target['ori_image'].cpu().permute(1, 2, 0).numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_ = copy.deepcopy(img)
        draw = ImageDraw.Draw(img)

        body_boxes = result['body_boxes']
        body_scores = result['body_scores']

        head_boxes = result['head_boxes']
        head_scores = result['head_scores']
        body_boxes_to_head = result['body_boxes_to_head']

        body_ids_match = result['match_result']['row_ind']
        head_ids_match = result['match_result']['col_ind']

        if plot_boxes:
            for i, (body_box, body_score) in enumerate(zip(body_boxes, body_scores)):
                if i not in body_ids_match:
                    x1, y1, x2, y2 = body_box.cpu().numpy()
                    draw.rectangle([x1, y1, x2, y2], outline = tuple(COLOR[0]), width = box_width)
                    draw.text((x1, y1-text_shift), '1_B', font = font, fill = tuple(COLOR[0]))

            for i, (head_box, body_box_to_head) in enumerate(zip(head_boxes, body_boxes_to_head)):
                if i not in head_ids_match:
                    x1, y1, x2, y2 = head_box.cpu().numpy()
                    draw.rectangle([x1, y1, x2, y2], outline = tuple(COLOR[1]), width = box_width)
                    draw.text((x1, y1-text_shift), '2_H', font = font, fill = tuple(COLOR[1]))

                    x1, y1, x2, y2 = body_box_to_head.cpu().numpy()
                    draw.rectangle([x1, y1, x2, y2], outline = tuple(COLOR[1]), width = box_width)
                    draw.text((x1, y1-text_shift), '2_B', font = font, fill = tuple(COLOR[1]))

            for body_id_match, head_id_match in zip(body_ids_match, head_ids_match):
                x1, y1, x2, y2 = head_boxes[head_id_match].cpu().numpy()
                draw.rectangle([x1, y1, x2, y2], outline = tuple(COLOR[2]), width = box_width)
                draw.text((x1, y1 - text_shift), '3_H', font = font, fill = tuple(COLOR[2]))

                x1, y1, x2, y2 = body_boxes_to_head[head_id_match].cpu().numpy()
                draw.rectangle([x1, y1, x2, y2], outline = tuple(COLOR[2]), width = box_width)
                draw.text((x1, y1 - text_shift), '3_B1', font = font, fill = tuple(COLOR[2]))

                x1, y1, x2, y2 = body_boxes[body_id_match].cpu().numpy()
                draw.rectangle([x1, y1, x2, y2], outline = tuple(COLOR[2]), width = box_width)
                draw.text((x1, y1 - text_shift), '3_B2', font = font, fill = tuple(COLOR[2]))

            img.save(f'{save_path}/{file_name}_1.png')

        body_match_mask = torch.zeros(len(body_boxes)).bool()
        head_match_mask = torch.zeros(len(head_boxes)).bool()

        body_match_mask[body_ids_match] = True
        head_match_mask[head_ids_match] = True

        unmatch_body_boxes = body_boxes[~body_match_mask]
        unmatch_body_scores = body_scores[~body_match_mask]

        unmatch_head_boxes = head_boxes[~head_match_mask]
        unmatch_head_scores = head_scores[~head_match_mask]
        unmatch_body_boxes_to_head = body_boxes_to_head[~head_match_mask]

        match_head_boxes = head_boxes[head_ids_match]
        match_head_scores = head_scores[head_ids_match]
        match_body_boxes_to_head = body_boxes_to_head[head_ids_match]
        if plot_boxes:
            count = 0
            draw = ImageDraw.Draw(img_)
            for i, unmatch_body_box in enumerate(unmatch_body_boxes.cpu().numpy(), count):
                count = count + 1
                x1, y1, x2, y2 = unmatch_body_box
                draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = box_width)
                draw.text((x1, y1 - text_shift), 'B', font = font, fill = tuple(COLORS[i]))

            for j, (unmatch_head_box, unmatch_body_to_head) in enumerate(zip(unmatch_head_boxes.cpu().numpy(), unmatch_body_boxes_to_head.cpu().numpy()), count):
                count = count + 1
                x1, y1, x2, y2 = unmatch_head_box
                draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[j]), width = box_width)
                draw.text((x1, y1 - text_shift), 'MH', font = font, fill = tuple(COLORS[j]))

                x1, y1, x2, y2 = unmatch_body_to_head
                draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[j]), width = box_width)
                draw.text((x1, y1 - text_shift), 'MB', font = font, fill = tuple(COLORS[j]))

            for k, (match_head_box, match_body_box_to_head) in enumerate(zip(match_head_boxes.cpu().numpy(), match_body_boxes_to_head.cpu().numpy()), count):
                count = count + 1
                x1, y1, x2, y2 = match_head_box
                draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[k]), width = box_width)
                draw.text((x1, y1 - text_shift), 'MH', font = font, fill = tuple(COLORS[k]))

                x1, y1, x2, y2 = match_body_box_to_head
                draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[k]), width = box_width)
                draw.text((x1, y1 - text_shift), 'MB', font = font, fill = tuple(COLORS[k]))

            img_.save(f'{save_path}/{file_name}_2.png')

        out = {'match' : {'body' : torch.cat([unmatch_body_boxes_to_head, match_body_boxes_to_head]),
                          'head' : torch.cat([unmatch_head_boxes, match_head_boxes]),
                          'score' : torch.cat([unmatch_head_scores, match_head_scores])},
               'unmatch' : {'body' : unmatch_body_boxes,
                            'score' : unmatch_body_scores}
               }
    outs.append(out)
    return outs
#
def plot_boxes_wo_bh(out, target, image_path, save_path, font_path):
    COLOR = [(255, 255, 0), (0, 255, 0)]
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    font = ImageFont.truetype(font = font_path, size = 20)
    text_shift = 20
    box_width = 3

    body_boxes = out['body_boxes']
    body_scores = out['body_scores']

    head_boxes = out['head_boxes']
    head_scores = out['head_scores']

    file_name = target['file_name']
    file_prefix, file_posfix = file_name.split('.')
    img = Image.open(os.path.join(image_path, file_name))
    img0 = copy.deepcopy(img)
    draw = ImageDraw.Draw(img)

    if 'boxes' in target:
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        gt_tags = target['tags']
        img_h, img_w = target['orig_size']

        for gt_box, gt_label, gt_tag in zip(gt_boxes, gt_labels, gt_tags):

            xc = gt_box[0] * img_w
            yc = gt_box[1] * img_h
            w = gt_box[2] * img_w
            h = gt_box[3] * img_h

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[gt_tag]), width = box_width)
            draw.text((x1, y1 - text_shift), f'{gt_label}', font = font, fill = tuple(COLORS[gt_tag]))
        img.save(os.path.join(save_path, f'{file_prefix}_gt.{file_posfix}'))

    draw = ImageDraw.Draw(img0)
    for body_box, body_score in zip(body_boxes, body_scores):
        x1, y1, x2, y2 = body_box
        draw.rectangle([x1, y1, x2, y2], outline = tuple(COLOR[0]), width = box_width)
        draw.text((x1, y1 - text_shift), f'0:{body_score:.2f}', font = font, fill = tuple(COLOR[0]))

    for head_box, head_score in zip(head_boxes, head_scores):
        x1, y1, x2, y2 = head_box
        draw.rectangle([x1, y1, x2, y2], outline = tuple(COLOR[1]), width = box_width)
        draw.text((x1, y1 - text_shift), f'1:{head_score:.2f}', font = font, fill = tuple(COLOR[1]))
    img0.save(os.path.join(save_path, f'{file_prefix}_dt.{file_posfix}'))

def plot_boxes_w_bh_head_decoder(outs, outputs_head_decoder, targets, image_path, save_path, font_path):
    COLOR = [(255, 255, 0), (0, 255, 0)]
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    font = ImageFont.truetype(font = font_path, size = 20)
    text_shift = 20
    box_width = 3

    for out, output_head_decoder, target in zip(outs, outputs_head_decoder, targets):

        bboxs = out['boxes']
        scores = out['scores']
        labels = out['labels']

        head_mask = (labels == 1) & (scores > 0.01)
        head_boxes = bboxs[head_mask]
        body_boxes = output_head_decoder['boxes'][head_mask]

        file_name = target['file_name']
        file_prefix, file_posfix = file_name.split('.')
        img = Image.open(os.path.join(image_path, file_name))
        draw = ImageDraw.Draw(img)

        for i, (head_box, body_box) in enumerate(zip(head_boxes, body_boxes)):
            x1, y1, x2, y2 = head_box
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = box_width)

            x1, y1, x2, y2 = body_box
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = box_width)
        img.save(os.path.join(save_path, f'{file_prefix}_second_match.{file_posfix}'))

def plot_boxes_w_bh(out, target, image_path, save_path, font_path):
    COLOR = [(255, 255, 0), (0, 255, 0)]
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    font = ImageFont.truetype(font = font_path, size = 20)
    text_shift = 20
    box_width = 3

    body_boxes = out['body_boxes']
    body_scores = out['body_scores']

    head_boxes = out['head_boxes']
    head_scores = out['head_scores']

    file_name = target['file_name']
    file_prefix, file_posfix = file_name.split('.')
    img = Image.open(os.path.join(image_path, file_name))
    img0 = copy.deepcopy(img)
    draw = ImageDraw.Draw(img)

    if 'boxes' in target:
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        gt_tags = target['tags']
        img_h, img_w = target['orig_size']

        for gt_box, gt_label, gt_tag in zip(gt_boxes, gt_labels, gt_tags):

            xc = gt_box[0] * img_w
            yc = gt_box[1] * img_h
            w = gt_box[2] * img_w
            h = gt_box[3] * img_h

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[gt_tag]), width = box_width)
            draw.text((x1, y1 - text_shift), f'{gt_label}', font = font, fill = tuple(COLORS[gt_tag]))
        img.save(os.path.join(save_path, f'{file_prefix}_gt.{file_posfix}'))

    draw = ImageDraw.Draw(img0)
    for i, (body_box, body_score) in enumerate(zip(body_boxes, body_scores)):
        x1, y1, x2, y2 = body_box
        draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = box_width)

    for i, (head_box, head_score) in enumerate(zip(head_boxes, head_scores)):
        x1, y1, x2, y2 = head_box
        draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = box_width)
    img0.save(os.path.join(save_path, f'{file_prefix}_match_dt.{file_posfix}'))

def plot_boxes_one_class(out, target, image_path, save_path, font_path):
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    font = ImageFont.truetype(font = font_path, size = 20)
    text_shift = 20
    box_width = 5

    boxes = out['boxes']
    scores = out['scores']

    file_name = target['file_name']
    file_prefix, file_posfix = file_name.split('.')
    img = Image.open(os.path.join(image_path, file_name))
    img0 = copy.deepcopy(img)
    draw = ImageDraw.Draw(img)

    if 'boxes' in target:
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        gt_tags = target['tags']
        img_h, img_w = target['orig_size']

        for gt_box, gt_label, gt_tag in zip(gt_boxes, gt_labels, gt_tags):

            xc = gt_box[0] * img_w
            yc = gt_box[1] * img_h
            w = gt_box[2] * img_w
            h = gt_box[3] * img_h

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[gt_tag]), width = box_width)
            draw.text((x1, y1 - text_shift), f'{gt_label}', font = font, fill = tuple(COLORS[gt_tag]))
        img.save(os.path.join(save_path, f'{file_prefix}_gt.{file_posfix}'))

    draw = ImageDraw.Draw(img0)
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[i]), width = box_width)
        draw.text((x1, y1 - text_shift), f'{score:.2f}', font = font, fill = tuple(COLORS[i]))
    img0.save(os.path.join(save_path, f'{file_prefix}_dt.{file_posfix}'))

def vis_match_result(targets, match_indice, outputs_without_aux, path, font_path):
    COLORS = np.random.randint(0, 255, size = (100000, 3))
    font = ImageFont.truetype(font = font_path, size = 20)
    text_shift = 10
    out_boxes = outputs_without_aux['pred_boxes']

    for indice, target, out_box in zip(match_indice, targets, out_boxes) :
        img = target['ori_image'].cpu().permute(1, 2, 0).numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_w, img_h = img.size
        boxes = target['boxes']

        draw = ImageDraw.Draw(img)

        out_ids = indice[0]
        target_ids = indice[1]
        file_name = target['file_name']
        for idx, (out_id, target_id) in enumerate(zip(out_ids, target_ids)):
            xc = max(0, boxes[target_id][0]) * img_w
            yc = max(0, boxes[target_id][1]) * img_h
            w  = max(0, boxes[target_id][2]) * img_w
            h  = max(0, boxes[target_id][3]) * img_h
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[idx]), width = 2)
            draw.text((x1, y1 - text_shift), f'{idx}', font = font, fill = tuple(COLORS[idx]))

            xc = max(0, out_box[out_id][0]) * img_w
            yc = max(0, out_box[out_id][1]) * img_h
            w  = max(0, out_box[out_id][2]) * img_w
            h  = max(0, out_box[out_id][3]) * img_h
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            draw.rectangle([x1, y1, x2, y2], outline = tuple(COLORS[idx]), width = 2)
            draw.text((x1, y1 - text_shift), f'{idx}', font = font, fill = tuple(COLORS[idx]))
        img.save(f'{path}/{file_name}.png')