# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import copy
import os

import torch
import torch.nn.functional as F
from torch import nn

from lib.util import box_ops
from lib.util.misc import (NestedTensor, accuracy, dice_loss, get_world_size,
                           interpolate, is_dist_avail_and_initialized,
                           nested_tensor_from_tensor_list, focal_loss)

from lib.util.vis import vis_match_result, vis_body_and_head_match
class DETR(nn.Module):
    """ This is the DETR module that performs object detection. """

    def __init__(self, backbone, transformer, num_classes, num_queries, bh,
                 aux_loss=False, overflow_boxes=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO, we
                         recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)

        if bh:
            self.head_class_embed = nn.Linear(self.hidden_dim, 2)
            self.head_bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)

        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        # match interface with deformable DETR
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model

    @property
    def fpn_channels(self):
        """ Returns FPN channels. """
        return self.backbone.num_channels[:3][::-1]
        # return [1024, 512, 256]

    def forward(self, samples: NestedTensor, targets: list = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                               containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized
                               in [0, 1], relative to the size of each individual image
                               (disregarding possible padding). See PostProcess for information
                               on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It
                                is a list of dictionnaries containing the two above keys for
                                each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        # src = self.input_proj[-1](src)
        src = self.input_proj(src)
        pos = pos[-1]

        batch_size, _, _, _ = src.shape

        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = None
        if targets is not None and 'track_query_hs_embeds' in targets[0]:
            # [BATCH_SIZE, NUM_PROBS, 4]
            track_query_hs_embeds = torch.stack([t['track_query_hs_embeds'] for t in targets])

            num_track_queries = track_query_hs_embeds.shape[1]

            track_query_embed = torch.zeros(
                num_track_queries,
                batch_size,
                self.hidden_dim).to(query_embed.device)
            query_embed = torch.cat([
                track_query_embed,
                query_embed], dim=0)

            tgt = torch.zeros_like(query_embed)
            tgt[:num_track_queries] = track_query_hs_embeds.transpose(0, 1)

            for i, target in enumerate(targets):
                target['track_query_hs_embeds'] = tgt[:, i]

        assert mask is not None
        hs, hs_without_norm, memory = self.transformer(
            src, mask, query_embed, pos, tgt)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'hs_embed': hs_without_norm[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out, targets, features, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module) :
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, args, weight_dict, losses, head_decoder_losses, bh, focal_alpha = 0.25, focal_gamma = 2) :
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.args = args
        self.weight_dict = weight_dict
        self.losses = losses
        self.head_decoder_losses = head_decoder_losses
        self.bh = bh
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def loss_labels(self, outputs, targets, indices, num_boxes, log = True) :
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype = torch.int64, device = src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype = src_logits.dtype, layout = src_logits.layout,
                                            device = src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        loss_ce = focal_loss(src_logits, target_classes_onehot, num_boxes, alpha = self.focal_alpha,
                             gamma = self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce' : loss_ce}

        if log :
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes) :
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device = device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error' : card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes) :
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim = 0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction = 'none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes) :
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size = target_masks.shape[-2 :],
                                mode = "bilinear", align_corners = False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask" : focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice" : dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_head_decoder_labels(self, outputs_decoder_head, targets):
        assert 'pred_logits' in outputs_decoder_head
        src_logits = outputs_decoder_head['pred_logits']

        losses = {}

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]], dtype = src_logits.dtype,
                                            layout = src_logits.layout,  device = src_logits.device)
        target_label = torch.stack([target['out_ids_respond_to_head_mask'] for target in targets]).long()
        target_classes_onehot.scatter_(2, target_label.unsqueeze(-1), 1)
        loss_ce = focal_loss(src_logits, target_classes_onehot, src_logits.shape[1],
                             alpha = self.focal_alpha, gamma = 2) * src_logits.shape[1]
        losses['loss_head_decoder_ce'] = loss_ce
        return losses

    def loss_head_decoder_boxes(self, outputs_decoder_head, targets) :
        assert 'pred_boxes' in outputs_decoder_head
        src_boxes = torch.cat([pred_boxes[target['out_ids_respond_to_head']] for pred_boxes, target in
                               zip(outputs_decoder_head['pred_boxes'], targets)])
        target_boxes = torch.cat([t['target_body_boxes'] for t in targets], dim = 0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction = 'none')
        num_boxes = max(len(loss_bbox), 0.00001)
        losses['loss_head_decoder_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_head_decoder_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_head_decoder_boxes_angle(self, outputs_decoder_head, targets) :
        assert 'pred_boxes' in outputs_decoder_head
        src_boxes = torch.cat([pred_boxes[target['out_ids_respond_to_head']] for pred_boxes, target in
                               zip(outputs_decoder_head['pred_boxes'], targets)])
        target_boxes = torch.cat([t['target_body_boxes'] for t in targets], dim = 0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction = 'none')
        num_boxes = max(len(loss_bbox), 0.00001)
        losses['loss_head_decoder_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_head_decoder_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices) :
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices) :
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs) :
        loss_map = {
            'labels' : self.loss_labels,
            'cardinality' : self.loss_cardinality,
            'boxes' : self.loss_boxes,
            'masks' : self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_head_decoder_loss(self, loss, outputs_decoder_head, targets):
        loss_map = {
            'head_decoder_labels': self.loss_head_decoder_labels,
            'head_decoder_boxes': self.loss_head_decoder_boxes}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs_decoder_head, targets)

    def forward(self, outputs, outputs_head_decoder, targets) :
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        device = outputs['pred_boxes'].device
        outputs_without_aux = {k : v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        match_indices = self.matcher(outputs_without_aux, targets)

        if self.args.train_cfg.vis_match_result:
            vis_match_result_path = os.path.join(self.args.root, self.args.train_cfg.vis_match_result_path)
            font_path = os.path.join(self.args.root, self.args.font_path)
            vis_match_result(targets, match_indices, outputs_without_aux, vis_match_result_path, font_path)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype = torch.float, device = next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized() :
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min = 1).item()

        # Compute all the requested losses
        for loss in self.losses :
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, match_indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs :
            for i, aux_outputs in enumerate(outputs['aux_outputs']) :
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses :
                    kwargs = {}
                    if loss == 'labels' :
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}' : v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if self.bh:
            for k, (indice, target) in enumerate(zip(match_indices, targets)):
                out_match_indice, target_match_indice = indice[0].to(device), indice[1] .to(device)
                target_body_mask = target['labels'] == 1
                target_head_mask = target['labels'] == 2

                target_body_ids = torch.where(target_body_mask)[0]
                target_head_ids = torch.where(target_head_mask)[0]

                # 将1000个输出按照target_head_ids的顺序排列
                head_target_match_matrix = target_head_ids.unsqueeze(dim = 1).eq(target_match_indice)
                head_target_matching = head_target_match_matrix.nonzero()[:, 1]
                out_ids_respond_to_head = out_match_indice[head_target_matching]

                target_head_tags = target['tags'][target_head_ids]
                match_result = (target_head_tags.unsqueeze(dim = 1).eq(target['tags']) & target_body_mask).nonzero()

                keep = match_result[:, 0]

                target_head_ids = target_head_ids[keep]
                out_ids_respond_to_head = out_ids_respond_to_head[keep]
                out_ids_respond_to_head_mask = torch.zeros(outputs_head_decoder['pred_boxes'].shape[1], device = device)
                out_ids_respond_to_head_mask[out_ids_respond_to_head] = 1
                target_body_ids = match_result[:, 1]

                target['target_head_ids'] = target_head_ids
                target['out_ids_respond_to_head'] = out_ids_respond_to_head
                target['out_ids_respond_to_head_mask'] = out_ids_respond_to_head_mask
                target['target_body_ids'] = target_body_ids
                target['target_body_boxes'] = target['boxes'][target_body_ids]
                target['output_head_boxes'] = outputs['pred_boxes'][k, out_ids_respond_to_head]
                target['target_head_boxes'] = target['boxes'][target_head_ids]

            if self.args.train_cfg.vis_body_and_head_match :
                vis_body_and_head_match_path = os.path.join(self.args.root, self.args.train_cfg.vis_body_and_head_match_path)
                font_path = os.path.join(self.args.root, self.args.font_path)
                vis_body_and_head_match(targets, vis_body_and_head_match_path, font_path)

            for loss in self.head_decoder_losses :
                losses.update(self.get_head_decoder_loss(loss, outputs_head_decoder, targets))

            if 'aux_outputs' in outputs_head_decoder :
                for i, aux_outputs in enumerate(outputs_head_decoder['aux_outputs']) :
                    for loss in self.head_decoder_losses: # ['head_decoder_boxes'] :
                        l_dict = self.get_head_decoder_loss(loss, aux_outputs, targets)
                        l_dict = {k + f'_{i}' : v for k, v in l_dict.items()}
                        losses.update(l_dict)
        return losses

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def process_boxes(self, boxes, target_sizes):
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return boxes

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of
                          each images of the batch For evaluation, this must be the
                          original image size (before any data augmentation) For
                          visualization, this should be the image size after data
                          augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits[:, :, 1:]
        scores, labels = prob.max(-1)
        boxes = self.process_boxes(out_bbox, target_sizes)
        results = [
            {'scores' : s, 'boxes' : b, 'labels': l}
            for s, b, l in zip(scores, boxes, labels)]
        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
