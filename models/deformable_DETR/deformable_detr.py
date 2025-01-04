# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from lib.util import box_ops
from lib.util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from models.deformable_DETR.detr import DETR, PostProcess, SetCriterion


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(DETR):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, bh, num_feature_levels,
                 head_query_false_positive_prob, aux_loss=True, with_box_refine = False, overflow_boxes=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO,
                         we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__(backbone, transformer, num_classes, num_queries, bh, aux_loss)
        self.overflow_boxes = overflow_boxes
        self.with_box_refine = with_box_refine
        self.bh = bh
        self._head_query_false_positive_prob = head_query_false_positive_prob
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)
        num_channels = backbone.num_channels[-3:]
        if num_feature_levels > 1:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            num_backbone_outs = len(backbone.strides) - 1

            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones_like(self.class_embed.bias) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for
        # region proposal generation
        num_pred = transformer.decoder.num_layers

        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2 :], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed

            if self.bh:
                self.head_class_embed = _get_clones(self.head_class_embed, num_pred)
                self.head_bbox_embed = _get_clones(self.head_bbox_embed, num_pred)
                nn.init.constant_(self.head_bbox_embed[0].layers[-1].bias.data[2 :], -2.0)
                self.transformer.decoder_head.bbox_embed = self.head_bbox_embed

        else :
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2 :], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

            if self.bh:
                nn.init.constant_(self.head_bbox_embed.layers[-1].bias.data[2 :], -2.0)
                self.head_class_embed = nn.ModuleList([self.head_class_embed for _ in range(num_pred)])
                self.head_bbox_embed = nn.ModuleList([self.head_bbox_embed for _ in range(num_pred)])
                self.transformer.decoder_head.bbox_embed = None


    def forward_once(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
                       - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
                       - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

                    It returns a dict with the following elements:
                       - "pred_logits": the classification logits (including no-object) for all queries.
                                        Shape= [batch_size x num_queries x (num_classes + 1)]
                       - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                                       (center_x, center_y, height, width). These values are normalized in [0, 1],
                                       relative to the size of each individual image (disregarding possible padding).
                                       See PostProcess for information on how to retrieve the unnormalized bounding box.
                       - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                        dictionnaries containing the two above keys for each decoder layer.
                """
        if not isinstance(samples, NestedTensor) :
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        features = features[-3 :]

        src_list = []
        mask_list = []
        pos_list = []

        pos_list.extend(pos[-3 :])

        for l, feat in enumerate(features) :
            src, mask = feat.decompose
            src_list.append(self.input_proj[l](src))
            mask_list.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(features) :
            _len_srcs = len(features)
            for l in range(_len_srcs, self.num_feature_levels) :
                if l == _len_srcs :
                    src = self.input_proj[l](features[-1].tensors)
                else :
                    src = self.input_proj[l](src_list[-1])
                _, m = features[0].decompose
                mask = F.interpolate(m[None].float(), size = src.shape[-2 :]).to(torch.bool)[0]

                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                src_list.append(src)
                mask_list.append(mask)
                pos_list.append(pos_l)

        query_embeds = self.query_embed.weight
        hs, memory, init_reference, inter_references, spatial_shapes, valid_ratios, mask_flatten = self.transformer(src_list, mask_list, pos_list, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]) :
            if lvl == 0 :
                reference = init_reference
            else :
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_class = torch.softmax(outputs_class, dim = -1)
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4 :
                tmp += reference
            else :
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits' : outputs_class[-1], 'pred_boxes' : outputs_coord[-1], 'memory' : memory, 'hs' : hs[-1],
               'spatial_shapes' : spatial_shapes, 'valid_ratios' : valid_ratios, 'mask_flatten' : mask_flatten}

        if self.aux_loss :
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def forward_decoder_head(self, outputs) :
        spatial_shapes = outputs['spatial_shapes']
        valid_ratios = outputs['valid_ratios']
        mask_flatten = outputs['mask_flatten']
        memory = outputs['memory']
        tgt = outputs['hs']
        reference_points = outputs['pred_boxes'][:, :, :2]
        init_reference_out = reference_points
        query_embed = None
        hs, inter_references = self.transformer.decoder_head(tgt, reference_points, memory,
                                                             spatial_shapes, valid_ratios, query_embed, mask_flatten)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]) :
            if lvl == 0 :
                reference = init_reference_out
            else :
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.head_class_embed[lvl](hs[lvl])
            outputs_class = torch.softmax(outputs_class, dim = -1)
            tmp = self.head_bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4 :
                tmp += reference
            else :
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out_head_decoder = {'pred_logits' : outputs_class[-1], 'pred_boxes' : outputs_coord[-1]}
        if self.aux_loss :
            out_head_decoder['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out_head_decoder

    def forward(self, samples: NestedTensor):
        outputs = self.forward_once(samples)
        if self.bh:
            outputs_head_decoder = self.forward_decoder_head(outputs)
        else:
            outputs_head_decoder = None
        return outputs, outputs_head_decoder

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DeformablePostProcess(PostProcess):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)

        # scores, labels = prob[..., 0:1].max(-1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {'scores': s, 'scores_no_object': 1 - s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results
