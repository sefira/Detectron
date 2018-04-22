# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Functions for using a Path Aggregation Network (PAN)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np

from core.config import cfg
from modeling.generate_anchors import generate_anchors
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import modeling.ResNet as ResNet
import modeling.FPN as FPN
import utils.blob as blob_utils
import utils.boxes as box_utils

# ---------------------------------------------------------------------------- #
# PAN with FPN with ResNet
# ------------------------------------------------------------ #

class PAN_LEVEL_INFO(object):
    class __OnlyOne:
        def __init__(self):
            self.val = None
        def __str__(self):
            return self.val
    instance = None
    def __new__(cls):
        if not PAN_LEVEL_INFO.instance:
            PAN_LEVEL_INFO.instance = PAN_LEVEL_INFO.__OnlyOne()
        return PAN_LEVEL_INFO.instance
    def __getattr__(self, name):
        return getattr(self.instance, name)
    def __setattr__(self, name):
        return setattr(self.instance, name)

def add_pan_fpn_ResNet50_conv5_body(model):
    level_info = PAN_LEVEL_INFO()
    level_info.val = pan_level_info_ResNet50_conv5
    return add_pan_onto_fpn_body(
        model, FPN.add_fpn_ResNet50_conv5_body, pan_level_info_ResNet50_conv5
    )

def add_pan_fpn_ResNet101_conv5_body(model):
    level_info = PAN_LEVEL_INFO()
    level_info.val = pan_level_info_ResNet101_conv5
    return add_pan_onto_fpn_body(
        model, FPN.add_fpn_ResNet101_conv5_body, pan_level_info_ResNet101_conv5
    )

def add_pan_fpn_ResNet152_conv5_body(model):
    level_info = PAN_LEVEL_INFO()
    level_info.val = pan_level_info_ResNet152_conv5
    return add_pan_onto_fpn_body(
        model, FPN.add_fpn_ResNet152_conv5_body, pan_level_info_ResNet152_conv5
    )

def add_pan_roi_fpn_ResNet50_conv5_head(model, blob_in, dim_in, spatial_scale):
    print("# ********************** DEPRECATED FUNCTIONALITY add_pan_roi_fpn_ResNet50_conv5_head ********************** #")
    return add_pan_head_onto_fpn_body(
        model, blob_in, dim_in, spatial_scale, pan_level_info_ResNet50_conv5
    )

def add_pan_roi_fpn_ResNet101_conv5_head(model, blob_in, dim_in, spatial_scale):
    print("# ********************** DEPRECATED FUNCTIONALITY add_pan_roi_fpn_ResNet101_conv5_head ********************** #")
    return add_pan_head_onto_fpn_body(
        model, blob_in, dim_in, spatial_scale, pan_level_info_ResNet101_conv5
    )

def add_pan_roi_fpn_ResNet152_conv5_head(model, blob_in, dim_in, spatial_scale):
    print("# ********************** DEPRECATED FUNCTIONALITY add_pan_roi_fpn_ResNet152_conv5_head ********************** #")
    return add_pan_head_onto_fpn_body(
        model, blob_in, dim_in, spatial_scale, pan_level_info_ResNet152_conv5
    )

# ---------------------------------------------------------------------------- #
# Functions for bolting PAN onto a FPN backbone architectures
# ---------------------------------------------------------------------------- #

def add_pan_onto_fpn_body(model, fpn_body_func, pan_level_info_func):
    """Add the PAN levels to the FPN levels.
    """
    # Note: blobs_conv is in order: [pan_2, pan_3, pan_4, pan_5]
    # similarly for dims_conv: [256, 256, 256, 256]
    # similarly for spatial_scales_pan: [1/4, 1/8, 1/16, 1/32]
    assert cfg.PAN.BottomUp_ON, "BottomUp_ON = False, can not use PAN body"

    blobs_fpn, dim_fpn, spatial_scales_fpn = fpn_body_func(model)
    blobs_pan, dim_pan, spatial_scales_pan = add_pan_bottom_up_path_lateral(
        model, pan_level_info_func()
    )

    # Return all fpn and pan blobs in a dictionary,
    # then let model_build.py to use it selectively according to config
    # for example, RPN uses fpn, while roi uses pan
    # return {"FPN": blobs_fpn, "PAN": blobs_pan}, {"FPN": dim_fpn, "PAN": dim_pan}, {"FPN": spatial_scales_fpn, "PAN": spatial_scales_pan}

    # If PAN_RPN_ON, return pan level blobs to RPN, then do adaptive pooling on pan level
    # otherwise return fpn level blobs, then do adaptive pooling on pan level
    if cfg.PAN.PAN_RPN_ON:
        return blobs_pan, dim_pan, spatial_scales_pan
    else:
        return blobs_fpn, dim_fpn, spatial_scales_fpn

def add_pan_head_onto_fpn_body(
    model, blobs_fpn, dim_fpn, spatial_scales_fpn, pan_level_info_func
):
    print("# ********************** DEPRECATED FUNCTIONALITY add_pan_head_onto_fpn_body ********************** #")
    """Add the specified conv body to the model and then add FPN levels to it.
    Then add PAN levels to it.
    And fuse these PAN levels using max or sum according cfg
    """
    # Note: blobs_conv is in order: [pan_2, pan_3, pan_4, pan_5]
    # similarly for dims_conv: [256, 256, 256, 256]
    # similarly for spatial_scales_pan: [1/4, 1/8, 1/16, 1/32]
    blobs, dim, spatial_scales = blobs_fpn, dim_fpn, spatial_scales_fpn

    if cfg.PAN.BottomUp_ON:
        blobs, dim, spatial_scales = add_pan_bottom_up_path_lateral(
            model, pan_level_info_func()
        )

    if cfg.PAN.AdaptivePooling_ON:
        blobs_out, dim_out = add_adaptive_pooling_box_head(
            model, blobs, dim, spatial_scales
        )

    return blobs_out, dim_out

# ---------------------------------------------------------------------------- #
# Functions for PAN head, specific adaptive pooling head
# ---------------------------------------------------------------------------- #

def add_adaptive_pooling_box_head(model, blobs_pan, dim_pan, spatial_scales_pan):
    """Fuse all PAN extra lateral level using a adaptive pooling"""
    # Fusion method is indicated in cfg.PAN.FUSION_METHOD
    assert cfg.PAN.AdaptivePooling_ON, "AdaptivePooling_ON = False, can not use PAN head"

    pan_level_info = PAN_LEVEL_INFO().val()
    # If BottomUp_ON, adaptive pooling on pan level
    # otherwise adaptive pooling on fpn level
    if cfg.PAN.BottomUp_ON:
        perfix = 'pan_'
    else:
        perfix = ''
    blobs_pan = [
        perfix + (s)
        for s in pan_level_info.blobs
    ]
    # For the finest FPN level: N2 = P2 only seeds recursion
    blobs_pan[0] = pan_level_info.blobs[0]
    dim_pan = pan_level_info.dims[0]
    spatial_scales_pan = pan_level_info.spatial_scales
    fusion_method = cfg.PAN.FUSION_METHOD
    assert fusion_method in {'Sum', 'Max', 'Mean'}, \
        'Unknown fusion method: {}'.format(fusion_method)
    adaptive_pooling_place = cfg.PAN.AdaptivePooling_Place
    assert adaptive_pooling_place in {'BeforeFC1', 'AfterFC1'}, \
        'Unknown adaptive pooling place: {}'.format(adaptive_pooling_place)
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if adaptive_pooling_place == 'AfterFC1':
        # AfterFC1 means adaptive pooling fc1 feature
        # which is in cls and bbox head branch
        roi_feat = model.RoIFeatureTransform(
            blobs_pan,
            'roi_feat',
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=roi_size,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scales_pan
        )
        fc6_list = []
        for i in range(len(roi_feat)):
            fc6_name = 'fc6_' + str(roi_feat[i])
            model.FC(roi_feat[i], fc6_name, dim_pan * roi_size * roi_size, hidden_dim)
            fc6_list += [fc6_name]
        pan_adaptive_pooling = model.net.__getattr__(fusion_method)(
            fc6_list, "pan_adaptive_pooling"
        )
        model.Relu(pan_adaptive_pooling, pan_adaptive_pooling)
        model.FC(pan_adaptive_pooling, 'fc7', hidden_dim, hidden_dim)
        model.Relu('fc7', 'fc7')

    elif adaptive_pooling_place == 'BeforeFC1':
        # BeforeFC1 means directly adaptive pooling conv feature map,
        # which can be simplify as follow:
        # First fuse then RoI pooling
        num_backbone_stages = len(blobs_pan)
        resized_pan_stages = []
        spatial_scales = []
        # Keep N2 as it is
        resized_pan_stages += [blobs_pan[0]]
        spatial_scales += [spatial_scales_pan[0]]
        # Resize all other stage into N2 scale
        for i in range(1, num_backbone_stages):
            resized = model.net.UpsampleNearest(
                blobs_pan[i],
                blobs_pan[i] + '_reszied',
                scale=int(spatial_scales_pan[0] / spatial_scales_pan[i])
            )
            resized_pan_stages += [resized]
            spatial_scales += [spatial_scales_pan[0]]

        # Fusion all resized stages directly, the apply RoIPooling defined in detector.py
        # TODO(buxingyuan): Think Twice, it seems equipollent to original Adaptive Pooling:
        # [1. Distribute RoI into different level
        # 2. RoIPooling in different level
        # 3. adaptive fusion all level featue]
        pan_adaptive_pooling = model.net.__getattr__(fusion_method)(
            resized_pan_stages, "pan_adaptive_pooling"
        )
        roi_feat = model.RoIFeatureTransform(
            pan_adaptive_pooling,
            'roi_feat',
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=roi_size,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scales_pan[0]
        )
        model.FC(roi_feat, 'fc6', dim_pan * roi_size * roi_size, hidden_dim)
        model.Relu('fc6', 'fc6')
        model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
        model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim

def add_adaptive_pooling_mask_head_v1up4convs(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2."""
    return adaptive_pooling_mask_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 4
    )


def add_adaptive_pooling_mask_head_v1up(model, blob_in, dim_in, spatial_scale):
    """v1up design: 2 * (conv 3x3), convT 2x2."""
    return adaptive_pooling_mask_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 2
    )


def adaptive_pooling_mask_head_v1upXconvs(model, blobs_pan, dim_pan, spatial_scales_pan):
    """Fuse all PAN extra lateral level using a adaptive pooling"""
    # Fusion method is indicated in cfg.PAN.FUSION_METHOD
    assert cfg.MODEL.MASK_ON, "MODEL.MASK_ON = False, can not use PAN mask head"
    assert cfg.PAN.MASK_ON, "PAN.MASK_ON = False, can not use PAN mask head"

    pan_level_info = PAN_LEVEL_INFO().val()
    # If BottomUp_ON, adaptive pooling on pan level
    # otherwise adaptive pooling on fpn level
    if cfg.PAN.BottomUp_ON:
        perfix = 'pan_'
    else:
        perfix = ''
    blobs_pan = [
        perfix + (s)
        for s in pan_level_info.blobs
    ]
    # For the finest FPN level: N2 = P2 only seeds recursion
    blobs_pan[0] = pan_level_info.blobs[0]
    dim_pan = pan_level_info.dims[0]
    spatial_scales_pan = pan_level_info.spatial_scales
    fusion_method = cfg.PAN.FUSION_METHOD
    assert fusion_method in {'Sum', 'Max', 'Mean'}, \
        'Unknown fusion method: {}'.format(fusion_method)
    # In mask branch, we fix the fusion place between the first and second conv layers
    # adaptive_pooling_place = cfg.PAN.AdaptivePooling_Place

    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    mask_roi_feat = model.RoIFeatureTransform(
        blobs_pan,
        blob_out='_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scales_pan
    )

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    # independent fcn1 for all levels
    mask_fcn1_list = []
    for i in range(len(mask_roi_feat)):
        mask_fcn1_name = '_[mask]_fcn1' + str(mask_roi_feat[i])
        model.Conv(
            mask_roi_feat[i],
            mask_fcn1_name,
            dim_pan,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        mask_fcn1_list += [mask_fcn1_name]
    # fuse
    pan_adaptive_pooling_mask_fcn1 = model.net.__getattr__(fusion_method)(
        mask_fcn1_list, "pan_adaptive_pooling_mask_fcn1"
    )
    model.Relu(pan_adaptive_pooling_mask_fcn1, pan_adaptive_pooling_mask_fcn1)

    current = pan_adaptive_pooling_mask_fcn1
    for i in range(1, num_convs):
        current = model.Conv(
            '_[mask]_fcn' + str(i),
            '_[mask]_fcn' + str(i + 1),
            dim_inner,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner

def add_pan_bottom_up_path_lateral(model, pan_level_info):
    """Add PAN connections based on the model described in the PAN paper."""
    # PAN levels are built starting from the finest level of the FPN.
    # First we recurisvely constructing higher resolution FPN levels.
    # In details:
    # N2 = P2, 
    # N3 = Conv(Conv(N2, 3x3, s=2) + P3, 3x3, s=1)
    # N4 = Conv(Conv(N3, 3x3, s=2) + P4, 3x3, s=1)
    # N5 = Conv(Conv(N4, 3x3, s=2) + P5, 3x3, s=1)
    # It seems there is no higher level than N5 (i.e. P5) in PAN
    pan_dim = cfg.PAN.DIM
    xavier_fill = ('XavierFill', {})
    num_backbone_stages = (
        len(pan_level_info.blobs)# - (min_level - LOWEST_BACKBONE_LVL)
    )

    fpn_input_blobs = pan_level_info.blobs
    pan_blobs = [
        'pan_{}'.format(s)
        for s in pan_level_info.blobs
    ]
    spatial_scales = [
        sp
        for sp in pan_level_info.spatial_scales
    ]
    pan_dim_lateral = pan_level_info.dims

    # For the finest FPN level: N2 = P2 only seeds recursion
    pan_blobs[0] = pan_level_info.blobs[0]

    # For other levels add bottom-up path
    for i in range(num_backbone_stages - 1):
        # Buttom-up 3x3 subsample conv
        subsample = model.Conv(
            pan_blobs[i],
            pan_blobs[i] + '_sub',
            dim_in=pan_dim,
            dim_out=pan_dim_lateral[i],
            kernel=3,
            pad=1,
            stride=2,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
        model.Relu(subsample, subsample)
        # Sum lateral and buttom-up subsampled conv
        model.net.Sum([subsample, fpn_input_blobs[i + 1]], pan_blobs[i] + '_sum')

        # Post-hoc scale-specific 3x3 convs
        pan_blob = model.Conv(
            pan_blobs[i] + '_sum',
            pan_blobs[i + 1],
            dim_in=pan_dim_lateral[i],
            dim_out=pan_dim,
            kernel=3,
            pad=1,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
        model.Relu(pan_blob, pan_blob)

    return pan_blobs, pan_dim, spatial_scales

# ---------------------------------------------------------------------------- #
# PAN level info for stages 5, 4, 3, 2 for select models (more can be added)
# ---------------------------------------------------------------------------- #

PanLevelInfo = collections.namedtuple(
    'PanLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)


def pan_level_info_ResNet50_conv5():
    return PanLevelInfo(
        #blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        blobs=('fpn_res2_2_sum', 'fpn_res3_3_sum', 'fpn_res4_5_sum', 'fpn_res5_2_sum'),
        #dims=(2048, 1024, 512, 256),
        dims=(256, 256, 256, 256),
        #spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
        spatial_scales=(1. / 4., 1. / 8., 1. / 16., 1. / 32.)
    )


def pan_level_info_ResNet101_conv5():
    return PanLevelInfo(
        #blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        blobs=('fpn_res2_2_sum', 'fpn_res3_3_sum', 'fpn_res4_22_sum', 'fpn_res5_2_sum'),
        #dims=(2048, 1024, 512, 256),
        dims=(256, 256, 256, 256),
        #spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
        spatial_scales=(1. / 4., 1. / 8., 1. / 16., 1. / 32.)
    )


def pan_level_info_ResNet152_conv5():
    return PanLevelInfo(
        #blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        blobs=('fpn_res2_2_sum', 'fpn_res3_7_sum', 'fpn_res4_35_sum', 'fpn_res5_2_sum'),
        #dims=(2048, 1024, 512, 256),
        dims=(256, 256, 256, 256),
        #spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
        spatial_scales=(1. / 4., 1. / 8., 1. / 16., 1. / 32.)
    )
