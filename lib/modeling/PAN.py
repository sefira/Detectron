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

def add_pan_roi_fpn_ResNet50_conv5_head(model, blob_in, dim_in, spatial_scale):
    return add_pan_head_onto_fpn_body(
        model, blob_in, dim_in, spatial_scale, pan_level_info_ResNet50_conv5
    )

def add_pan_roi_fpn_ResNet101_conv5_head(model, blob_in, dim_in, spatial_scale):
    return add_pan_head_onto_fpn_body(
        model, blob_in, dim_in, spatial_scale, pan_level_info_ResNet101_conv5
    )

def add_pan_roi_fpn_ResNet152_conv5_head(model, blob_in, dim_in, spatial_scale):
    return add_pan_head_onto_fpn_body(
        model, blob_in, dim_in, spatial_scale, pan_level_info_ResNet152_conv5
    )

# ---------------------------------------------------------------------------- #
# Functions for bolting PAN onto a FPN backbone architectures
# ---------------------------------------------------------------------------- #

def add_pan_head_onto_fpn_body(
    model, blobs_fpn, dim_fpn, spatial_scales_fpn, pan_level_info_func
):
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
            model, pan_level_info_func(), blobs_fpn
        )

    if cfg.PAN.AdaptivePooling_ON:
        blobs_out, dim_out = add_adaptive_pooling_head(
            model, blobs, dim, spatial_scales
        )

    return blobs_out, dim_out

def add_adaptive_pooling_head(model, blobs_pan, dim_pan, spatial_scales_pan):
    """Fuse all PAN extra lateral level using a adaptive pooling"""
    # Fusion method is indicated in cfg.PAN.FUSION_METHOD
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
            model.Relu(fc6_name, fc6_name)
            fc6_list += fc6_name
        pan_adaptive_pooling = model.net.__getattr__(fusion_method)(
            fc6_list, "pan_adaptive_pooling"
        )
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


def add_pan_bottom_up_path_lateral(model, pan_level_info, blobs_fpn):
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
