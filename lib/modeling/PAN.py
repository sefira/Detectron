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

# Lowest and highest pyramid levels in the backbone network. For FPN, we assume
# that all networks have 5 spatial reductions, each by a factor of 2. Level 1
# would correspond to the input image, hence it does not make sense to use it.
LOWEST_BACKBONE_LVL = 2   # E.g., "conv2"-like level
HIGHEST_BACKBONE_LVL = 5  # E.g., "conv5"-like level


# ---------------------------------------------------------------------------- #
# PAN with FPN with ResNet
# ------------------------------------------------------------ #

def add_pan_fpn_ResNet50_conv5_body(model):
    return add_pan_onto_fpn_body(
        model, FPN.add_fpn_ResNet50_conv5_body, pan_level_info_ResNet50_conv5
    )

def add_pan_fpn_ResNet101_conv5_body(model):
    return add_pan_onto_fpn_body(
        model, FPN.add_fpn_ResNet101_conv5_body, pan_level_info_ResNet101_conv5
    )

def add_pan_fpn_ResNet152_conv5_body(model):
    return add_pan_onto_fpn_body(
        model, FPN.add_fpn_ResNet152_conv5_body, pan_level_info_ResNet152_conv5
    )

# ---------------------------------------------------------------------------- #
# Functions for bolting PAN onto a FPN backbone architectures
# ---------------------------------------------------------------------------- #

def add_pan_onto_fpn_body(
    model, fpn_body_func, pan_level_info_func
):
    """Add the specified conv body to the model and then add FPN levels to it.
    Then add PAN levels to it.
    """
    # Note: blobs_conv is in revsersed order: [fpn5, fpn4, fpn3, fpn2]
    # similarly for dims_conv: [2048, 1024, 512, 256]
    # similarly for spatial_scales_fpn: [1/32, 1/16, 1/8, 1/4]

    fpn_body_func(model)
    blobs_fpn, dim_fpn, spatial_scales_fpn = add_pan(
        model, pan_level_info_func()
    )

    return blobs_pan, dim_pan, spatial_scales_pan


def add_pan(model, pan_level_info):
    """Add PAN connections based on the model described in the PAN paper."""
    # PAN levels are built starting from the finest level of the FPN.
    # First we recurisvely constructing higher resolution FPN levels. 
    # In details:
    # N2 = P2, 
    # N3 = Conv(Conv(N2, 3x3, s=2) + P3, 3x3, s=1) 
    # ...
    # N5 = ...
    # It seems there is no higher level than N5 (i.e. P5) in PAN
    pan_dim = cfg.PAN.DIM    
    num_backbone_stages = (
        len(pan_level_info.blobs)# - (min_level - LOWEST_BACKBONE_LVL)
    )

    lateral_input_blobs = pan_level_info.blobs
    output_blobs = [
        'pan_inner_{}'.format(s)
        for s in pan_level_info.blobs
    ]
    pan_dim_lateral = pan_level_info.dims
    xavier_fill = ('XavierFill', {})

    # For the finest FPN level: N2 = P2 only seeds recursion
    output_blobs[0] = lateral_input_blobs[0]

    # For other levels add bottom-up and lateral connections
    for i in range(num_backbone_stages - 1):
        add_bottomup_lateral_module(
            model,
            output_blobs[i],             # bottom-up blob
            lateral_input_blobs[i + 1],  # lateral blob
            output_blobs[i + 1],         # next output blob
            pan_dim,                     # output dimension
            pan_dim_lateral[i + 1]       # lateral input dimension
        )

    # Post-hoc scale-specific 3x3 convs, exclude N2
    blobs_pan = []
    spatial_scales = []
    for i in range(1, num_backbone_stages):
        pan_blob = model.Conv(
            output_blobs[i],
            'pan_{}'.format(pan_level_info.blobs[i]),
            dim_in=pan_dim,
            dim_out=pan_dim,
            kernel=3,
            pad=1,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
        model.Relu(pan_blob, pan_blob)
        blobs_pan += [pan_blob]
        spatial_scales += [pan_level_info.spatial_scales[i]]

    return blobs_pan, pan_dim, spatial_scales


def add_bottomup_lateral_module(
    model, pan_buttom_input, lateral_input, pan_up_output, dim_pan_input, dim_lateral_input
):
    """Add a buttom-up lateral module."""
    # Buttom-up 3x3 conv
    bu = model.Conv(
        pan_buttom_input,
        pan_up_output + '_bu',
        dim_in=dim_pan_input,
        dim_out=dim_lateral_input,
        kernel=3,
        pad=1,
        stride=2,
        weight_init=xavier_fill,
        bias_init=const_fill(0.0)
    )
    model.Relu(bu, bu)
    # Sum lateral and buttom-up
    model.net.Sum([bu, lateral_input], pan_up_output)


# ---------------------------------------------------------------------------- #
# PAN level info for stages 5, 4, 3, 2 for select models (more can be added)
# ---------------------------------------------------------------------------- #

FpnLevelInfo = collections.namedtuple(
    'FpnLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)


def fpn_level_info_ResNet50_conv5():
    return FpnLevelInfo(
        #blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        blobs=('fpn_res2_2_sum', 'fpn_res3_3_sum', 'fpn_res4_5_sum', 'fpn_res5_2_sum'),
        #dims=(2048, 1024, 512, 256),
        dims=(256, 256, 256, 256),
        #spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
        spatial_scales=(1. / 4., 1. / 8., 1. / 16., 1. / 32.)
    )


def fpn_level_info_ResNet101_conv5():
    return FpnLevelInfo(
        #blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        blobs=('fpn_res2_2_sum', 'fpn_res3_3_sum', 'fpn_res4_22_sum', 'fpn_res5_2_sum'),
        #dims=(2048, 1024, 512, 256),
        dims=(256, 256, 256, 256),
        #spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
        spatial_scales=(1. / 4., 1. / 8., 1. / 16., 1. / 32.)
    )


def fpn_level_info_ResNet152_conv5():
    return FpnLevelInfo(
        #blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        blobs=('fpn_res2_2_sum', 'fpn_res3_7_sum', 'fpn_res4_35_sum', 'fpn_res5_2_sum'),
        #dims=(2048, 1024, 512, 256),
        dims=(256, 256, 256, 256),
        #spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
        spatial_scales=(1. / 4., 1. / 8., 1. / 16., 1. / 32.)
    )
