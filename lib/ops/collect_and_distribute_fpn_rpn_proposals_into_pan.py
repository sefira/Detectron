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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from core.config import cfg
from datasets import json_dataset
import modeling.FPN as fpn
import roi_data.fast_rcnn
import utils.blob as blob_utils
from ops.collect_and_distribute_fpn_rpn_proposals \
    import CollectAndDistributeFpnRpnProposalsOp


class CollectAndDistributeFpnRpnProposalsIntoPANOp(CollectAndDistributeFpnRpnProposalsOp):
    def __init__(self, train):
        self._train = train

    def distribute(self, rois, label_blobs, outputs, train):
        """To understand the output blob order see return value of
        roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
        """
        outputs[0].reshape(rois.shape)
        outputs[0].data[...] = rois
