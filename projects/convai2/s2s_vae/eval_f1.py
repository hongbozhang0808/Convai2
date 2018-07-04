# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for f1 metric.
This seq2seq model was trained on convai2:self.
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_f1 import setup_args, eval_f1
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput
import os
import platform


# def getSeparator():
#     if 'Windows' in platform.system():
#         separator = '\\'
#     else:
#         separator = '/'
#     return separator
#
# def findPath(file):
#     o_path = os.getcwd()
#     separator = getSeparator()
#     str = o_path
#     str = str.split(separator)
#     while len(str) > 0:
#         spath = separator.join(str)+separator+file
#         leng = len(str)
#         if os.path.exists(spath):
#             return '/'.join(str)
#         del str[-1]


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        model='parlai.agents.keras_convai2.s2s_vae_v1_3_7_4:RNNAgent',
        model_file='models:convai2/seq2seq-model/weights_s2s_v1_3_7_5.hdf5',
        dict_file='models:convai2/dict_both_revised_polished.txt',
        word_embedding_size = 100,
        embedding_file = '',
        batchsize=16,
        hiddensize=512,
        topic_model_dir='/models/convai2/topic-model',
        speaker_model_dir='/models/convai2/cluster-model'
    )
    opt = parser.parse_args(print_args=True)
    opt['topic_model_dir'] = opt['datapath'] + opt['topic_model_dir']
    opt['speaker_model_dir'] = opt['datapath'] + opt['speaker_model_dir']
    eval_f1(opt, print_parser=parser)
