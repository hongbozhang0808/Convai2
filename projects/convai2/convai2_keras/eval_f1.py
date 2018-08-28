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
import os
# from pycallgraph import PyCallGraph
# from pycallgraph import Config
# from pycallgraph.output import GraphvizOutput

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        model='parlai.agents.convai2_keras.attention_seq2seq_v1_2_f1_ppl:AttentionSeq2seqAgent',
        model_file='models:convai2/attentionSeq2seq_v1.2_weights_only.hdf5',
        dict_file='models:convai2/new_dict_original.txt',
        word_embedding_size = 300,
        embedding_file = '',
        batchsize=256,
        hiddensize=512,
        persona_select_model_file='models/convai2/persona_select_model_v1_2_shuffle',
        batch_sort=True
    )
    opt = parser.parse_args(print_args=False)
    opt['persona_select_model_file'] = os.path.join(opt['datapath'], opt['persona_select_model_file'])
    eval_f1(opt, print_parser=parser)
