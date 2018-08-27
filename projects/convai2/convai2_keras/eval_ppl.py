# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for ppl metric.
This seq2seq model was trained on convai2:self.
"""
from parlai.core.build_data import download_models
from projects.convai2.eval_ppl import setup_args, eval_ppl



if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        model='parlai.agents.convai2_keras.attention_seq2seq_v1_2_f1_ppl:PerplexityEvaluatorAgent',
        model_file='/data/ParlAI/mymodels/A_seq2seq/model/attentionSeq2seq_v1.2_weights_only.hdf5',
        dict_file='/data/ParlAI/mymodels/A_seq2seq/data/new_dict_original.txt',
        word_embedding_size=300,
        embedding_file='',
        batchsize=1,
        hiddensize=512,
        persona_select_model_file='/data/ParlAI/mymodels/persona_select/model/persona_select_model_v1_2_shuffle'
    )
    opt = parser.parse_args()
    eval_ppl(opt)
