# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for ppl metric.
This seq2seq model was trained on convai2:self.
"""
# from parlai.core.build_data import download_models
# from parlai.core.dict import DictionaryAgent
# from parlai.core.params import ParlaiParser, modelzoo_path
# from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from parlai.agents.keras_convai2.s2s_vae_v1_3_7_4 import RNNAgent
from projects.convai2.build_dict import build_dict
from projects.convai2.eval_ppl import setup_args, eval_ppl
# import torch.nn.functional as F


class Seq2seqEntry(RNNAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if shared:
            self.probs = shared['probs']
        else:
            # default minimum probability mass for all tokens
            self.probs = {k: 1e-7 for k in build_dict().keys()}

    def share(self):
        shared = super().share()
        shared['probs'] = self.probs.copy()
        return shared

    def next_word_probability(self, observation, partial_out):
        """Return probability distribution over next words given an input and
        partial true output. This is used to calculate the per-word perplexity.

        Arguments:
        observation -- input observation dict
        partial_out -- list of previous "true" words

        Returns a dict, where each key is a word and each value is a probability
        score for that word. Unset keys assume a probability of zero.

        e.g.
        {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
        """
        if not hasattr(self, 'prev_enc'):
            self.prev_enc = None
            self.last_text = None
        if observation['text'] != self.last_text:
            self.prev_enc = None
            self.last_text = observation.get('text')
            self.observe(observation)

        obs = self.observation
        obs['eval_labels'] = [' '.join(['__START__']+partial_out)]
        obs['eval_ppl'] = True

        topic_input, speaker_classes, p_np, c_np, t_np, ys, valid_inds = self.batchify([obs])
        probs = self.train(topic_input, speaker_classes, p_np, c_np, t_np, ys, is_training=False, next_i=len(partial_out))
        dist = dict()
        for i in range(len(self.dict)):
            dist[self.dict[i]] = probs[i]
        return dist


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        model='projects.convai2.s2s_vae.eval_ppl:Seq2seqEntry',
        model_file='models:convai2/seq2seq-model/weights_s2s_v1_3_7_5.hdf5',
        dict_file='models:convai2/dict_both_revised_polished.txt',
        word_embedding_size=100,
        embedding_file='',
        batchsize=1,
        hiddensize=512,
        topic_model_dir='/models/convai2/topic-model',
        speaker_model_dir='/models/convai2/cluster-model'
    )
    opt = parser.parse_args(print_args=True)
    opt['topic_model_dir'] = opt['datapath'] + opt['topic_model_dir']
    opt['speaker_model_dir'] = opt['datapath'] + opt['speaker_model_dir']
    eval_ppl(opt)
