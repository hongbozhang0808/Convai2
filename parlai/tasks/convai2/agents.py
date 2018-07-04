# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os
import re

'''All teachers have a version with and without label candidates. Each teacher
defaults to using a dataset with label candidates. To use a dataset without
label candidates, specify this using the task flag:

--task convai2:{TEACHER_NAME}:no_cands

where TEACHER_NAME is None, SelfOriginal (Self), or SelfRevised.
'''

def _path(opt, persona, use_cands):
    # Build the data if it doesn't exist.
    build(opt)
    datatype =  opt['datatype'].split(':')[0]
    if datatype == 'test':
        print("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    dt = datatype + '_' + persona
    cands = '' if use_cands else '_no_cands'
    return os.path.join(opt['datapath'], 'ConvAI2', dt + cands + '.txt')


#########
def preprocess(txt):
    l1 = ['comparen\'tes', 'i\'ver', 'n t ', 'can t ', 'hi\'m', '\'ssue', '\'sland', 'won\'tice', 'ii\'m',
          'n\'thing', 'she\'sn\'t', 'they\'ren\'t', 'i\'ven t', 'i\'dn\'t', 'won’t', 'won\'t', 'wouldn’t',
          'wouldn\'t', '’m', '’re', '’ve', '’ll', '’s', '’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s',
          '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,']
    l2 = ['compare notes', 'i have never', ' not ', 'can not ', 'hi i\'m', ' issue', 'island',
          'won\'t notice', 'i am', ' nothing', 'she is not', 'they are not', 'i have not', 'i do not', 'will not',
          'will not', 'would not', 'would not', ' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am',
          ' are',
          ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.',
          ',']
    l3 = ['-', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
    for j, term in enumerate(l1):
        txt = txt.replace(term, l2[j])
    for term in l3:
        txt = txt.replace(term, ' ')
    # txt = txt.lower()
    txt = txt.replace('  ', ' ')
    txt = txt.replace('\n', ' ')
    txt = re.sub('( !)+', ' !', txt)
    txt = re.sub('( \?)+', ' ?', txt)
    return (txt,)

class NoneTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'none_original', use_cands)
        super().__init__(opt, shared)

class SelfOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_original', use_cands)
        super().__init__(opt, shared)

class SelfTeacher(SelfOriginalTeacher):
    pass

class SelfRevisedTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_revised', use_cands)
        super().__init__(opt, shared)

    def observe(self, observation):
        """Process observation for metrics."""
        if self.use_batch_act:
            if self.lastYs[self.batchindex] is None:
                self.lastY =self.lastYs[self.batchindex]
            else:
                self.lastY = preprocess(self.lastYs[self.batchindex][0])
            self.lastYs[self.batchindex] = None

        if hasattr(self, 'lastY') and self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

################################
class BothRevisedTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_revised', use_cands)
        super().__init__(opt, shared)


    def observe(self, observation):
        """Process observation for metrics."""
        if self.use_batch_act:
            if self.lastYs[self.batchindex] is None:
                self.lastY =self.lastYs[self.batchindex]
            else:
                self.lastY = preprocess(self.lastYs[self.batchindex][0])
            self.lastYs[self.batchindex] = None

        if hasattr(self, 'lastY') and self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

class DefaultTeacher(SelfOriginalTeacher):
    pass
