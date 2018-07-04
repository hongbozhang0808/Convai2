# based on model 1.3.7.4

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import copy
import os
import random
from keras.layers import LSTM, GRU, Dense, Embedding, Input, Concatenate, RepeatVector, TimeDistributed, Reshape, CuDNNLSTM, Lambda
from keras.models import Model, save_model, load_model, Sequential
# from tensorflow.contrib.keras.python.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import pdb
import re
import keras.backend as KTF
import os
from keras import optimizers
from keras_attention_block.attention import Attention1DLayer
import pickle
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
import math
from parlai.core.metrics import Metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)


class RNNAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('RNN Arguments')
        agent.add_argument('-rnn', '--rnntype', type=str, default='GRU',
                           help='choose GRU or LSTM')
        agent.add_argument('-hs', '--hiddensize', type=int, default=512,
                           help='size of the hidden layers and embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.5,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.2,
                           help='dropout rate')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('--embedding-file', type=str, default='',
                           help='embedding file used to init embedding layer')
        agent.add_argument('--word-embedding-size', type=int, default=100,
                           help='the word embedding size')
        agent.add_argument('--topic-model-dir', type=str,
                           help='LDA model file')
        agent.add_argument('--topic-embedding-size', type=int,
                           default=100,
                           help='The output size of topic embedding')
        agent.add_argument('--speaker-model-dir', type=str,
                           help='Speaker cluster model dir')
        agent.add_argument('--speaker-embedding-size', type=int,
                           default=100,
                           help='The output size of speaker embedding')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            # don't enter this loop for shared instantiations
            local_device_protos = device_lib.list_local_devices()
            available = [x.name for x in local_device_protos if x.device_type == 'GPU']
            opt['cuda'] = not opt['no_cuda'] and available
            if opt['cuda']:
                print('[ Using CUDA ]')
                print('[ Using Device:', ', '.join(available), ']')
            else:
                print('[ NO CUDA ]')

            self.id = 'RNN'
            self.dict = DictionaryAgent(opt)
            self.observation = {}
            self.rnn_type = opt['rnntype']
            self.hidden_size = opt['hiddensize']
            self.num_layers = opt['numlayers']
            self.dropout_rate = opt['dropout']
            self.learning_rate = opt['learningrate']
            self.path = opt.get('model_file', None)
            self.reuse = None if opt['datatype'] == 'train' else True
            self.t_max_len = 25
            self.p_max_len = 20
            self.c_max_len = 100
            self.p_max_sentence_num = 6
            self.batch_size = opt['batchsize']
            self.embedding_file = opt['embedding_file']
            self.word_embedding_size = opt['word_embedding_size']
            self.topic_model_dir = opt['topic_model_dir']
            self.topic_embedding_size = opt['topic_embedding_size']
            self.speaker_model_dir = opt['speaker_model_dir']
            self.speaker_embedding_size = opt['speaker_embedding_size']
            # if shared and shared.get('metrics'):
            #     self.metrics = shared['metrics']
            # else:
            #     self.metrics = Metrics(opt)

            self.load_pre_model()
            self.create_model()

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
                self.load(opt['model_file'])
        self.episode_done = True

    def txt2vec(self, txt):
        l1 = ['comparen\'tes', 'i\'ver', 'n t ', 'can t ', 'hi\'m', '\'ssue', '\'sland', 'won\'tice',
              'ii\'m', 'n\'thing', 'she\'sn\'t', 'they\'ren\'t', 'i\'ven t', 'i\'dn\'t', 'won’t', 'won\'t',
              'wouldn’t', 'wouldn\'t', '’m', '’re', '’ve', '’ll', '’s', '’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll',
              '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,']
        l2 = ['compare notes', 'i have never', ' not ', 'can not ', 'hi i\'m', ' issue', 'island', 'won\'t notice',
              'i am', ' nothing', 'she is not', 'they are not', 'i have not', 'i do not', 'will not', 'will not',
              'would not', 'would not', ' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have',
              ' will',
              ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',']
        l3 = ['-', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
        for j, term in enumerate(l1):
            txt = txt.replace(term, l2[j])
        for term in l3:
            txt = txt.replace(term, ' ')
        # txt = txt.lower()
        txt = txt.replace('  ', ' ')
        txt = txt.replace('\n', ' ')

        # # 语料里面有个SB，专门为他写的规则
        txt = re.sub('( !)+', ' !', txt)
        txt = re.sub('( \?)+', ' ?', txt)

        words = txt.split(' ')
        vec = [0] * len(words)
        for iw, w in enumerate(words):
            if self.dict.tok2ind.get(w) != None:
                vec[iw] = self.dict.tok2ind[w]
            elif w != '__NULL__' and w != '__SILENCE__':
                # id of '__UNK__'
                vec[iw] = 3
        return vec

    def vec2txt(self, vec):
        txt = self.dict.vec2txt(vec)
        return txt

    def zero_grad(self):
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        for optimizer in self.optims.values():
            optimizer.step()

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        # At this moment `self.episode_done` is the previous example
        if not self.episode_done:
            # If the previous example is not the end of the episode,
            # we need to recall the `text` mentioned in the previous example.
            # At this moment `self.observation` is the previous example.
            prev_dialogue = self.observation['text']
            # Add the previous and current `text` and update current `text`
            observation['text'] = prev_dialogue + '\n' + observation['text']
        # Overwrite with current example
        self.observation = observation
        # The last example of an episode is provided as `{'episode_done': True}`
        self.episode_done = observation['episode_done']
        return observation

    def create_model(self):
        vs = len(self.dict)
        hs = self.hidden_size
        t_ml = self.t_max_len
        c_ml = self.c_max_len
        p_num = self.p_max_sentence_num
        p_ml = self.p_max_len
        drop_out = self.dropout_rate
        speaker_num = self.speaker_num
        speaker_embedding_size = self.speaker_embedding_size
        topic_num = self.topic_num
        topic_embedding_size = self.topic_embedding_size

        if self.embedding_file == '':
            word_embedding = Embedding(input_dim=vs, output_dim=self.word_embedding_size, mask_zero=False,
                                               embeddings_initializer='glorot_normal', trainable=True)
        else:
            with open(self.embedding_file, 'rb') as f:
                embedding = pickle.load(f)
            self.word_embedding_size = embedding.shape[1]
            word_embedding = Embedding(input_dim=vs, output_dim=self.word_embedding_size, mask_zero=False,
                                       weights=[embedding], trainable=False)
        speaker_embedding = Embedding(input_dim=speaker_num, output_dim=speaker_embedding_size,
                                      embeddings_initializer='glorot_normal', trainable=True)
        topic_embedding = Embedding(input_dim=topic_num + 1, output_dim=topic_embedding_size,
                                    embeddings_initializer='glorot_normal', trainable=True)
        encoder_lstm1 = LSTM(hs, return_sequences=True, return_state=True,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal')
        encoder_lstm2 = LSTM(hs, return_sequences=False, return_state=True,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal')
        context_lstm1 = LSTM(hs, return_sequences=True, return_state=False,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal')
        context_lstm2 = LSTM(hs, return_sequences=False, return_state=False,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal')
        persona_lstm1 = LSTM(hs, return_sequences=True, return_state=False,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal')
        persona_lstm2 = LSTM(hs, return_sequences=False, return_state=False,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal')
        decoder_lstm1 = LSTM(hs, return_sequences=True, return_state=True,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal')
        decoder_lstm2 = LSTM(hs, return_sequences=True, return_state=True,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal')
        decoder_dense_1 = Dense(int(vs / 2), activation='relu', kernel_initializer='glorot_normal')
        decoder_dense_2 = Dense(vs, activation='softmax', kernel_initializer='glorot_normal')
        topic_map_dense = Dense(hs, kernel_initializer='glorot_normal')
        attention_p_topic = Attention1DLayer(similarity='additive', dropout_rate=drop_out,
                                             kernel_initializer='glorot_normal',
                                             wk_kernel_initializer='glorot_normal',
                                             wq_kernel_initializer='glorot_normal')

        # encode text sentence
        t_input = Input((t_ml,))
        t_x = word_embedding(t_input)
        t_x, init_h1, init_c1 = encoder_lstm1(t_x)
        t_x, init_h2, init_c2 = encoder_lstm2(t_x)
        self.encoder_model = Model(t_input, [init_h1, init_h2, init_c1, init_c2])

        # encode context
        c_input = Input((c_ml,))
        c_x = word_embedding(c_input)
        c_x = context_lstm1(c_x)
        c_x = context_lstm2(c_x)
        repeat_c_x = RepeatVector(t_ml)(c_x)
        self.context_model = Model(c_input, c_x)

        # encode persona
        def slice(x, index):
            return x[:, index, :]

        p_input = Input((p_num, p_ml))
        p_x = []
        for i in range(int(p_input.shape[1])):
            tmp = Lambda(slice, arguments={'index': i})(p_input)
            tmp = word_embedding(tmp)
            tmp = persona_lstm1(tmp)
            tmp = persona_lstm2(tmp)
            tmp = Reshape((1, hs))(tmp)
            p_x.append(tmp)
        p_x = Concatenate(1)(p_x)

        # get speaker vector
        s_input = Input((1,))
        s_x = speaker_embedding(s_input)
        s_x = Reshape((speaker_embedding_size,))(s_x)
        repeat_s_x = RepeatVector(t_ml)(s_x)

        # get topic vector
        topic_intput = Input((1,))
        topic_x = topic_embedding(topic_intput)
        topic_x = topic_map_dense(topic_x)
        a_p_topic = attention_p_topic([p_x, topic_x])
        reshape_a_p_topic = Reshape((hs,))(a_p_topic)
        repeat_a_p_topic = RepeatVector(t_ml)(reshape_a_p_topic)
        self.persona_topic_model = Model([p_input, topic_intput], a_p_topic)

        # decode response
        r_input = Input((t_ml,))
        r_x = word_embedding(r_input)
        r_x = Concatenate(2)([r_x, repeat_s_x, repeat_c_x, repeat_a_p_topic])
        r_x, _, __ = decoder_lstm1(r_x, initial_state=[init_h1, init_c1])
        r_x, _, __ = decoder_lstm2(r_x, initial_state=[init_h2, init_c2])
        decoder_output = TimeDistributed(decoder_dense_1)(r_x)
        decoder_output = TimeDistributed(decoder_dense_2)(decoder_output)
        self.model = Model([t_input, s_input, topic_intput, p_input, c_input, r_input], decoder_output)

        # response generate model
        rg_init_h1_input = Input((hs,))
        rg_init_h2_input = Input((hs,))
        rg_init_c1_input = Input((hs,))
        rg_init_c2_input = Input((hs,))
        rg_s_input = Input((1,))
        rg_s_x = speaker_embedding(rg_s_input)
        rg_p_topic_input = Input((1, hs))
        rg_c_input = Input((hs,))
        rg_c_x = Reshape((1, -1))(rg_c_input)
        rg_r_input = Input((1,))
        rg_r_x = word_embedding(rg_r_input)
        rg_r_x = Concatenate(2)([rg_r_x, rg_s_x, rg_c_x, rg_p_topic_input])
        rg_r_x, rg_h1, rg_c1 = decoder_lstm1(rg_r_x, initial_state=[rg_init_h1_input, rg_init_c1_input])
        rg_r_x, rg_h2, rg_c2 = decoder_lstm2(rg_r_x, initial_state=[rg_init_h2_input, rg_init_c2_input])
        rg_decoder_output = decoder_dense_1(rg_r_x)
        rg_decoder_output = decoder_dense_2(rg_decoder_output)
        self.rg_model = Model([rg_init_h1_input, rg_init_h2_input, rg_init_c1_input, rg_init_c2_input,
                               rg_s_input, rg_p_topic_input, rg_c_input, rg_r_input],
                              [rg_decoder_output, rg_h1, rg_h2, rg_c1, rg_c2])

    def train(self, topic_input, speaker_classes, p_np, c_np, t_np, ys, is_training=True, next_i=None):
        vs = len(self.dict)
        if is_training:
            def splitLabelsText(labels):
                labels_for_input = []
                labels_for_target = []
                for l in labels:
                    tmp = [0] * len(l)
                    tmp[0] = 1
                    tmp[1:len(l)] = l[0:(len(l) - 1)]
                    labels_for_input.append(tmp)

                    tmp = [0] * len(l)
                    tmp[0:len(l)] = l
                    l_one_hot = np.zeros((len(l), vs))
                    l_one_hot[np.arange(len(l)), tmp] = 1
                    labels_for_target.append(l_one_hot)
                return labels_for_input, labels_for_target

            ys_input, ys_target = splitLabelsText(ys)

            adam = optimizers.adam(lr=self.learning_rate)

            self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.fit([np.array(t_np), np.array(speaker_classes), np.array(topic_input),
                            np.array(p_np), np.array(c_np), np.array(ys_input)],
                           np.array(ys_target),
                           batch_size=len(p_np), epochs=1)
            idx_prob = self.model.predict([np.array(t_np), np.array(speaker_classes), np.array(topic_input),
                                           np.array(p_np), np.array(c_np), np.array(ys_input)],
                                          batch_size=len(p_np))
            idx = np.argmax(idx_prob, 2)
            preds = [self.vec2txt(i) for i in idx]
            for i in range(3):
                print(preds[i])
            return preds
        else:
            pre_probs = self.model.predict([np.array(t_np), np.array(speaker_classes), np.array(topic_input),
                                            np.array(p_np), np.array(c_np), np.array(ys)],
                                           batch_size=len(p_np))
            # pre_probs = pre_probs.reshape((self.t_max_len, -1))
            print(self.vec2txt([ys[0, next_i], pre_probs[0, next_i, :].argmax()]))
            # print(self.vec2txt([pre_probs[0, next_i, :].argmax()]))
            return pre_probs[0, next_i, :]

    def predict(self, topic_input, speaker_classes, p_np, c_np, t_np):
        batch_size = len(p_np)
        t_max_len = self.t_max_len

        [init_h1, init_h2, init_c1, init_c2] = self.encoder_model.predict(np.array(t_np), batch_size=batch_size)
        c_x = self.context_model.predict(c_np, batch_size=batch_size)
        persona_topic = self.persona_topic_model.predict([np.array(p_np), np.array(topic_input)], batch_size=batch_size)
        ys_input = np.ones((t_np.shape[0], 1), dtype='int32')
        pred_word_ids = np.zeros((t_np.shape[0], t_max_len), dtype='int32')
        for i in range(t_max_len):
            pred_probs, _h1, _h2, _c1, _c2 = self.rg_model.predict(
                [init_h1, init_h2, init_c1, init_c2,
                 np.array(speaker_classes), persona_topic, c_x, ys_input],
                batch_size=batch_size)
            tmp = np.squeeze(pred_probs)
            tmp = np.argmax(tmp, axis=len(tmp.shape)-1)
            pred_word_ids[:, i] = tmp
            ys_input = tmp
            init_h1 = _h1
            init_h2 = _h2
            init_c1 = _c1
            init_c2 = _c2

        preds = []
        for ys in pred_word_ids:
            tmp = []
            for y in ys:
                if y == 1:
                    continue
                elif y == 2:
                    break
                else:
                    tmp.append(y)
            preds.append(tmp)

        preds = [self.vec2txt(i) for i in preds]
        for i in range(2):
            print('\n')
            for pi in p_np[i]:
                print('your persona: '+re.sub('__NULL__ ', '', self.vec2txt(pi)))
            print('context: '+re.sub('__NULL__ ', '', self.vec2txt(c_np[i])))
            print('text: '+re.sub('__NULL__ ', '', self.vec2txt(t_np[i])))
            print('resp: '+preds[i])
            print('\n')
        return preds

    def getLDAtopic(self, text):
        # print('Get topic information...')
        l1 = ['comparen\'tes', 'i\'ver', 'n t ', 'can t ', 'hi\'m', '\'ssue', '\'sland', 'won\'tice',
              'ii\'m', 'n\'thing', 'she\'sn\'t', 'they\'ren\'t', 'i\'ven t', 'i\'dn\'t', 'won’t', 'won\'t',
              'wouldn’t', 'wouldn\'t', '’m', '’re', '’ve', '’ll', '’s', '’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll',
              '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,',
              ' don t ', 'i m ', '\t']
        l2 = ['compare notes', 'i have never', ' not ', 'can not ', 'hi i\'m', ' issue', 'island', 'won\'t notice',
              'i am', ' nothing', 'she is not', 'they are not', 'i have not', 'i do not', 'will not', 'will not',
              'would not', 'would not', ' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have',
              ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',',
              ' do not ', 'i am ', ' ']
        l3 = ['-', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .',
              ',', '.', '!', '?', ';', ':', '\'', '__NULL__', '__START__', '__END__', '__UNK__']

        for i, txt in enumerate(text):
            for j, term in enumerate(l1):
                txt = txt.replace(term, l2[j])
            for term in l3:
                txt = txt.replace(term, '')
            txt = re.sub('( )+', ' ', txt)
            txt = txt.replace('\n', ' ')
            txt = txt.strip()

            # 语料里面有个SB，专门为他写的规则
            txt = re.sub('( !)+', ' !', txt)
            txt = re.sub('( \?)+', ' ?', txt)
            text[i] = txt
        # print("Remove stop words from original text...")
        train = []
        for line in text:
            train.append([w for w in line.split(' ') if w not in set(stopwords.words('english'))])
        corpus = [self.lda_dict.doc2bow(t) for t in train]
        train_topic = []
        for c in corpus:
            tmp = self.lda_model.get_document_topics(c, per_word_topics=False)
            if len(tmp) > 1:
                max_p = tmp[0][1]
                max_i = 0
                i = 1
                while i < len(tmp):
                    if tmp[i][1] > max_p:
                        max_i = i
                        max_p = tmp[i][1]
                    i += 1
                train_topic.append(tmp[max_i][0] + 1)
            elif len(tmp) == 0:
                train_topic.append(0)
            else:
                train_topic.append(tmp[0][0] + 1)
        return train_topic

    def getSpeakerClass(self, text):
        l1 = ['comparen\'tes', 'i\'ver', 'n t ', 'can t ', 'hi\'m', '\'ssue', '\'sland', 'won\'tice',
              'ii\'m', 'n\'thing', 'she\'sn\'t', 'they\'ren\'t', 'i\'ven t', 'i\'dn\'t', 'won’t', 'won\'t',
              'wouldn’t', 'wouldn\'t', '’m', '’re', '’ve', '’ll', '’s', '’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll',
              '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,']
        l2 = ['compare notes', 'i have never', ' not ', 'can not ', 'hi i\'m', ' issue', 'island', 'won\'t notice',
              'i am', ' nothing', 'she is not', 'they are not', 'i have not', 'i do not', 'will not', 'will not',
              'would not', 'would not', ' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have',
              ' will',
              ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',']
        l3 = ['-', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
        persona = []
        for line in text:
            txt = line.strip()
            for j, term in enumerate(l1):
                txt = txt.replace(term, l2[j])
            for term in l3:
                txt = txt.replace(term, ' ')
            # txt = txt.lower()
            txt = txt.replace('  ', ' ')
            txt = txt.replace('\n', ' ')

            # # 语料里面有个SB，专门为他写的规则
            txt = re.sub('( !)+', ' !', txt)
            txt = re.sub('( \?)+', ' ?', txt)
            txt.strip()
            persona.append(txt)
        persona = ' '.join(persona)
        vectorized = self.vec.transform([persona])
        lable = self.km.predict(vectorized)
        return lable[0]

    def batchify(self, obs):
        """Convert batch observations `text` and `label` to rank 2 tensor `xs` and `ys`
        """
        # max_len = self.max_len

        p_max_sentence_num = self.p_max_sentence_num
        p_max_len = self.p_max_len
        t_max_len = self.t_max_len
        c_max_len = self.c_max_len

        def txt2np(txt, max_len, use_offset=True, use_max_len=True):
            vec = [self.txt2vec(t) for t in txt]
            if use_max_len == False:
                max_len = max([len(v) for v in vec])
            arr = np.zeros((len(vec), max_len)).astype(np.int32)  # 0 filled rank 2 tensor
            for i, v in enumerate(vec):
                if use_max_len and len(v) > max_len:
                    v = v[(len(v) - max_len):len(v)]
                offset = 0
                if use_offset:
                    offset = max_len - len(v)  # Right justified
                for j, idx in enumerate(v):
                    if j >= max_len:
                        break
                    try:
                        arr[i][j + offset] = idx
                    except:
                        print(txt[i])
                        os._exit()
            return arr  # batch x time

        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        if len(exs) == 0:
            return (None,) * 3

        xs = [ex['text'] for ex in exs]
        persona = []
        context = []
        text = []
        speaker_classes = []
        for x in xs:
            tmp_p = []
            tmp_c = []
            x_split = x.split('\n')
            for s in x_split:
                if 'your persona: ' in s:
                    tmp_p.append(re.sub('your persona: ', '', s))
                else:
                    tmp_c.append(s)
            speaker_classes.append(self.getSpeakerClass(tmp_p))
            if len(tmp_p) < p_max_sentence_num:
                for _ in range(p_max_sentence_num - len(tmp_p)):
                    tmp_p.append(' '.join(['__NULL__'] * p_max_len))
            elif len(tmp_p) > p_max_sentence_num:
                tmp_p = tmp_p[0:p_max_sentence_num]
                print("warning: The number of persona sentences is greater than " + str(p_max_sentence_num) + "!")
            persona.append(tmp_p)
            if len(tmp_c) == 1:
                context.append([' '.join(['__NULL__'] * p_max_len)])
                text.append(tmp_c[0])
            else:
                context.append(tmp_c[0:(len(tmp_c) - 1)])
                text.append(tmp_c[len(tmp_c) - 1])

        p_np = [txt2np(p, p_max_len, use_offset=True, use_max_len=True) for p in persona]
        for i, c in enumerate(context):
            tmp = ' __END__ '.join(c)
            context[i] = tmp
        c_np = txt2np(context, c_max_len, use_offset=True, use_max_len=True)
        t_np = txt2np(text, t_max_len, use_offset=True, use_max_len=True)
        ys = None
        if 'labels' in exs[0]:
            ys = [' '.join(ex['labels']) + ' __END__' for ex in exs]
            ys = txt2np(ys, t_max_len, use_offset=False, use_max_len=True)
        elif 'eval_labels' in exs[0] and 'eval_ppl' in exs[0]:
            if exs[0]['eval_ppl']:
                ys = [' '.join(ex['eval_labels']) for ex in exs]
                ys = txt2np(ys, t_max_len, use_offset=False, use_max_len=True)
        topic_input = self.getLDAtopic(text)
        return topic_input, speaker_classes, p_np, c_np, t_np, ys, valid_inds

    def batch_act(self, observations):
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        topic_input, speaker_classes, p_np, c_np, t_np, ys, valid_inds = self.batchify(observations)

        if t_np is None:
            return batch_reply

        if ys is not None:
            preds = self.train(topic_input, speaker_classes, p_np, c_np, t_np, ys)  # ['bedroom', ...]
        else:
            preds = self.predict(topic_input, speaker_classes, p_np, c_np, t_np)

        for i in range(len(preds)):
            batch_reply[valid_inds[i]]['text'] = preds[i]

        return batch_reply  # [{'text': 'bedroom', 'id': 'RNN'}, ...]

    def act(self):
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        path = self.path if path is None else path
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

    def load_pre_model(self):
        if os.path.exists(self.topic_model_dir):
            self.lda_model = LdaModel.load(os.path.join(self.topic_model_dir, 'train_none_original_no_cands.lda.model'))
            self.lda_dict = Dictionary.load(os.path.join(self.topic_model_dir, 'train_none_original_no_cands.lda.dict'))
            self.topic_num = self.lda_model.num_topics
        else:
            print('error: topic model dir '+self.topic_model_dir+' is not exist.')


        if os.path.exists(self.speaker_model_dir):
            with open(os.path.join(self.speaker_model_dir, "model_tfidf_vectorizer.pkl"), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open(os.path.join(self.speaker_model_dir, "model_vec.pkl"), 'rb') as f:
                self.vec = pickle.load(f)
            with open(os.path.join(self.speaker_model_dir, "model_km.pkl"), 'rb') as f:
                self.km = pickle.load(f)
            self.speaker_num = self.km.n_clusters
        else:
            print("error: speaker model dir "+self.speaker_model_dir+" is not exist.")
            exit(0)

    # def report(self):
    #     """Report loss and perplexity from model's perspective.
    #
    #     Note that this includes predicting __END__ and __UNK__ tokens and may
    #     differ from a truly independent measurement.
    #     """
    #     tmp = self.metrics.report()
    #     return tmp
