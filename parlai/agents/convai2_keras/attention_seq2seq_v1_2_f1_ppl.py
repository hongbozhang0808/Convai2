# based on model 1.3.7.4

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

from parlai.agents.convai2_keras.attention import *
from parlai.agents.convai2_keras.utils import *
from parlai.agents.convai2_keras.SCN import SCN
import tensorflow as tf
import keras.backend as KTF
import copy
from keras.layers import  Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)


class AttentionSeq2seqAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('RNN Arguments')
        agent.add_argument('-rnn', '--rnntype', type=str, default='LSTM',
                           help='choose GRU or LSTM')
        agent.add_argument('-hs', '--hiddensize', type=int, default=0,
                           help='size of the hidden layers and embeddings')
        agent.add_argument('-dr', '--dropout', type=float, default=0,
                           help='dropout rate')
        agent.add_argument('--embedding-file', type=str, default='',
                           help='embedding file used to init embedding layer')
        agent.add_argument('--word-embedding-size', type=int, default=0,
                           help='the word embedding size')
        agent.add_argument('--persona-select-model-file', type=str, default='',
                           help='persona select model file for SCN')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            self.id = 'RNN'
            self.dict = DictionaryAgent(opt)
            self.vocabulary_size = len(self.dict)
            self.observation = {}
            self.rnn_type = opt['rnntype']
            self.hidden_size = opt['hiddensize']
            self.dropout_rate = opt['dropout']
            self.path = opt.get('model_file', None)
            self.reuse = None if opt['datatype'] == 'train' else True
            self.r_max_len = 30
            self.p_max_len = 60
            self.c_max_len = 100
            self.batch_size = opt['batchsize']
            self.embedding_file = opt['embedding_file']
            self.word_embedding_size = opt['word_embedding_size']
            self.persona_select_model = SCN()
            self.persona_select_model.model_file = opt['persona_select_model_file']
            self.persona_select_model.config = config
            self.persona_select_model.BuildModel()
            self.persona_select_model_sess = self.persona_select_model.LoadModel()
            self.create_model()

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from \'%s\' and \'%s\''%(opt['model_file'], opt['persona_select_model_file']))
                self.model_file = opt['model_file']
                self.load()
        self.episode_done = True

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
        hidden_size = self.hidden_size
        drop_out = self.dropout_rate
        embedding_size = self.word_embedding_size
        vocabulary_size = self.vocabulary_size
        c_max_len = self.c_max_len
        p_max_len = self.p_max_len
        r_max_len = self.r_max_len

        if self.embedding_file == '':
            word_embedding = Embedding(input_dim=self.vocabulary_size, output_dim=self.word_embedding_size, mask_zero=False,
                                       embeddings_initializer='glorot_normal', trainable=True)
        else:
            with open(self.embedding_file, 'rb') as f:
                embedding = pickle.load(f)
            self.word_embedding_size = embedding.shape[1]
            word_embedding = Embedding(input_dim=self.vocabulary_size, output_dim=self.word_embedding_size, mask_zero=False,
                                       weights=[embedding], trainable=False)

        persona_encoder_lstm = Bidirectional(LSTM(int(hidden_size / 2), return_state=False, return_sequences=True,
                                                  kernel_initializer='glorot_normal',
                                                  recurrent_initializer='glorot_normal',
                                                  name='persona_encoder_lstm', recurrent_dropout=drop_out),
                                             merge_mode='concat')

        context_encoder_lstm1 = LSTM(hidden_size, return_state=True, return_sequences=True,
                                     kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal',
                                     name='context_encoder_lstm1', recurrent_dropout=drop_out)
        context_encoder_lstm2 = LSTM(hidden_size, return_state=True, return_sequences=True,
                                     kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal',
                                     name='context_encoder_lstm2', recurrent_dropout=drop_out)

        decoder_lstm1 = LSTM(hidden_size, return_state=True, return_sequences=True,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal',
                             name='decoder_lstm1', recurrent_dropout=drop_out)
        decoder_lstm2 = LSTM(hidden_size, return_state=True, return_sequences=True,
                             kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal',
                             name='decoder_lstm2', recurrent_dropout=drop_out)

        decoder_dense1 = Dense(embedding_size, activation='linear', kernel_initializer='glorot_normal',
                               name='decoder_dense1')
        decoder_dense2 = Dense(vocabulary_size, activation='softmax', kernel_initializer='glorot_normal',
                               name='decoder_dense2')

        # encode context text
        c_input = Input((c_max_len,), name='context_input')
        c_x = word_embedding(c_input)
        c_x = Dropout(drop_out)(c_x)
        c_x, c_h_1, c_c_1 = context_encoder_lstm1(c_x)
        c_x, c_h_2, c_c_2 = context_encoder_lstm2(c_x)

        # encode persona text
        p_input = Input((p_max_len,), name='persona_input')
        p_x = word_embedding(p_input)
        p_x = Dropout(drop_out)(p_x)
        p_x = persona_encoder_lstm(p_x)

        # generate response
        r_input = Input((r_max_len,), name='response_input')
        r_x = word_embedding(r_input)
        r_x = Dropout(drop_out)(r_x)
        r_x, r_h_1, r_c_1 = decoder_lstm1(r_x, initial_state=[c_h_1, c_c_1])
        r_x, r_h_1, r_c_1 = decoder_lstm2(r_x, initial_state=[c_h_2, c_c_2])

        attention_c_r = attention_3d_block(c_x, r_x, c_max_len, r_max_len, False, drop_out=drop_out)
        attention_p_r = attention_3d_block(p_x, r_x, p_max_len, r_max_len, False, drop_out=drop_out)
        t_x = Concatenate(axis=-1)([r_x, attention_c_r, attention_p_r])
        t_x = TimeDistributed(decoder_dense1)(t_x)
        t_x = Dropout(drop_out)(t_x)
        decoder_output = TimeDistributed(decoder_dense2)(t_x)
        self.model = Model([c_input, p_input, r_input], decoder_output)

    def train(self, context_input, persona_input, response_input):
        # vs = len(self.dict)
        # if is_training:
        #     def splitLabelsText(labels):
        #         labels_for_input = []
        #         labels_for_target = []
        #         for l in labels:
        #             tmp = [0] * len(l)
        #             tmp[0] = 1
        #             tmp[1:len(l)] = l[0:(len(l) - 1)]
        #             labels_for_input.append(tmp)
        #
        #             tmp = [0] * len(l)
        #             tmp[0:len(l)] = l
        #             l_one_hot = np.zeros((len(l), vs))
        #             l_one_hot[np.arange(len(l)), tmp] = 1
        #             labels_for_target.append(l_one_hot)
        #         return labels_for_input, labels_for_target
        #
        #     ys_input, ys_target = splitLabelsText(ys)
        #
        #     adam = optimizers.adam(lr=self.learning_rate)
        #
        #     self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        #     self.model.fit([np.array(t_np), np.array(speaker_classes), np.array(topic_input),
        #                     np.array(p_np), np.array(c_np), np.array(ys_input)],
        #                    np.array(ys_target),
        #                    batch_size=len(p_np), epochs=1)
        #     idx_prob = self.model.predict([np.array(t_np), np.array(speaker_classes), np.array(topic_input),
        #                                    np.array(p_np), np.array(c_np), np.array(ys_input)],
        #                                   batch_size=len(p_np))
        #     idx = np.argmax(idx_prob, 2)
        #     preds = [self.vec2txt(i) for i in idx]
        #     for i in range(3):
        #         print(preds[i])
        #     return preds
        # else:
        #     pre_probs = self.model.predict([np.array(t_np), np.array(speaker_classes), np.array(topic_input),
        #                                     np.array(p_np), np.array(c_np), np.array(ys)],
        #                                    batch_size=len(p_np))
        #     # pre_probs = pre_probs.reshape((self.t_max_len, -1))
        #     print(self.vec2txt([ys[0, next_i], pre_probs[0, next_i, :].argmax()]))
        #     # print(self.vec2txt([pre_probs[0, next_i, :].argmax()]))
        #     return pre_probs[0, next_i, :]
        return None

    def predict(self, context_input, persona_input):
        response_input = np.zeros((context_input.shape[0], self.r_max_len), dtype='int32')
        response_input[:, 0] = 1
        pred_word_ids = np.zeros_like(response_input)
        for i in range(self.r_max_len):
            pred_probs = self.model.predict({'context_input':context_input,
                                             'persona_input':persona_input,
                                             'response_input':response_input},
                                            batch_size=self.batch_size)
            pred_probs = pred_probs.argmax(axis=-1)
            pred_probs = pred_probs.squeeze()
            if response_input.shape[0] == 1:
                pred_probs = np.expand_dims(pred_probs, axis=0)
            pred_word_ids[:, i] = pred_probs[:, i]
            if i < self.r_max_len-1:
                response_input[:, i+1] = pred_probs[:, i]

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

        preds = [vec2text(i, self.dict.ind2tok) for i in preds]
        # for i in range(2):
        #     print('\n')
        #     print('persona: ' + re.sub('__NULL__ ', '', ' '.join(vec2text(persona_input[i], self.dict.ind2tok))))
        #     print('context: ' + re.sub('__NULL__ ', '', ' '.join(vec2text(context_input[i], self.dict.ind2tok))))
        #     print('response: ' + ' '.join(preds[i]))
        #     print('\n')
        return [' '.join(s) for s in preds]

    def persona_selection(self, context_input, persona_input, top_n=2):
        p_num = [len(p) for p in persona_input]
        extend_context_input = []
        extend_persona_input = []
        for i, n in enumerate(p_num):
            extend_context_input += [context_input[i] for _ in range(n)]
            extend_persona_input += persona_input[i]
        scores = self.persona_select_model.MakePrediction(extend_context_input, extend_persona_input)
        scores = list(scores)

        start_i = 0
        selected_persona = []
        for i, n in enumerate(p_num):
            tmp_score = [-scores[x] for x in range(start_i, start_i+n)]
            start_i += n

            tmp_p = []
            for y in list(np.argsort(tmp_score))[0:top_n]:
                tmp_p += persona_input[i][y]

            selected_persona.append(tmp_p)
        return selected_persona

    def batchify(self, obs):
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        """Convert batch observations `text` and `label` to rank 2 tensor `xs` and `ys`
        """
        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        if len(exs) == 0:
            return (None,) * 3

        xs = [ex['text'] for ex in exs]
        persona = []
        context = []
        for x in xs:
            tmp_p = []
            tmp_c = []
            x_split = x.split('\n')
            for s in x_split:
                if 'your persona: ' in s:
                    tmp_p.append(re.sub('your persona: ', '', s))
                else:
                    tmp_c.append(s)
            context.append(tmp_c)
            persona.append(tmp_p)

        context = [preprocess(s) for s in context]
        persona = [preprocess(s) for s in persona]
        response = None
        if 'labels' in exs[0]:
            tmp = [' '.join(ex['labels']) for ex in exs]
            tmp = preprocess(tmp)
            response = [list(text2vec([t], self.dict.tok2ind, 30, False, False)[0]) for t in tmp]
            response = pad_sequences(response, self.r_max_len, dtype='int32', padding='post')

        # select useful persona
        context_vetors = [[list(text2vec([s], self.dict.tok2ind, 30, False, False)[0]) for s in ss] for ss in context]
        persona_vetors = [[list(text2vec([s], self.dict.tok2ind, 30, False, False)[0]) for s in ss] for ss in persona]
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        selcted_persona_vectors = self.persona_selection(copy.deepcopy(context_vetors),
                                                         copy.deepcopy(persona_vetors))
        # print('--------------------------------------------------------------')

        context_vetors = [np.concatenate(np.array(s)).tolist() for s in context_vetors]
        context_vetors = pad_sequences(context_vetors, self.c_max_len, dtype='int32', padding='pre')
        selcted_persona_vectors = pad_sequences(selcted_persona_vectors, self.p_max_len, dtype='int32', padding='pre')

        return context_vetors, selcted_persona_vectors, response, valid_inds

    def batch_act(self, observations):
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        context_input, persona_input, response_input, valid_inds = self.batchify(observations)

        if context_input is None:
            return batch_reply

        if response_input is not None:
            preds = self.train(context_input, persona_input, response_input)  # ['bedroom', ...]
        else:
            preds = self.predict(context_input, persona_input)

        for i in range(len(preds)):
            batch_reply[valid_inds[i]]['text'] = preds[i]

        return batch_reply  # [{'text': 'bedroom', 'id': 'RNN'}, ...]

    def act(self):
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        path = self.path if path is None else path
        self.model.save_weights(path)

    def load(self):
        self.model.load_weights(self.model_file)

    # def report(self):
    #     """Report loss and perplexity from model's perspective.
    #
    #     Note that this includes predicting __END__ and __UNK__ tokens and may
    #     differ from a truly independent measurement.
    #     """
    #     tmp = self.metrics.report()
    #     return tmp

class mydefaultdict(defaultdict):
    """Custom defaultdict which overrides defaults requested by the get
    function with the default factory.
    """
    def get(self, key, default=None):
        # override default from "get" (like "__getitem__" already is)
        return super().get(key, default or self.default_factory())

class PerplexityEvaluatorAgent(AttentionSeq2seqAgent):
    """Subclass for doing standardized perplexity evaluation.

    This is designed to be used in conjunction with the PerplexityWorld at
    parlai/scripts/eval_ppl.py. It uses the `next_word_probability` function
    to calculate the probability of tokens one token at a time.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        # self.prev_enc = None
        # self.last_xs = None

    def next_word_probability(self, partial_out):
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
        obs = self.observation
        next_index = 0
        if len(partial_out) > 0:
            obs['labels'] = ['__start__ '+' '.join(partial_out)]
            next_index = len(partial_out)
        else:
            obs['labels'] = ['__start__']
        context_vetors, selcted_persona_vectors, response, _ = self.batchify([obs])
        probs = self.model.predict({'context_input': context_vetors,
                                    'persona_input': selcted_persona_vectors,
                                    'response_input': response},
                                   batch_size=self.batch_size)
        probs = probs[0, next_index, :].flatten()

        dist = mydefaultdict(lambda: 1e-7)  # default probability for any token
        for i in range(len(probs)):
            dist[self.dict[i]] = probs[i].item()
        return dist
