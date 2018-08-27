# based on model 1.3.7.4

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

from parlai.agents.convai2_keras.attention import *
from parlai.agents.convai2_keras.utils import *
from parlai.agents.convai2_keras.SCN import SCN
from parlai.agents.convai2_keras.Text_match import *
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
        agent.add_argument('-rnn', '--rnntype', type=str, default='GRU',
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
        agent.add_argument('--beamsearch', type=bool, default=False,
                           help='use beamsearch during prediciton')
        agent.add_argument('--beamsize', type=int, default=10,
                           help='the beam size of beamsearch')
        agent.add_argument('--match-model-file', type=str, default='',
                           help='model file for text matching')
        agent.add_argument('--match-persona', type=bool, default=False,
                           help='need use the match model or not')

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
            self.beamsearch = opt['beamsearch']
            self.beamsize = opt['beamsize']
            self.match_persona = opt['match_persona']

            self.persona_select_model = SCN()
            self.persona_select_model.model_file = opt['persona_select_model_file']
            self.persona_select_model.config = config
            self.persona_select_model.BuildModel()
            self.persona_select_model_sess = self.persona_select_model.LoadModel()

            self.match_model_file = opt['match_model_file']
            self.match_model = buildMatchModel(hidden_size=[40, 30, 30],
                                               vocabulary_size=self.vocabulary_size,
                                               word_embedding_size=self.word_embedding_size,
                                               max_len=self.r_max_len)
            self.match_model.load_weights(self.match_model_file)

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

    def predict_beamsearch(self, context_input, persona_input, beam_size = 10):
        strat_id = 1
        end_id = 2

        sample_num = len(context_input)
        live_k = [1 for _ in range(sample_num)]
        deak_k = [0 for _ in range(sample_num)]
        dead_cands = [[] for _ in range(sample_num)]
        dead_scores = [[] for _ in range(sample_num)]
        live_cands = [[[strat_id]] for _ in range(sample_num)]
        live_scores = [[0.0] for _ in range(sample_num)]

        have_lives = True
        enough_deads = False
        current_position = -1
        while have_lives and (not enough_deads):
            tmp_context_input = []
            tmp_persona_input = []
            tmp_response_input = []
            for i_s, s_n in enumerate(live_k):
                if s_n > 0:
                    for i_c in range(s_n):
                        tmp_context_input.append(context_input[i_s])
                        tmp_persona_input.append(persona_input[i_s])
                        tmp = live_cands[i_s][i_c] + [0 for _ in range(self.r_max_len-len(live_cands[i_s][i_c]))]
                        tmp_response_input.append(tmp)

            pred_probs = self.model.predict({'context_input': np.array(tmp_context_input),
                                             'persona_input': np.array(tmp_persona_input),
                                             'response_input': np.array(tmp_response_input)},
                                            batch_size=self.batch_size)
            current_position += 1
            split_index = 0
            for i_s, s_n in enumerate(live_k):
                if s_n > 0:
                    probs = pred_probs[split_index:split_index+s_n, current_position, :]
                    # total score for every candidate is sum of -log of word prob
                    probs = probs.reshape((s_n, -1))
                    cand_scores = np.array(live_scores[i_s])[:, None] - np.log(probs)
                    cand_flat = cand_scores.flatten()

                    # get the best (lowest) scores we have from all possible candidates and new words
                    ranks_flat = cand_flat.argsort()[:(beam_size - deak_k[i_s])]
                    live_scores[i_s] = cand_flat[ranks_flat]

                    # append the new words to their appropriate live candidates
                    vs = self.vocabulary_size
                    live_cands[i_s] = [live_cands[i_s][r // vs] + [r % vs] for r in ranks_flat]

                    # find the dead candidates in live ones
                    zombies = [s[-1] == end_id or len(s) >= self.r_max_len for s in live_cands[i_s]]

                    # add the zombies to dead candidates
                    dead_cands[i_s] += [s[1:len(s)] for s, z in zip(live_cands[i_s], zombies) if z]
                    dead_scores[i_s] += [s for s, z in zip(live_scores[i_s], zombies) if z]
                    deak_k[i_s] = len(dead_cands[i_s])

                    # remove the zombies from the live candidates
                    live_cands[i_s] = [s for s, z in zip(live_cands[i_s], zombies) if not z]
                    live_scores[i_s] = [s for s, z in zip(live_scores[i_s], zombies) if not z]
                    live_k[i_s] = len(live_cands[i_s])
                split_index += s_n
            if sum(live_k) == 0:
                have_lives = False
            else:
                have_lives = True

            if all([k == beam_size for k in deak_k]):
                enough_deads = True
            else:
                enough_deads = False

        cands = []
        scores = []
        for i in range(sample_num):
            tmp_cands = dead_cands[i]
            tmp_scores = dead_scores[i]
            if len(live_cands[i]) > 0:
                tmp_cands += live_cands[i]
                tmp_scores += live_scores[i]
            cands.append(tmp_cands)
            scores.append(tmp_scores)

        pred_word_ids = []
        for cand, score in zip(cands, scores):
            pred_word_ids.append(cand[np.argsort(score)[0]])

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

    def predict_beamsearch_match_persona(self, context_input, persona_input, beam_size = 10):
        strat_id = 1
        end_id = 2

        sample_num = len(context_input)
        live_k = [1 for _ in range(sample_num)]
        deak_k = [0 for _ in range(sample_num)]
        dead_cands = [[] for _ in range(sample_num)]
        dead_scores = [[] for _ in range(sample_num)]
        live_cands = [[[strat_id]] for _ in range(sample_num)]
        live_scores = [[0.0] for _ in range(sample_num)]

        have_lives = True
        enough_deads = False
        current_position = -1
        while have_lives and (not enough_deads):
            tmp_context_input = []
            tmp_persona_input = []
            tmp_response_input = []
            for i_s, s_n in enumerate(live_k):
                if s_n > 0:
                    for i_c in range(s_n):
                        tmp_context_input.append(context_input[i_s])
                        tmp_persona_input.append(persona_input[i_s])
                        tmp = live_cands[i_s][i_c] + [0 for _ in range(self.r_max_len-len(live_cands[i_s][i_c]))]
                        tmp_response_input.append(tmp)

            pred_probs = self.model.predict({'context_input': np.array(tmp_context_input),
                                             'persona_input': np.array(tmp_persona_input),
                                             'response_input': np.array(tmp_response_input)},
                                            batch_size=self.batch_size)
            current_position += 1
            split_index = 0
            for i_s, s_n in enumerate(live_k):
                if s_n > 0:
                    probs = pred_probs[split_index:split_index+s_n, current_position, :]
                    # total score for every candidate is sum of -log of word prob
                    probs = probs.reshape((s_n, -1))
                    cand_scores = np.array(live_scores[i_s])[:, None] - np.log(probs)
                    cand_flat = cand_scores.flatten()

                    # get the best (lowest) scores we have from all possible candidates and new words
                    ranks_flat = cand_flat.argsort()[:(beam_size - deak_k[i_s])]
                    live_scores[i_s] = cand_flat[ranks_flat]

                    # append the new words to their appropriate live candidates
                    vs = self.vocabulary_size
                    live_cands[i_s] = [live_cands[i_s][r // vs] + [r % vs] for r in ranks_flat]

                    # find the dead candidates in live ones
                    zombies = [s[-1] == end_id or len(s) >= self.r_max_len for s in live_cands[i_s]]

                    # add the zombies to dead candidates
                    dead_cands[i_s] += [s[1:len(s)] for s, z in zip(live_cands[i_s], zombies) if z]
                    dead_scores[i_s] += [s for s, z in zip(live_scores[i_s], zombies) if z]
                    deak_k[i_s] = len(dead_cands[i_s])

                    # remove the zombies from the live candidates
                    live_cands[i_s] = [s for s, z in zip(live_cands[i_s], zombies) if not z]
                    live_scores[i_s] = [s for s, z in zip(live_scores[i_s], zombies) if not z]
                    live_k[i_s] = len(live_cands[i_s])
                split_index += s_n
            if sum(live_k) == 0:
                have_lives = False
            else:
                have_lives = True

            if all([k == beam_size for k in deak_k]):
                enough_deads = True
            else:
                enough_deads = False

        cands = []
        scores = []
        for i in range(sample_num):
            tmp_cands = dead_cands[i]
            tmp_scores = dead_scores[i]
            if len(live_cands[i]) > 0:
                tmp_cands += live_cands[i]
                tmp_scores += live_scores[i]
            cands.append(tmp_cands)
            scores.append(tmp_scores)

        pred_word_ids = []
        for i in range(sample_num):
            seq1 = cands[i]
            seq1 = pad_sequences(seq1, maxlen=self.r_max_len)
            tmp = persona_input[i]
            persona_sentences = []
            p = []
            for w in tmp:
                if w != 0:
                    p.append(w)
                    if w == 2:
                        persona_sentences.append(p)
                        p = []
            persona1 = [persona_sentences[0] for _ in range(len(seq1))]
            persona2 = [persona_sentences[1] for _ in range(len(seq1))]
            persona1 = pad_sequences(np.array(persona1), maxlen=self.r_max_len)
            persona2 = pad_sequences(np.array(persona2), maxlen=self.r_max_len)

            seq1_diff, seq2_diff, seq1_overlap, seq2_overlap = comp_diff_overlap(np.append(seq1, seq1, axis=0),
                                                                                 np.append(persona1, persona2, axis=0))
            match_scores = self.match_model.predict({'seq1': np.append(seq1, seq1, axis=0),
                                                     'seq2': np.append(persona1, persona2, axis=0),
                                                     'seq1_diff': np.array(seq1_diff),
                                                     'seq2_diff': np.array(seq2_diff),
                                                     'seq1_overlap': np.array(seq1_overlap),
                                                     'seq2_overlap': np.array(seq2_overlap)},
                                                    batch_size=len(seq1))
            match_scores = np.reshape(-match_scores.flatten(), (2, -1))
            generate_scores = scores[i]
            combined_scores = np.add(generate_scores, match_scores[0, :])
            combined_scores = np.add(combined_scores, match_scores[1, :])
            pred_word_ids.append(cands[i][np.argsort(combined_scores)[0]])

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
            tmp = [preprocess(' '.join(ex['labels'])) for ex in exs]
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
            if self.beamsearch:
                if self.match_persona:
                    preds = self.predict_beamsearch_match_persona(context_input, persona_input, self.beamsize)
                else:
                    preds = self.predict_beamsearch(context_input, persona_input, self.beamsize)
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
