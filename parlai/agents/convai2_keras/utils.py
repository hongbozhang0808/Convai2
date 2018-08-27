import re
import os
import numpy as np
import pandas as pd
import codecs
from collections import defaultdict
import pickle
import random
from keras.preprocessing.sequence import pad_sequences

# preprocess the input english/spanish sentences,
# there are some additional methods for processing spanish in .senti/classifier/preProcessing.py, which could be useful.
def preprocess(text, lan='english'):
    eng1 = ['comparen\'tes', 'i\'ver', 'n t ', 'can t ', 'hi\'m', '\'ssue', '\'sland', 'won\'tice',
            'ii\'m', 'n\'thing', 'she\'sn\'t', 'they\'ren\'t', 'i\'ven t', 'i\'dn\'t', 'won’t', 'won\'t',
            'wouldn’t', 'wouldn\'t', '’m', '’re', '’ve', '’ll', '’s', '’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll',
            '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'cannot']
    eng2 = ['compare notes', 'i have never', ' not ', 'can not ', 'hi i\'m', ' issue', 'island', 'won\'t notice',
            'i am', ' nothing', 'she is not', 'they are not', 'i have not', 'i do not', 'will not', 'will not',
            'would not', 'would not', ' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have',
            ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', 'can not']
    eng3 = ['-', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .', '\'']

    spa1 = [',', ';', '.', '?', '!', ':', '¿']
    spa2 = [' ,', ' ;', ' .', ' ?', ' !', ' :', '¿ ']
    spa3 = ['-', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .', '\'', '/']

    if lan=='english':
        text = [t.lower() for t in text]
        text = [re.sub('![!]*', '!', t) for t in text]
        text = [re.sub('\?[?]*', ' ? ', t) for t in text]
        text = [re.sub('\.[.]*', '.', t) for t in text]
        for j, term in enumerate(eng1):
            text = [t.replace(term, eng2[j]) for t in text]
        for term in eng3:
            text = [t.replace(term, ' ') for t in text]
        text = [t.replace('\n', ' ') for t in text]
        text = [re.sub(' [ ]*', ' ', t) for t in text]
        text = [t.strip() for t in text]
        text = [t+' __END__' for t in text]
        return text
    elif lan=='spanish':
        ###
        # preprocess spanish here
        text = [t.lower() for t in text]
        text = [re.sub('![!]*', '!', t) for t in text]
        text = [re.sub('\?[?]*', ' ? ', t) for t in text]
        text = [re.sub('\.[.]*', '.', t) for t in text]
        for j, term in enumerate(spa1):
            text = [t.replace(term, spa2[j]) for t in text]
        for term in spa3:
            text = [t.replace(term, ' ') for t in text]
        text = [t.replace('\n', ' ') for t in text]
        text = [re.sub(' [ ]*', ' ', t) for t in text]
        text = [t.strip() for t in text]
        text = [t+' __END__' for t in text]
        ###
        return text
    else:
        os._exit(0)

# load existing dictionary to construct tok2ind, ind2tok
def loadDict(dict_file):
    def unescape(s):
        return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    print('load dictionary from '+dict_file)
    freq = defaultdict(int)
    tok2ind = {}
    ind2tok = {}
    with codecs.open(dict_file, 'r', encoding='utf-8') as read:
        for line in read:
            split = line.strip().split('\t')
            token = unescape(split[0])
            cnt = int(split[1]) if len(split) > 1 else 0
            freq[token] = cnt
            if token not in tok2ind:
                index = len(tok2ind)
                tok2ind[token] = index
                ind2tok[index] = token
    print('number of words: %d' % tok2ind[token])
    return tok2ind, ind2tok

# load given word2vetor file
def loadGivenEmbedding(embedding_file):
    print('load embedding from ' + embedding_file)
    embedding_index = {}
    with codecs.open(embedding_file, 'r', encoding='utf-8') as f:
        # skip the first line
        line = f.readline()
        embedding_size = int(line.split(' ')[1])
        line = f.readline()
        print('--------these lines have exceptions in the given embedding--------')
        while line:
            values = line.split()
            if len(values) > embedding_size+1:
                word = ''.join(values[0:(len(values)-embedding_size)])
                coefs = np.array(values[(len(values)-embedding_size):len(values)], dtype='float32')
                print("longer line: "+word)
            elif len(values) < embedding_size+1:
                print("shorter line: "+line)
            else:
                word = values[0]
                coefs = np.array(values[1:], dtype='float32')
            embedding_index[word] = coefs
            line = f.readline()
        print('------------------------------------------------------------------\n\n')
    return embedding_index

# using the given pretrained word2vetor to construct initial embedding weights
def construtInitEmbedding(ind2tok, embedding_index):
    dict_size = len(ind2tok)
    embedding_size = len(embedding_index[list(embedding_index.keys())[0]])
    embedding = np.zeros((dict_size, embedding_size), dtype='float32')
    # preprocess '__null__', '__start__', '__end__', '__unk__'
    embedding[0, :] = [0.0] * embedding_size
    embedding[1, :] = np.random.standard_normal(embedding_size)
    embedding[2, :] = np.random.standard_normal(embedding_size)
    embedding[3, :] = np.random.standard_normal(embedding_size)
    print('--------these words are not in the given embedding--------')
    for id in range(4, dict_size):
        if embedding_index.__contains__(ind2tok[id]):
            embedding[id, :] = embedding_index.get(ind2tok[id])
        else:
            embedding[id, :] = np.random.standard_normal(embedding_size)
            print(ind2tok[id])
    print('----------------------------------------------------------\n\n')
    return embedding, len(embedding), embedding_size

# transform the sentences to their id representations
def text2vec(text, tok2ind, max_len, use_offset=True, use_max_len=True):
    def t2v(t):
        words = t.split(' ')
        v = [0 for _ in range(len(words))]
        for iw, w in enumerate(words):
            if tok2ind.__contains__(w):
                v[iw] = tok2ind[w]
            elif w == '__start__':
                v[iw] = 1
            else:
                # '__nuk__'
                v[iw] = 3
        return v

    vec = [t2v(t) for t in text]
    if use_max_len == False:
        max_len = max([len(v) for v in vec])
    arr = np.zeros((len(vec), max_len), dtype='int32')
    for iv, v in enumerate(vec):
        if use_max_len and len(v) > max_len:
            v = v[(len(v) - max_len):len(v)]
        offset = 0
        if use_offset:
            offset = max_len - len(v)
        for j, idx in enumerate(v):
            if j >= max_len:
                break;
            try:
                arr[iv][j+offset] = idx
            except:
                print(text[iv])
                os._exit(0)
    return arr

# transform their id representations to the word sentences
def vec2text(vec, ind2tok):
    text = ['__NULL__'] * len(vec)
    for i, v in enumerate(vec):
        text[i] = ind2tok[v]
    return text

# split the target language vector into two vectors, and one of the vectors shift one time step with the other one:
# the two vectors are used as the input of decoder_model and the ground truth respectively
# the ground truth must be tranformed as one-hot matrix whose shape is [batch_size, max_seq_len, vocabulary_size]
def splitTargetText(text, vs):
    text_for_input = []
    text_for_label = np.zeros((len(text), len(text[0]), vs), dtype='int8')
    for i, t in enumerate(text):
        tmp = [0] * len(t)
        tmp[0] = 1
        tmp[1:len(t)] = t[0:(len(t)-1)]
        text_for_input.append(tmp)

        tmp = t
        # tmp[len(t)] = 2
        t_onehot = np.zeros((len(t), vs), dtype='int8')
        t_onehot[np.arange(len(t)), tmp] = 1
        text_for_label[i, :, :] = t_onehot
    return text_for_input, text_for_label

def multi_sequences_padding(all_sequences, max_sentence_len=50, max_num_utterance=10):
    # max_num_utterance = 10
    PAD_SEQUENCE = [0] * max_sentence_len
    padded_sequences = []
    sequences_length = []
    for sequences in all_sequences:
        sequences_len = len(sequences)
        sequences_length.append(get_sequences_length(sequences, maxlen=max_sentence_len))
        if sequences_len < max_num_utterance:
            sequences += [PAD_SEQUENCE] * (max_num_utterance - sequences_len)
            sequences_length[-1] += [0] * (max_num_utterance - sequences_len)
        else:
            sequences = sequences[-max_num_utterance:]
            sequences_length[-1] = sequences_length[-1][-max_num_utterance:]
        sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
        padded_sequences.append(sequences)
    return padded_sequences, sequences_length


def get_sequences_length(sequences, maxlen):
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return sequences_length

def read_unprocessed_conversation(input_file):
    conv_data = []
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        conv_i = 0
        my_persona = []
        pa_persona = []
        my_say = []
        pa_say = []
        for line in f.readlines():
            tmp = re.sub('\n', '', line)
            tmp = tmp.split(' ')
            line_id = int(tmp[0])
            line = ' '.join(tmp[1:])
            if line_id == 1:
                if conv_i > 0:
                    conv_data.append([my_persona, pa_persona, my_say, pa_say])
                conv_i += 1
                my_persona = []
                pa_persona = []
                my_say = []
                pa_say = []
            if 'your persona: ' in line:
                my_persona.append(re.sub('your persona: ', '', line))
            elif 'partner\'s persona: ' in line:
                pa_persona.append(re.sub('partner\'s persona: ', '', line))
            else:
                tmp = line.split('\t')
                if len(tmp) != 2:
                    print('error line: '+line)
                    os._exit(0)
                else:
                    pa_say.append(tmp[0])
                    my_say.append(tmp[1])
        conv_data.append([my_persona, pa_persona, my_say, pa_say])
    return conv_data

# conv_data [[my_persona, pa_persona, my_say, pa_say]]
# match_data [[id, say, per, score]]
def construct_input_for_SCN(conv_data, match_data, output_file, tok2ind):
    history = []
    response = []
    label = []
    # match_data = pd.DataFrame

    match_data_start_index = 0
    for i_c, conv in enumerate(conv_data):
        # prepare sentences
        [my_persona, pa_persona, my_say, pa_say] = conv
        need_size = len(my_persona) * len(my_say) + len(pa_persona) * len(pa_say)

        # get corresponding match data
        tmp_match_data = []
        for i_m in range(match_data_start_index, match_data_start_index+need_size):
            tmp_match_data.append(list(match_data.iloc[i_m]))
        match_data_start_index += need_size

        # check if conv_data and match_data matching
        [id_in_match, say_in_match, per_in_match, _] = tmp_match_data[0]
        if id_in_match != str(i_c)+'_m_0_0' or say_in_match != my_say[0] or per_in_match != my_persona[0]:
            print('error: conv_data and match_data are not matching at beginning!')
            print('conv_data: %s'%('\t'.join([str(i_c)+'_m_0_0', my_say[0], my_persona[0]])))
            print('match_data: %s'%('\t'.join([id_in_match, say_in_match, per_in_match])))
            os._exit(0)

        [id_in_match, say_in_match, per_in_match, _] = tmp_match_data[-1]
        if id_in_match != str(i_c)+'_p_'+str(len(pa_say)-1)+'_'+str(len(pa_persona)-1) or say_in_match != pa_say[-1] or per_in_match != pa_persona[-1]:
            print('error: conv_data and match_data are not matching at end!')
            print('conv_data: %s'%('\t'.join([str(i_c)+'_p_'+str(len(pa_say)-1)+'_'+str(len(pa_persona)-1), pa_say[-1], pa_persona[-1]])))
            print('match_data: %s'%('\t'.join([id_in_match, say_in_match, per_in_match])))
            os._exit(0)

        # preprocess and translate sentences into vectors
        my_persona = preprocess(my_persona)
        pa_persona = preprocess(pa_persona)
        my_say = preprocess(my_say)
        pa_say = preprocess(pa_say)

        my_persona_vec = [text2vec([s], tok2ind, 30, False, False) for s in my_persona]
        my_persona_vec = [list(s[0]) for s in my_persona_vec]

        pa_persona_vec = [text2vec([s], tok2ind, 30, False, False) for s in pa_persona]
        pa_persona_vec = [list(s[0]) for s in pa_persona_vec]

        my_say_vec = [text2vec([s], tok2ind, 30, False, False) for s in my_say]
        my_say_vec = [list(s[0]) for s in my_say_vec]

        pa_say_vec = [text2vec([s], tok2ind, 30, False, False) for s in pa_say]
        pa_say_vec = [list(s[0]) for s in pa_say_vec]

        # joint conv_data and match_data
        tmp_history = []
        tmp_response = []
        tmp_label = []
        for i_s, say in enumerate(my_say_vec):
            h = [[]]
            if i_s == 0:
                h.append(pa_say_vec[0])
                # h.append(say)
            else:
                for i in range(i_s):
                    h.append(pa_say_vec[i])
                    h.append(my_say_vec[i])
                h.append(pa_say_vec[i_s])
                # h.append(say)

            for i in range(len(my_persona_vec)):
                tmp_history.append(h)
                tmp_response.append(my_persona_vec[i])
                tmp_label.append(float(tmp_match_data[i_s*len(my_persona)+i][3]))

        history += tmp_history
        response += tmp_response
        label += tmp_label

        if i_c%100 == 0:
            print('%d/%d done!'%(i_c+1, len(conv_data)))


    with open(output_file, 'wb') as f:
        pickle.dump([history, response, label], f)

    return history, response, label