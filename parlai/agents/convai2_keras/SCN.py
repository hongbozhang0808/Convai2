# softmax, classification
import tensorflow as tf
import pickle
from parlai.agents.convai2_keras.utils import *
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import gc

# those files must be given during model training
embedding_file = None
evaluate_file = None
train_file = None
# embedding_file = '../data/original_embedding_full.pkl'
evaluate_file = '/data/ParlAI/mymodels/persona_select/data/valid_both_original_no_cands.txt_SCN.pkl'
# train_file = '../data/train_both_original_no_cands.txt_SCN.pkl'
# # train_file = '../data/valid_both_original_no_cands.txt_SCN.pkl'

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SCN():
    def __init__(self):
        self.max_num_utterance = 10
        self.max_sentence_len = 50
        self.word_embedding_size = 300
        self.rnn_units = 200
        self.total_words = 19304
        self.batch_size = 256
        self.epochs = 30
        self.min_valid_loglss = 10.0
        self.current_valid_logloss = 100.0
        self.model_file = None
        self.threshold = 0.7
        self.config = None

    def LoadModel(self):
        # init = tf.global_variables_initializer()
        # saver = tf.train.Saver()
        # # sess = tf.Session()
        # with tf.Session() as sess:
        #     sess.run(init)
        #     saver.restore(sess, self.model_file)
        #     return sess

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.sess = tf.Session(config=self.config)
        self.sess.run(init)
        saver.restore(self.sess, self.model_file)


    def BuildModel(self):
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
        self.y_true = tf.placeholder(tf.int32, shape=(None, ))
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        self.response_len = tf.placeholder(tf.int32, shape=(None,))
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))
        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words,self.
                                                                      word_embedding_size), dtype=tf.float32, trainable=False)
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        reuse = None

        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings, sequence_length=self.response_len, dtype=tf.float32,
                                                       scope='sentence_GRU')
        self.response_embedding_save = response_GRU_embeddings
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings, sequence_length=utterance_len, dtype=tf.float32,
                                                            scope='sentence_GRU')
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse, name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(matching_vectors, axis=0, name='matching_stack'), dtype=tf.float32,
                                           time_major=True, scope='final_GRU')  # TODO: check time_major
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
        self.y_pred = tf.nn.softmax(logits)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits)
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.total_loss)

    def Evaluate(self):
        with open(evaluate_file, 'rb') as f:
           history, true_utt, labels = pickle.load(f)
           tmp_labels = []
           for l in labels:
               if l >= self.threshold:
                   tmp_labels.append(1)
               else:
                   tmp_labels.append(0)
           labels = tmp_labels
        self.all_candidate_scores = []
        self.all_losses = []
        history, history_len = multi_sequences_padding(history, self.max_sentence_len)
        history, history_len = np.array(history), np.array(history_len)
        true_utt_len = np.array(get_sequences_length(true_utt, maxlen=self.max_sentence_len))
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
        low = 0
        while True:
            n_sample = min(low + self.batch_size, history.shape[0]) - low
            feed_dict = {self.utterance_ph: np.concatenate([history[low:low + n_sample]], axis=0),
                         self.all_utterance_len_ph: np.concatenate([history_len[low:low + n_sample]], axis=0),
                         self.response_ph: np.concatenate([true_utt[low:low + n_sample]], axis=0),
                         self.response_len: np.concatenate([true_utt_len[low:low + n_sample]], axis=0),
                         self.y_true: np.concatenate([labels[low:low + n_sample]], axis=0)
                         }
            candidate_scores = self.sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])

            losses = self.sess.run(self.losses, feed_dict=feed_dict)
            self.all_losses.append(losses)
            low = low + n_sample
            if low >= history.shape[0]:
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)
        all_losses = np.concatenate(self.all_losses, axis=0).mean()
        print('mean log loss on valid data: %f'%(all_losses))
        self.current_valid_logloss = all_losses
        return labels, all_candidate_scores

    def MakePrediction(self, history, true_utt):
        history, history_len = multi_sequences_padding(history, self.max_sentence_len)
        history, history_len = np.array(history), np.array(history_len)
        true_utt_len = np.array(get_sequences_length(true_utt, maxlen=self.max_sentence_len))
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
        low = 0
        all_candidate_scores = []
        while True:
            n_sample = min(low + self.batch_size, history.shape[0]) - low
            feed_dict = {self.utterance_ph: np.concatenate([history[low:low + n_sample]], axis=0),
                         self.all_utterance_len_ph: np.concatenate([history_len[low:low + n_sample]], axis=0),
                         self.response_ph: np.concatenate([true_utt[low:low + n_sample]], axis=0),
                         self.response_len: np.concatenate([true_utt_len[low:low + n_sample]], axis=0)
                         }
            # print('1111111111111111111111111111111111111111111111111')
            candidate_scores = self.sess.run(self.y_pred, feed_dict=feed_dict)
            # print('2222222222222222222222222222222222222222222222222')
            all_candidate_scores.append(candidate_scores[:, 1])
            low = low + n_sample
            if low >= history.shape[0]:
                break
        all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
        return all_candidate_scores


    def TrainModel(self, countinue_train = False, previous_modelpath = "model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        # with tf.Session(config=config) as sess:
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("output2", sess.graph)
            train_writer = tf.summary.FileWriter('output2', sess.graph)
            with open(embedding_file, 'rb') as f:
                embeddings = pickle.load(f,encoding="bytes")
            with open(train_file, 'rb') as f:
                history, true_utt, labels = pickle.load(f)
                tmp_labels = []
                for l in labels:
                    if l >= self.threshold:
                        tmp_labels.append(1)
                    else:
                        tmp_labels.append(0)
                labels = tmp_labels

            history, history_len = multi_sequences_padding(history, self.max_sentence_len)
            true_utt_len = np.array(get_sequences_length(true_utt, maxlen=self.max_sentence_len))
            true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
            history, history_len = np.array(history), np.array(history_len)
            if countinue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: embeddings})
            else:
                saver.restore(sess,previous_modelpath)
            low = 0
            epoch = 1
            while epoch < self.epochs:
                n_sample = min(low + self.batch_size, history.shape[0]) - low
                feed_dict = {self.utterance_ph: np.concatenate([history[low:low + n_sample]], axis=0),
                             self.all_utterance_len_ph: np.concatenate([history_len[low:low + n_sample]], axis=0),
                             self.response_ph: np.concatenate([true_utt[low:low + n_sample]], axis=0),
                             self.response_len: np.concatenate([true_utt_len[low:low + n_sample]], axis=0),
                             self.y_true: np.concatenate([labels[low:low + n_sample]], axis=0)
                             }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                if low % (self.batch_size*10) == 0:
                    print("Training loss: ",sess.run(self.total_loss, feed_dict=feed_dict))
                    # self.Evaluate(sess)
                if low >= history.shape[0]:
                    print("Evaluation accuracy: ")
                    self.Evaluate(sess)
                    low = 0
                    if self.current_valid_logloss < self.min_valid_loglss:
                        print('better valid logloss %f vs %f in epoch %d'%(self.current_valid_logloss, self.min_valid_loglss, epoch))
                        saver.save(sess, self.model_file)
                        self.min_valid_loglss = self.current_valid_logloss
                    # saver.save(sess,"model/model.{0}".format(epoch))
                    # print(sess.run(self.total_loss, feed_dict=feed_dict))
                    print('epoch={i}'.format(i=epoch))

                    shuffle_index = list(range(len(history)))
                    np.random.shuffle(shuffle_index)
                    shuffle_history = np.zeros_like(history)
                    shuffle_history_len = np.zeros_like(history_len)
                    shuffle_true_utt = np.zeros_like(true_utt)
                    shuffle_true_utt_len = np.zeros_like(true_utt_len)
                    shuffle_labels = np.zeros_like(labels)

                    for i in shuffle_index:
                        shuffle_history[i] = history[i]
                        shuffle_history_len[i] = history_len[i]
                        shuffle_true_utt[i] = true_utt[i]
                        shuffle_true_utt_len[i] = true_utt_len[i]
                        shuffle_labels[i] = labels[i]

                    history = shuffle_history
                    history_len = shuffle_history_len
                    true_utt = shuffle_true_utt
                    true_utt_len = shuffle_true_utt_len
                    labels = shuffle_labels

                    del shuffle_history, shuffle_history_len, shuffle_true_utt, shuffle_true_utt_len, shuffle_labels
                    gc.collect()

                    epoch += 1

# if __name__ == "__main__":
#     scn =SCN()
#     scn.model_file = '/data/ParlAI/mymodels/persona_select/model/persona_select_model_v1_2_shuffle'
#     scn.BuildModel()
#     # scn.TrainModel()
#
#     sess = scn.LoadModel()
#     with open(evaluate_file, 'rb') as f:
#         history, true_utt, labels = pickle.load(f)
#
#     scores = scn.MakePrediction(history, true_utt)
#     print(scores)



    # true_labels, pred_labels = scn.Evaluate(sess)
    # r = ['\t'.join([str(true_labels[i]), str(pred_labels[i])]) for i in range(len(true_labels))]
    # with open(evaluate_file+'_pred.txt', 'w') as f:
    #     f.writelines('\n'.join(r))
