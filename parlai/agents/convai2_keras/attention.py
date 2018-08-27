from keras.models import Sequential, Model
from keras.layers import Dense, GRU, LSTM
from keras.layers import Input, TimeDistributed, Embedding, RepeatVector, Lambda, Bidirectional, Dropout
from keras.layers import Flatten, Reshape, Permute, Activation
from keras.layers import Dot, Concatenate, Multiply
from keras.layers import merge
from keras.callbacks import EarlyStopping
from keras import backend as K


# def attention_3d_block(source, time_steps_s, query=None, time_steps_q=None, self_attention=True):
#     # source.shape = (batch_size, time_steps_s, input_dim)
#     # query.shape = (batch_size, time_steps_q, input_dim)
#     if self_attention:
#         query = source
#         time_steps_q = time_steps_s
#
#     input_dim = int(source.shape[2])
#     # time_steps_s = int(source.shape[1])
#     # time_steps_q = int(query.shape[1])
#     a = Permute((2, 1))(query)
#     a = Reshape((input_dim, time_steps_q))(a)  # this line is not useful. It's just to know which dimension is what.
#     a = Dense(time_steps_s, activation='softmax', kernel_initializer='glorot_normal')(a)
#     a_probs = Permute((2, 1))(a)
#     output_attention_mul = merge([source, a_probs], mode='mul')
#     return output_attention_mul

# attention block for seq2seq
# H: the inputs from encoder
# S: the inputs from decoder
# return the attention-based encode vectors
def attention_3d_block(H, S, H_max_len, S_max_len, return_alpha=False, drop_out=0.1):
    m = H_max_len
    n = S_max_len
    d1 = int(H.shape[2])
    d2 = int(S.shape[2])
    # concatenate all H = (h_i) to all S = (s_j) ===> H_S = [s_(j-1) h_i] (S lag one phase behind)
    # S = Input((n, d2,))
    # S_shift = Lambda(lambda x: K.concatenate([K.zeros_like(K.expand_dims(x[:, 0], 1)), x[:, :-1]], 1))(S)
    S_flat = Flatten()(S)
    S_flat_rep = RepeatVector(m)(S_flat)
    S_rep_n = Reshape((m, n, d2))(S_flat_rep)
    # (m, n, d2,)

    # H = Input((m, d1,))
    H_flat = Flatten()(H)
    H_flat_rep = RepeatVector(n)(H_flat)
    H_flat_rep_ = Reshape((n, m, d1))(H_flat_rep)
    H_rep_m = Permute((2, 1, 3))(H_flat_rep_)
    # (m, n, d1,)

    # concatenate everything with everything:
    S_H_ = Concatenate(-1)([S_rep_n, H_rep_m])
    # (m, n, d1+d2)
    S_H_flat = Flatten()(S_H_)
    S_H = Reshape((m*n, d1+d2))(S_H_flat)
    # (m*n, (d1+d2),)

    # make the e_ji
    E_T_1 = TimeDistributed(Dense(d1+d2, activation='tanh'))(S_H)
    E_T_1 = Dropout(drop_out)(E_T_1)
    E_T_ = TimeDistributed(Dense(1, activation='linear'))(E_T_1)
    E_T = Reshape((m, n))(E_T_)
    E = Permute((2, 1))(E_T)  # E = {E_j} = {{ e_{ji} }}
    # (n, m,)

    # the alignemtns
    alpha = TimeDistributed(Activation('softmax'))(E)  # alpha_j = softmax(E_j}
    # (n, m,)
    if return_alpha:
        return alpha

    C = Dot((2, 1))([alpha, H])
    # (n, d1,)

    # attention_model = Model([S, H], C)
    return C


def attention_3d_model(H_max_len, S_max_len, H_d, S_d, return_alpha=False):
    m = H_max_len
    n = S_max_len
    d1 = H_d
    d2 = S_d
    H = Input((H_max_len, H_d))
    S = Input((S_max_len, S_d))
    # concatenate all H = (h_i) to all S = (s_j) ===> H_S = [s_(j-1) h_i] (S lag one phase behind)
    # S = Input((n, d2,))
    # S_shift = Lambda(lambda x: K.concatenate([K.zeros_like(K.expand_dims(x[:, 0], 1)), x[:, :-1]], 1))(S)
    S_flat = Flatten()(S)
    S_flat_rep = RepeatVector(m)(S_flat)
    S_rep_n = Reshape((m, n, d2))(S_flat_rep)
    # (m, n, d2,)

    # H = Input((m, d1,))
    H_flat = Flatten()(H)
    H_flat_rep = RepeatVector(n)(H_flat)
    H_flat_rep_ = Reshape((n, m, d1))(H_flat_rep)
    H_rep_m = Permute((2, 1, 3))(H_flat_rep_)
    # (m, n, d1,)

    # concatenate everything with everything:
    S_H_ = Concatenate(-1)([S_rep_n, H_rep_m])
    # (m, n, d1+d2)
    S_H_flat = Flatten()(S_H_)
    S_H = Reshape((m*n, d1+d2))(S_H_flat)
    # (m*n, (d1+d2),)

    # make the e_ji
    E_T_1 = TimeDistributed(Dense(d1+d2, activation='tanh'))(S_H)
    E_T_ = TimeDistributed(Dense(1, activation='linear'))(E_T_1)
    E_T = Reshape((m, n))(E_T_)
    E = Permute((2, 1))(E_T)  # E = {E_j} = {{ e_{ji} }}
    # (n, m,)

    # the alignemtns
    alpha = TimeDistributed(Activation('softmax'))(E)  # alpha_j = softmax(E_j}
    # (n, m,)
    if return_alpha:
        return Model([H, S], alpha)

    C = Dot((2, 1))([alpha, H])
    # (n, d1,)

    # attention_model = Model([S, H], C)
    return Model([H, S], C)
