# encoding: utf-8

import time, os
import gensim
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils import save_embedding


class STNE(object):
    def __init__(self, hidden_dim, node_num, fea_dim, seq_len, depth=1, node_fea=None, node_fea_trainable=False):
        self.node_num, self.fea_dim, self.seq_len = node_num, fea_dim, seq_len

        self.input_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_seq')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        if node_fea is not None:
            assert self.node_num == node_fea.shape[0] #and self.fea_dim == node_fea.shape[1]
            self.embedding_W = tf.Variable(initial_value=node_fea, name='encoder_embed', trainable=node_fea_trainable)
        else:
            self.embedding_W = tf.Variable(initial_value=tf.random_uniform(shape=(self.node_num, self.fea_dim)),
                                           name='encoder_embed', trainable=node_fea_trainable)
        input_seq_embed = tf.nn.embedding_lookup(self.embedding_W, self.input_seqs, name='input_embed_lookup')
        # input_seq_embed = tf.layers.dense(input_seq_embed, units=1200, activation=None)

        # encoder
        encoder_cell_fw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)
        encoder_cell_bw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)
        if depth == 1:
            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0])
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0])
        else:
            encoder_cell_fw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)
            encoder_cell_bw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)

            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0] + [encoder_cell_fw_1] * (depth - 1))
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0] + [encoder_cell_bw_1] * (depth - 1))

        encoder_outputs, encoder_final = bi_rnn(encoder_cell_fw_all, encoder_cell_bw_all, inputs=input_seq_embed,
                                                dtype=tf.float32)
        c_fw_list, h_fw_list, c_bw_list, h_bw_list = [], [], [], []
        for d in range(depth):
            (c_fw, h_fw) = encoder_final[0][d]
            (c_bw, h_bw) = encoder_final[1][d]
            c_fw_list.append(c_fw)
            h_fw_list.append(h_fw)
            c_bw_list.append(c_bw)
            h_bw_list.append(h_bw)

        decoder_init_state = tf.concat(c_fw_list + c_bw_list, axis=-1), tf.concat(h_fw_list + h_bw_list, axis=-1)
        decoder_cell = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim * 2), output_keep_prob=1 - self.dropout)
        decoder_init_state = LSTMStateTuple(
            tf.layers.dense(decoder_init_state[0], units=hidden_dim * 2, activation=None),
            tf.layers.dense(decoder_init_state[1], units=hidden_dim * 2, activation=None))

        self.encoder_output = tf.concat(encoder_outputs, axis=-1)
        encoder_output_T = tf.transpose(self.encoder_output, [1, 0, 2])  # h

        new_state = decoder_init_state
        outputs_list = []
        for i in range(seq_len):
            new_output, new_state = decoder_cell(tf.zeros(shape=tf.shape(encoder_output_T)[1:]), new_state)  # None
            outputs_list.append(new_output)

        decoder_outputs = tf.stack(outputs_list, axis=0)  # seq_len * batch_size * hidden_dim
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])  # batch_size * seq_len * hidden_dim
        self.decoder_outputs = decoder_outputs
        output_preds = tf.layers.dense(decoder_outputs, units=self.node_num, activation=None)
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs, logits=output_preds)
        self.loss_ce = tf.reduce_mean(loss_ce, name='loss_ce')

        self.global_step = tf.Variable(1, name="global_step", trainable=False)


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)


class Classifier(object):
    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        # averages = ["micro", "macro", "samples", "weighted"]
        # f1_results = {}
        # pre_results = {}
        # rec_results = {}
        # acc_results = accuracy_score(Y, Y_)
        f1_macro = f1_score(Y, Y_, average="macro")
        f1_micro = f1_score(Y, Y_, average="micro")
        # for average in averages:
        #      f1_results[average] = f1_score(Y, Y_, average=average)
        #     pre_results[average] = precision_score(Y, Y_, average=average)
        #     rec_results[average] = recall_score(Y, Y_, average=average)
        # print 'Results, using embeddings of dimensionality', len(self.embeddings[X[0]])
        # print '-------------------'
        # print('\nF1 Score: ')
        # print(f1_results)
        # print('\nPrecision Score:')
        # print(pre_results)
        # print('\nRecall Score:')
        # print(rec_results)
        # print('Accuracy Score:', acc_results)

        # return f1_results, pre_results, rec_results, acc_results
        return f1_micro, f1_macro
        # print '-------------------'

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = np.random.get_state()

        training_size = int(train_precent * len(X))
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size + 1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')

        if len(vec) == 2:
            X.append(int(vec[0]))
            Y.append([int(v) for v in vec[1:]])
    fin.close()
    return X, Y


def read_node_features(filename):
    fea = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        fea.append(np.array([float(x) for x in vec[1:]]))
    fin.close()
    return np.array(fea, dtype='float32')


def read_node_sequences(filename):
    seq = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        seq.append(np.array([int(x) for x in vec]))
    fin.close()
    return np.array(seq)


def reduce_seq2seq_hidden_mean(seq, seq_h, node_num, seq_num, seq_len):
    node_dict = {}
    for i_seq in range(seq_num):
        for j_node in range(seq_len):
            nid = seq[i_seq, j_node]
            if nid in node_dict:
                node_dict[nid].append(seq_h[i_seq, j_node, :])
            else:
                node_dict[nid] = [seq_h[i_seq, j_node, :]]
    vectors = []
    for nid in range(node_num):
        vectors.append(np.average(np.array(node_dict[nid]), 0))
    return np.array(vectors)


def reduce_seq2seq_hidden_add(sum_dict, count_dict, seq, seq_h_batch, seq_len, batch_start):
    for i_seq in range(seq_h_batch.shape[0]):
        for j_node in range(seq_len):
            nid = seq[i_seq + batch_start, j_node]
            if nid in sum_dict:
                sum_dict[nid] = sum_dict[nid] + seq_h_batch[i_seq, j_node, :]
            else:
                sum_dict[nid] = seq_h_batch[i_seq, j_node, :]
            if nid in count_dict:
                count_dict[nid] = count_dict[nid] + 1
            else:
                count_dict[nid] = 1
    return sum_dict, count_dict


def reduce_seq2seq_hidden_avg(sum_dict, count_dict, node_num):
    vectors = []
    for nid in range(node_num):
        vectors.append(sum_dict[nid] / count_dict[nid])
    return np.array(vectors)


def node_classification(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)
    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def node_classification_d(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_dec.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_dec = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_dec.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)

    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def node_classification_hd(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    dec_sum_dict = {}
    node_cnt_enc = {}
    node_cnt_dec = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt_enc = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt_enc, sequences,
                                                               batch_enc.astype('float32'), seq_len, s_idx)

        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        dec_sum_dict, node_cnt_dec = reduce_seq2seq_hidden_add(dec_sum_dict, node_cnt_dec, sequences,
                                                               batch_dec.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt_enc = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt_enc, sequences,
                                                               batch_enc.astype('float32'), seq_len, s_idx)

        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        dec_sum_dict, node_cnt_dec = reduce_seq2seq_hidden_add(dec_sum_dict, node_cnt_dec, sequences,
                                                               batch_dec.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt_enc, node_num=node_n)
    node_dec_mean = reduce_seq2seq_hidden_avg(sum_dict=dec_sum_dict, count_dict=node_cnt_dec, node_num=node_n)

    node_mean = np.concatenate((node_enc_mean, node_dec_mean), axis=1)
    lr = Classifier(vectors=node_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def check_all_node_trained(trained_set, seq_list, total_node_num):
    for seq in seq_list:
        trained_set.update(seq)
    if len(trained_set) == total_node_num:
        return True
    else:
        return False


def get_embed(session, bs, seqne, sequences, seq_len, node_n):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)
    return node_enc_mean


if __name__ == '__main__':
    folder = '/home/hezhicheng/PycharmProjects/STNE/data/cora/'
    fn = '/home/hezhicheng/PycharmProjects/STNE/data/cora/result.txt'

    dpt = 1            # Depth of both the encoder and the decoder layers (MultiCell RNN)
    h_dim = 500        # Hidden dimension of encoder LSTMs
    s_len = 10         # Length of input node sequence
    epc = 2            # Number of training epochs
    trainable = False  # Node features trainable or not
    dropout = 0.2      # Dropout ration
    clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]  # Ration of training samples in subsequent classification
    b_s = 128          # Size of batches
    lr = 0.001         # Learning rate of RMSProp
    save_emb_file = '../data/cora/embed/stne_vec_nc.txt'
    save_emb = len(save_emb_file) > 0

    start = time.time()
    fobj = open(fn, 'w')
    X, Y = read_node_label(folder + 'labels.txt')
    node_fea = read_node_features(folder + 'cora.features')
    node_seq = read_node_sequences(folder + 'node_sequences_10_10.txt')

    embed_dim = node_fea.shape[1] # Embedding Dimension

    with tf.Session() as sess:
        model = STNE(hidden_dim=h_dim, node_fea_trainable=trainable, seq_len=s_len, depth=dpt, node_fea=node_fea,
                     node_num=node_fea.shape[0], fea_dim=embed_dim)
        train_op = tf.train.RMSPropOptimizer(lr).minimize(model.loss_ce, global_step=model.global_step)
        sess.run(tf.global_variables_initializer())

        trained_node_set = set()
        all_trained = False
        for epoch in range(epc):
            start_idx, end_idx = 0, b_s
            print('Epoch,\tStep,\tLoss,\t#Trained Nodes')
            while end_idx < len(node_seq):
                _, loss, step = sess.run([train_op, model.loss_ce, model.global_step], feed_dict={
                    model.input_seqs: node_seq[start_idx:end_idx], model.dropout: dropout})

                if not all_trained:
                    all_trained = check_all_node_trained(trained_node_set, node_seq[start_idx:end_idx],
                                                         node_fea.shape[0])

                if step % 10 == 0:
                    print(epoch, '\t', step, '\t', loss, '\t', len(trained_node_set))
                    if all_trained:
                        f1_mi = []
                        for ratio in clf_ratio:
                            f1_mi.append(node_classification(session=sess, bs=b_s, seqne=model, sequences=node_seq,
                                                             seq_len=s_len, node_n=node_fea.shape[0], samp_idx=X,
                                                             label=Y, ratio=ratio))

                        print('step ', step)
                        fobj.write('step ' + str(step) + ' ')
                        for f1 in f1_mi:
                            print(f1)
                            fobj.write(str(f1) + ' ')
                        fobj.write('\n')
                start_idx, end_idx = end_idx, end_idx + b_s

            if start_idx < len(node_seq):
                sess.run([train_op, model.loss_ce, model.global_step],
                         feed_dict={model.input_seqs: node_seq[start_idx:len(node_seq)], model.dropout: dropout})

            minute = np.around((time.time() - start) / 60)
            print('\nepoch ', epoch, ' finished in ', str(minute), ' minutes\n')

            f1_mi = []
            for ratio in clf_ratio:
                f1_mi.append(
                    node_classification(session=sess, bs=b_s, seqne=model, sequences=node_seq, seq_len=s_len,
                                        node_n=node_fea.shape[0], samp_idx=X, label=Y, ratio=ratio))

            fobj.write(str(epoch) + ' ')
            print('Classification results on current ')
            for f1 in f1_mi:
                print(f1)
                fobj.write(str(f1) + ' ')
            fobj.write('\n')
            minute = np.around((time.time() - start) / 60)

            fobj.write(str(minute) + ' minutes' + '\n')
            print('\nClassification finished in ', str(minute), ' minutes\n')

        fobj.close()
        minute = np.around((time.time() - start) / 60)
        print('Total time: ' + str(minute) + ' minutes')
        if save_emb:
            print(f'Saving embedding to {save_emb_file}')
            embedding = get_embed(session=sess, bs=b_s, seqne=model, sequences=node_seq, seq_len=s_len, node_n=node_fea.shape[0])
            learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(embed_dim)
            nodes = np.arange(embedding.shape[0])
            learned_embed.add([str(node) for node in nodes], embedding)
            save_embedding(learned_embed, save_emb_file, binary=(os.path.splitext(save_emb_file)[1]== 'bin'))

