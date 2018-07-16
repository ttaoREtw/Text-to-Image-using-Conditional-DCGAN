# ---------------
# Author: Tu, Tao
# Reference: https://github.com/zsdonghao/text-to-image/blob/master/model.py
# ---------------

import Layer
import tensorflow as tf

""" 
This file contains following models 
  - imageEncoder:  to encode images into features
  - textEncoder:   to encode texts into features
  - generator:     to generate images given noises and condition texts
  - discriminator: to discriminate real and fake images given condition texts

"""


def imageEncoder(x, out_dim=128, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    with tf.variable_scope('ImageEncoder', reuse=reuse):
        h0 = Layer.conv2d(x, act=tf.nn.leaky_relu, filter_shape=[4, 4, x.get_shape()[-1], df_dim], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h0/conv2d')
        # 1
        h1 = Layer.conv2d(h0, act=tf.identity, filter_shape=[4, 4, df_dim, df_dim * 2], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h1/conv2d')
        h1 = Layer.batch_norm(h1, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h1/batch_norm')
        # 2
        h2 = Layer.conv2d(h1, act=tf.identity, filter_shape=[4, 4, df_dim * 2, df_dim * 4], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h2/conv2d')
        h2 = Layer.batch_norm(h2, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h2/batch_norm')
        # 3
        h3 = Layer.conv2d(h2, act=tf.identity, filter_shape=[4, 4, df_dim * 4, df_dim * 8], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h3/conv2d')
        h3 = Layer.batch_norm(h3, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h3/batch_norm')
        h3_flat = Layer.flatten(h3, name='h3/flatten')
        # 4
        h4 = Layer.dense(h3_flat, output_dim=out_dim,
                         W_init=w_init, b_init=None, name='h4/dense')
    return h4


def textEncoder(txt, vocab_size, with_matrix=False, reuse=tf.AUTO_REUSE, pad_token=0, bidirectional=False, word_dim=256, sent_dim=128):
    with tf.variable_scope('TextEncoder', reuse=reuse):
        if with_matrix:
            w_embed_seq, w_matrix = _word_embedding(
                txt, word_dim, with_matrix=True)
        else:
            w_embed_seq = _word_embedding(
                txt, word_dim, vocab_size, with_matrix=False)
        w_seq_len = Layer.retrieve_seq_length(txt, pad_val=pad_token)
        s_embed = _sent_embedding(
            w_embed_seq, sent_dim, w_seq_len, bidirectional)
        # according to the phi function of paper "Generative Adversarial Text
        # to Image Synthesis"
        code = Layer.dense(
            s_embed[:, -1, :], output_dim=sent_dim, act=tf.nn.leaky_relu, name='code')
    return code if not with_matrix else (code, w_matrix)


def _word_embedding(txt, w_dim, vocab_size, with_matrix):
    with tf.variable_scope('word_embedding'):
        embed_matrix = tf.get_variable('word_embed_matrix',
                                       shape=(vocab_size, w_dim),
                                       initializer=tf.random_normal_initializer(
                                           stddev=0.02),
                                       dtype=tf.float32)
        w_embeds = tf.nn.embedding_lookup(embed_matrix, txt)
    return w_embeds if not with_matrix else (w_embeds, embed_matrix)


def _sent_embedding(w_embed_seq, s_dim, w_seq_len, bidirectional=False):
    with tf.variable_scope('sent_embedding') as scope:
        batch_size = tf.shape(w_embed_seq)[0]
        if bidirectional:
            cell_fw = tf.contrib.rnn.BasicLSTMCell(s_dim // 2)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(s_dim // 2)
            init_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            init_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
            outputs_tuple, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=w_embed_seq,
                sequence_length=w_seq_len,
                initial_state_fw=init_state_fw,
                initial_state_bw=init_state_bw,
                dtype=tf.float32,
                time_major=False,
                scope=scope)
            output_fw, output_bw = outputs_tuple
            outputs = tf.concat([output_fw, output_bw], axis=2)
        else:
            cell_fw = tf.contrib.rnn.BasicLSTMCell(s_dim)
            init_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(
                cell=cell_fw,
                inputs=w_embed_seq,
                sequence_length=w_seq_len,
                initial_state=init_state_fw,
                dtype=np.float32,
                time_major=False,
                scope=scope)
    return outputs


def generator(z, txt, img_height, img_width, img_depth=3, gf_dim=128, is_train=True, reuse=tf.AUTO_REUSE):
    """ Generate image given a code [z, txt]. """
    H, W, D = img_height, img_width, img_depth
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    H2, H4, H8, H16 = int(H / 2), int(H / 4), int(H / 8), int(H / 16)
    W2, W4, W8, W16 = int(W / 2), int(W / 4), int(W / 8), int(W / 16)
    with tf.variable_scope('Generator', reuse=reuse):
        code = tf.concat([z, txt], axis=1, name='code')
        h0 = Layer.dense(code, gf_dim * 8 * H16 * W16, act=tf.identity,
                         W_init=w_init, b_init=None, name='h0/dense')
        h0 = Layer.batch_norm(h0, is_train=is_train, gamma_init=gamma_init,
                              name='h0/batch_norm')
        h0 = tf.reshape(
            h0, shape=[-1, H16, W16, gf_dim * 8], name='h0/reshape')
        # 1
        h1 = Layer.conv2d(h0, act=tf.identity, filter_shape=[1, 1, gf_dim * 8, gf_dim * 2],
                          strides=[1, 1, 1, 1], padding='VALID', W_init=w_init, b_init=None, name='h1/conv2d')
        h1 = Layer.batch_norm(h1, act=tf.nn.relu, is_train=is_train,
                              gamma_init=gamma_init, name='h1/batch_norm')
        # 2
        h2 = Layer.conv2d(h1, act=tf.identity, filter_shape=[3, 3, gf_dim * 2, gf_dim * 2],
                          strides=[1, 1, 1, 1], padding='SAME', W_init=w_init, b_init=None, name='h2/conv2d')
        h2 = Layer.batch_norm(h2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='h2/batch_norm')
        # 3
        h3 = Layer.conv2d(h2, act=tf.identity, filter_shape=[3, 3, gf_dim * 2, gf_dim * 8],
                          strides=[1, 1, 1, 1], padding='SAME', W_init=w_init, b_init=None, name='h3/conv2d')
        h3 = Layer.batch_norm(h3, is_train=is_train,
                              gamma_init=gamma_init, name='h3/batch_norm')
        h3_res = tf.add(h3, h0, name='h3/residue_add')
        h3_out = tf.nn.relu(h3_res)
        # 4
        h4 = Layer.deconv2d(h3_out, act=tf.identity, filter_shape=[4, 4, gf_dim * 4, gf_dim * 8],
                            output_shape=[tf.shape(h3_out)[0], H8, W8, gf_dim * 4], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h4/deconv2d')
        h4 = Layer.batch_norm(h4, act=tf.identity, is_train=is_train,
                              gamma_init=gamma_init, name='h4/batch_norm')
        # 5
        h5 = Layer.conv2d(h4, act=tf.identity, filter_shape=[1, 1, gf_dim * 4, gf_dim],
                          strides=[1, 1, 1, 1], padding='VALID', W_init=w_init, b_init=None, name='h5/conv2d')
        h5 = Layer.batch_norm(h5, act=tf.nn.relu, is_train=is_train,
                              gamma_init=gamma_init, name='h5/batch_norm')
        # 6
        h6 = Layer.conv2d(h5, act=tf.identity, filter_shape=[3, 3, gf_dim, gf_dim],
                          strides=[1, 1, 1, 1], padding='SAME', W_init=w_init, b_init=None, name='h6/conv2d')
        h6 = Layer.batch_norm(h6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='h6/batch_norm')
        # 7
        h7 = Layer.conv2d(h6, act=tf.identity, filter_shape=[3, 3, gf_dim, gf_dim * 4],
                          strides=[1, 1, 1, 1], padding='SAME', W_init=w_init, b_init=None, name='h7/conv2d')
        h7 = Layer.batch_norm(h7, is_train=is_train,
                              gamma_init=gamma_init, name='h7/batch_norm')
        h7_res = tf.add(h7, h4, name='h7/residue_add')
        h7_out = tf.nn.relu(h7_res)
        # 8
        h8 = Layer.deconv2d(h7_out, act=tf.identity, filter_shape=[4, 4, gf_dim * 2, gf_dim * 4],
                            output_shape=[tf.shape(h7_out)[0], H4, W4, gf_dim * 2], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h8/deconv2d')
        h8 = Layer.batch_norm(h8, act=tf.nn.relu, is_train=is_train,
                              gamma_init=gamma_init, name='h8/batch_norm')
        # 9
        h9 = Layer.deconv2d(h8, act=tf.identity, filter_shape=[4, 4, gf_dim, gf_dim * 2],
                            output_shape=[tf.shape(h8)[0], H2, W2, gf_dim], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h9/deconv2d')
        h9 = Layer.batch_norm(h9, act=tf.nn.relu, is_train=is_train,
                              gamma_init=gamma_init, name='h9/batch_norm')
        # 10
        h10 = Layer.deconv2d(h9, act=tf.identity, filter_shape=[4, 4, D, gf_dim],
                             output_shape=[tf.shape(h9)[0], H, W, D], strides=[1, 2, 2, 1], padding='SAME',
                             W_init=w_init, name='h10/deconv2d')
        logits = h10
        outputs = tf.nn.tanh(logits)
    return outputs, logits


def discriminator(x, txt, img_height, img_width, img_depth=3, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE):
    """ Determine if an image x condition on txt is real or fake. """
    H, W, D = img_height, img_width, img_depth
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    # H2, H4, H8, H16 = int(H / 2), int(H / 4), int(H / 8), int(H / 16)
    # W2, W4, W8, W16 = int(W / 2), int(W / 4), int(W / 8), int(W / 16)
    with tf.variable_scope('Discriminator', reuse=reuse):
        h0 = Layer.conv2d(x, act=tf.nn.leaky_relu, filter_shape=[4, 4, D, df_dim], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h0/conv2d')
        # 1
        h1 = Layer.conv2d(h0, act=tf.identity, filter_shape=[4, 4, df_dim, df_dim * 2], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h1/conv2d')
        h1 = Layer.batch_norm(h1, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h1/batch_norm')
        # 2
        h2 = Layer.conv2d(h1, act=tf.identity, filter_shape=[4, 4, df_dim * 2, df_dim * 4], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h2/conv2d')
        h2 = Layer.batch_norm(h2, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h2/batch_norm')
        # 3
        h3 = Layer.conv2d(h2, act=tf.identity, filter_shape=[4, 4, df_dim * 4, df_dim * 8], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h3/conv2d')
        h3 = Layer.batch_norm(h3, act=tf.identity, is_train=is_train,
                              gamma_init=gamma_init, name='h3/batch_norm')
        # 4
        h4 = Layer.conv2d(h3, act=tf.identity, filter_shape=[1, 1, df_dim * 8, df_dim * 2], strides=[1, 1, 1, 1],
                          padding='VALID', W_init=w_init, b_init=None, name='h4/conv2d')
        h4 = Layer.batch_norm(h4, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h4/batch_norm')
        # 5
        h5 = Layer.conv2d(h4, act=tf.identity, filter_shape=[3, 3, df_dim * 2, df_dim * 2], strides=[1, 1, 1, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h5/conv2d')
        h5 = Layer.batch_norm(h5, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h5/batch_norm')
        # 6
        h6 = Layer.conv2d(h5, act=tf.identity, filter_shape=[3, 3, df_dim * 2, df_dim * 8], strides=[1, 1, 1, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h6/conv2d')
        h6 = Layer.batch_norm(h6, is_train=is_train,
                              gamma_init=gamma_init, name='h6/batch_norm')
        # residual
        h6_res = tf.add(h6, h3, name='h6/residue_add')
        h6_out = tf.nn.leaky_relu(h6_res)
        # txt: [batch_size, s_dim]
        # h6_out: [batch_size, _, _, df_dim*8]
        txt_expand = tf.expand_dims(txt, axis=1, name='txt_expand_1')
        txt_expand = tf.expand_dims(txt_expand, axis=1, name='txt_expand_2')
        txt_expand = tf.tile(txt_expand, multiples=[
                             1, h6_out.get_shape()[1], h6_out.get_shape()[2], 1], name='txt_tile')
        h_txt = tf.concat([h6_out, txt_expand], axis=3)
        # 7
        h7 = Layer.conv2d(h_txt, act=tf.identity, filter_shape=[1, 1, h_txt.get_shape()[-1], df_dim * 8],
                          strides=[1, 1, 1, 1], padding='VALID', W_init=w_init, b_init=None, name='h7/conv2d')
        h7 = Layer.batch_norm(h7, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h7/batch_norm')
        h_flat = Layer.flatten(h7, name='h7/flat')
        logits = Layer.dense(h_flat, output_dim=1)
        outputs = tf.nn.sigmoid(logits)
    return outputs, logits
