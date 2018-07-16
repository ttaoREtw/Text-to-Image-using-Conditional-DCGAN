# ---------------
# Author: Tu, Tao
# Reference: https://github.com/zsdonghao/text-to-image/blob/master/train_txt2im.py
# ---------------
from Layer import cosine_similarity as cos_loss
import tensorflow as tf
import os
import logging
import time
import random
import numpy as np
from models_simple import imageEncoder, textEncoder, generator as G, discriminator as D
from util import sent2IdList, save_images

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)

# ---- Args ----
# G: z, txt, img_height, img_width, img_depth=3, gf_dim=128, is_train=True, reuse=tf.AUTO_REUSE
# D: x, txt, img_height, img_width, img_depth=3, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE
# textEncoder: txt, vocab_size, with_matrix=False, reuse=tf.AUTO_REUSE, pad_token=0,
#              bidirectional=False, word_dim=256, sent_dim=128
# imageEncoder: x, out_dim=128, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE
# --------------


DIR_RECORD = './tfrecords/'
hps_list = {
    'lr': 2e-4,
    'beta1': 0.5,
    'beta2': 0.9,
    'clip_norm': 1e-1,
    'n_critic': 1,
    'imH': 64,
    'imW': 64,
    'imD': 3,
    'max_seq_len': 20,
    'z_dim': 100,
    's_dim': 128,
    'w_dim': 256,
    'gf_dim': 128,
    'df_dim': 64,
    'voc_size': 6375,
    'pad_token': 6372
}


def hparas(hps_list):
    class Hparas(object):
        pass
    hps = Hparas()
    for hp in hps_list:
        setattr(hps, hp, hps_list[hp])
    return hps


class ModelWrapper(object):
    """ For convenience. """

    def __init__(self, sess, hps, train_files, batch_size, use_bn=True, is_train=True):
        self.hps = hps
        self.sess = sess
        self.saver = None
        self.files_tr = train_files
        self.batch_size = batch_size
        self.use_bn = use_bn
        self.is_train = is_train

    def _build_dataset(self):
        def crop(img):
            img = tf.decode_raw(img, tf.uint8)
            img = tf.reshape(img, [self.hps.imH, self.hps.imW, self.hps.imD])
            img = tf.cast(img, tf.float32)
            img = tf.div(img, 255.0)
            #img = tf.image.resize_images(img, size=[self.hps.imH+15, self.hps.imW+15])
            #img = tf.random_crop(img, [self.hps.imH, self.hps.imW, self.hps.imD])
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            return img

        def parser(record):
            features = {
                "image": tf.FixedLenFeature([], dtype=tf.string),
                "caption": tf.FixedLenFeature([20], dtype=tf.int64),
                "image_wrong": tf.FixedLenFeature([], dtype=tf.string),
                "caption_wrong": tf.FixedLenFeature([20], dtype=tf.int64)}

            # features contains - 'img', 'caption'
            ex = tf.parse_single_example(record, features=features)

            img = crop(ex['image'])
            img_w = crop(ex['image_wrong'])
            caption = ex['caption']
            caption_w = ex['caption_wrong']

            return img, caption, img_w, caption_w

        dataset = tf.data.TFRecordDataset(self.files_tr)
        dataset = dataset.map(parser, num_parallel_calls=8).repeat().shuffle(
            buffer_size=10 * self.batch_size).batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        item = iterator.get_next()
        self.sess.run(iterator.initializer)
        return item

    def build(self):
        hps = self.hps
        with tf.variable_scope('inputs'):
            if self.is_train:
                self.img, self.cap, self.img_w, self.cap_w = self._build_dataset()
            else:
                self.cap = tf.placeholder(
                    tf.int64, shape=[None, hps.max_seq_len], name='caption_inference')
            self.z = tf.placeholder(
                tf.float32, shape=[None, hps.z_dim], name='noise')

        with tf.variable_scope('models'):
            # x: image embedding
            # v: text embedding
            if self.is_train:
                #self.x = imageEncoder(self.img, out_dim=hps.s_dim, df_dim=hps.df_dim, is_train=True)
                #self.x_w = imageEncoder(self.img_w, out_dim=hps.s_dim, df_dim=hps.df_dim, is_train=True, reuse=True)
                self.v = textEncoder(self.cap, hps.voc_size, seq_len=hps.max_seq_len, pad_token=hps.pad_token, attention=True,
                                     bidirectional=True, word_dim=hps.w_dim, sent_dim=hps.s_dim)
                self.v_w = textEncoder(self.cap_w, hps.voc_size, seq_len=hps.max_seq_len, attention=True,
                                       bidirectional=True, pad_token=hps.pad_token, word_dim=hps.w_dim,
                                       sent_dim=hps.s_dim, reuse=True)
                # interpolation
                self.v_i = 0.5 * self.v + 0.5 * self.v_w

                self.img_fake = G(self.z, self.v, img_height=hps.imH, img_width=hps.imW,
                                  img_depth=hps.imD, gf_dim=hps.gf_dim, is_train=self.use_bn)
                self.img_fake_i = G(self.z, self.v_i, img_height=hps.imH, img_width=hps.imW,
                                    img_depth=hps.imD, gf_dim=hps.gf_dim, is_train=self.use_bn)

                # real data
                self.d_real, self.logits_real = D(self.img, self.v, img_height=hps.imH, img_width=hps.imW,
                                                  img_depth=hps.imD, df_dim=hps.df_dim, is_train=self.use_bn)
                # fake data from generator
                _, self.logits_fake = D(self.img_fake, self.v, img_height=hps.imH, img_width=hps.imW,
                                        img_depth=hps.imD, df_dim=hps.df_dim, is_train=self.use_bn, reuse=True)
                _, self.logits_fake_i = D(self.img_fake_i, self.v, img_height=hps.imH, img_width=hps.imW,
                                          img_depth=hps.imD, df_dim=hps.df_dim, is_train=self.use_bn, reuse=True)
                # mismatched data
                _, self.logits_mis_v = D(self.img, self.v_w, img_height=hps.imH, img_width=hps.imW,
                                         img_depth=hps.imD, df_dim=hps.df_dim, is_train=self.use_bn, reuse=True)
                _, self.logits_mis_im = D(self.img_w, self.v, img_height=hps.imH, img_width=hps.imW,
                                          img_depth=hps.imD, df_dim=hps.df_dim, is_train=self.use_bn, reuse=True)
            else:
                self.v = textEncoder(self.cap, hps.voc_size, seq_len=hps.max_seq_len, pad_token=hps.pad_token,
                                     attention=True, bidirectional=True, word_dim=hps.w_dim, sent_dim=hps.s_dim)
                self.img_fake = G(self.z, self.v, img_height=hps.imH, img_width=hps.imW,
                                  img_depth=hps.imD, gf_dim=hps.gf_dim, is_train=self.use_bn)

        with tf.variable_scope('losses'):
            if self.is_train:
                # encoder loss
                # alpha = 0.2 # margin alpha
                # self.enr_loss = tf.reduce_mean(alpha - cos_loss(self.x, self.v) + cos_loss(self.x, self.v_w)) + \
                #    tf.reduce_mean(alpha - cos_loss(self.x, self.v) + cos_loss(self.x_w, self.v))
                # loss of generator
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_fake, labels=tf.ones_like(self.logits_fake), name='d_loss_fake'))
                self.g_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_fake_i, labels=tf.ones_like(self.logits_fake_i), name='d_loss_fake_i'))
                # loss of discriminator
                self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_real, labels=tf.ones_like(self.logits_real), name='d_loss_real'))
                self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_fake, labels=tf.zeros_like(self.logits_fake), name='d_loss_fake'))
                self.d_loss_mis_v = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_mis_v, labels=tf.zeros_like(self.logits_mis_v), name='d_loss_mismatch_text'))
                self.d_loss_mis_im = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_mis_im, labels=tf.zeros_like(self.logits_mis_im), name='d_loss_mismatch_image'))
                self.d_loss = self.d_loss_real + 0.5 * \
                    (self.d_loss_fake + self.d_loss_mis_v + self.d_loss_mis_im)

        # the power of name scope
        # self.cnn_vars = [var for var in tf.get_collection(
        # tf.GraphKeys.GLOBAL_VARIABLES, scope='wrapper/models/ImageEncoder')]
        self.rnn_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='wrapper/models/TextEncoder')]
        self.g_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='wrapper/models/Generator')]
        self.d_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='wrapper/models/Discriminator')]

        with tf.variable_scope('optimizers'):
            if self.is_train:
                self.lr_op = tf.Variable(hps.lr, trainable=False)
                # optimizer for textEncoder
                self.enr_opt = tf.train.AdamOptimizer(
                    self.lr_op, beta1=hps.beta1, beta2=hps.beta2)
                # def clipIfNotNone(grad, norm):
                #    if grad is None:
                #        return grad
                #    return tf.clip_by_norm(grad, norm)
                grads_and_vars = self.enr_opt.compute_gradients(
                    self.g_loss + self.d_loss, self.rnn_vars)
                clipped_grads_and_vars = [
                    (tf.clip_by_norm(gv[0], hps.clip_norm), gv[1]) for gv in grads_and_vars]
                # apply gradient and variables to optimizer
                self.enr_opt = self.enr_opt.apply_gradients(
                    clipped_grads_and_vars)

                # optimizer for generator
                self.g_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps.beta1,
                                                    beta2=hps.beta2).minimize(self.g_loss, var_list=self.g_vars)
                # optimizer for discriminator
                self.d_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps.beta1,
                                                    beta2=hps.beta2).minimize(self.d_loss, var_list=self.d_vars)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(
            var_list=self.rnn_vars + self.g_vars + self.d_vars, max_to_keep=20)

    def train(self, num_train_example, ep, ckpt_dir='./ckpt_model', log=True, load_idx=None):
        hps = self.hps
        num_batch_per_epoch = num_train_example // self.batch_size
        sample_size = self.batch_size

        # train
        if load_idx:
            self.restore(ckpt_dir, idx=load_idx)
        else:
            self.restore(ckpt_dir)
        for step in range(num_batch_per_epoch):
            # get noise (normal(mean=0, stddev=1.0))
            b_z = np.random.normal(loc=0.0, scale=1.0, size=(
                sample_size, hps.z_dim)).astype(np.float32)
            step_time = time.time()
            # update D
            errD, _ = self.sess.run(
                [self.d_loss, self.d_opt],
                feed_dict={self.z: b_z})
            if step % hps.n_critic == 0:
                errG, _ = self.sess.run(
                    [self.g_loss, self.g_opt], feed_dict={self.z: b_z})
            # update textEncoder
            self.sess.run(self.enr_opt, feed_dict={self.z: b_z})
            if log and step % hps.n_critic == 0:
                logging.info(
                    'Epoch: %2d [%4d/%4d] time: %4.4fs, g_loss: %2.6f, d_loss: %2.6f'
                    % (ep, step, num_batch_per_epoch, time.time() - step_time, errG, errD))
        self.save(ckpt_dir=ckpt_dir, idx=ep)

    def _test(self, epoch, save_path='result/'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        hps = self.hps
        sample_size = self.batch_size
        sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(
            8 * sample_size, hps.z_dim)).astype(np.float32)
        sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * sample_size + \
            ["this flower has petals that are yellow, white and purple and has dark lines"] * sample_size + \
            ["the petals on this flower are white with a yellow center"] * sample_size + \
            ["this flower has a lot of small round pink petals."] * sample_size + \
            ["this flower is orange in color, and has petals that are ruffled and rounded."] * sample_size + \
            ["the flower has yellow petals and the center of it is brown."] * sample_size + \
            ["this flower has petals that are blue and white."] * sample_size + \
            ["these white flowers have petals that start off white in color and end in a white towards the tips."] * sample_size
        for i, sent in enumerate(sample_sentence):
            sample_sentence[i] = sent2IdList(sent, hps.max_seq_len)
        img_gen = self.sess.run(self.img_fake, feed_dict={
            self.cap: sample_sentence,
            self.z: sample_seed})
        save_images(img_gen, [8, sample_size],
                    save_path + 'train_%d.png' % epoch)
        logging.info('Done test')
        return img_gen

    def save(self, ckpt_dir='ckpt/', idx=0):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.saver.save(self.sess, os.path.join(
            ckpt_dir, 'model-%d.ckpt' % idx))

    def restore(self, ckpt_dir='ckpt/', idx=None):
        if idx:
            self.saver.restore(self.sess, os.path.join(
                ckpt_dir, 'model-%d.ckpt' % idx))
            return True
        else:
            latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
            if latest_ckpt:
                self.saver.restore(self.sess, latest_ckpt)
                return True
        return False


if __name__ == '__main__':
    tf.reset_default_graph()
    filenames = [DIR_RECORD + 'train_data-%d.tfrecords' % i for i in range(20)]
    hps = hparas(hps_list)
    epoch = 1000
    gpu_ratio = 0.4
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.variable_scope('wrapper'):
        model_tr = ModelWrapper(sess, hps, filenames,
                                batch_size=256, is_train=True)
        model_tr.build()

    with tf.variable_scope('wrapper', reuse=True):
        model_vis = ModelWrapper(
            sess, hps, None, batch_size=10, is_train=False)
        model_vis.build()

    for ep in range(epoch):
        model_tr.train(num_train_example=70504, ep=ep,
                       ckpt_dir='ckpt_model_attentionLarge_biRnn_simple')
        model_vis._test(epoch=ep, save_path='result_simpleLarge/')
    sess.close()
