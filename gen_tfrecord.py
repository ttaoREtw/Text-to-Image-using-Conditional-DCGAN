# ---------------
# Author: Tu, Tao
# ---------------
import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import _pickle as pkl

DIR_DICT = './dictionary/'
DIR_DATA = './dataset/'
NUM_FILE = 20


def _int64_feature(value, pass_list=False):
    if not pass_list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(X, name):
    imgs = X['image']
    caps = X['caption']
    imgs_w = X['image_wrong']
    caps_w = X['caption_wrong']

    filename = os.path.join(DIR_DATA, name + '.tfrecords')
    with tf.python_io.TFRecordWriter(filename) as writer:
        for idx in range(len(X)):
            image_raw = imgs[idx].tostring()
            image_raw_w = imgs_w[idx].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(tf.compat.as_bytes(image_raw)),
                        'caption': _int64_feature(caps[idx], pass_list=True),
                        'image_wrong': _bytes_feature(tf.compat.as_bytes(image_raw_w)),
                        'caption_wrong': _int64_feature(caps_w[idx], pass_list=True)
                    }))
            writer.write(example.SerializeToString())


def make_examples(df):
    examples = []
    cnt = 0
    for _, item in df.iterrows():
        st_time = time.time()
        caps = item['Captions']
        im_path = item['ImagePath']
        im = Image.open(DIR_DATA + im_path).resize((64, 64))
        # dtype: uint8
        im = np.asarray(im)
        for cap in caps:
            cap = [int(x) for x in cap]
            im_w = Image.open(
                DIR_DATA + df['ImagePath'].iloc[np.random.randint(len(df))]).resize((64, 64))
            im_w = np.asarray(im_w)
            cap_w_candidates = df['Captions'].iloc[np.random.randint(len(df))]
            cap_w = cap_w_candidates[np.random.randint(len(cap_w_candidates))]
            cap_w = [int(x) for x in cap_w]
            examples.append([im, cap, im_w, cap_w])
            cnt += 1
            if cnt % 1000 == 0:
                print('finished: %d, time: %4.4fs' %
                      (cnt, time.time() - st_time))
    return examples


if __name__ == '__main__':
    df = pd.read_pickle(DIR_DATA + 'text2ImgData.pkl')
    ex_per_file = len(df) // NUM_FILE

    # could use threads
    for i in range(NUM_FILE):
        examples = make_examples(df[i*ex_per_file : (i+1)*ex_per_file])
        X = pd.DataFrame(examples, columns=[
                         'image', 'caption', 'image_wrong', 'caption_wrong'])
        st_time = time.time()
        convert_to(X, 'train_data-%d' % i)
        print('Create %s, time: %4.4fs' %
              (DIR_DATA + 'train_data-%d.tfrecords' % i, time.time() - st_time))
