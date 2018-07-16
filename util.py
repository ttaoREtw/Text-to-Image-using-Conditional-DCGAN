import re
import string
import numpy as np
import scipy

DIR_DICT = './dictionary/'

enc_map = dict(np.load(DIR_DICT + 'word2Id.npy'))
dec_map = dict(np.load(DIR_DICT + 'id2Word.npy'))


def sent2IdList(line, word2Id_dict, MAX_SEQ_LENGTH=20):
    word2Id_dict = enc_map
    MAX_SEQ_LIMIT = MAX_SEQ_LENGTH
    padding = 0
    prep_line = re.sub('[%s]' % re.escape(
        string.punctuation), ' ', line.rstrip())
    prep_line = prep_line.replace('-', ' ')
    prep_line = prep_line.replace('-', ' ')
    prep_line = prep_line.replace('  ', ' ')
    prep_line = prep_line.replace('.', '')
    tokens = prep_line.split(' ')
    tokens = [
        tokens[i] for i in range(len(tokens))
        if tokens[i] != ' ' and tokens[i] != ''
    ]
    l = len(tokens)
    padding = MAX_SEQ_LIMIT - l
    for i in range(padding):
        tokens.append('<PAD>')
    line = [
        word2Id_dict[tokens[k]]
        if tokens[k] in word2Id_dict else word2Id_dict['<RARE>']
        for k in range(len(tokens))
    ]

    return line

# Save images


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def save_images(images, size, image_path):
    return imsave(images, size, image_path)
