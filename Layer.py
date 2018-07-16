# ---------------
# Author: Tu, Tao
# Reference: TensorLayer (https://tensorlayer.readthedocs.io/en/latest/index.html)
# ---------------
import tensorflow as tf
from tensorflow.python.training import moving_averages


def dense(
        x,
        output_dim=100,
        act=tf.identity,
        W_init=tf.truncated_normal_initializer(stddev=0.1),
        b_init=tf.constant_initializer(value=0.0),
        with_all=False,
        dtype=tf.float32,
        name='Fully-Connected-Layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=(
            x.get_shape()[-1], output_dim), initializer=W_init, dtype=dtype)
        b = tf.get_variable(name='b', shape=(output_dim),
                            initializer=b_init, dtype=dtype)
        if b_init:
            outputs = act(tf.matmul(x, W) + b)
        else:
            outputs = act(tf.matmul(x, W))

    return outputs if not with_all else (outputs, W, b)


def conv2d(
        x,
        act=tf.identity,
        filter_shape=[5, 5, 1, 100],
        strides=[1, 1, 1, 1],
        padding='SAME',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        with_all=False,
        dtype=tf.float32,
        name='2D-Convolution-Layer'):
    # filter_shape: [height, width, in_channels, out_channels]
    with tf.variable_scope(name):
        W = tf.get_variable(name='W_conv2d', shape=filter_shape,
                            initializer=W_init, dtype=dtype)
        b = tf.get_variable(
            name='b_conv2d', shape=filter_shape[-1], initializer=b_init, dtype=dtype)
        if b_init:
            outputs = act(tf.nn.conv2d(
                x, W, strides=strides, padding=padding) + b)
        else:
            outputs = act(tf.nn.conv2d(x, W, strides=strides, padding=padding))

    return outputs if not with_all else (outputs, W, b)


def deconv2d(
        x,
        act=tf.identity,
        filter_shape=[3, 3, 128, 256],
        output_shape=[1, 256, 256, 128],
        strides=[1, 2, 2, 1],
        padding='SAME',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        with_all=False,
        dtype=tf.float32,
        name='2D-Deconvolution-Layer'):
    # filter_shape: [height, width, output_channels, in_channels]
    with tf.variable_scope(name):
        W = tf.get_variable(name='W_deconv2d', shape=filter_shape,
                            initializer=W_init, dtype=dtype)
        b = tf.get_variable(name='b_deconv2d',
                            shape=filter_shape[-2], initializer=b_init, dtype=dtype)
        if b_init:
            outputs = act(tf.nn.conv2d_transpose(
                x, W, output_shape=output_shape, strides=strides, padding=padding) + b)
        else:
            outputs = act(tf.nn.conv2d_transpose(
                x, W, output_shape=output_shape, strides=strides, padding=padding))

    return outputs if not with_all else (outputs, W, b)


def batch_norm(
        x,
        decay=0.9,
        eps=1e-5,
        act=tf.identity,
        is_train=False,
        beta_init=tf.zeros_initializer(),
        gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
        dtype=tf.float32,
        name='Batch-Norm-Layer'):
    with tf.variable_scope(name):
        x_shape = x.get_shape()
        axis = list(range(len(x_shape) - 1))
        beta = tf.get_variable('beta', shape=x_shape[-1],
                               initializer=beta_init,
                               dtype=dtype,
                               trainable=is_train)
        gamma = tf.get_variable('gamma', shape=x_shape[-1],
                                initializer=gamma_init,
                                dtype=dtype,
                                trainable=is_train)
        moving_mean = tf.get_variable('moving_mean',
                                      x_shape[-1],
                                      initializer=tf.zeros_initializer(),
                                      dtype=dtype,
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance',
                                          x_shape[-1],
                                          initializer=tf.constant_initializer(
                                              1.),
                                          dtype=dtype,
                                          trainable=False)
        # Be preformed only when training
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean,
                                                                   decay,
                                                                   zero_debias=False)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance,
                                                                       variance,
                                                                       decay,
                                                                       zero_debias=False)

        def mean_var_with_update():
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)
        if is_train:
            mean, var = mean_var_with_update()
            outputs = act(tf.nn.batch_normalization(
                x, mean, var, beta, gamma, eps))
        else:
            outputs = act(tf.nn.batch_normalization(
                x, moving_mean, moving_variance, beta, gamma, eps))
    return outputs


def flatten(x, name='Flatten-Layer'):
    x_shape = x.get_shape().as_list()[1:]
    flat_dim = 1
    for dim in x_shape:
        flat_dim *= dim
    outputs = tf.reshape(x, shape=[-1, flat_dim], name=name)
    return outputs


def retrieve_seq_length(x, pad_val=0):
    return tf.reduce_sum(tf.cast(tf.not_equal(x, pad_val), dtype=tf.int32), 1)


def cosine_similarity(v1, v2):
    return tf.reduce_sum(tf.multiply(v1, v2), 1) / (tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1)) * tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1)))


