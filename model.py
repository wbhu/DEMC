#!/usr/bin/env python
"""
    File Name   :   DEMC-model
    date        :   14/7/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""
import tensorflow as tf


def feature_cnn(inputData, is_training=True, output_channels=1):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(inputData, 64, 3, padding='same', activation=tf.nn.relu,
                                  kernel_regularizer=regularizer)
    for layers in xrange(2, 3 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers,
                                      kernel_regularizer=regularizer)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block4'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same', activation=tf.nn.relu,
                                  kernel_regularizer=regularizer)
        # output = tf.pow(output, 1.0 / 2.2)
    return output


def dualEncoder(x, edge_map, is_training):
    conv_layers, skip_layers = encoder(x, name='RGB')
    edge_conv, edge_skip = encoder(edge_map, name='edge_map')
    # network = skip_connection_layer(conv_layers,edge_conv,str='encoder/connect_rgb_edge')
    network = tf.layers.conv2d(conv_layers, 512, kernel_size=3, padding='same', name='encoder/h6/conv')
    network = tf.layers.batch_normalization(network, training=is_training, name='encoder/h6/batch_norm')
    network = tf.nn.relu(network, name='encoder/h6/relu')
    # Decoder network
    network = decoder(network, skip_layers, edge_skip, is_training=is_training)
    return network


def encoder(inputData, name):
    # Convolutional layers size 1
    network = conv_layer(inputData, [3, 64], '%s/encoder/h1/conv_1' % name)
    beforepool1 = conv_layer(network, [64, 64], '%s/encoder/h1/conv_2' % name)
    network = pool_layer(beforepool1, '%s/encoder/h1/pool' % name)

    # Convolutional layers size 2
    network = conv_layer(network, [64, 128], '%s/encoder/h2/conv_1' % name)
    beforepool2 = conv_layer(network, [128, 128], '%s/encoder/h2/conv_2' % name)
    network = pool_layer(beforepool2, '%s/encoder/h2/pool' % name)

    # Convolutional layers size 3
    network = conv_layer(network, [128, 256], '%s/encoder/h3/conv_1' % name)
    network = conv_layer(network, [256, 256], '%s/encoder/h3/conv_2' % name)
    beforepool3 = conv_layer(network, [256, 256], '%s/encoder/h3/conv_3' % name)
    network = pool_layer(beforepool3, '%s/encoder/h3/pool' % name)

    # Convolutional layers size 4
    network = conv_layer(network, [256, 512], '%s/encoder/h4/conv_1' % name)
    network = conv_layer(network, [512, 512], '%s/encoder/h4/conv_2' % name)
    beforepool4 = conv_layer(network, [512, 512], '%s/encoder/h4/conv_3' % name)
    network = pool_layer(beforepool4, '%s/encoder/h4/pool' % name)

    # Convolutional layers size 5
    network = conv_layer(network, [512, 512], '%s/encoder/h5/conv_1' % name)
    network = conv_layer(network, [512, 512], '%s/encoder/h5/conv_2' % name)
    beforepool5 = conv_layer(network, [512, 512], '%s/encoder/h5/conv_3' % name)
    network = pool_layer(beforepool5, '%s/encoder/h5/pool' % name)

    return network, (inputData, beforepool1, beforepool2, beforepool3, beforepool4, beforepool5)


# Decoder network
def decoder(input_layer, skip_layers, edge_skip, is_training=False):
    sb, sx, sy, sf = input_layer.shape.as_list()
    alpha = 0.0

    # Upsampling 1
    network = deconv_layer(input_layer, sf, 'decoder/h1/decon2d', alpha, is_training)

    # Upsampling 2
    network = skip_connection_layer(network, tf.concat([skip_layers[5], edge_skip[5]], axis=3),
                                    'decoder/h2/fuse_skip_connection')
    network = deconv_layer(network, sf, 'decoder/h2/decon2d', alpha, is_training)

    # Upsampling 3
    network = skip_connection_layer(network, tf.concat([skip_layers[4], edge_skip[4]], axis=3),
                                    'decoder/h3/fuse_skip_connection')
    network = deconv_layer(network, sf / 2, 'decoder/h3/decon2d', alpha, is_training)

    # Upsampling 4
    network = skip_connection_layer(network, tf.concat([skip_layers[3], edge_skip[3]], axis=3),
                                    'decoder/h4/fuse_skip_connection')
    network = deconv_layer(network, sf / 4, 'decoder/h4/decon2d', alpha,
                           is_training)

    # Upsampling 5
    network = skip_connection_layer(network, tf.concat([skip_layers[2], edge_skip[2]], axis=3),
                                    'decoder/h5/fuse_skip_connection', )
    network = deconv_layer(network, sf / 8, 'decoder/h5/decon2d', alpha,
                           is_training)

    # Skip-connection at full size
    network = skip_connection_layer(network, tf.concat([skip_layers[1], edge_skip[1]], axis=3),
                                    'decoder/h6/fuse_skip_connection')
    # Final convolution
    network = tf.layers.conv2d(network, 3, kernel_size=3, padding='same', activation=tf.nn.relu,
                               name='decoder/h7/conv2d')
    return network


# === Layers ==================================================================

# Convolutional layer
def conv_layer(input_layer, sz, str):
    network = tf.layers.conv2d(input_layer, sz[1], 3, padding='same', activation=tf.nn.relu, name=str)
    return network


# Max-pooling layer
def pool_layer(input_layer, str):
    network = tf.layers.max_pooling2d(input_layer, pool_size=2, strides=2, padding='same', name=str)
    return network


# Concatenating fusion of skip-connections
def skip_connection_layer(input_layer, skip_layer, str):
    _, sx, sy, sf = input_layer.shape.as_list()
    _, sx_, sy_, sf_ = skip_layer.shape.as_list()

    assert (sx_, sy_) == (sx, sy)
    network = tf.concat([input_layer, skip_layer], axis=3, name='%s/skip_connection' % str)
    network = tf.layers.conv2d(network, filters=sf, kernel_size=1, padding='same', name=str)
    return network


# Deconvolution layer
def deconv_layer(input_layer, num_out_channels, str, alpha, is_training=True):
    scale = 2
    filter_size = (2 * scale - scale % 2)
    network = tf.layers.conv2d_transpose(input_layer, num_out_channels, kernel_size=filter_size, strides=scale,
                                         padding='same', name=str)
    network = tf.layers.batch_normalization(network, training=is_training, name='%s/batch_norm_dc' % str)
    network = tf.maximum(alpha * network, network, name='%s/leaky_relu_dc' % str)
    return network
