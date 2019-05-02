import tensorflow as tf


class CorrelationPredictor:
    def classify(self, input_batch, is_training):
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            # Input shape: N, 1, 127, 127
            net = tf.layers.conv2d(input_batch, filters=5, kernel_size=3, strides=2, activation=tf.nn.relu,
                                   trainable=is_training, data_format='channels_first')
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, data_format='channels_first')

            net = tf.layers.conv2d(net, filters=10, kernel_size=3, activation=tf.nn.relu, trainable=is_training,
                                   data_format='channels_first')
            net = tf.layers.batch_normalization(net, training=is_training)

            net = tf.layers.conv2d(net, filters=20, kernel_size=3, activation=tf.nn.relu, trainable=is_training,
                                   data_format='channels_first')
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, data_format='channels_first')

            net = tf.layers.flatten(net)
            net = tf.layers.dense(net, 450, activation=tf.nn.relu, trainable=is_training)

            net = tf.layers.dense(net, 1, activation=tf.nn.tanh, trainable=is_training)
            net = tf.squeeze(net)

            return net
