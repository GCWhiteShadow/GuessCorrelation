import tensorflow as tf
from glob import glob

from model import CorrelationPredictor


def _parse_function(example_proto):
    features = {"raw_img": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.float32, default_value=0.)}

    parsed_features = tf.parse_single_example(example_proto, features)

    img = tf.decode_raw(parsed_features['raw_img'], tf.uint8)
    img = tf.reshape(img, (1, 127, 127))
    img = tf.cast(img, tf.float32)
    img /= 255
    img -= 0.5

    return img, parsed_features["label"]


def get_soft_sign(signal):
    return tf.tanh(signal * 300)


def train():
    train_record_list = glob('data/records/[0-9][1-9].tfrecords')
    test_record_list = glob('data/records/[0-9]0.tfrecords')

    dataset = tf.data.TFRecordDataset(train_record_list)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(64)
    iterator = dataset.make_one_shot_iterator()
    input_batch, label = iterator.get_next()

    test_set = tf.data.TFRecordDataset(test_record_list)
    test_set = test_set.map(_parse_function)
    test_set = test_set.batch(64)
    test_itr = test_set.make_one_shot_iterator()
    test_input, test_label = test_itr.get_next()

    model = CorrelationPredictor()

    prediction = model.classify(input_batch, is_training=True)
    loss = tf.losses.mean_squared_error(label, prediction)

    test_output = model.classify(test_input, is_training=False)
    test_loss = tf.losses.mean_squared_error(test_label, test_output)

    lr = tf.train.exponential_decay(1e-5, tf.train.get_or_create_global_step(), 500, 0.75)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    train_op = tf.group([train_op, update_ops])

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for step in range(int(1e4)):
            _ = sess.run([train_op])

            if step % 200 == 0:
                loss_value = sess.run(test_loss)
                print('Step {:d}: {:.3f}'.format(step, loss_value))


if __name__ == '__main__':
    train()
