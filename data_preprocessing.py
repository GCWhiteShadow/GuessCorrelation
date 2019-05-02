import numpy as np
import os
import csv
from PIL import Image
import tensorflow as tf


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


data_dir = 'data/'
os.makedirs(data_dir+'records/', exist_ok=True)

csv_file = open('train_responses.csv', 'r')
csv_reader = csv.reader(csv_file, )
next(csv_reader)

data_counter = 0
data_list = []

tf_record_counter = 0

for data_row in csv_reader:
    data_name = data_row[0]
    data_cov = float(data_row[1])

    img = Image.open(data_dir + data_name + '.png')
    img = np.array(img)
    img = img[21:-2, 21:-2]

    raw_img = img.tostring()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'label': _floats_feature(data_cov),
        'raw_img': _bytes_feature(raw_img)}))

    data_list.append(tf_example)

    data_counter += 1

    if data_counter == 1500:
        data_counter = 0

        writer = tf.python_io.TFRecordWriter(data_dir + 'records/' + '{:02d}.tfrecords'.format(tf_record_counter))
        tf_record_counter += 1

        for example_data in data_list:
            writer.write(example_data.SerializeToString())

        data_list = []
