from PIL import Image
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floatList_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def data_writer(img, position):
    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()

    record = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'position': _floatList_feature(position.flatten().tolist())
        }))
    return record

def data_parser(record):
    features = tf.parse_single_example(
      record,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'position': tf.FixedLenFeature([384], tf.float32)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([224, 224, 3])
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, dtype=tf.float32)
    position = tf.reshape(features['position'], tf.constant([3,128]))
    position = tf.transpose(position)
    return image, position


