from PIL import Image
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floatList_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def data_writer(img1, img2):
    height = img1.shape[0]
    width = img1.shape[1]
    img1_raw = img1.tostring()
    img2_raw = img2.tostring()
    record = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image1_raw': _bytes_feature(img1_raw),
        'image2_raw': _bytes_feature(img2_raw),
        }))
    return record

def data_parser(record):
    features = tf.parse_single_example(
      record,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image1_raw': tf.FixedLenFeature([], tf.string),
        'image2_raw': tf.FixedLenFeature([], tf.string)
        })

    image1 = tf.decode_raw(features['image1_raw'], tf.uint8)
    image2 = tf.decode_raw(features['image2_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([224, 224, 3])
    image1 = tf.reshape(image1, image_shape)
    image1 = tf.cast(image1, dtype=tf.float32)
    image2 = tf.reshape(image2, image_shape)
    image2 = tf.cast(image2, dtype=tf.float32)
    return image1, image2


