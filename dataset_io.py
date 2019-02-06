from PIL import Image
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floatList_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def data_writer(img, position, knots):
    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()

    record = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'position': _floatList_feature(position.flatten().tolist()),
        'knots': _floatList_feature(knots.flatten().tolist())}))
    return record

def data_parser(record):
    features = tf.parse_single_example(
      record,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'position': tf.FixedLenFeature([256], tf.float32),
        'knots': tf.FixedLenFeature([8], tf.float32)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([224, 224, 3])
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, dtype=tf.float32)
    position = tf.reshape(features['position'], tf.constant([2,128]))
    position = tf.transpose(position)
    knots = tf.reshape(features['knots'], tf.constant([2,4]))
    knots = tf.transpose(knots)
    # add random rotation of 90 degrees
    rotate = tf.random_uniform([]) > 0.5
#    rotate = tf.constant(False)
    image, position, knots = tf.cond(rotate,
        true_fn=lambda: (tf.image.rot90(image),
                         position[:,::-1]*tf.constant([1.0, -1.0]),
                         knots[:,::-1]*tf.constant([1.0, -1.0])),
        false_fn=lambda: (image, position, knots))
    return image, position, knots


