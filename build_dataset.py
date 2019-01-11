from PIL import Image
import numpy as np
import tensorflow as tf
from dataset_io import data_writer

def write_record(image_file, position_file, knot_file, writer):
    img = Image.open(image_file)
    img = img.resize((224,224), resample=Image.LANCZOS)
    img = np.array(img)[:,:,0:3] # exclude A channel
#    for x in range(224):
#        for y in range(224):
#            if np.sum(img[x,y])>720:
#                img[x,y,:]=np.array([100,200,100])
    position = np.loadtxt(position_file)
    position = position.transpose()
    if knot_file is not None:
        knots = np.loadtxt(knot_file)
        knots = knots.transpose()
    else:
        knots = np.zeros((2,4))

    record = data_writer(img, position, knots)
    writer.write(record.SerializeToString())


tfrecords_filename = 'cloth2d_%s_temp.tfrecords'
img_pattern = "/scr-ssd/mengyuan/gen_data/data_with_knots/%04d.png"
position_pattern = "/scr-ssd/mengyuan/gen_data/data_with_knots/%04d.txt"
knots_pattern = "/scr-ssd/mengyuan/gen_data/data_with_knots/%04d_knots.txt"

train_writer = tf.python_io.TFRecordWriter(tfrecords_filename % ('train'))
for i in range(9000):
    write_record(img_pattern % (i),
                 position_pattern % (i),
                 knots_pattern % (i) if knots_pattern else None,
                 train_writer)
train_writer.close()

test_writer = tf.python_io.TFRecordWriter(tfrecords_filename % ('test'))
for i in range(9000, 10000):
    write_record(img_pattern % (i),
                 position_pattern % (i),
                 knots_pattern % (i) if knots_pattern else None,
                 test_writer)
test_writer.close()

