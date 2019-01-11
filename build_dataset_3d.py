from PIL import Image
import numpy as np
import tensorflow as tf
from dataset_io_3d import data_writer

def write_record(image_file, position_file, intersect_file, writer):
    img = Image.open(image_file)
    img = img.resize((224,224), resample=Image.LANCZOS)
    img = np.array(img)[:,:,0:3] # exclude A channel
#    for x in range(224):
#        for y in range(224):
#            if np.sum(img[x,y])>720:
#                img[x,y,:]=np.array([100,200,100])
    position = np.loadtxt(position_file)
    position = position[:,:2].transpose()
    intersect = np.loadtxt(intersect_file)
    intersect = intersect.flatten().astype(np.int64)
    if position[0,0] > position[0,-1]:
        position = position[:,::-1]
        intersect = intersect[::-1]
    record = data_writer(img, position, intersect)
    writer.write(record.SerializeToString())


tfrecords_filename = 'cloth2d_%s_3d_depth2.tfrecords'
img_pattern = "/scr-ssd/mengyuan/gen_data/data_intersection_2/%04d_depth.png"
position_pattern = "/scr-ssd/mengyuan/gen_data/data_intersection_2/%04d.txt"
intersect_pattern = "/scr-ssd/mengyuan/gen_data/data_intersection_2/%04d_intersect.txt"

train_writer = tf.python_io.TFRecordWriter(tfrecords_filename % ('train'))
for i in range(9000):
    write_record(img_pattern % (i),
                 position_pattern % (i),
                 intersect_pattern % (i),
                 train_writer)
train_writer.close()

test_writer = tf.python_io.TFRecordWriter(tfrecords_filename % ('test'))
for i in range(9000, 10000):
    write_record(img_pattern % (i),
                 position_pattern % (i),
                 intersect_pattern % (i),
                 test_writer)
test_writer.close()

