from PIL import Image
import numpy as np
import tensorflow as tf
from dataset_io import data_writer

def write_record(image_file, position_file, knot_file, writer):
    img = Image.open(image_file)
    img = img.resize((224,224), resample=Image.LANCZOS)
    img = np.array(img)[:,:,0:3] # exclude A channel
    position = np.loadtxt(position_file)
    position = position.transpose()
    if knot_file is not None:
        knots = np.loadtxt(knot_file)
        knots = knots.transpose()
    else:
        knots = np.zeros((2,4))

    record = data_writer(img, position, knots)
    writer.write(record.SerializeToString())


tfrecords_filename = 'cloth2d_%s_sim_seq_small.tfrecords'
img_pattern = "/scr1/mengyuan/data/sim_sequence_bending1000/%04d/%04d_%d.png"
position_pattern = "/scr1/mengyuan/data/sim_sequence_bending1000/%04d/%04d_%d.txt"

"""parameters
for big train: run in range(950), t in range(100).
for medium train: run in range(95), t in range(100).
for small train: run in range(19), t in range(0,100,2).
for test: run in range(950,1000), t in range(100).
"""

train_writer = tf.python_io.TFRecordWriter(tfrecords_filename % ('train'))
images = []
positions= []
for run in range(19):
    for t in range(0,100,2):
        images.append(img_pattern % (run, run, t))
        positions.append(position_pattern % (run, run, t))
idx = np.arange(len(images))
np.random.shuffle(idx)

for i in idx:
    write_record(images[i],
                 positions[i],
                 None,
                 train_writer)
train_writer.close()


test_writer = tf.python_io.TFRecordWriter(tfrecords_filename % ('test'))
images = []
positions= []
for run in range(950, 1000):
    for t in range(100):
        images.append(img_pattern % (run, run, t))
        positions.append(position_pattern % (run, run, t))
idx = np.arange(len(images))
np.random.shuffle(idx)
for i in idx:
    write_record(images[i],
                 positions[i],
                 None,
                 test_writer)
test_writer.close()

