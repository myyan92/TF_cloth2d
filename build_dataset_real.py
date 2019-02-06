from PIL import Image
import numpy as np
import tensorflow as tf
from dataset_io import data_writer
import os


#tfrecords_filename = 'cloth2d_test_real_11.tfrecords'
tfrecords_filename = 'cloth2d_test_real_ours_rect_2.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

#img_pattern = "/scr-ssd/mengyuan/gen_data/real_rope/run%02d/img_%04d.jpg"
#img_pattern = "/scr-ssd/mengyuan/gen_data/data_reproduce_real/%04d.png"
img_pattern = "/scr1/mengyuan/data/real_rope_ours_2/seq_m%d/image_%d.png"

for run in [9]: #range(1,9): #[9]
    i = 0
    while os.path.isfile(img_pattern%(run, i)):
        img = Image.open(img_pattern % (run, i))
#        img = img.crop((0,20,200,220))
#        img = img.crop((650,300,1350,1000))
        img = img.resize((224,224), resample=Image.LANCZOS)
        img = np.array(img)[:,:,0:3] # exclude A channel
        position = np.zeros((2,128))
        knots = np.zeros((2,4))

        record = data_writer(img, position, knots)
        writer.write(record.SerializeToString())
        i += 1

writer.close()
