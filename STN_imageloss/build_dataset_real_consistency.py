from PIL import Image
import numpy as np
import tensorflow as tf
from dataset_io_consistency import data_writer
import os


#tfrecords_filename = 'cloth2d_test_real_11.tfrecords'
#tfrecords_filename = 'cloth2d_test_real_ours_rect_2.tfrecords'
#tfrecords_filename = 'cloth2d_train_real_ours_with_occlusion.tfrecords'
tfrecords_filename = 'cloth2d_train_real_ours_with_occlusion_consistency_pairs.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

#img_pattern = "/scr-ssd/mengyuan/gen_data/real_rope/run%02d/img_%04d.jpg"
#img_pattern = "/scr-ssd/mengyuan/gen_data/data_reproduce_real/%04d.png"
#img_pattern = "/scr1/mengyuan/data/real_rope_ours_2/seq_m%d/image_%d.png"
img_pattern = "/scr1/mengyuan/data/real_rope_with_occlusion-new/run_%d/%02d.png"

for run in range(1,27): #range(1,9): #[9]
    i = 0
    prev_img = None
    while os.path.isfile(img_pattern%(run, i)):
        img = Image.open(img_pattern % (run, i))
        img = img.resize((224,224), resample=Image.LANCZOS)
        img = np.array(img)[:,:,0:3] # exclude A channel
        if prev_img is not None:
            record = data_writer(img, prev_img)
            writer.write(record.SerializeToString())
        prev_img = img
        i += 1

writer.close()
