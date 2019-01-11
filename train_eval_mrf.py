import tensorflow as tf
from TF_cloth2d.models.model_VGG import Model
from TF_cloth2d.models.model_MRF import Model_MRF
from dataset_io import data_parser
import losses
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gin, argparse, os
import pdb

@gin.configurable
class Trainner():
    def __init__(self,
                 train_dataset,
                 eval_dataset,
                 model, # A model instance constructed by gin config
                 num_epoch,
                 batch_size,
                 save_dir,
                 snapshot=None):

        # create TensorFlow Dataset objects
        tr_data = tf.data.TFRecordDataset(train_dataset)
        tr_data = tr_data.map(data_parser)
        tr_data = tr_data.shuffle(buffer_size=5000)
        tr_data = tr_data.batch(batch_size)
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser)
        val_data = val_data.batch(1) # batch_size
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                                   tr_data.output_shapes)
        self.next_image, self.next_position, self.next_knot = iterator.get_next()
        # create two initialization ops to switch between the datasets
        self.training_init_op = iterator.make_initializer(tr_data)
        self.validation_init_op = iterator.make_initializer(val_data)

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.num_epoch = num_epoch
        self.model = model
        self.model.training=True  # hack
        self.model.build(rgb=self.next_image)
        self.model.setup_optimizer()
        self.global_step = 0
        self.train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'tfboard'), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.snapshot = snapshot
        config_str = gin.operative_config_str()
        with open(os.path.join(save_dir, '0.gin'), 'w') as f:
            f.write(config_str)

    def train_epoch(self):
        self.sess.run(self.training_init_op)
        losses = []
        while True:
            try:
                summary, _, loss, cov = self.sess.run(
                    [self.model.merged_summary, self.model.optimizer, self.model.loss, self.model.cov_negative])
                self.train_writer.add_summary(summary, self.global_step)
                self.global_step += 1
                losses.append(loss)
            except tf.errors.OutOfRangeError:
                break
        print("train batch loss this epoch: %f" %(np.mean(losses)))

    def eval(self):
        self.sess.run(self.validation_init_op)
        total_loss = 0
        total_count = 0
        while True:
            try:
                pred, gt, loss = self.sess.run([self.model.segmentation, self.next_image, self.model.loss])
                classification = self.sess.run(self.model.conv3_1, feed_dict={self.model.rgb:gt})
                count = pred.shape[0]
                total_loss += loss
                total_count += count
                fig, ax = plt.subplots(2,1)
                ax[0].imshow(pred[0,:,:,0], vmin=0.0, vmax=1.0)
                ax[1].imshow(gt[0,:,:,:]/256.0)
                plt.show()
                pdb.set_trace()
            except tf.errors.OutOfRangeError:
                break
        print("eval average node L2 loss: %f" % (total_loss/total_count))

    def train(self):
        for i in range(self.num_epoch):
            self.train_epoch()
            self.eval()
            self.model.save(self.sess, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=["train", "eval"], help="running train or eval.")
    parser.add_argument('--gin_config', default='', help="path to gin config file.")
    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)
    trainner = Trainner()
    if args.action == "train":
        trainner.train()
    else:
        trainner.model.load(trainner.sess, trainner.snapshot)
        trainner.eval()

