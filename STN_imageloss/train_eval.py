import tensorflow as tf
from TF_cloth2d.models.model_VGG_STN_imageloss_EM import Model_IM_EM
from TF_cloth2d.models.model_VGG_STN_2_imageloss_EM import Model_IM_EM_v2
from TF_cloth2d.dataset_io import data_parser
from TF_cloth2d.sort_nodes import sort_nodes
import numpy as np
from PIL import Image
import argparse, gin, os
import pdb
import matplotlib.pyplot as plt

@gin.configurable
class Trainner():
    def __init__(self, train_dataset, eval_dataset,
                 model, num_epoch, batch_size, save_dir, snapshot):

        # create TensorFlow Dataset objects
        tr_data = tf.data.TFRecordDataset(train_dataset)
        tr_data = tr_data.map(data_parser)
        tr_data = tr_data.shuffle(buffer_size=5000)
        tr_data = tr_data.batch(batch_size)
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser)
        val_data = val_data.batch(batch_size)
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                                   tr_data.output_shapes)
        self.next_image, self.next_position, _ = iterator.get_next()
        self.next_position = self.next_position[:,::2,:]
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
        #self.model = Model_IM_EM_v2('/scr-ssd/mengyuan/TF_cloth2d/models/vgg16_weights.npz',
        #                         fc_sizes=[1024, 256], learning_rate=0.001,
        #                         loss_type='imageEM', save_dir='./',
        #                         train_scale=True, train_rotation=True)
        self.model.build(self.next_image)
        self.model.setup_optimizer()
        self.global_step = 0
        self.train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'tboard'), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.model.load(self.sess, snapshot)
        self.loss_threshold=-8.55 # curriculum
        self.num_train_instance=0 # curriculum
        config_str = gin.operative_config_str()
        with open(os.path.join(save_dir, '0.gin'), 'w') as f:
            f.write(config_str)

    def train_epoch(self):
        self.sess.run(self.training_init_op)
        num_train_instance = 0
        image_losses = []
        reg_losses = []
        centroid_losses = []
        while True:
            try:
                image, image_loss_2 = self.sess.run([self.next_image, self.model.image_loss_e])
                idx = image_loss_2 < self.loss_threshold
                num_train_instance += np.sum(idx)
                image = image[idx,:,:,:]
                if np.sum(idx)>0:
                    summary, _, image_loss, reg_loss, centroid_loss = self.sess.run([self.model.merged_summary, self.model.optimizer,
                                                                 self.model.image_loss, self.model.reg_loss, self.model.reg_loss_l2],
                                                                 feed_dict={self.model.rgb:image})
                    self.train_writer.add_summary(summary, self.global_step)
                    self.global_step += 1
                    image_losses.append(image_loss)
                    reg_losses.append(reg_loss)
                    centroid_losses.append(centroid_loss)
            except tf.errors.OutOfRangeError:
                break
        if self.num_train_instance >= num_train_instance:
            self.loss_threshold *= 0.98
        self.num_train_instance = num_train_instance
        print("train batch image / stiffness / centroid loss this epoch: %f %f %f"
              %(np.mean(image_losses), np.mean(reg_losses), np.mean(centroid_losses)))
        print("trained on %d instances"%(num_train_instance))

    def eval(self):
        self.sess.run(self.validation_init_op)
        total_loss = 0
        total_loss_sorted = 0
        total_count = 0
        while True:
            try:
                gt, pred = self.sess.run([self.next_position, self.model.pred])
                loss = np.sum(np.square(gt-pred))
                total_loss += loss
                pred_s = [sort_nodes(p) for p in pred]
                loss_s = np.sum(np.square(gt-np.array(pred_s)))
                total_loss_sorted += loss_s
                total_count += gt.shape[0]*gt.shape[1]
            except tf.errors.OutOfRangeError:
                break
        print("eval average node L2 loss / sorted L2 loss: %f %f" % (total_loss/total_count, total_loss_sorted/total_count))

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

#    train_dataset = '/scr-ssd/mengyuan/TF_cloth2d/cloth2d_train_sim_seq.tfrecords'
#    test_dataset = '/scr-ssd/mengyuan/TF_cloth2d/cloth2d_test_sim_seq.tfrecords'
#    snapshot = '../b-spline_data_pred_node_STN_2/model-28'

    trainner = Trainner()
    trainner.train()

