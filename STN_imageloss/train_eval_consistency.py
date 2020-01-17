import tensorflow as tf
from TF_cloth2d.models.model_VGG_STN_2_imageloss_EM_consistency import Model_IM_EM_v2
from TF_cloth2d.STN_imageloss.dataset_io_consistency import data_parser
from TF_cloth2d.sample_spline_TF import sample_equdistance
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
        tr_data = tr_data.batch(batch_size//2)
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser)
        val_data = val_data.batch(batch_size//2)
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                                   tr_data.output_shapes)
        self.next_image_first, self.next_image_second = iterator.get_next()
        self.next_image_input = tf.concat([self.next_image_first, self.next_image_second], axis=0)
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
        self.model.build(self.next_image_input)
        self.model.setup_optimizer()
        self.global_step = 0
        self.train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'tboard'), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.model.load(self.sess, snapshot)
        self.loss_threshold=-4.39  #-4.75 # -8.55 # curriculum
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
                image, pred, image_loss_2 = self.sess.run([self.next_image_input, self.model.pred, self.model.image_loss_e])
                idx = image_loss_2 < self.loss_threshold
                num_train_instance += np.sum(idx)
                batch_size = image.shape[0]
                if np.sum(idx)>0:
                    image_loss_train_image = image[idx,:,:,:]
                    summary, _, image_loss, reg_loss, centroid_loss = self.sess.run([self.model.merged_summary, self.model.optimizer,
                                                                 self.model.image_loss, self.model.reg_loss, self.model.reg_loss_l2],
                                                                 feed_dict={self.model.rgb:image_loss_train_image})
                    self.train_writer.add_summary(summary, self.global_step)
                    self.global_step += 1
                    image_losses.append(image_loss)
                    reg_losses.append(reg_loss)
                    centroid_losses.append(centroid_loss)
                    consistency_loss_train_image = []
                    consistency_loss_train_label = []
                    for i in range(batch_size):
                        if idx[i] and not idx[(i+batch_size//2)%batch_size]:
                            consistency_loss_train_image.append(image[(i+batch_size//2)%batch_size,:,:,:])
                            consistency_loss_train_label.append(pred[i,:,:])
                    if len(consistency_loss_train_label) > 0:
                        consistency_loss_train_image = np.array(consistency_loss_train_image)
                        consistency_loss_train_label = np.array(consistency_loss_train_label)
                        consistency_loss_train_label, _ = sample_equdistance(consistency_loss_train_label, np.zeros((64,64)), 64)
                        consistency_loss_train_label = np.transpose(consistency_loss_train_label, (0,2,1))
                        #pdb.set_trace()
                        summary, _, image_loss, reg_loss, centroid_loss = self.sess.run([self.model.merged_summary, self.model.consistency_optimizer,
                                                                                         self.model.image_loss, self.model.reg_loss, self.model.reg_loss_l2],
                                                                                         feed_dict={self.model.rgb:consistency_loss_train_image,
                                                                                                    self.model.pred_target:consistency_loss_train_label})
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
        total_image_loss = 0
        total_count = 0
        while True:
            try:
                pred, image_loss = self.sess.run([self.model.pred, self.model.image_loss])
                total_loss += 0
                total_image_loss += image_loss*pred.shape[0]*pred.shape[1] # hack to work with count
                total_loss_sorted += 0
                total_count += pred.shape[0]*pred.shape[1]
            except tf.errors.OutOfRangeError:
                break
        print("eval average image loss / node L2 loss / sorted L2 loss: %f %f %f" % (
                total_image_loss/total_count, total_loss/total_count, total_loss_sorted/total_count))

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
    if args.action == "train":
        trainner.train()
    else:
        trainner.eval()
