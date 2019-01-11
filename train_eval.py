import tensorflow as tf
from TF_cloth2d.models.model_VGG import Model
from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2
from TF_cloth2d.models.model_REINFORCE import make_RL_model
from dataset_io import data_parser
import losses
import numpy as np
from PIL import Image
import gin, argparse, os
import pdb

@gin.configurable
class Trainner():
    def __init__(self,
                 train_dataset,
                 eval_dataset,
                 model, # A model instance constructed by gin config
                 pred_target, # knot or node
                 use_RL,
                 loss,
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
        val_data = val_data.batch(batch_size)
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                                   tr_data.output_shapes)
        self.next_image, self.next_position, self.next_knot = iterator.get_next()
        if pred_target == "node":
            assert(self.next_position.shape[1] % model.num_points == 0)
            skip = self.next_position.shape[1] // model.num_points
            self.next_position = self.next_position[:, ::skip, :]
            self.label = self.next_position
        else:
            if 'l2' in loss:
                assert(self.next_knot.shape[1] == model.num_points)
            self.label = self.next_knot
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
        self.loss_fn = getattr(losses, loss, None)
        if self.loss_fn is None:
            self.model.setup_optimizer(self.label)
        else:
            self.model.setup_optimizer()
        self.use_RL = use_RL
        self.pred_target = pred_target
        self.global_step = 0
        self.train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'tfboard'), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.snapshot = snapshot
        if snapshot is not None:
            self.model.load(self.sess, snapshot)
        config_str = gin.operative_config_str()
        with open(os.path.join(save_dir, '0.gin'), 'w') as f:
            f.write(config_str)

    def train_epoch(self):
        self.sess.run(self.training_init_op)
        losses = []
        while True:
            try:
                if self.loss_fn is None:
                    summary, _, loss = self.sess.run(
                        [self.model.merged_summary, self.model.optimizer, self.model.loss])
                elif not self.use_RL:
                    image, GT_node, GT_knot, pred = self.sess.run(
                        [self.next_image, self.next_position, self.next_knot, self.model.pred])
                    if self.pred_target == 'node':
                        pred_knot, pred_node = None, pred
                    else:
                        pred_knot, pred_node = pred, None
                    loss, gradient = self.loss_fn(pred_node, pred_knot, GT_node, GT_knot)
                    self.sess.run(self.model.optimizer, feed_dict={self.model.rgb:image,
                                                               self.model.pred_grad:gradient})
                    summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)])
                else:
                    image, GT_node, GT_knot = self.sess.run(
                        [self.next_image, self.next_position, self.next_knot])
                    pred, pred_value = self.model.predict_batch(self.sess, image, sampling=True)
                    if self.pred_target == 'node':
                        pred_knot, pred_node = None, pred
                    else:
                        pred_knot, pred_node = pred, None
                    if self.pred_target == 'knot':
                        cost = self.loss_fn(pred_node, pred_knot, GT_node, GT_knot)
                        cost_adv = (cost - np.mean(cost)) / np.std(cost) # pred_value.flatten()
                    else: # pointwise loss is used when predicting nodes
                        cost_per_node, cost_sum = self.loss_fn(pred_node, pred_knot, GT_node, GT_knot)
                        cost_adv = (cost_per_node - np.mean(cost_per_node, axis=0)) / np.std(cost_per_node, axis=0)
                        cost_adv = np.clip(cost_adv, -3, 3)
                        cost = cost_sum
                    summary, loss, ent_loss = self.model.fit(self.sess, image, pred, cost_adv, cost)

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
                if self.loss_fn is None:
                    gt, loss = self.sess.run(
                        [self.label, self.model.loss])
                    count = gt.shape[0]*gt.shape[1]
                else:
                    GT_node, GT_knot, pred = self.sess.run(
                        [self.next_position, self.next_knot, self.model.pred])
                    if self.pred_target == 'node':
                        pred_knot, pred_node = None, pred
                    else:
                        pred_knot, pred_node = pred, None
                    if self.use_RL:
                        if self.pred_target == 'knot':
                            loss = self.loss_fn(pred_node, pred_knot, GT_node, GT_knot)
                        else:
                            _, loss = self.loss_fn(pred_node, pred_knot, GT_node, GT_knot)
                    else:
                        loss, _ = self.loss_fn(pred_node, pred_knot, GT_node, GT_knot)
                    loss = np.sum(loss)  # Loss is sometimes scalar and sometimes arrays.
                    if "l2" in self.loss_fn.__name__:
                        count = pred.shape[0]*pred.shape[1]
                    else:
                        count = GT_node.shape[0]*GT_node.shape[1]

                total_loss += loss
                total_count += count
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

