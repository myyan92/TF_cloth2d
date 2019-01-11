import numpy as np
import tensorflow as tf
import gin, gin.tf
from losses_TF import node_l2loss
from TF_cloth2d.models.spatial_transformer import transformer
from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2
import pdb

@gin.configurable
class Model_STNv2_3d(Model_STNv2):

    def build(self, rgb=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
        """
        super().build(rgb)
        with tf.variable_scope(self.scope):

            # Create boxes where point[i] and point[i+1] from pred are centers of opposite sides.
# LOWRES            pred_previous = self.pred_layers[-1][:,:-1,:]  # remove the last segment
            pred_previous = self.pred_layers[-2] # LOWRES
            centers = (pred_previous[:,:-1,:] + pred_previous[:,1:,:])*0.5
            diffs = (pred_previous[:,1:,:] - pred_previous[:,:-1,:])*0.5
            box_features = tf.concat([centers, diffs], axis=-1)
            transforms = tf.layers.dense(box_features, 6,
                         kernel_initializer=tf.constant_initializer([0,0,-1,0,0,0,
                                                                     0,0,0,0,0,-1,
                                                                     1,0,0,0,1,0,
                                                                     0,-1,0,1,0,0]),
                         bias_initializer=tf.constant_initializer(0.0),
                         trainable=False, name='transforms_4')

# LOWRES            self.rois = transformer(self.conv1_2, transforms, (7,7)) # make it class variable to debug and visualize

            self.rois_raw = transformer(self.rgb, transforms, (13,13))
# LOWRES            fc_1 = self.dense(tf.layers.flatten(self.rois), 'fc1_1', 512, tf.nn.relu)
            fc_1 = self.dense(tf.layers.flatten(self.rois[-1]), 'fc1_1', 512, tf.nn.relu) # LOWRES
            fc_2 = self.dense(fc_1, 'fc1_2', 128, tf.nn.relu) # 64
            fc_3 = self.dense(fc_2, 'fc1_3', 3, None)

            self.pred_intersect_logit = tf.reshape(fc_3, [-1, transforms.shape[-2], 3])
            self.pred_intersect = tf.nn.softmax(self.pred_intersect_logit, axis=-1)

            saver_var_list = self.get_trainable_variables()
            if not self.use_vgg:
                vars = self.get_variables()
                vars = [v for v in vars if 'moving_mean' in v.name or 'moving_variance' in v.name]
                saver_var_list = saver_var_list + vars
            self.saver = tf.train.Saver(var_list=saver_var_list, max_to_keep=50)
            # summaries
            for var in saver_var_list:
                summary = tf.summary.histogram('summary'+var.name, var, family='weights')
            for activation in [self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2]:
                summary = tf.summary.histogram('summary'+activation.name, activation, family='activations')

    def predict_single(self, sess, input, training=False):
        pred, pred_intersect = sess.run([self.pred, self.pred_intersect],
                                        feed_dict={self.rgb:input[None], self.training:training})
        return pred[0], pred_intersect[0]

    def predict_batch(self, sess, inputs, training=False):
        pred, pred_intersect = sess.run([self.pred, self.pred_intersect],
                                        feed_dict={self.rgb:inputs, self.training:training})
        return pred, pred_intersect

    def setup_optimizer(self, GT_position=None, GT_intersect=None):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if GT_position is not None:
            self.gt_pred = GT_position
        else:
            self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=self.pred.shape)
        if GT_intersect is not None:
            self.gt_pred_intersect = GT_intersect
        else:
            self.gt_pred_intersect = tf.placeholder(name="gt_pred_intersect",
                                                    dtype=tf.float32,
                                                    shape=self.pred_intersect.shape)
        self.pred_layers_losses = []
        for pred in self.pred_layers:
            loss = node_l2loss(pred*6.0, self.gt_pred, resample_equdistance=True)
            self.pred_layers_losses.append(loss)
        self.pred_loss = self.pred_layers_losses[-1]
        self.reg_loss = tf.losses.get_regularization_loss()
        classification_weight = tf.reduce_sum(self.gt_pred_intersect*tf.constant([1.0, 0.01, 1.0]), axis=-1)
        self.classification_loss = tf.losses.softmax_cross_entropy(
                                       self.gt_pred_intersect,
                                       self.pred_intersect_logit,
                                       weights=classification_weight)

        self.loss = tf.add_n(self.pred_layers_losses)*0.25 + self.reg_loss*1.0 + self.classification_loss*500.0
        #self.loss = self.classification_loss
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=self.momentum, epsilon=0.01).minimize(self.loss)
        tf.summary.scalar('loss', self.pred_layers_losses[-1])
        tf.summary.scalar('loss_reg', self.reg_loss)
        tf.summary.scalar('loss_CE', self.classification_loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, gt_position, gt_intersect):
        _, loss = sess.run([self.optimizer, self.loss],
                           feed_dict={self.rgb:inputs,
                                      self.gt_pred:gt_position,
                                      self.gt_pred_intersect:gt_intersect,
                                      self.training:True})


if __name__ == "__main__":
    model = Model_STNv2_3d('vgg16_weights.npz', fc_sizes=[1024, 256], loss_type='l2', save_dir='./')
    input = tf.placeholder(dtype=tf.float32, shape=(None, 224,224,3))
    model.build(input)
    output = tf.placeholder(dtype=tf.float32, shape=(None, 64,2))
    model.setup_optimizer(output)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    pdb.set_trace()

