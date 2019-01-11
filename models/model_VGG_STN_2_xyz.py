import numpy as np
import tensorflow as tf
import gin, gin.tf
from losses_TF import node_l2loss
from TF_cloth2d.models.spatial_transformer import transformer
from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2
import pdb

@gin.configurable
class Model_STNv2_xyz(Model_STNv2):

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

            num_points = int(transforms.shape[-2])*3
            pred_next = tf.reshape(fc_3, [-1,num_points,1])

            # A hacky way to average endpoints from neighboring boxes.
            pred_gather_indexes = []
            pred_gather_indexes.append([0,0])
            source_index = 1
            while source_index+2 < num_points:
                pred_gather_indexes.append([source_index, source_index])
                pred_gather_indexes.append([source_index+1, source_index+2])
                source_index += 3
            pred_gather_indexes.append([source_index, source_index])
            pred_gather_indexes.append([source_index+1, source_index+1])
            pred_gather = tf.gather(pred_next, pred_gather_indexes, axis=1)
            pred_next_averaged = tf.reduce_mean(pred_gather, axis=2)
            self.pred_z = pred_next_averaged[:,:-1,:]

            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=50)

    def predict_single(self, sess, input):
        pred, pred_z = sess.run([self.pred, self.pred_z],
                                        feed_dict={self.rgb:input[None]})
        return pred[0], pred_z[0]

    def predict_batch(self, sess, inputs):
        pred, pred_z = sess.run([self.pred, self.pred_z],
                                        feed_dict={self.rgb:inputs})
        return pred, pred_z

    def setup_optimizer(self, GT_position=None):
        if GT_position is not None:
            self.gt_pred = GT_position
        else:
            placeholder_shape = tf.stack([self.pred.shape[0], self.pred.shape[1], 3])
            self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=placeholder_shape)

        self.gt_pred_xy = self.gt_pred[:,:,0:2]
        self.gt_pred_z = self.gt_pred[:,:,2:]
        #self.pred_layers_losses = []
        #for pred in self.pred_layers:
        #    loss = node_l2loss(pred*6.0, self.gt_pred, resample_equdistance=True)
        #    self.pred_layers_losses.append(loss)
        self.pred_loss = tf.nn.l2_loss(self.pred-self.gt_pred_xy)
        self.pred_z_loss = tf.nn.l2_loss(self.pred_z-self.gt_pred_z)
        self.reg_loss = tf.losses.get_regularization_loss()

        self.loss = self.pred_loss + self.pred_z_loss*100 + self.reg_loss*5.0
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=self.momentum, epsilon=0.01).minimize(self.loss)
        tf.summary.scalar('loss', self.pred_loss+self.pred_z_loss)
        tf.summary.scalar('loss_reg', self.reg_loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, gt_position):
        _, loss = sess.run([self.optimizer, self.loss],
                           feed_dict={self.rgb:inputs,
                                      self.gt_pred:gt_position})


if __name__ == "__main__":
    model = Model_STNv2_xyz('vgg16_weights.npz', fc_sizes=[1024, 256], loss_type='l2', save_dir='./')
    input = tf.placeholder(dtype=tf.float32, shape=(None, 224,224,3))
    model.build(input)
    output = tf.placeholder(dtype=tf.float32, shape=(None, 64,3))
    model.setup_optimizer(output)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    pdb.set_trace()

