import numpy as np
import tensorflow as tf
import gin
from TF_cloth2d.losses_TF import node_l2loss
from TF_cloth2d.models.spatial_transformer import transformer
from TF_cloth2d.models.model_VGG import Model, VGG_MEAN

@gin.configurable
class Model_STN(Model):
    def __init__(self, **kwargs):
        # fc_sizes should be [1024, 256]
        self.train_scale = kwargs.pop('train_scale', True)
        self.train_rotation = kwargs.pop('train_rotation', True)
        kwargs['num_points'] = 64 # The pred layer from VGG will be wasted.
        super().__init__(**kwargs)

    def build(self, rgb=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
        """
        super().build(rgb)
        with tf.variable_scope(self.scope):

            self.pred_1_t = self.dense(self.last_feature, 'fc%d_trans'%(6+len(self.fc_sizes)), 16, None)
            self.pred_1_t = tf.reshape(self.pred_1_t, [-1,8,2])  # x and y in [-1,1]*[-1,1]
            # Pred_1_s adds to [0,0] and [1,1] of the transformation matrix.
            self.pred_1_s = self.dense(self.last_feature, 'fc%d_scale'%(6+len(self.fc_sizes)), 8, None, scale=0.1)
            self.pred_1_s = tf.expand_dims(self.pred_1_s, -1)
            if not self.train_scale:
                self.pred_1_s = tf.clip_by_value(self.pred_1_s, 0.0, 0.0)
            # Pred_1_r changes [0,1] amd [1,0] of the transformation matrix.
            # This is only enabled when train_scale is True.
            self.pred_1_r = self.dense(self.last_feature, 'fc%d_rotate'%(6+len(self.fc_sizes)), 8, None, scale=0.1)
            self.pred_1_r = tf.expand_dims(self.pred_1_r, -1)
            if (not self.train_scale) or (not self.train_rotation):
                self.pred_1_r = tf.clip_by_value(self.pred_1_r, 0.0, 0.0)

            self.pred_1 = tf.concat([self.pred_1_t, self.pred_1_s, self.pred_1_r], axis=2)
            self.transforms_1 = tf.layers.dense(self.pred_1, 6,
                                kernel_initializer=tf.constant_initializer([0,0,-1,0,0,0,
                                                                            0,0,0,0,0,-1,
                                                                            1,0,0,0,1,0,
                                                                            0,-1,0,1,0,0]),
                                bias_initializer=tf.constant_initializer([0.25,0,0,0,0.25,0]),
                                trainable=False, name='transforms')

            self.rois_1 = transformer(self.conv4_3, self.transforms_1, (7,7))
            self.fc4_1 = self.dense(tf.layers.flatten(self.rois_1), 'fc4_1', 1024, tf.nn.relu)
            self.fc4_2 = self.dense(self.fc4_1, 'fc4_2', 256, tf.nn.relu)
            self.fc4_3 = self.dense(self.fc4_2, 'fc4_3', 16, None)
            self.pred_1_expand = tf.expand_dims(self.pred_1_t, 2)
            self.pred_1_expand = tf.reshape(tf.tile(self.pred_1_expand, [1,1,8,1]), [-1,64,2])
            self.pred_1_trans_expand = tf.expand_dims(self.transforms_1, 2)
            self.pred_1_trans_expand = tf.reshape(tf.tile(self.pred_1_trans_expand, [1,1,8,1]), [-1,64,2,3])
            self.pred_2 = tf.reshape(self.fc4_3, [-1,64,2,1])
            self.pred_2_scaled = tf.matmul(self.pred_1_trans_expand[:,:,:,0:2], self.pred_2)
            self.pred_2_comp = self.pred_2_scaled + tf.expand_dims(self.pred_1_expand, -1)     # must do the reshaping to use matmul
            self.pred_2_comp = tf.reshape(self.pred_2_comp, [-1,64,2])
            self.pred = self.pred_2_comp*6.0 # 6.0 is because of data generation.
            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=50)

    def setup_optimizer(self, GT_position=None):
        if GT_position is not None:
            self.gt_pred = GT_position
        else:
            self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=self.pred.shape)
        self.pred_loss = node_l2loss(self.pred, self.gt_pred, resample_equdistance=False)
        centroids_l2 = tf.reduce_mean(tf.reshape(self.pred_2_scaled, [-1,8,8,2]), axis=2)
        self.reg_loss = tf.nn.l2_loss(centroids_l2, "reg_loss")
        self.loss = self.pred_loss + self.reg_loss * 10
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        tf.summary.scalar('loss', self.pred_loss)
        tf.summary.scalar('loss_reg', self.reg_loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, annos):
        _, loss = sess.run([self.optimizer, self.loss],
                           feed_dict={self.rgb:inputs,
                                      self.gt_pred:annos})
        return loss

if __name__ == "__main__":
    model = Model_STN(vgg16_npy_path='vgg16_weights.npz', fc_sizes=[1024, 256],
                      loss_type='l2', save_dir='./tmp',
                      train_scale=True, train_rotation=True)
    input = tf.placeholder(dtype=tf.float32, shape=(None, 224,224,3))
    model.build(input)
    output = tf.placeholder(dtype=tf.float32, shape=(None, 64,2))
    model.setup_optimizer(output)
