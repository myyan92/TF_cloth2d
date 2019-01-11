import numpy as np
import tensorflow as tf
import gin, gin.tf
from TF_cloth2d.models.model_VGG import Model, VGG_MEAN

@gin.configurable
class Model_3d(Model):

    def build(self, rgb=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
        """
        super().build(rgb)
        with tf.variable_scope(self.scope):

            self.pred_intersect = self.dense(self.last_feature,
                                             'fc%d_intersect'%(6+len(self.fc_sizes)),
                                             (self.num_points-1)*3, None)
            self.pred_intersect_logit = tf.reshape(self.pred_intersect, [-1,self.num_points-1,3])
            self.pred_intersect = tf.nn.softmax(self.pred_intersect_logit, axis=-1)

            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=50)

    def predict_single(self, sess, input):
        pred, pred_intersect = sess.run([self.pred, self.pred_intersect],
                                        feed_dict={self.rgb:input[None]})
        return pred[0], pred_intersect[0]

    def predict_batch(self, sess, inputs):
        pred, pred_intersect = sess.run([self.pred, self.pred_intersect],
                                        feed_dict={self.rgb:inputs})
        return pred, pred_intersect

    def setup_optimizer(self, GT_position=None, GT_intersect=None):
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

        self.pred_loss = tf.nn.l2_loss(self.gt_pred-self.pred, "loss")
        classification_weight = tf.reduce_sum(self.gt_pred_intersect*tf.constant([1.0, 0.01, 1.0]), axis=-1)
        self.classification_loss = tf.losses.softmax_cross_entropy(
                                       self.gt_pred_intersect,
                                       self.pred_intersect_logit,
                                       weights=classification_weight)
        self.loss = self.pred_loss + self.classification_loss*100.0
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        tf.summary.scalar('loss_Euc', self.pred_loss)
        tf.summary.scalar('loss_CE', self.classification_loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, gt_position, gt_intersect):
        _, loss = sess.run([self.optimizer, self.loss],
                           feed_dict={self.rgb:inputs,
                                      self.gt_pred:gt_position,
                                      self.gt_pred_intersect:gt_intersect})
        return loss

if __name__ == "__main__":
    model = Model_3d('vgg16_weights.npz', fc_sizes=[1024, 1024], num_points=128, loss_type='l2', save_dir='./tmp-3d')
    input = tf.placeholder(dtype=tf.float32, shape=(None, 224,224,3))
    model.build(input)
    output = tf.placeholder(dtype=tf.float32, shape=(None, 128,2))
    output_intersect = tf.placeholder(dtype=tf.float32, shape=(None, 127,3))
    model.setup_optimizer(output, output_intersect)
