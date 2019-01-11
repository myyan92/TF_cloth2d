import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gin, gin.tf
import os


@gin.configurable
class Model_MRF:
    def __init__(self,
                 save_dir,
                 learning_rate=0.001,
                 momentum=0.9):
        # For predicting nodes, use fc_sizes=[1024, 1024], num_points=128.
        # For predicting knots, use fc_sizes=[1024, 1024, 128, 128], num_points=4.
        self.scope='vgg'
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.save_dir = save_dir

    def build(self, rgb=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
        """
        with tf.variable_scope(self.scope):
            if rgb is not None:
                self.rgb = rgb
            else:
                self.rgb = tf.placeholder(dtype=tf.float32, shape=[None,224,224,3])

            self.conv1_1 = self.conv_layer(self.rgb/256.0, "conv1_1", 64)
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2", 64)
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1", 128)
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2", 128)
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1", 20)
            self.conv3_1 = tf.nn.sigmoid(self.conv3_1 - 1.0)
            self.activate = tf.where(tf.greater(self.conv3_1, 0.5),
                                     self.conv3_1, tf.zeros_like(self.conv3_1))

            # construct patches with segments in different direction
            pix_x = tf.lin_space(-3.0,3.0,7)
            pix_y = tf.lin_space(-3.0,3.0,7)
            pix_x, pix_y = tf.meshgrid(pix_x, pix_y)
            pix = tf.stack([pix_x,pix_y], axis=-1)


            angle = tf.lin_space(-np.pi / 2.0, np.pi / 2.0, 21)[:-1]
            cos = tf.cos(angle)
            sin = tf.sin(angle)

            pix = tf.reshape(pix, [7,7,1,2]) # to broadcast
            # perpendicular distance to segments
            pix_to_segment = (pix[:,:,:,0]*sin-pix[:,:,:,1]*cos)
            self.width = tf.get_variable('width',dtype=tf.float32, shape=(), initializer=tf.constant_initializer(2.0))
            width = tf.maximum(self.width, 1.0)
            pix_to_segment = tf.maximum(pix_to_segment - width, 0.0)

            pix_prob = tf.exp((-tf.square(pix_to_segment))*8.0) # should be 7x7x20
            pix_prob = tf.expand_dims(pix_prob, axis=-2)
            batch_size = tf.shape(self.rgb)[0]
            deconv_shape = tf.stack([batch_size, 224, 224, 1])
            self.segmentation = tf.nn.conv2d_transpose(self.activate, pix_prob,
                                                       output_shape=deconv_shape,
                                                       strides=[1,4,4,1]) # shoud be Bx224x224x1

            self.segmentation = tf.clip_by_value(self.segmentation, 0, 1)

            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=50)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, channels, kernel=3, stride=1):
        with tf.variable_scope(name):
            k_init = tf.variance_scaling_initializer()
            b_init = tf.zeros_initializer()
            output = tf.layers.conv2d(bottom, channels, kernel_size=kernel, strides=stride, padding='SAME',
                                      activation=None, kernel_initializer=k_init, bias_initializer=b_init,
                                      trainable=True)
            output = tf.layers.batch_normalization(output, training=self.training, trainable=True)
            output = tf.nn.relu(output)
        return output

    def dense(self, bottom, name, channels, activation, scale=1.0):
        with tf.variable_scope(name):
            k_init = tf.variance_scaling_initializer(scale)
            b_init = tf.zeros_initializer()
            output = tf.layers.dense(bottom, channels, activation=None,
                                     kernel_initializer=k_init, bias_initializer=b_init,
                                     trainable = True)
            output = tf.layers.batch_normalization(output, training=self.training, trainable=True)
            if activation is not None:
                output = activation(output)
        return output

    def predict_single(self, sess, input):
        pred, = sess.run([self.segmentation], feed_dict={self.rgb:input[None]})
        return pred[0]

    def predict_batch(self, sess, inputs):
        pred, = sess.run([self.segmentation], feed_dict={self.rgb:inputs})
        return pred

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def setup_optimizer(self):

        mean_positive = tf.reduce_mean(self.segmentation, axis=[1,2], keepdims=True) # B x 1 x 1 x 1
        mean_negative = tf.reduce_mean(1-self.segmentation, axis=[1,2], keepdims=True)
        self.norm_rgb = self.rgb / 256.0
        pix_positive = tf.reduce_mean(self.segmentation*self.norm_rgb, axis=[1,2], keepdims=True) / mean_positive
        pix_negative = tf.reduce_mean((1-self.segmentation)*self.norm_rgb, axis=[1,2], keepdims=True) / mean_negative
        self.pix_negative = pix_negative
        mean_positive = tf.squeeze(mean_positive, axis=[1,2,3])
        mean_negative = tf.squeeze(mean_negative, axis=[1,2,3])
        segmentation = tf.expand_dims(self.segmentation, -1) # annoying reshaping to broadcast correctly
        mean_positive = tf.expand_dims(tf.expand_dims(mean_positive, -1), -1)
        mean_negative = tf.expand_dims(tf.expand_dims(mean_negative, -1), -1)
        cov_positive = tf.reduce_mean(segmentation*tf.matmul(
                                               tf.expand_dims(self.norm_rgb-pix_positive, -1),
                                               tf.expand_dims(self.norm_rgb-pix_positive, -1),
                                               transpose_b=True), axis=[1,2]) / mean_positive
        cov_negative = tf.reduce_mean((1-segmentation)*tf.matmul(
                                               tf.expand_dims(self.norm_rgb-pix_negative, -1),
                                               tf.expand_dims(self.norm_rgb-pix_negative, -1),
                                               transpose_b=True), axis=[1,2]) / mean_negative
        self.cov_negative = cov_negative
        positive_gaussian = tfp.distributions.MultivariateNormalFullCovariance(
                                       loc=tf.squeeze(pix_positive, axis=[1,2]),
                                       covariance_matrix=cov_positive+tf.eye(3)*0.0001)
        negative_gaussian = tfp.distributions.MultivariateNormalFullCovariance(
                                       loc=tf.squeeze(pix_negative, axis=[1,2]),
                                       covariance_matrix=cov_negative+tf.eye(3)*0.0001)

        mean_positive = tf.reshape(mean_positive, [1,1,-1])
        mean_negative = tf.reshape(mean_negative, [1,1,-1])
        segmentation = tf.transpose(tf.squeeze(segmentation, axis=[3,4]), perm=[1,2,0])
        image_reshape = tf.transpose(self.norm_rgb, perm=[1,2,0,3]) # H x W x B x 3
        prob_positive = mean_positive * positive_gaussian.prob(image_reshape) # H x W x B
        prob_negative = mean_negative * negative_gaussian.prob(image_reshape)
        prob_positive = tf.maximum(prob_positive, 1e-30)
        prob_negative = tf.maximum(prob_negative, 1e-30)

        self.image_loss_e = - tf.reduce_mean(tf.log(segmentation*prob_positive + (1-segmentation)*prob_negative), axis=[0,1]) # image loss per instance
        self.image_loss = tf.reduce_mean(self.image_loss_e)
        self.reg_loss = tf.reduce_mean(tf.abs(self.conv3_1))
        self.loss = self.image_loss + self.reg_loss*0.1
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.momentum). \
                                  minimize(self.loss)
        tf.summary.scalar('image_loss', self.image_loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs):
        _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.rgb:inputs})
        return loss

    def save(self, sess, step):
        self.saver.save(sess, os.path.join(self.save_dir, 'model'), global_step=step)

    def load(self, sess, snapshot):
        self.saver.restore(sess, snapshot)

if __name__ == "__main__":
    model = Model_MRF('../tmp')
    model.training=True
    input = tf.placeholder(dtype=tf.float32, shape=(None, 224,224,3))
    model.build(input)
    model.setup_optimizer()
