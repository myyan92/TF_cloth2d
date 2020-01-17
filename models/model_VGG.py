import numpy as np
import tensorflow as tf
import gin
import os, pdb

VGG_MEAN = [103.939, 116.779, 123.68]

@gin.configurable
class Model:
    def __init__(self,
                 vgg16_npy_path,
                 fc_sizes,
                 num_points,
                 save_dir,
                 loss_type='l2',
                 learning_rate=0.001,
                 momentum=0.9,
                 use_vgg=True,
                 finetune_vgg=False):
        # For predicting nodes, use fc_sizes=[1024, 1024], num_points=128.
        # For predicting knots, use fc_sizes=[1024, 1024, 128, 128], num_points=4.
        self.data_dict = np.load(vgg16_npy_path)
        self.scope='vgg'
        self.fc_sizes = fc_sizes
        self.num_points = num_points
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_vgg = use_vgg
        self.finetune_vgg = finetune_vgg
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
            self.training = tf.placeholder(tf.bool)
            # Convert RGB to BGR
            bgr = tf.reverse(self.rgb, axis=[-1]) - tf.constant(VGG_MEAN)
            bgr = bgr / 256.0
            #bgr = (self.rgb-100.0)/10.0
            trainable = self.finetune_vgg or (not self.use_vgg)
            self.conv1_1 = self.conv_layer(bgr, "conv1_1", 64, load_weight=self.use_vgg, trainable=trainable)
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2", 64, load_weight=self.use_vgg, trainable=trainable)
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1", 128, load_weight=self.use_vgg, trainable=trainable)
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2", 128, load_weight=self.use_vgg, trainable=trainable)
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1", 256, load_weight=self.use_vgg, trainable=trainable)
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2", 256, load_weight=self.use_vgg, trainable=trainable)
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3", 256, load_weight=self.use_vgg, trainable=trainable)
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1", 512, load_weight=self.use_vgg, trainable=trainable)
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2", 512, load_weight=self.use_vgg, trainable=trainable)
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3", 512, load_weight=self.use_vgg, trainable=trainable)
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1", 512, load_weight=self.use_vgg, trainable=trainable)
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2", 512, load_weight=self.use_vgg, trainable=trainable)
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3", 512, load_weight=self.use_vgg, trainable=trainable)
            self.pool5 = self.max_pool(self.conv5_3, 'pool5')

            net = tf.layers.flatten(self.pool5)
            self.fcs = []
            for i, h in enumerate(self.fc_sizes):
                net = self.dense(net, 'fc%d'%(6+i), h, tf.nn.relu)
                self.fcs.append(net)
            self.pred = self.dense(net, 'fc%d'%(6+len(self.fc_sizes)), 2*self.num_points, None)
            self.pred = tf.reshape(self.pred, [-1,self.num_points,2])

            self.last_feature = self.fcs[-1]
            saver_var_list = self.get_trainable_variables()
            if not self.use_vgg:
                vars = self.get_variables()
                vars = [v for v in vars if 'moving_mean' in v.name or 'moving_variance' in v.name]
                saver_var_list = saver_var_list + vars
            self.saver = tf.train.Saver(var_list=saver_var_list, max_to_keep=20)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, channels, kernel=3, stride=1, load_weight=True, trainable=False):
        with tf.variable_scope(name):
            if load_weight:
                k_init = tf.constant_initializer(self.data_dict[name+'_W'], verify_shape=True)
                b_init = tf.constant_initializer(self.data_dict[name+'_b'], verify_shape=True)
            else:
                k_init = tf.variance_scaling_initializer()
                b_init = tf.zeros_initializer()
            output = tf.layers.conv2d(bottom, channels, kernel_size=kernel, strides=stride, padding='SAME',
                                      activation=None, kernel_initializer=k_init, bias_initializer=b_init,
                                      trainable=trainable)
            if not load_weight:
                output = tf.layers.batch_normalization(output, momentum=0.9, epsilon=1e-6,
                                                       training=self.training, trainable=True)
            output = tf.nn.relu(output)
        return output

    def dense(self, bottom, name, channels, activation, scale=1.0, load_weight=False, trainable=True):
        with tf.variable_scope(name):
            if load_weight:
                k_init = tf.constant_initializer(self.data_dict[name+'_W'], verify_shape=True)
                b_init = tf.constant_initializer(self.data_dict[name+'_b'], verify_shape=True)
            else:
                k_init = tf.variance_scaling_initializer(scale)
                b_init = tf.zeros_initializer()
            output = tf.layers.dense(bottom, channels, activation=None,
                                     kernel_initializer=k_init, bias_initializer=b_init,
                                     trainable = trainable)
            #if not load_weight:    #
            #    output = tf.layers.batch_normalization(output, training=self.training, trainable=trainable)  #
            if activation is not None:
                output = activation(output)
        return output

    def predict_single(self, sess, input, training=False):
        pred, = sess.run([self.pred], feed_dict={self.rgb:input[None], self.training:training})
        return pred[0]

    def predict_batch(self, sess, inputs, training=False):
        pred, = sess.run([self.pred], feed_dict={self.rgb:inputs, self.training:training})
        return pred

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def setup_optimizer(self, GT_position=None):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.loss_type == "l2":
            if GT_position is not None:
                self.gt_pred = GT_position
            else:
                self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=[None, self.num_points, 2])
            self.loss = tf.nn.l2_loss(self.gt_pred-self.pred, "loss_knots")
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.momentum). \
                                          minimize(self.loss)
            tf.summary.scalar('loss', self.loss)
        elif self.loss_type == "gradient":
            self.pred_grad = tf.placeholder(name="pred_grad", dtype=tf.float32, shape=self.pred.shape)
            self.parameters = self.get_trainable_variables()
            self.gradients = tf.gradients(self.pred, self.parameters, self.pred_grad)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.momentum). \
                                          apply_gradients(zip(self.gradients, self.parameters))
        else:
            raise ValueError("Unsupported loss type!")
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, annos):
        if self.loss_type == "l2":
            _, loss = sess.run([self.optimizer, self.loss],
                               feed_dict={self.rgb:inputs,
                                          self.gt_pred:annos,
                                          self.training:True})
        elif self.loss_type == "gradient":
            _, = sess.run([self.optimizer], feed_dict={self.rgb:inputs,
                                                       self.pred_grad:annos,
                                                       self.training:True})
            loss = None
        return loss

    def save(self, sess, step):
        self.saver.save(sess, os.path.join(self.save_dir, 'model'), global_step=step)

    def load(self, sess, snapshot):
        self.saver.restore(sess, snapshot)

if __name__ == "__main__":
    model = Model('vgg16_weights.npz', [1024, 1024, 128, 128], 4, '../tmp', use_vgg=False)
    input = tf.placeholder(dtype=tf.float32, shape=(None, 224,224,3))
    model.build(input)
    output = tf.placeholder(dtype=tf.float32, shape=(None, 4, 2))
    model.setup_optimizer(output)
