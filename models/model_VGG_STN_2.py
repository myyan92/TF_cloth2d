import numpy as np
import tensorflow as tf
import gin, gin.tf
from TF_cloth2d.losses_TF import node_l2loss
from TF_cloth2d.models.spatial_transformer import transformer
from TF_cloth2d.models.model_VGG import Model, VGG_MEAN
import pdb

@gin.configurable
class Model_STNv2(Model):
    def __init__(self, **kwargs):
        # fc_sizes should be [1024, 256]
        self.stop_gradient = kwargs.pop('stop_gradient', True)
        kwargs['num_points'] = 64 # The pred layer from VGG will be wasted.
        super().__init__(**kwargs)

    def build(self, rgb=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
        """
        super().build(rgb)
        with tf.variable_scope(self.scope):

            self.pred_layers = []
            self.raw_pred_layers = []
            self.raw_pred_transformed = []
            self.transformations = []
            self.rois = [] # LOWRES
            pred_1 = self.dense(self.last_feature, 'fc%d_trans'%(6+len(self.fc_sizes)), 18, None)
            pred_1 = tf.reshape(pred_1, [-1,9,2])  # x and y in [-1,1]*[-1,1]
            self.pred_layers.append(pred_1)
            pull_feature = [self.conv4_3, self.conv3_3, self.conv2_2]
            fc_scale = [0.3/1.04, 0.3/6.73, 0.3/4.3] # Because VGG conv activations are not scale 1.0
            for hierarchy in range(3):
                # Create boxes where point[i] and point[i+1] from pred_1 are centers of opposite sides.
                pred_previous = self.pred_layers[-1]
                centers = (pred_previous[:,:-1,:] + pred_previous[:,1:,:])*0.5
                diffs = (pred_previous[:,1:,:] - pred_previous[:,:-1,:])*0.5
                box_features = tf.concat([centers, diffs], axis=-1)
                transforms = tf.layers.dense(box_features, 6,
                             kernel_initializer=tf.constant_initializer([0,0,-1,0,0,0,
                                                                         0,0,0,0,0,-1,
                                                                         1,0,0,0,1,0,
                                                                         0,-1,0,1,0,0]),
                             bias_initializer=tf.constant_initializer(0.0),
                             trainable=False, name='transforms_%d'%(hierarchy+1))
                self.transformations.append(transforms)
                if self.stop_gradient:
                    transforms = tf.stop_gradient(transforms)
                rois = transformer(pull_feature[hierarchy], transforms, (7,7))
                self.rois.append(rois) # LOWRES
                fc_1 = self.dense(tf.layers.flatten(rois), 'fc%d_1'%(4-hierarchy), 1024, tf.nn.tanh,
                                  scale=fc_scale[hierarchy])
                fc_2 = self.dense(fc_1, 'fc%d_2'%(4-hierarchy), 256, tf.nn.tanh)
                fc_3 = self.dense(fc_2, 'fc%d_3'%(4-hierarchy), 6, None, scale=0.01)
                fc_3 = fc_3 + tf.constant([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                regularizing_loss = tf.reduce_mean((fc_3[:,0:2]-tf.constant([-1.0,0.0]))**2)
                tf.losses.add_loss(regularizing_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
                regularizing_loss = tf.reduce_mean((fc_3[:,4:6]-tf.constant([1.0,0.0]))**2)
                tf.losses.add_loss(regularizing_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
                regularizing_loss = tf.reduce_mean((fc_3[:,2:4]-tf.constant([0.0,0.0]))**2)
                tf.losses.add_loss(regularizing_loss*0.1, tf.GraphKeys.REGULARIZATION_LOSSES)

                num_points = int(transforms.shape[-2])*3

                transform_expand = tf.expand_dims(transforms, 2)
                transform_expand = tf.reshape(tf.tile(transform_expand, [1,1,3,1]), [-1,num_points,2,3])
                pred_next = tf.reshape(fc_3, [-1,num_points,2])
                self.raw_pred_layers.append(pred_next)
                pred_next = tf.layers.dense(pred_next, 3,
                              kernel_initializer=tf.constant_initializer([1,0,0,
                                                                          0,1,0]),
                              bias_initializer=tf.constant_initializer([0,0,-1]),
                              trainable=False, name='homography_%d'%(hierarchy+1))

                pred_next_transformed = tf.matmul(transform_expand, tf.expand_dims(pred_next, -1))
                pred_next_transformed = tf.reshape(pred_next_transformed, [-1,num_points,2])
                self.raw_pred_transformed.append(pred_next_transformed)
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

                pred_gather = tf.gather(pred_next_transformed, pred_gather_indexes, axis=1)
                pred_next_averaged = tf.reduce_mean(pred_gather, axis=2)
                self.pred_layers.append(pred_next_averaged)

            self.pred = self.pred_layers[-1][:,:-1,:]*6.0 # 6.0 is because of data generation.
            saver_var_list = self.get_trainable_variables()
            if not self.use_vgg:
                vars = self.get_variables()
                vars = [v for v in vars if 'moving_mean' in v.name or 'moving_variance' in v.name]
                saver_var_list = saver_var_list + vars
            self.saver = tf.train.Saver(var_list=saver_var_list, max_to_keep=50)

    def setup_optimizer(self, GT_position=None):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if GT_position is not None:
            self.gt_pred = GT_position
        else:
            self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=self.pred.shape)
        self.pred_layers_losses = []
        for pred in self.pred_layers:
            loss = node_l2loss(pred*6.0, self.gt_pred, resample_equdistance=True)
            self.pred_layers_losses.append(loss)
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = tf.add_n(self.pred_layers_losses)*0.25 + self.reg_loss*5.0
        self.debug_gradients = tf.gradients(self.loss, self.pred_layers)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=self.momentum, epsilon=0.01).minimize(self.loss)
        tf.summary.scalar('loss', self.pred_layers_losses[-1])
        tf.summary.scalar('loss_reg', self.reg_loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, annos):
        _, loss = sess.run([self.optimizer, self.loss],
                           feed_dict={self.rgb:inputs,
                                      self.gt_pred:annos,
                                      self.training:True})
        return loss

if __name__ == "__main__":
    model = Model_STNv2(vgg16_npy_path='vgg16_weights.npz', fc_sizes=[1024, 256],
                        loss_type='l2', save_dir='./')
    input = tf.placeholder(dtype=tf.float32, shape=(None, 224,224,3))
    model.build(input)
    output = tf.placeholder(dtype=tf.float32, shape=(None, 64,2))
    model.setup_optimizer(output)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    pdb.set_trace()

