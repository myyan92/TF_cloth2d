from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2, VGG_MEAN
import tensorflow as tf
import tensorflow_probability as tfp
import gin, gin.tf

@gin.configurable
class Model_IM_EM_v2(Model_STNv2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup_optimizer(self):
        pix_x = tf.lin_space(-6.0,6.0,224)
        pix_y = tf.lin_space(-6.0,6.0,224)
        pix_x, pix_y = tf.meshgrid(pix_x, pix_y)
        pix = tf.stack([pix_x,pix_y], axis=-1)

        # batch of [1,63] gaussian distributions in R2, one for each edge
        self.image_pred = - self.pred  # annotated configuration need to be negated to match render
        self.norm_rgb = self.rgb/255

        center = (self.image_pred[:,1:,:]+self.image_pred[:,:-1,:]) / 2.0
        diff = (self.image_pred[:,1:,:]-self.image_pred[:,:-1,:]) / 2.0
        dist = tf.norm(diff, axis=2)
        self.dist = dist

        pix = tf.reshape(pix, [224,224,1,1,2]) # to broadcast
        pix_to_center = pix - center
        # perpendicular distance to segments
        # cross(pix-center, diff) / dist
        pix_to_segment_p = (pix_to_center[:,:,:,:,0]*diff[:,:,1]-pix_to_center[:,:,:,:,1]*diff[:,:,0]) / dist
        # longitutional distance to segments
        # max{ abs[ dot(pix-center, diff) / dist ] - dist, 0 }
        pix_to_segment_l = (pix_to_center[:,:,:,:,0]*diff[:,:,0]+pix_to_center[:,:,:,:,1]*diff[:,:,1]) / dist
        pix_to_segment_l = tf.maximum(tf.abs(pix_to_segment_l)-dist, 0)

        #sigma = tf.constant(0.1, dtype=tf.float32)
        self.sigma = tf.get_variable('sigma',dtype=tf.float32, shape=(), initializer=tf.constant_initializer(0.1))
        sigma = tf.maximum(self.sigma, 0.1)

        pix_prob = tf.exp((-tf.square(pix_to_segment_p)-tf.square(pix_to_segment_l))/(2*sigma*sigma))
        self.reg_loss_e = tf.reduce_sum(pix_prob[:,:,:,1:]*pix_prob[:,:,:,:-1], axis=[0,1,3]) / 20000.0
        pix_prob = tf.reduce_max(pix_prob, axis=3) # shape should be [224,224,batch]
        pix_prob = tf.transpose(pix_prob, perm=[2,0,1]) # [batch, 224,224]
        pix_prob = tf.clip_by_value(pix_prob, 0, 1)
        pix_prob = tf.expand_dims(pix_prob, axis=3)
        self.pix_prob=pix_prob # debug
        # simple case, assume foreground and background color is known
        #pix_positive = tf.constant([1,0,0], dtype=tf.float32) # red pixel
        #pix_negative = tf.constant([1,1,1], dtype=tf.float32) # white pixel
        # harder case, foreground / background color by averaging prediction
        mean_positive = tf.reduce_mean(pix_prob, axis=[1,2], keep_dims=True) # B x 1 x 1 x 1
        mean_negative = tf.reduce_mean(1-pix_prob, axis=[1,2], keep_dims=True)
        pix_positive = tf.reduce_mean(pix_prob*self.norm_rgb, axis=[1,2], keep_dims=True) / mean_positive
        pix_negative = tf.reduce_mean((1-pix_prob)*self.norm_rgb, axis=[1,2], keep_dims=True) / mean_negative
        # render is for visualization, not in loss
        self.render = pix_prob * pix_positive + (1-pix_prob) * pix_negative
        self.render = tf.clip_by_value(self.render, 0, 1)

        mean_positive = tf.squeeze(mean_positive, axis=[1,2,3])
        mean_negative = tf.squeeze(mean_negative, axis=[1,2,3])
        pix_prob = tf.squeeze(pix_prob, axis=3)
        pix_prob = tf.expand_dims(tf.expand_dims(pix_prob, -1), -1) # annoying reshaping to broadcast correctly
        mean_positive = tf.expand_dims(tf.expand_dims(mean_positive, -1), -1)
        mean_negative = tf.expand_dims(tf.expand_dims(mean_negative, -1), -1)
        cov_positive = tf.reduce_mean(pix_prob*tf.matmul(
                                               tf.expand_dims(self.norm_rgb-pix_positive, -1),
                                               tf.expand_dims(self.norm_rgb-pix_positive, -1),
                                               transpose_b=True), axis=[1,2]) / mean_positive
        cov_negative = tf.reduce_mean((1-pix_prob)*tf.matmul(
                                               tf.expand_dims(self.norm_rgb-pix_negative, -1),
                                               tf.expand_dims(self.norm_rgb-pix_negative, -1),
                                               transpose_b=True), axis=[1,2]) / mean_negative

        positive_gaussian = tfp.distributions.MultivariateNormalFullCovariance(
                                       loc=tf.squeeze(pix_positive, axis=[1,2]),
                                       covariance_matrix=cov_positive+tf.eye(3)*0.0001)
        negative_gaussian = tfp.distributions.MultivariateNormalFullCovariance(
                                       loc=tf.squeeze(pix_negative, axis=[1,2]),
                                       covariance_matrix=cov_negative+tf.eye(3)*0.0001)

        mean_positive = tf.reshape(mean_positive, [1,1,-1])
        mean_negative = tf.reshape(mean_negative, [1,1,-1])
        pix_prob = tf.transpose(tf.squeeze(pix_prob, axis=[3,4]), perm=[1,2,0])
        image_reshape = tf.transpose(self.norm_rgb, perm=[1,2,0,3]) # H x W x B x 3
        prob_positive = mean_positive * positive_gaussian.prob(image_reshape) # H x W x B
        prob_negative = mean_negative * negative_gaussian.prob(image_reshape)
        prob_positive = tf.maximum(prob_positive, 1e-30)
        prob_negative = tf.maximum(prob_negative, 1e-30)
        # for debuging
        self.pix_prob=pix_prob
        self.prob_positive=prob_positive
        self.prob_negative=prob_negative
        self.image_loss_e = - tf.reduce_mean(tf.log(pix_prob*prob_positive + (1-pix_prob)*prob_negative), axis=[0,1]) # image loss per instance
        self.image_loss = tf.reduce_mean(self.image_loss_e)
#        self.reg_loss_e = tf.reduce_mean(dist*dist, axis=1)     # - tf.reduce_mean(tf.reduce_mean(dist,axis=[1])*tf.reduce_mean(dist,axis=[1]), axis=0)*0.95
        self.reg_loss = tf.reduce_mean(self.reg_loss_e)
        #skip_diff = (self.image_pred[:,2:,:]-self.image_pred[:,:-2,:]) / 2.0
        #skip_dist = tf.norm(skip_diff, axis=2)
        #self.bending_loss = tf.reduce_mean(tf.square(dist[:,1:]+dist[:,:-1]-skip_dist))
#        centroids_l2 = tf.reduce_mean(tf.reshape(self.pred_2_scaled, [-1,8,8,2]), axis=2)
#        self.reg_loss_l2 = tf.reduce_mean(tf.square(centroids_l2, "reg_loss"))
        self.reg_loss_l2 = tf.losses.get_regularization_loss()

        prob_grad = tf.gradients(self.image_loss, pix_prob)[0]
        self.prob_grad=prob_grad
        prob_grad = tf.clip_by_value(prob_grad, -0.01, 0.01)
        self.reg_losses = self.reg_loss + 0.1*self.reg_loss_l2
        self.adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.01)
        gradvars = self.adam_optimizer.compute_gradients(self.reg_losses)
        vars = [v for g,v in gradvars]
        image_grads = tf.gradients(pix_prob, vars, grad_ys=prob_grad)
        gradvars_new = []
        for (g,v), ig in zip(gradvars, image_grads):
            if g is not None and ig is not None:
                gradvars_new.append((g+ig, v))
            elif g is not None:
                gradvars_new.append((g,v))
            elif ig is not None:
                gradvars_new.append((ig,v))
        self.optimizer = self.adam_optimizer.apply_gradients(gradvars_new)

        tf.summary.scalar('image_loss', self.image_loss)
        tf.summary.scalar('reg_loss', self.reg_loss)
        tf.summary.scalar('centroid_loss', self.reg_loss_l2)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs):
        _, summary, image_loss, reg_loss, centroid_loss = sess.run([self.optimizer, self.merged_summary, self.image_loss, self.reg_loss, self.reg_loss_l2],
                                                                  feed_dict={self.rgb:inputs})
        return summary, image_loss, reg_loss, centroid_loss


if __name__ == "__main__":
    sess = tf.Session()
    model = Model_IM_EM('../vgg16_weights.npz', fc_sizes=[1024, 256],
                      loss_type='imageEM', save_dir='./tmp',
                      train_scale=True, train_rotation=True)
    model.build()
    model.setup_optimizer()
    sess.run(tf.global_variables_initializer())

