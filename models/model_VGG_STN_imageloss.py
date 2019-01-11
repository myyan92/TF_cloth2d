from TF_cloth2d.models.model_VGG_STN import Model_STN, VGG_MEAN
import tensorflow as tf
import gin, gin.tf

@gin.configurable
class Model_IM_L1(Model_STN):

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
        sigma = tf.get_variable('sigma',dtype=tf.float32, shape=(), initializer=tf.constant_initializer(0.5))
        sigma = tf.maximum(sigma, 0.1)

        pix_prob = tf.exp((-tf.square(pix_to_segment_p)-tf.square(pix_to_segment_l))/(2*sigma*sigma))
        pix_prob = tf.reduce_max(pix_prob, axis=3) # shape should be [224,224,batch]
        pix_prob = tf.transpose(pix_prob, perm=[2,0,1]) # [batch, 224,224]
        pix_prob = tf.clip_by_value(pix_prob, 0, 1)
        pix_prob = tf.expand_dims(pix_prob, axis=3)
        # simple case, assume foreground and background color is known
        #pix_positive = tf.constant([1,0,0], dtype=tf.float32) # red pixel
        #pix_negative = tf.constant([1,1,1], dtype=tf.float32) # white pixel
        # harder case, foreground / background color by averaging prediction
        pix_positive = tf.reduce_sum(pix_prob*self.norm_rgb, axis=[1,2], keep_dims=True)  \
                       / tf.reduce_sum(pix_prob, axis=[1,2], keep_dims=True)
        pix_negative = tf.reduce_sum((1-pix_prob)*self.norm_rgb, axis=[1,2], keep_dims=True)  \
                       / tf.reduce_sum((1-pix_prob), axis=[1,2], keep_dims=True)
        self.render = pix_prob * pix_positive + (1-pix_prob) * pix_negative
        self.render = tf.clip_by_value(self.render, 0, 1)

        self.image_loss = tf.losses.absolute_difference(self.norm_rgb,self.render)
        self.reg_loss = tf.reduce_mean(dist*dist) # - tf.reduce_mean(tf.reduce_mean(dist,axis=[1])*tf.reduce_mean(dist,axis=[1]), axis=0)*0.95
        skip_diff = (self.image_pred[:,2:,:]-self.image_pred[:,:-2,:]) / 2.0
        skip_dist = tf.norm(skip_diff, axis=2)
        self.bending_loss = tf.reduce_mean(tf.square(dist[:,1:]+dist[:,:-1]-skip_dist))
        centroids_l2 = tf.reduce_mean(tf.reshape(self.pred_2_scaled, [-1,8,8,2]), axis=2)
        self.reg_loss_l2 = tf.reduce_mean(tf.square(centroids_l2, "reg_loss"))
        self.loss = self.image_loss + 0.05*self.reg_loss + 0.3*self.reg_loss_l2 # + 0.1*self.bending_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.01).minimize(self.loss)
        tf.summary.scalar('image_loss', self.image_loss)
        tf.summary.scalar('reg_loss', self.reg_loss)
        tf.summary.scalar('bending_loss', self.bending_loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs):
        _, summary, image_loss, reg_loss, bending_loss = sess.run([self.optimizer, self.merged_summary, self.image_loss, self.reg_loss, self.bending_loss],
                                                                  feed_dict={self.rgb:inputs})
        return summary, image_loss, reg_loss, bending_loss


if __name__ == "__main__":
    sess = tf.Session()
    model = Model_IM_L1('vgg16_weights.npz', fc_sizes=[1024, 256],
                      loss_type='imageL1', save_dir='./tmp',
                      train_scale=True, train_rotation=True)
    model.build()
    model.setup_optimizer()
    sess.run(tf.global_variables_initializer())

