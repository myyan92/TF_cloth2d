from TF_cloth2d.models.model_VGG import Model, VGG_MEAN
from TF_cloth2d.models.model_VGG_STN import Model_STN
import tensorflow as tf
import gin, gin.tf
import os

@gin.configurable
def make_RL_model(base_model,
                  save_dir,
                  variance_type="constant",
                  variance_per_point=False,
                  pointwise_cost=False,
                  entropy_weight=0.2,
                  learning_rate=0.001,
                  momentum=0.9):

    class Model_RL(base_model):
        def __init__(self):
            super().__init__()
            if variance_type in ["constant", "independent", "dependent"]:
                self.variance_type = variance_type
            else:
                raise ValueError("variance_type should be one of constant, independent, and dependent")
            self.variance_per_point = variance_per_point
            self.pointwise_cost = pointwise_cost
            self.entropy_weight = entropy_weight
            self.learning_rate = learning_rate # Overwrite supervised model learning rate.
            self.momentum = momentum # Overwrite supervised model momentum.
            self.save_dir = save_dir # Overwrite model saving directory.

        def build(self, rgb=None):
            super().build(rgb)
            with tf.variable_scope(self.scope):
                if self.variance_per_point:
                    dim = self.pred.shape[1]
                else:
                    dim = 1
                if self.variance_type == "constant":
                    self.variance = tf.constant(1.0, dtype=tf.float32, name='variance')
                elif self.variance_type == "independent":
                    self.variance = tf.get_variable('variance', shape=(dim,1), dtype=tf.float32, initializer=tf.constant_initializer(1.0))
                else:
                    self.variance = self.dense(self.last_feature, 'variance', dim, tf.nn.softplus) # log(exp(x)+1)
                    self.variance = tf.expand_dims(self.variance, axis=-1)
                self.variance = tf.clip_by_value(self.variance, 0.01, 10.0)
                self.pd_pred = tf.distributions.Normal(loc=self.pred, scale=self.variance)  # element-wise distribution, need to sum over log-prob
                self.pred_sample = self.pd_pred.sample()

                self.value = self.dense(self.last_feature, 'value', 1, None)  # value function as baseline
                self.value = tf.reshape(self.value, [-1])
                trainable_vars = self.get_trainable_variables()
                trainable_vars = [var for var in trainable_vars if 'variance' in var.name or 'value' in var.name]
                self.saver_pd = tf.train.Saver(var_list=trainable_vars, max_to_keep=50)

        def predict_single(self, sess, input, sampling=True):
            if sampling:
                pred, value = sess.run([self.pred_sample, self.value], feed_dict={self.rgb:input[None]})
                pred = pred[0]
                value = value[0]
            else:
                pred = super().predict_single(sess, input)
                value = None
            return pred, value

        def predict_batch(self, sess, inputs, sampling=True):
            if sampling:
                pred, value = sess.run([self.pred_sample, self.value], feed_dict={self.rgb:inputs})
            else:
                pred = super().predict_batch(sess, inputs)
                value = None
            return pred, value

        def setup_optimizer(self):
            self.sample_used = tf.placeholder(dtype=tf.float32, shape=self.pred.shape)
            if self.pointwise_cost:
                self.log_prob = tf.reduce_sum(self.pd_pred.log_prob(self.sample_used), axis=[2])
                self.cost_adv = tf.placeholder(dtype=tf.float32, shape=[None,self.pred.shape[1]]) # already subtracted baseline
            else:
                self.log_prob = tf.reduce_sum(self.pd_pred.log_prob(self.sample_used), axis=[1,2])
                self.cost_adv = tf.placeholder(dtype=tf.float32, shape=[None,]) # already subtracted baseline
            self.cost = tf.placeholder(dtype=tf.float32, shape=[None,]) # for value function loss
            self.rl_loss = tf.reduce_sum(self.cost_adv*self.log_prob)
            self.entropy = tf.reduce_sum(self.pd_pred.entropy())
            self.vf_loss = tf.nn.l2_loss(self.value-self.cost)
            self.loss = self.rl_loss - entropy_weight * self.entropy # + self.vf_loss
            if hasattr(self, "pred_2_scaled"):
                centroids_l2 = tf.reduce_mean(tf.reshape(self.pred_2_scaled, [-1,8,8,2]), axis=2)
                self.reg_loss = tf.nn.l2_loss(centroids_l2)
                self.loss += 10*self.reg_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1 = self.momentum,
                                                    epsilon = 1e-6).minimize(self.loss)
            tf.summary.scalar('rl_loss', self.rl_loss)
            tf.summary.scalar('cost', tf.reduce_mean(self.cost))
            tf.summary.scalar('entropy', self.entropy)
            # tf.summary.scalar('value_loss', self.vf_loss)
            self.merged_summary = tf.summary.merge_all()

        def fit(self, sess, inputs, samples, costs_adv, costs):
            _, summary, rl_loss, entropy, vf_loss = sess.run([self.optimizer, self.merged_summary, self.rl_loss, self.entropy, self.vf_loss],
                                                feed_dict={self.rgb:inputs,
                                                           self.sample_used:samples,
                                                           self.cost_adv:costs_adv,
                                                           self.cost:costs})
            return summary, rl_loss, entropy

        def save(self, sess, step):
            super().save(sess, step)
            self.saver_pd.save(sess, os.path.join(self.save_dir, 'model-RL'), global_step=step)

        def load(self, sess, snapshot_mean, snapshot_var):
            super().load(sess, snapshot_mean)
            if snapshot_var is not None:
                self.saver_pd.restore(sess, snapshot_var)

    return Model_RL()

if __name__ == "__main__":
    gin.bind_parameter("Model.fc_sizes", [1024,1024,128,128])
    gin.bind_parameter("Model.num_points", 4)
    gin.bind_parameter("Model.loss_type", 'l2')
    sess = tf.Session()
    model = make_RL_model(Model)
    model.build()
    model.setup_optimizer(0.001, 1.0)
    sess.run(tf.global_variables_initializer())

