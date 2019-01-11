from model_VGG import Model, VGG_MEAN
import tensorflow as tf
import numpy as np

class Model_PPO(Model):
    def __init__(self, vgg16_npy_path, variance_type="constant", variance_per_point=False):
        super().__init__(vgg16_npy_path)
        if variance_type in ["constant", "independent", "dependent"]:
            self.variance_type = variance_type
        else:
            raise ValueError("variance_type should be one of constant, independent, and dependent")
        self.variance_per_point = variance_per_point

    def build(self):
        super().build(None) # should not connect with data flow here, because we must feed pairing input/reward for optimization
        with tf.variable_scope(self.scope):
            if self.variance_per_point:
                dim = self.pred.shape[-1]
            else:
                dim = 1
            if self.variance_type == "constant":
                self.variance = tf.constant(1.0, dtype=tf.float32, name='variance')
            elif self.variance_type == "independent":
                self.variance = tf.get_variable('variance', shape=(dim,), dtype=tf.float32, initializer=tf.constant_initializer(1.0))
            else:
                self.variance = self.dense(self.last_feature, 'variance', dim, tf.nn.softplus) # log(exp(x)+1)
            self.pd_pred = tf.distributions.Normal(loc=self.pred, scale=self.variance)  # element-wise distribution, need to sum over log-prob
            self.pred_sample = self.pd_pred.sample()
            self.pred_logprob = tf.reduce_sum(self.pd_pred.log_prob(self.pred_sample), axis=[1,2])

            self.value = self.dense(self.last_feature, 'value', 1, None)  # value function as baseline
            self.value = tf.reshape(self.value, [-1])
            trainable_vars = self.get_trainable_variables()
            trainable_vars = [var for var in trainable_vars if 'variance' in var.name or 'value' in var.name]
            self.saver_pd = tf.train.Saver(var_list=trainable_vars, max_to_keep=50)

    def predict_single(self, sess, input, sampling=True):
        if sampling:
            pred, value, logprob = sess.run([self.pred_sample, self.value, self.pred_logprob],
                                            feed_dict={self.rgb:input[None]})
            pred, value, logprob = pred[0], value[0], logprob[0]
        else:
            pred = super().predict_single(sess, input)
            value, logprob = None, None
        return pred, value, logprob

    def predict_batch(self, sess, inputs, sampling=True):
        if sampling:
            pred, value, logprob = sess.run([self.pred_sample, self.value, self.pred_logprob],
                                            feed_dict={self.rgb:inputs})
            mu, var = sess.run([self.pred, self.variance], feed_dict={self.rgb:inputs})
        else:
            pred = super().predict_batch(sess, inputs)
            value, logprob = None, None
        return pred, value, logprob

    def setup_optimizer(self, learning_rate, clip_range, entropy_weight):
        self.samples_used = tf.placeholder(name='samples_used', dtype=tf.float32, shape=self.pred.shape)
        self.cost_adv = tf.placeholder(name='adv', dtype=tf.float32, shape=[None,]) # already subtracted baseline
        self.cost = tf.placeholder(name='cost', dtype=tf.float32, shape=[None,]) # for value function loss
        self.oldLogProb = tf.placeholder(name='old_logprob', dtype=tf.float32, shape=[None,])
        self.oldValue = tf.placeholder(name='old_value', dtype=tf.float32, shape=[None,])
        self.log_prob = tf.reduce_sum(self.pd_pred.log_prob(self.samples_used), axis=[1,2])

        self.entropy = tf.reduce_sum(self.pd_pred.entropy())
        self.vf_loss_1 = tf.square(self.value-self.cost)
        self.value_clipped = self.oldValue + tf.clip_by_value(self.value-self.oldValue,
                                                              -clip_range, clip_range)
        self.vf_loss_2 = tf.square(self.value_clipped-self.cost)
        self.vf_loss = 0.5*tf.reduce_sum(tf.maximum(self.vf_loss_1, self.vf_loss_2))
        self.ratio = tf.exp(self.log_prob-self.oldLogProb)
        self.rl_loss_1 = self.cost_adv * self.ratio
        self.rl_loss_2 = self.cost_adv * tf.clip_by_value(self.ratio,
                                                          1.0-clip_range, 1.0+clip_range)
        self.rl_loss = tf.reduce_sum(tf.maximum(self.rl_loss_1, self.rl_loss_2))
        self.loss = self.rl_loss - entropy_weight * self.entropy  + self.vf_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        tf.summary.scalar('rl_loss', self.rl_loss*50)
        tf.summary.scalar('cost', tf.reduce_mean(self.cost)*50)
        tf.summary.scalar('entropy', self.entropy)
        var_grad = tf.gradients(self.rl_loss, self.variance)
        tf.summary.scalar("variance_gradient", var_grad[0][0])
        # tf.summary.scalar('value_loss', self.vf_loss*2500)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, samples, costs, logprob, value):
        _, summary, rl_loss, entropy, vf_loss = sess.run([self.optimizer, self.merged_summary, self.rl_loss, self.entropy, self.vf_loss],
                                                         feed_dict={self.rgb:inputs,
                                                                    self.samples_used:samples,
                                                                    self.cost_adv:costs-value,
                                                                    self.cost:costs,
                                                                    self.oldLogProb:logprob,
                                                                    self.oldValue:value})
        return summary, rl_loss, entropy

    def save(self, sess, step):
        super().save(sess, step)
        self.saver_pd.save(sess, './model-RL', global_step=step)

    def load(self, sess, snapshot_mean, snapshot_var):
        super().load(sess, snapshot_mean)
        self.saver_pd.restore(sess, snapshot_var)

if __name__ == "__main__":
    sess = tf.Session()
    model = Model_PPO('vgg16_weights.npz', variance_type='dependent', variance_per_point=True)
    model.build()
    model.setup_optimizer(0.001, 1.0)
    sess.run(tf.global_variables_initializer())

