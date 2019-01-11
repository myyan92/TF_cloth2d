import numpy as np
from losses_TF import *
from sample_spline import sample_equdistance
import matplotlib.pyplot as plt
import pdb

# test resampling with linear interpolation.
# Starting from a random curve with 9 points, optimize towards a line y=0.
samples = tf.get_variable('sample', shape=[4, 9, 2], dtype=tf.float32,
                          initializer=tf.constant_initializer(0.0))
gt_sub_samples = tf.placeholder(shape=[None, 64, 2], dtype=tf.float32)
loss = node_l2loss(samples, gt_sub_samples, resample_equdistance=True)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
samples_val = np.random.random((4,9,2))*5
gt_subsamples = np.zeros((1,64,2))
gt_subsamples[0,:,0] = np.linspace(0.0, 5.0, 64)

for i in range(4):
    plt.plot(samples_val[i,:,0], samples_val[i,:,1])

sess.run(tf.assign(samples, samples_val))
for i in range(5001):
    loss_val, _ = sess.run([loss, optimizer],
                                       feed_dict={gt_sub_samples:gt_subsamples})
    print(loss_val)
samples_result = sess.run(samples)
for i in range(4):
    plt.plot(samples_result[i,:,0], samples_result[i,:,1])

plt.show()

tf.reset_default_graph()

# testing resampling with b-spline interpolation.
# Starting from a random curve with 4 knots, optimizer towards a sample in data_with_knots.
knots_input = tf.get_variable('knots', shape=[1, 4, 2], dtype=tf.float32,
                              initializer = tf.constant_initializer(0.0))
GT_samples = tf.placeholder(shape=[None, 64, 2], dtype=tf.float32)
loss = node_l2loss(knots_input, GT_samples, resample_b_spline=True,
                   resample_equdistance=True)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
knots_val = np.loadtxt('/scr-ssd/mengyuan/gen_data/data_with_knots/0100_knots.txt')
samples_val = np.loadtxt('/scr-ssd/mengyuan/gen_data/data_with_knots/0100.txt')
samples_val = samples_val[np.newaxis, ::2,:]
sess.run(tf.assign(knots_input, np.random.random((1,4,2))*5))

for i in range(4000):
    loss_val, _ = sess.run([loss, optimizer], feed_dict={GT_samples:samples_val})
    print(loss_val)

knots_result = sess.run(knots_input)
plt.plot(knots_val[:,0], knots_val[:,1])
plt.plot(knots_result[0,:,0], knots_result[0,:,1])
plt.show()

