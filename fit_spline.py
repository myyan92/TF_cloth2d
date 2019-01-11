import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sample_spline import sample_b_spline, sample_equdistance

control = tf.get_variable(name='control_points', dtype=tf.float32, shape=[1, 2, 4])
t = tf.get_variable(name='spline_t', dtype=tf.float32, shape=[1, 1, 128])
pointcloud = tf.get_variable(name='point_cloud', dtype=tf.float32, shape=[1, 2, 128], trainable=False)
t_tile = tf.tile(t, [1,6,1]) - tf.constant([-1,0,1,2,3,4], dtype=tf.float32, shape=[1,6,1])
weights =( tf.maximum(tf.pow((t_tile-2),3),0) -
         4*tf.maximum(tf.pow((t_tile-1),3),0) +
         6*tf.maximum(tf.pow((t_tile  ),3),0) -
         4*tf.maximum(tf.pow((t_tile+1),3),0) +
           tf.maximum(tf.pow((t_tile+2),3),0) ) / 6.0
control_expand = tf.concat([control[:,:,0:1], control, control[:,:,-1:]], axis=2)
samples = tf.matmul(control_expand, weights)
loss = tf.nn.l2_loss(samples-pointcloud) + tf.nn.l2_loss(t[:,:,1:]-t[:,:,:-1]) \
      + tf.nn.l2_loss(t[:,:,0]) + tf.nn.l2_loss(t[:,:,-1]-3)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(loss)


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())

def test_fit(node_file, knot_file, gt_node_file, gt_knot_file):
    with open(node_file) as f:
        position = f.readlines()
    position = [p.strip().split() for p in position]
    position = [[float(pp) for pp in p] for p in position]
    position = np.array(position).transpose()
    with open(knot_file) as f:
        knots = f.readlines()
    knots = [k.strip().split() for k in knots]
    knots = [[float(kk) for kk in k] for k in knots]
    knots = np.array(knots).transpose()
    knots_t = [knots[:,0], knots[:,0], knots[:,1],
               knots[:,2], knots[:,3], knots[:,3]]
    ss, ws = sample_b_spline(knots_t)
    ss, ws = sample_equdistance(ss, ws, 128)
#    plt.plot(ss[0,:],ss[1,:], label='pred knot')
#    plt.plot(position[0,:],position[1,:], label='pred node')

    t_np= np.linspace(0,3,128)
    sess.run(tf.assign(control, knots[np.newaxis,:,:]))
    sess.run(tf.assign(t, t_np[np.newaxis,np.newaxis,:]))
    sess.run(tf.assign(pointcloud, position[np.newaxis,:,:]))
    for i in range(800):
        l, _ = sess.run([loss, optimizer])
#        if i % 200 == 0:
#            print(l)
#            s,ts = sess.run([samples,control])
#            plt.scatter(s[0,0,:], s[0,1,:], label='fit')
#            plt.scatter(position[0,:], position[1,:], label='samples')
#            print(ts)
#            plt.legend()
#            plt.show()
    fitted_ss, fitted_knots, t_out = sess.run([samples, control, t])
    knots_t = [fitted_knots[0,:,0], fitted_knots[0,:,0], fitted_knots[0,:,1],
               fitted_knots[0,:,2], fitted_knots[0,:,3], fitted_knots[0,:,3]]
    ss, ws = sample_b_spline(knots_t)
    ss, ws = sample_equdistance(ss, ws, 128)

    with open(gt_node_file) as f:
        position = f.readlines()
    position = [p.strip().split() for p in position]
    position = [[float(pp) for pp in p] for p in position]
    position = np.array(position).transpose()
    with open(gt_knot_file) as f:
        knots = f.readlines()
    knots = [k.strip().split() for k in knots]
    knots = [[float(kk) for kk in k] for k in knots]
    knots = np.array(knots).transpose()
    loss_knots = np.sum(np.square(knots-fitted_knots))/4.0
    loss_nodes = np.sum(np.square(position-ss))/128.0
#    plt.plot(position[0,:],position[1,:], label='gt')
#    plt.plot(ss[0,:],ss[1,:], label='fitted')
#    plt.legend()
#    plt.show()
    print("l2 loss per node: ", loss_nodes)
#    print("l2 loss per knot: ", loss_knots)

    return

for i in range(1000):
    test_fit('./pred/%04d_nodes.txt'%(i+1),
             './pred/%04d_knots.txt'%(i+1),
             '../gen_data/data_with_knots/%04d.txt'%(i+9000),
             '../gen_data/data_with_knots/%04d_knots.txt'%(i+9000))


