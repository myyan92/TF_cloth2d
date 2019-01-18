import tensorflow as tf

def sample_b_spline(knots_input):
    # tensorflow implementation of sample_b_spline
    knots = tf.concat([knots_input[:,0:1,:],
                       knots_input,
                       knots_input[:,-1:,:]], axis=-2)

    # construct weights
    num_pts = 100
    ts = tf.linspace(0.0, 1.0, num_pts+1)[:-1]
    offsets = tf.constant([1.0, 0.0, -1.0, -2.0])
    ts = tf.expand_dims(ts, axis=-2) + tf.expand_dims(offsets, axis=-1)
    weights = tf.maximum(0.0, (ts-2)**3)
    weights -= 4*tf.maximum(0.0, (ts-1)**3)
    weights += 6*tf.maximum(0.0, ts**3)
    weights -= 4*tf.maximum(0.0, (ts+1)**3)
    weights += tf.maximum(0.0, (ts+2)**3)
    weights = weights / 6.0

    # turn weights into a 1d conv filter
    # 4 x pts -> 4 x 2 x (pts*2)
    padding = tf.zeros_like(weights)
    filter = tf.stack([weights, padding, padding, weights], axis=-1)
    filter = tf.reshape(filter, [4, num_pts, 2, 2])
    filter = tf.transpose(filter, (0,2,1,3))
    filter = tf.reshape(filter, [4, 2, num_pts*2])
    samples = tf.nn.conv1d(knots, filter, stride=1, padding='VALID')
    samples_shape = tf.stack([tf.shape(samples)[0],
                              tf.shape(samples)[1]*num_pts,
                              2])
    samples = tf.reshape(samples, samples_shape)
    return samples

def sample_equdistance(samples, num_pts):
    # tensorflow implementation of sample_equdistance
    length = samples[:, 1:,:]-samples[:, :-1,:]
    length = tf.norm(length, axis=-1)
    length = tf.cumsum(length, axis=-1)
    segments = tf.map_fn(lambda l: tf.linspace(0.0, l[-1], num_pts), length)
    compare = tf.greater(tf.expand_dims(segments, -1),
                         tf.expand_dims(length, -2))
    idx = tf.reduce_sum(tf.cast(compare, tf.int32), axis=-1)
    idx_0 = tf.ones_like(idx)
    idx_0 = tf.cumsum(idx_0, axis=0)-1
    idx = tf.stack([idx_0, idx], axis=-1)

    length_left = tf.gather_nd(length, idx[:,1:-1,:]-tf.constant([0,1]))
    length_right = tf.gather_nd(length, idx[:,1:-1,:])
    alpha = (segments[:, 1:-1] - length_left) / (length_right - length_left)
    alpha = tf.expand_dims(alpha, axis=-1)
    sample_left = tf.gather_nd(samples, idx[:,1:-1:,])
    sample_right = tf.gather_nd(samples, idx[:,1:-1,:]+tf.constant([0,1]))

    sub_samples = sample_left * (1-alpha) + sample_right * alpha
    sub_samples = tf.concat([samples[:,0:1,:],
                             sub_samples,
                             samples[:,-1:,:]], axis=-2)
    return sub_samples

def node_l2loss(pred, GT,
                resample_b_spline=False,
                resample_equdistance=False):
    """L2 loss between prediction and groud truth.

    If resample_b_spline, interpolate pred by b-spline interpolation,
    then resample to be equdistance.
    If resample_equdistance, interpolate linearly and be equdistance.
    Otherwise use raw prediction for l2 loss.
    """
    num_pts = int(GT.shape[1])
    if resample_b_spline:
        samples = sample_b_spline(pred)
        samples = sample_equdistance(samples, num_pts)
    elif resample_equdistance:
        samples = sample_equdistance(pred, num_pts)
    else:
        samples = pred

    loss_1 = tf.reduce_sum(tf.square(samples-GT), axis=[1,2])
    loss_2 = tf.reduce_sum(tf.square(samples-GT[:,::-1,:]), axis=[1,2])
    loss = tf.minimum(loss_1, loss_2)
    loss = tf.reduce_sum(loss) / 2.0
    return loss
