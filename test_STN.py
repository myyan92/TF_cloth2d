import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pdb

im1 = Image.open('../../gen_data/data_with_knots/9000.png')
im1 = im1.resize((224,224))
im1 = np.array(im1).reshape(1, 224, 224, 3)
im1 = im1.astype('float32')
#im2 = Image.open('../../gen_data/data_with_knots/9002.png')
#im2 = im2.resize((224,224))
#im2 = np.array(im2).reshape(1, 224, 224, 3)
#im2 = im2.astype('float32')
batch = im1

out_size = (56,56)
centers = np.array([[[-0.81018408, -0.14346762],
                  [-0.65621038, -0.05408934],
                  [-0.46516188, -0.20582525],
                  [-0.27603996, -0.36419026],
                  [-0.06075768, -0.4804699 ],
                  [ 0.17994416, -0.48848208],
                  [ 0.40426286, -0.39059381],
                  [ 0.60163176, -0.24306651]]])


x = tf.placeholder(tf.float32, [None, 224, 224, 3])
center = tf.placeholder(tf.float32, [None, 8, 2])
transforms = tf.layers.dense(center, 6,
                             kernel_initializer=tf.constant_initializer([0,0,-1,0,0,0,0,0,0,0,0,-1]),
                             bias_initializer=tf.constant_initializer([0.25,0,0,0,0.25,0]),
                             trainable=False, name='transforms')

# %% Create localisation network and convolutional layer
with tf.variable_scope('spatial_transformer_0'):
    output = transformer(x, transforms, out_size)

# %% Run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y = sess.run(output, feed_dict={x: batch, center: centers})
#pdb.set_trace()
print(y.shape)
for patch in y:
    plt.imshow(patch)
    plt.show()
