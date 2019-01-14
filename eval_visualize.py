import tensorflow as tf
from TF_cloth2d.models.model_VGG import Model
from TF_cloth2d.models.model_VGG_STN import Model_STN
from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2
from dataset_io import data_parser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.transforms
from sample_spline import sample_b_spline, sample_equdistance
from physbam_python.rollout_physbam import rollout_single
import gin, argparse
import pdb

@gin.configurable
class Visualizer():
    def __init__(self,
                 eval_dataset,
                 eval_snapshot,
                 model,
                 pred_target,
                 use_physbam):

        # create TensorFlow Dataset objects
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser)
        val_data = val_data.batch(1)
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(val_data.output_types,
                                                   val_data.output_shapes)
        self.next_image, self.next_position, self.next_knot = iterator.get_next()
        self.eval_init_op = iterator.make_initializer(val_data)

        if pred_target == "node":
            assert(self.next_position.shape[1] % model.num_points == 0)
            skip = self.next_position.shape[1] // model.num_points
            self.next_position = self.next_position[:, ::skip, :]

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.model = model
        self.model.build(self.next_image)
        self.pred_target = pred_target
        self.use_physbam = use_physbam
        self.sess.run(tf.global_variables_initializer())
        self.model.load(self.sess, eval_snapshot)

    def eval(self):
        self.sess.run(self.eval_init_op)
        total_loss = 0
        total_count = 0
        while True:
            try:
                image, position, pred = self.sess.run(
                    [self.next_image, self.next_position, self.model.pred])
                if self.pred_target == 'knot':
                    pred_knots = [pred[0,0,:]]+list(pred[0,:,:]) + [pred[0,-1,:]]
                    samples,weights = sample_b_spline(pred_knots)
                    samples,weights = sample_equdistance(samples, weights, position.shape[1])
                    samples = samples.transpose()
                else:
                    samples = pred[0]
                loss = np.sum(np.square(samples-position[0,:,:]))
                total_loss += loss
                total_count += position.shape[0] * position.shape[1]
                if "STNv2" in self.model.__class__.__name__:
                    # hack recompute loss after sampling.
                    samples,_ = sample_equdistance(samples, np.zeros((64,64)), 64)
                    samples = samples.transpose()
                    total_loss -= loss
                    loss = np.sum(np.square(samples-position[0,:,:]))
                    total_loss += loss
                    pred = self.sess.run(self.model.pred_layers,
                                         feed_dict={self.model.rgb:image})
                    plt.plot(position[0,:,0], position[0,:,1])
                    plt.plot(pred[0][0,:,0]*6, pred[0][0,:,1]*6)
                    plt.plot(pred[1][0,:,0]*6, pred[1][0,:,1]*6)
                    plt.plot(pred[2][0,:,0]*6, pred[2][0,:,1]*6)
                    plt.plot(pred[3][0,:,0]*6, pred[3][0,:,1]*6)
                    plt.axis("equal")
                    #plt.show()
                    #plt.savefig('vis_%04d.png'%(total_count//position.shape[1]))
                    plt.close()
                    print(loss)
                elif "STN" in self.model.__class__.__name__:
                    transforms = self.sess.run(self.model.transforms_1,
                                               feed_dict={self.model.rgb:image})
                    fig, ax = plt.subplots(1)
                    ax.plot(position[0,:,0], position[0,:,1])
                    c = np.tile(np.arange(1,9),8)
                    ax.scatter(samples[:,0], samples[:,1], c=c)
                    boxes = []
                    for trans in transforms[0]:
                        scale = np.linalg.norm(trans[0:2])
                        angle = np.arctan2(trans[1], trans[0])
                        x, y = -trans[2], -trans[5]
                        rect=Rectangle(( (x-scale)*6, (y-scale)*6 ), scale*12, scale*12)
                        t = matplotlib.transforms.Affine2D().rotate_around(x*6, y*6, angle)
                        rect.set_transform(t)
                        boxes.append(rect)
                    pc = PatchCollection(boxes, facecolor='none', edgecolor='g')
                    ax.add_collection(pc)
                    plt.axis("equal")
                    #plt.show()
                    plt.close()
                    print(loss)
                else:
                    plt.plot(position[0,:,0], position[0,:,1])
                    plt.plot(samples[:,0], samples[:,1])
                    plt.axis("equal")
                    plt.show()
                if self.use_physbam:
                    nodes = sort_nodes(samples[0,:,:])
                    nodes_physbam=rollout_single(nodes, 20, [0.0,0.0], 1, physbam_args=" -disable_collisions")
                    position_physbam=rollout_single(position[0,:,:], 20, [0.0,0.0], 1, physbam_args=" -disable_collisions")
                    loss_phy = np.sum(np.square(nodes_physbam-position_physbam))
                    print("physbam: %f -> %f\n" %(loss, loss_phy))
                    plt.figure()
                    plt.plot(position_physbam[:,0], position_physbam[:,1])
                    plt.plot(nodes_physbam[:,0], nodes_physbam[:,1]) #, s=1, c="orange")
                    plt.axis("equal")
                    plt.show()

            except tf.errors.OutOfRangeError:
                break
        print("eval average node L2 loss: %f" % (total_loss/total_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_config', default='', help="path to gin config file.")
    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)

    vis = Visualizer()
    vis.eval()

