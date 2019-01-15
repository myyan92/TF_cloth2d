import tensorflow as tf
from TF_cloth2d.models.model_VGG_STN_imageloss_EM import Model_IM_EM
from TF_cloth2d.models.model_VGG_STN_2_imageloss_EM import Model_IM_EM_v2
from TF_cloth2d.dataset_io import data_parser
from TF_cloth2d.sort_nodes import sort_nodes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.transforms as transforms
import pdb, argparse, gin

def ploting(position, pred, transforms):
    fig, ax = plt.subplots(1)
    ax.plot(position[0,:,0], position[0,:,1])
    c = np.tile(np.arange(1,9),8)
    #ax.plot(pred[0,:,0], pred[0,:,1]) #, s=1, c="orange")
    ax.scatter(pred[0,:,0], pred[0,:,1], c=c)
    boxes = []
    for trans in transforms:
        scale = np.linalg.norm(trans[0:2])
        angle = np.arctan2(trans[1], trans[0])
        x, y = -trans[2], -trans[5]
        rect=Rectangle(( (x-scale)*6, (y-scale)*6 ), scale*12, scale*12)
        t = transforms.Affine2D().rotate_around(x*6, y*6, angle)
        rect.set_transform(t)
        boxes.append(rect)
    pc = PatchCollection(boxes, facecolor='none', edgecolor='g')
    ax.add_collection(pc)
    plt.axis("equal")
    plt.show()

@gin.configurable
class Visualizer():
    def __init__(self, eval_dataset, eval_snapshot, model):

        # create TensorFlow Dataset objects
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser)
        val_data = val_data.batch(1)
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(val_data.output_types,
                                                   val_data.output_shapes)
        self.next_image, self.next_position, self.next_knot = iterator.get_next()
        # create two initialization ops to switch between the datasets
        self.eval_init_op = iterator.make_initializer(val_data)

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.model = model
        self.model.build(rgb=self.next_image)
        self.model.setup_optimizer()
        self.sess.run(tf.global_variables_initializer())
        self.model.load(self.sess, eval_snapshot)
#        self.sess.run(tf.assign(self.model.sigma, tf.constant(0.05)))

    def eval(self):
        self.sess.run(self.eval_init_op)
        total_loss = 0
        total_count = 0
        while True:
            try:
                image, position, pred, render, image_loss, reg_loss = self.sess.run([self.next_image, self.next_position,
                                                  self.model.pred, self.model.render,
                                                  self.model.image_loss, self.model.reg_losses])

                fig, axes=plt.subplots(1,2)
                axes[0].imshow(image[0,:,:,:].astype(np.uint8))
                pred_im=-pred/6*112+112
                axes[0].plot(pred_im[0,:,0], pred_im[0,:,1])
                axes[1].imshow(render[0,:,:,:])
                plt.show()
                #plt.savefig('eval_%03d.png'%(total_count))
                #plt.close()
                total_count += 1
                print(image_loss, reg_loss)
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

