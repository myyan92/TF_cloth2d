import tensorflow as tf
from TF_cloth2d.models.model_VGG_STN_imageloss_EM import Model_IM_EM
from TF_cloth2d.models.model_VGG_STN_2_imageloss_EM import Model_IM_EM_v2
from TF_cloth2d.dataset_io import data_parser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.transforms
import pdb, argparse, gin

def patches(transforms, facecolors, edgecolors):
    boxes = []
    for trans, fc, ec in zip(transforms, facecolors, edgecolors):
        scale = np.linalg.norm(trans[0:2])*112
        angle = np.arctan2(-trans[1], trans[0])
        x, y = trans[2]*112+112, trans[5]*112+112
        rect=Rectangle(( (x-scale), (y-scale) ), scale*2, scale*2, alpha=0.2, facecolor=fc, edgecolor=ec)
        t = matplotlib.transforms.Affine2D().rotate_around(x, y, angle)
        rect.set_transform(t)
        boxes.append(rect)
    pc = PatchCollection(boxes, match_original=True)
    return pc

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
                image, position, pred_layers, transformations, raw_pred_layers = self.sess.run([self.next_image, self.next_position,
                                                  self.model.pred_layers, self.model.transformations, self.model.raw_pred_transformed])

                fig, axes=plt.subplots(1,5, figsize=(18, 3))
                axes[0].imshow(image[0,:,:,:].astype(np.uint8))
                colors = ['#0033ee', '#0055cc', '#0077aa', '#009988']
                for i,c in enumerate(colors):
                    pred_im=-pred_layers[i][0,:,:]*112+112
                    axes[i+1].plot(pred_im[:,0], pred_im[:,1], c=c, marker='x')
                    axes[i+1].axis([0,224,224,0], 'square')
#                plt.show()
                if (total_count % 20==0):
                    plt.savefig('vis1_%03d.png'%(total_count))
                    plt.close()
                fig, axes=plt.subplots(1,3, figsize=(12, 3))
                pred_im=-pred_layers[0][0,:,:]*112+112
                axes[0].plot(pred_im[:,0], pred_im[:,1], marker='x')
                colors = ['#dd2200', '#ff4400', '#ff6600', '#dd8800', '#bbaa00', '#99cc00', '#77ee00', '#55cc00']
                patches_im = patches(transformations[0][0], facecolors=colors,
                                     edgecolors=colors)
                axes[0].add_collection(patches_im)
                axes[0].axis([0,224,224,0], 'square')
#                axes[1].plot(pred_im[:,0], pred_im[:,1])
                patches_im = patches(transformations[0][0], facecolors=colors,
                                     edgecolors=colors)
                axes[1].add_collection(patches_im)
                raw_pred_im=-raw_pred_layers[0][0,:,:]*112+112
                for i,c in enumerate(colors):
                    axes[1].scatter(raw_pred_im[i*3:(i+1)*3,0], raw_pred_im[i*3:(i+1)*3,1], c=c, s=10)
                axes[1].axis([0,224,224,0], 'square')
#                axes[2].plot(pred_im[:,0], pred_im[:,1])
                patches_im = patches(transformations[0][0], facecolors=colors,
                                     edgecolors=colors)
                axes[2].add_collection(patches_im)
                pred_im=-pred_layers[1][0,:,:]*112+112
                axes[2].plot(pred_im[:,0], pred_im[:,1], marker='x')
                axes[2].axis([0,224,224,0], 'square')
                #plt.show()
                if (total_count % 20==0):
                    plt.savefig('vis2_%03d.png'%(total_count))
                    plt.close()
                total_count += 1
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

