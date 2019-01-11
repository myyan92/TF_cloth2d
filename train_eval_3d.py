import tensorflow as tf
from TF_cloth2d.models.model_VGG_3d import Model_3d
from TF_cloth2d.models.model_VGG_STN_2_3d import Model_STNv2_3d
#from models.model_REINFORCE import make_RL_model
from dataset_io_3d import data_parser
import numpy as np
from PIL import Image
import gin, argparse, os
import matplotlib.pyplot as plt
import pdb

@gin.configurable
class Trainner():
    def __init__(self,
                 train_dataset,
                 eval_dataset,
                 model, # A model instance constructed by gin config
                 use_RL,
                 loss,
                 num_epoch,
                 batch_size,
                 save_dir,
                 mode,
                 snapshot=None):

        # create TensorFlow Dataset objects
        tr_data = tf.data.TFRecordDataset(train_dataset)
        tr_data = tr_data.map(data_parser)
        tr_data = tr_data.shuffle(buffer_size=5000)
        tr_data = tr_data.batch(batch_size)
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser)
        val_data = val_data.batch(batch_size)
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                                   tr_data.output_shapes)
        self.next_image, self.next_position, self.next_intersect = iterator.get_next()
        assert(self.next_position.shape[1] % model.num_points == 0)
        skip = self.next_position.shape[1] // model.num_points
        self.next_position = self.next_position[:, ::skip, :]
        skip = skip * 2 # LOWRES
        if skip > 1:
            padding = tf.zeros_like(self.next_intersect[:,-1:,:])
            self.next_intersect = tf.concat(
                                  [self.next_intersect, padding], axis=-2)
#            self.next_intersect = self.next_intersect[:, :self.next_intersect.shape[1]-skip+1, :]
            self.next_intersect = tf.reshape(self.next_intersect,
                                             [-1, model.num_points//2, skip, 3]) # LOWRES
            self.next_intersect = tf.reduce_sum(self.next_intersect, axis=2)
            c0, c1, c2 = tf.split(self.next_intersect, 3, -1)
            assert_op = tf.Assert(tf.reduce_any(tf.less(c0+c2, 1.0)), [c0,c2])
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assert_op)
            c1 = 1.0 - c0 - c2
            self.next_intersect = tf.concat([c0,c1,c2], axis=-1)
# LOWRES    self.next_intersect = self.next_intersect[:, :-1, :]
        # create two initialization ops to switch between the datasets
        self.training_init_op = iterator.make_initializer(tr_data)
        self.validation_init_op = iterator.make_initializer(val_data)

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.num_epoch = num_epoch
        self.model = model
        self.model.build(rgb=self.next_image)
        assert(loss == 'l2') # only support l2 loss + cross entropy for now.
        self.model.setup_optimizer(self.next_position, self.next_intersect)
        self.use_RL = use_RL
        self.global_step = 0
        self.train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'tfboard'), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.snapshot = snapshot
        config_str = gin.operative_config_str()
        with open(os.path.join(save_dir, '0.gin'), 'w') as f:
            f.write(config_str)

    def train_epoch(self):
        self.sess.run(self.training_init_op)
        losses = []
        while True:
            try:
                if not self.use_RL:
                    #image, position, intersection, pred = self.sess.run(
                    #    [self.next_image, self.next_position, self.next_intersect, self.model.pred])
                    #pdb.set_trace()
                    image, positions, intersections = self.sess.run(
                                                      [self.next_image, self.next_position, self.next_intersect])
                    #additional = 2*positions[:,-1:,:]-positions[:,-2:-1,:]
                    #positions = np.concatenate([positions,additional],axis=-2)
                    #positions_in = positions[:, ::2, :] # LOWRES
                    summary, _, loss = self.sess.run(
                        [self.model.merged_summary, self.model.optimizer, self.model.loss],
                        feed_dict={self.model.rgb:image,
                                   #self.model.pred_layers[-2]:positions_in/6.0, # LOWRES
                                   self.model.gt_pred:positions,
                                   self.model.gt_pred_intersect:intersections,
                                   self.model.training:True})
                else:
                    image, GT_node, GT_intersect = self.sess.run(
                        [self.next_image, self.next_position, self.next_intersect])
                    pred, pred_intersect, pred_value = self.model.predict_batch(self.sess, image, sampling=True)
                    cost_per_node, cost_sum = self.loss_fn(pred, pred_intersect, GT_node, GT_intersect)
                    cost_adv = (cost_per_node - np.mean(cost_per_node, axis=0)) / np.std(cost_per_node, axis=0)
                    cost_adv = np.clip(cost_adv, -3, 3)
                    cost = cost_sum
                    summary, loss, ent_loss = self.model.fit(self.sess, image, pred, cost_adv, cost)

                self.train_writer.add_summary(summary, self.global_step)
                self.global_step += 1
                losses.append(loss)
            except tf.errors.OutOfRangeError:
                break
        print("train batch loss this epoch: %f" %(np.mean(losses)))

    def eval(self):
        self.sess.run(self.validation_init_op)
        total_euc_loss = 0
        total_euc_count = 0
        total_intersect_loss = 0
        total_intersect_count = 0
        confusion = np.zeros((3,3))
        while True:
            try:
                if not self.use_RL:
                    image, gt, gt_intersect = self.sess.run([self.next_image, self.next_position, self.next_intersect])
                    #additional = 2*gt[:,-1:,:]-gt[:,-2:-1,:]
                    #gt = np.concatenate([gt,additional],axis=-2)
                    #gt_in = gt[:, ::2, :] # LOWRES
                    pred_layers, pred_intersect, euc_loss, intersect_loss, roi, roi_raw = self.sess.run(
                        [self.model.pred_layers, self.model.pred_intersect, self.model.pred_loss, self.model.classification_loss, self.model.rois, self.model.rois_raw],
                        feed_dict={self.model.rgb:image,
                                   #self.model.pred_layers[-2]:gt_in/6.0, # LOWRES
                                   self.model.gt_pred:gt,
                                   self.model.gt_pred_intersect:gt_intersect,
                                   self.model.training:False})
                    euc_count = gt.shape[0]*gt.shape[1]
                    intersect_count = gt.shape[0]*(gt.shape[1]//2)
                    pred_layers = [p*6.0 for p in pred_layers]
                    plt.plot(gt[0,:,0],gt[0,:,1])
                    plt.plot(pred_layers[0][0,:,0], pred_layers[0][0,:,1], c='r')
                    plt.plot(pred_layers[1][0,:,0], pred_layers[1][0,:,1], c='#dd5500')
                    plt.plot(pred_layers[2][0,:,0], pred_layers[2][0,:,1], c='#dddd00')
                    #plt.plot(pred[0,:,0], pred[0,:,1])
                    for idx,pi in enumerate(pred_intersect[0]):
                        if np.argmax(pi)==0:
                            plt.plot(pred_layers[2][0,idx:idx+2,0], pred_layers[2][0,idx:idx+2,1], color='#00dd00')
                        if np.argmax(pi)==2:
                            plt.plot(pred_layers[2][0,idx:idx+2,0], pred_layers[2][0,idx:idx+2,1], color='#00dd00')
                    plt.axis('equal')
                    #plt.show()
                    #print(euc_loss, intersect_loss)
                    gt_class = np.argmax(gt_intersect, axis=-1)
                    pred_class = np.argmax(pred_intersect, axis=-1)
                    #confusion = np.zeros((3,3))
                    for g,p in zip(gt_class.flatten(), pred_class.flatten()):
                        confusion[g,p]+=1
                    #print(confusion)
                    #pdb.set_trace()
                else:
                    GT_node, GT_intersect, pred, pred_intersect = self.sess.run(
                        [self.next_position, self.next_intersect, self.model.pred, self.model.pred_intersect])
                    _, loss = self.loss_fn(pred_node, pred_intersect, GT_node, GT_intersect)
                    loss = np.sum(loss)  # Loss is sometimes scalar and sometimes arrays.
                    count = GT_node.shape[0]*GT_node.shape[1]

                total_euc_loss += euc_loss
                total_euc_count += euc_count
                total_intersect_loss += intersect_loss
                total_intersect_count += intersect_count

            except tf.errors.OutOfRangeError:
                break
        print(confusion)
        print("eval average node L2 loss: %f" % (total_euc_loss/total_euc_count))
        print("eval average intersection cross-entropy loss: %f" % (total_intersect_loss/total_intersect_count))

    def train(self):
        for i in range(self.num_epoch):
            self.train_epoch()
            self.eval()
            self.model.save(self.sess, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=["train", "eval"], help="running train or eval.")
    parser.add_argument('--gin_config', default='', help="path to gin config file.")
    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)
    trainner = Trainner(mode=args.action)
    if args.action == "train":
        trainner.train()
    elif args.action == 'resume':
        trainner.model.load(trainner.sess, trainner.snapshot)
        trainner.train()
    else:
        trainner.model.load(trainner.sess, trainner.snapshot)
        trainner.eval()

