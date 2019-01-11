import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from PIL import Image
import curve_pb2
from model_VGG import Model
import rollout_physbam
import os, sys

data_dir = "/scr-ssd/mengyuan/gen_data/data/"

eval_snapshot = "./b-spline_data_pred_node/model-29"
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)
model = Model('vgg16_weights.npz')  #load pretrained weights
model.build()
sess.run(tf.global_variables_initializer())
model.load(sess, eval_snapshot)

def save_animation(pred_data, gt_data, filename=None):
    fig, ax = plt.subplots()
    line_pred, = ax.plot(pred_data[0][:,0], pred_data[0][:,1])
    line_gt, = ax.plot(gt_data[0][:,0], gt_data[0][:,1])
    plt.axis('equal')
    plt.axis([-6,6,-6,6])
    def animate(i):
        line_pred.set_xdata(pred_data[i][:,0])
        line_pred.set_ydata(pred_data[i][:,1])
        line_gt.set_xdata(gt_data[i][:,0])
        line_gt.set_ydata(gt_data[i][:,1])
        return line_pred, line_gt

    ani = animation.FuncAnimation(fig, animate, np.arange(1, 100),
                                  interval=25)
    if filename is None:
        plt.show()
    else:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=24)
        ani.save(filename, writer=writer)
    plt.close()


def experiment(test_idx, option='diagnose'):
    im=Image.open(data_dir+"%04d.png"%(test_idx))
    im=im.resize((224,224), resample=Image.LANCZOS)
    im=np.array(im)[:,:,0:3]
    with open(data_dir+"%04d.txt"%(test_idx)) as f:
        position = f.readlines()
    position = [p.strip().split() for p in position]
    position = [[float(pp) for pp in p] for p in position]
    position = np.array(position).transpose()

    samples = model.predict_single(sess, im)
    # for _ in range(5)
    #     samples[:,1:-1] = (samples[:,2:]+2*samples[:,1:-1]+samples[:,:-2])/4.0

    if option == "animation":
        experiments, losses, actions = rollout_physbam.rollout_pair(samples, position, 1)
        sim_pred_data, sim_gt_data = experiments[0]
        save_animation(sim_pred_data, sim_gt_data)
    else:
        experiments, losses, actions = rollout_physbam.rollout_pair(samples, position, 20)
        output_dir = 'rollout_node_experiments/rollout_node_%d'%(test_idx)
        os.mkdir(output_dir)
        with open(os.path.join(output_dir, 'log_loss.txt'), 'w') as f:
            for l,a in zip(losses, actions):
                f.write('node %d, action %f %f\n'%(a[0], a[1][0], a[1][1]))
                f.write(" ".join(str(ll) for ll in l)+'\n')
        for l in losses:
            plt.plot(l)
        plt.savefig(os.path.join(output_dir,'diagnose.png'))
        plt.close()
        for i,exp in enumerate(experiments):
            if losses[i][-1] > 10*losses[i][0]:
                save_animation(exp[0], exp[1], os.path.join(output_dir, 'exp_%d.mp4'%(i+1)))

#option = "animation"
#option = "diagnose"

if __name__ == "__main__":
    for i in range(9000,9200):
        experiment(i)
