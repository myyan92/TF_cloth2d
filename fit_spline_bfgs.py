import numpy as np
from scipy.optimize import fmin_bfgs
from sample_spline import sample_b_spline, sample_equdistance
import matplotlib.pyplot as plt
import time

def func(x, samples, dof=4, reg=1.0):
    x = np.array(x)
    control=x[:2*dof].reshape((2,dof))
    control=np.concatenate([control[:,0:1], control, control[:,-1:]], axis=1)
    ts = x[2*dof:]
    reg_loss = reg*np.sum(np.square(ts[1:]-ts[:-1]))+10*ts[0]**2+10*(dof-1-ts[-1])**2
    ts = ts[:,np.newaxis]-np.arange(-1,dof+1).reshape((1,-1))
    weights = (   np.maximum(0,(ts-2)**3)
               -4*np.maximum(0,(ts-1)**3)
               +6*np.maximum(0, ts   **3)
               -4*np.maximum(0,(ts+1)**3)
                 +np.maximum(0,(ts+2)**3) ) / 6.0
    loss = np.sum(np.square(control.dot(weights.transpose())-samples))
    return loss + reg_loss

def func_grad(x, samples, dof=4, reg=1.0):
    x = np.array(x)
    control=x[:2*dof].reshape((2,dof))
    control=np.concatenate([control[:,0:1], control, control[:,-1:]], axis=1)
    ts = x[2*dof:]
    reg_grad = np.zeros_like(ts)
    reg_grad[1:-1] = 2*reg*(2*ts[1:-1]-ts[:-2]-ts[2:])
    reg_grad[0] = 2*reg*(ts[0]-ts[1])+20*ts[0]
    reg_grad[-1]= 2*reg*(ts[-1]-ts[-2])+20*(ts[-1]-dof+1)
    ts = ts[:,np.newaxis]-np.arange(-1,dof+1).reshape((1,-1))
    weights = (   np.maximum(0,(ts-2)**3)
               -4*np.maximum(0,(ts-1)**3)
               +6*np.maximum(0, ts   **3)
               -4*np.maximum(0,(ts+1)**3)
                 +np.maximum(0,(ts+2)**3) ) / 6.0
    residue = control.dot(weights.transpose())-samples
    control_grad = 2*np.dot(residue, weights)
    control_grad_f = control_grad[:,1:-1]
    control_grad_f[:,0] += control_grad[:,0]
    control_grad_f[:,-1] += control_grad[:,-1]
    weights_grad =  (   (ts-2)**2*(ts>2)
                     -4*(ts-1)**2*(ts>1)
                     +6*  ts**2  *(ts>0)
                     -4*(ts+1)**2*(ts>-1)
                       +(ts+2)**2*(ts>-2) ) / 2.0
    ts_grad = 2*np.sum(residue * control.dot(weights_grad.transpose()), axis=0)
    ts_grad += reg_grad
    return np.concatenate([control_grad_f.flatten(),ts_grad.flatten()], axis=0)

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
    samples, weights = sample_b_spline(knots_t)
    samples, weights = sample_equdistance(samples, weights, 128)
#    plt.plot(samples[0,:],samples[1,:], label='pred knot')

    t_np= np.linspace(0,3,128)
    start = time.time()
    initial = np.concatenate([knots.flatten(), t_np],axis=0)
    result = fmin_bfgs(func, initial, func_grad, args=(position,), gtol=1e-5, full_output=True, disp=False)
#    print("time: ", time.time()-start)
#    print("fit loss: ", result[1])
    fitted_knots = result[0][:8].reshape((2,4))
    knots_t = [fitted_knots[:,0], fitted_knots[:,0], fitted_knots[:,1],
               fitted_knots[:,2], fitted_knots[:,3], fitted_knots[:,3]]
    samples, weights = sample_b_spline(knots_t)
    samples, weights = sample_equdistance(samples, weights, 128)

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
    loss_nodes = np.sum(np.square(position-samples))/128.0
#    plt.plot(position[0,:],position[1,:], label='gt')
#    plt.plot(samples[0,:],samples[1,:], label='fitted')
#    plt.axis('equal')
#    plt.legend()
#    plt.show()
    print("l2 loss per node: ", loss_nodes)
#    print("l2 loss per knot: ", loss_knots)

if __name__ == "__main__":
    for i in range(10):
        test_fit('./pred/%04d_nodes.txt'%(i+1),
                 './pred/%04d_knots.txt'%(i+1),
                 '../gen_data/data_with_knots/%04d.txt'%(i+9000),
                 '../gen_data/data_with_knots/%04d_knots.txt'%(i+9000))
