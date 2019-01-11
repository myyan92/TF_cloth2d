from sample_spline_TF import sample_b_spline, sample_equdistance
from physbam_python.rollout_physbam import rollout_single
from multiprocessing import Pool
import numpy as np
import gin

def node_loss(pred_node=None, pred_knot=None, GT_node=None, GT_knot=None):
    if pred_knot is not None and GT_node is not None:
        samples, weights = sample_b_spline(pred_knot)
        samples, weights = sample_equdistance(samples, weights, GT_node.shape[1])
        samples = samples.transpose((0, 2, 1))
        loss = np.sum(np.square(samples-GT_node))/2.0
        grad = samples-GT_node
        knots_grad = [np.dot(weights[i,:,:], grad[i,:,:]) for i in range(samples.shape[0])]
        knots_grad = np.concatenate([g[np.newaxis,:,:] for g in knots_grad], axis=0)
    else:
        raise NotImplementedError('node_loss must have pred_knot and GT_node arguments')
    return loss, knots_grad

def RL_knot_l2loss(pred_node=None, pred_knot=None, GT_node=None, GT_knot=None):
    if pred_knot is not None and GT_knot is not None:
        cost = np.sum(np.square(pred_knot-GT_knot), axis=(1,2)) / 2.0
    else:
        raise NotImplementedError('RL_knot_l2loss must have pred_knot and GT_knot arguments')
    return cost

def RL_knot_nodeloss(pred_node=None, pred_knot=None, GT_node=None, GT_knot=None):
    if pred_knot is not None and GT_node is not None:
        samples, weights = sample_b_spline(pred_knot)
        samples, weights = sample_equdistance(samples, weights, GT_node.shape[1])
        samples = samples.transpose((0, 2, 1))
        cost = np.sum(np.square(samples-GT_node), axis=(1,2)) / 2.0
    else:
        raise NotImplementedError('RL_knot_nodeloss must have pred_knot and GT_node arguments')
    return cost

@gin.configurable
def RL_knot_simloss(pred_node=None, pred_knot=None, GT_node=None, GT_knot=None, physbam_args=""):
    if pred_knot is not None and GT_node is not None:
        samples, weights = sample_b_spline(pred_knot)
        samples, weights = sample_equdistance(samples, weights, GT_node.shape[1])

        args = []
        for i in range(pred_knot.shape[0]):
            action_node = np.random.randint(10,GT_node.shape[1]-10)
            ang = np.random.uniform(0,6.28)
            action = [np.sin(ang)*0.02, np.cos(ang)*0.02]
            args.append((samples[i,:,:], action_node, action, 1, physbam_args))
        pool = Pool(8)
        pred_sim_data = pool.starmap(rollout_single, args)
        args = [(GT_node[i,:,:],a[1],a[2],a[3], a[4]) for i,a in enumerate(args)]
        gt_sim_data = pool.starmap(rollout_single, args)
        losses = [ np.sum(np.square(pred-gt)) for pred, gt in zip(pred_sim_data, gt_sim_data)]
        pool.close()
        pool.join()

        cost = np.array(losses)
    else:
        raise NotImplementedError('RL_knot_simloss must have pred_knot and GT_node arguments')
    return cost

def RL_node_l2loss(pred_node=None, pred_knot=None, GT_node=None, GT_knot=None):
    if pred_node is not None and GT_node is not None:
        distance = np.sum(np.square(pred_node-GT_node), axis=(2)) / 2.0
        cost = np.copy(distance)
        cost[:,1:] += distance[:,:-1]
        cost[:,:-1] += distance[:,1:]
        cost[:,2:] += distance[:,:-2]
        cost[:,:-2] += distance[:,2:]
        cost_sum = np.sum(distance, axis=1)
    else:
        raise NotImplementedError('RL_node_l2loss must have pred_node and GT_node arguments')
    return cost, cost_sum

@gin.configurable
def RL_node_simloss(pred_node=None, pred_knot=None, GT_node=None, GT_knot=None, physbam_args=""):
    if pred_node is not None and GT_node is not None:
        args = []
        action_node = np.random.randint(10,GT_node.shape[1]-10)
        ang = np.random.uniform(0,6.28)
        action = [np.sin(ang)*0.05, np.cos(ang)*0.05]
        for i in range(pred_node.shape[0]):
            args.append((pred_node[i,:,:], action_node, action, 1, physbam_args))
        pool = Pool(8)
        pred_sim_data = pool.starmap(rollout_single, args)
        args = [(GT_node[i,:,:],a[1],a[2],a[3],a[4]) for i,a in enumerate(args)]
        gt_sim_data = pool.starmap(rollout_single, args)
        experiments = [(pred,gt) for pred, gt in zip(pred_sim_data, gt_sim_data)]
        distance = [ np.sum(np.square(pred-gt), axis=(1)) / 2.0
                     for pred, gt in zip(pred_sim_data, gt_sim_data)]
        pool.close()
        pool.join()

        distance = np.array(distance)
        cost = np.copy(distance)
        cost[:,1:] += distance[:,:-1]
        cost[:,:-1] += distance[:,1:]
        cost[:,2:] += distance[:,:-2]
        cost[:,:-2] += distance[:,2:]
        cost_sum = np.sum(distance, axis=1)
    else:
        raise NotImplementedError('RL_node_simloss must have pred_node and GT_node arguments')
    return cost, cost_sum

