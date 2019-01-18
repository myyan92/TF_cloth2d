import numpy as np
import matplotlib.pyplot as plt
import time

def normalize_shape(knots):
    if knots.shape[1] != 2 and knots.shape[1] != 3:
        knots = knots.transpose((0, 2, 1))
    if knots.shape[1] != 2 and knots.shape[1] != 3:
        raise ValueError("cannot handle knots shape")
    return knots

def sample_cubic_spline(knots):
    knots = normalize_shape(knots)
    knots = np.concatenate([knots[:,:,0:1], knots, knots[:,:,-1:]], axis=2)
    weights = []
    for i in range(2, knots.shape[-1]-1):
        num_pts = 4000
        ts = np.linspace(0, 1, num_pts, endpoint=False)
        ws = np.zeros((num_pts, knots.shape[-1]))
        ws[:,i-2] = (-0.5*ts+ts**2-0.5*ts**3)
        ws[:,i-1] = (1-2.5*ts**2+1.5*ts**3)
        ws[:,i]   = (0.5*ts+2*ts**2-1.5*ts**3)
        ws[:,i+1] = (-0.5*ts**2+0.5*ts**3)
        weights.append(ws)
    weights = np.concatenate(weights, axis=0).transpose()
    samples = np.dot(np.array(knots), weights)
    weights_t = weights[1:-1,:]
    weights_t[0,:] += weights[0,:]
    weights_t[-1,:] += weights[-1,:]
    return samples, weights_t


def sample_b_spline(knots):
    knots = normalize_shape(knots)
    knots = np.concatenate([knots[:,:,0:1], knots, knots[:,:,-1:]], axis=2)
    weights = []

    def basis(x):
        y =np.maximum(0, (x-2)**3)
        y-=4*np.maximum(0,(x-1)**3)
        y+=6*np.maximum(0,x**3)
        y-=4*np.maximum(0,(x+1)**3)
        y+=np.maximum(0, (x+2)**3)
        return y/6

    for i in range(2, knots.shape[-1]-1):
        num_pts = 4000
        ts = np.linspace(0, 1, num_pts, endpoint=False)
        ws = np.zeros((num_pts, knots.shape[-1]))
        ws[:,i-2] = basis(ts+1)
        ws[:,i-1] = basis(ts)
        ws[:,i]   = basis(ts-1)
        ws[:,i+1] = basis(ts-2)
        weights.append(ws)

    weights = np.concatenate(weights, axis=0).transpose()
    samples = np.dot(np.array(knots), weights)
    weights_t = weights[1:-1,:]
    weights_t[0,:] += weights[0,:]
    weights_t[-1,:] += weights[-1,:]
    return samples, weights_t

def sample_equdistance(samples, weights, num_pts):
    samples = normalize_shape(samples)
    samples = np.array(samples)
    lengths = samples[:,:,1:] - samples[:,:,:-1]
    lengths = np.linalg.norm(lengths, axis=1)
    lengths = np.cumsum(lengths, axis=1)
    lengths = np.insert(lengths, 0, 0, axis=1)
    segments = [np.linspace(0, l[-1], num_pts) for l in lengths]
    idx = [np.searchsorted(l,s)[1:-1] for l,s in zip(lengths, segments)]
    alpha = [(s[1:-1]-l[ii-1])/(l[ii]-l[ii-1]) for s,l,ii in zip(segments, lengths, idx)]
    p1 = [s[:,ii-1] for s,ii in zip(samples, idx)]
    p2 = [s[:,ii] for s,ii in zip(samples, idx)]
    w1 = [weights[:,ii-1] for ii in idx]
    w2 = [weights[:,ii] for ii in idx]
    alpha = np.concatenate([a.reshape((1,1,-1)) for a in alpha], axis=0)
    p1 = np.concatenate([p[np.newaxis,:,:] for p in p1], axis=0)
    p2 = np.concatenate([p[np.newaxis,:,:] for p in p2], axis=0)
    w1 = np.concatenate([w[np.newaxis,:,:] for w in w1], axis=0)
    w2 = np.concatenate([w[np.newaxis,:,:] for w in w2], axis=0)
    subsamples = p1*(1-alpha)+p2*alpha
    subweights = w1*(1-alpha)+w2*alpha
    subsamples = np.concatenate([samples[:,:,0:1], subsamples, samples[:,:,-1:]],axis=2)
    subweights = np.concatenate([np.tile(weights[:,0:1],(samples.shape[0],1,1)), subweights,
                                 np.tile(weights[:,-1:],(samples.shape[0],1,1))], axis=2)
    return subsamples, subweights

if __name__ == "__main__":
    knots1 = np.array([(-5,-5),(-5,5),(5,-5),(5,5)])
    knots2 = np.array([(-5,-5),(-5,5),(5,-5),(5,5)])
    knots = np.array([knots1.transpose(), knots2.transpose()])
    start = time.time()
    samples, weights = sample_b_spline(knots)
    print(time.time()-start)
    start = time.time()
    subsamples, weights = sample_equdistance(samples, weights, 128)
    print(time.time()-start)
    print(subsamples)
    plt.plot(samples[0,0,:], samples[0,1,:])
    plt.scatter(subsamples[0,0,:], subsamples[0,1,:], s=1, c='r')
    plt.axis('equal')
    plt.show()
    plt.plot(samples[1,0,:], samples[1,1,:])
    plt.scatter(subsamples[1,0,:], subsamples[1,1,:], s=1, c='r')
    plt.axis('equal')
    plt.show()

