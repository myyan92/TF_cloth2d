import numpy as np
import matplotlib.pyplot as plt
import time

def normalize_shape(knots):
    if knots.shape[0] != 2 and knots.shape[0] != 3:
        knots = knots.transpose()
    if knots.shape[0] != 2 and knots.shape[0] != 3:
        raise ValueError("cannot handle knots shape")
    return knots

def sample_cubic_spline(knots):
    if len(knots)<4:
        return []
    weights = []
    for i in range(2, len(knots)-1):
        dist = np.sqrt((knots[i][0]-knots[i-1][0])**2+(knots[i][1]-knots[i-1][1])**2)
        num_pts = int(dist*500)
        ts = np.linspace(0, 1, num_pts, endpoint=False)
        ws = np.zeros((num_pts, len(knots)))
        ws[:,i-2] = (-0.5*ts+ts**2-0.5*ts**3)
        ws[:,i-1] = (1-2.5*ts**2+1.5*ts**3)
        ws[:,i]   = (0.5*ts+2*ts**2-1.5*ts**3)
        ws[:,i+1] = (-0.5*ts**2+0.5*ts**3)
        weights.append(ws)
    weights = np.concatenate(weights, axis=0)
    samples = np.dot(weights, np.array(knots))
    return samples.transpose()


def sample_b_spline(knots):
    if len(knots)<4:
        return []
    weights = []

    def basis(x):
        y =np.maximum(0, (x-2)**3)
        y-=4*np.maximum(0,(x-1)**3)
        y+=6*np.maximum(0,x**3)
        y-=4*np.maximum(0,(x+1)**3)
        y+=np.maximum(0, (x+2)**3)
        return y/6

    for i in range(2, len(knots)-1):
        dist = np.sqrt((knots[i][0]-knots[i-1][0])**2+(knots[i][1]-knots[i-1][1])**2)
        num_pts = int(dist*500)
        ts = np.linspace(0, 1, num_pts)
        ws = np.zeros((num_pts, len(knots)))
        ws[:,i-2] = basis(ts+1)
        ws[:,i-1] = basis(ts)
        ws[:,i]   = basis(ts-1)
        ws[:,i+1] = basis(ts-2)
        weights.append(ws)

    weights = np.concatenate(weights, axis=0)
    samples = np.dot(weights, np.array(knots))
    return samples.transpose(), weights.transpose()

def linear_interpolate(p1, p2, w1, w2, alpha):
    p = p1*(1-alpha)+p2*alpha
    w = w1*(1-alpha)+w2*alpha
    return p, w

def sample_equdistance(samples, weights, num_pts):
    samples = np.array(samples)
    samples = normalize_shape(samples)
    lengths = samples[:,1:] - samples[:,:-1]
    lengths = np.linalg.norm(lengths, axis=0)
    lengths = np.cumsum(lengths)
    lengths = np.insert(lengths, 0, 0)
    #print("total curve length: ", lengths[-1])
    segments = np.linspace(0, lengths[-1], num_pts)
    idx = np.searchsorted(lengths, segments)[1:-1]
    alpha = (segments[1:-1]-lengths[idx-1]) / (lengths[idx]-lengths[idx-1])
    subsamples, subweights = linear_interpolate(samples[:,idx-1],samples[:,idx],
                                    weights[:,idx-1],weights[:,idx], alpha.reshape((1,num_pts-2)) )
    subsamples = np.concatenate([samples[:,0:1], subsamples, samples[:,-1:]],axis=1)
    subweights = np.concatenate([weights[:,0:1], subweights, weights[:,-1:]],axis=1)
    return subsamples, subweights

if __name__ == "__main__":
    knots = [(-5,-5),(-5,-5),(-5,5),(5,-5),(5,5),(5,5)]
    start = time.time()
    samples, weights = sample_b_spline(knots)
    print(time.time()-start)
    start = time.time()
    subsamples, weights = sample_equdistance(samples, weights, 128)
    print(time.time()-start)
    print(subsamples)
    plt.plot(samples[0], samples[1])
    plt.scatter(subsamples[0], subsamples[1], s=1, c='r')
    plt.axis('equal')
    plt.show()
