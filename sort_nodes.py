import numpy as np

def linear_interpolate(p1, p2, alpha):
    p = p1*(1-alpha)+p2*alpha
    return p

def sample_equdistance(samples, num_pts):
    samples = np.array(samples)
    lengths = samples[:,1:] - samples[:,:-1]
    lengths = np.linalg.norm(lengths, axis=0)
    lengths = np.cumsum(lengths)
    lengths = np.insert(lengths, 0, 0)
    segments = np.linspace(0, lengths[-1], num_pts)
    idx = np.searchsorted(lengths, segments)[1:-1]
    alpha = (segments[1:-1]-lengths[idx-1]) / (lengths[idx]-lengths[idx-1])
    subsamples = linear_interpolate(samples[:,idx-1],samples[:,idx],
                                    alpha.reshape((1,num_pts-2)) )
    subsamples = np.concatenate([samples[:,0:1], subsamples, samples[:,-1:]],axis=1)
    return subsamples

def sort_nodes(nodes):
    assert(nodes.ndim==2)
    if nodes.shape[0]==2:
        nodes = nodes.transpose()
    for _ in range(8):  # maximum 8 passes
        changes=False
        for i in range(1,nodes.shape[0]-2):
            vec1=nodes[i,:]-nodes[i-1,:]
            vec2=nodes[i+1,:]-nodes[i,:]
            vec3=nodes[i+2,:]-nodes[i+1,:]
            cos1=np.inner(vec1,vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
            cos2=np.inner(vec2,vec3) / np.linalg.norm(vec2) / np.linalg.norm(vec3)
            if cos1<-0.95 and cos2<-0.95:
                tmp=nodes[i+1,:].copy()
                nodes[i+1,:]=nodes[i,:].copy()
                nodes[i,:]=tmp
                changes=True
        if not changes:
            break
    #nodes[1:-1,:] = 0.25*(nodes[:-2,:]+2*nodes[1:-1,:]+nodes[2:,:])
    nodes = sample_equdistance(nodes.transpose(),64)
    return nodes.transpose()

if __name__=='__main__':
    nodes=np.array([[1,0],[2,0],[3,0],[1.5,0],[2.5,0],[4,0]])
    nodes = sort_nodes(nodes)
    print(nodes)
