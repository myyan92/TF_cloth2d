import matplotlib
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import numpy as np

def process_train_log(file_name):
    with open(file_name) as f:
        lines = f.readlines()

    train_loss=lines[0::3]
    train_iter=lines[1::3]
    eval_loss=lines[2::3]
    train_loss=[t.strip().split()[-3] for t in train_loss]
    train_loss=[float(t) for t in train_loss]
    train_iter=[t.strip().split()[-2] for t in train_iter]
    train_iter=[int(t) for t in train_iter]
    eval_loss=[t.strip().split()[-3] for t in eval_loss]
    eval_loss=[float(t) for t in eval_loss]
    train_iter=np.cumsum(train_iter)
    return train_iter, train_loss, eval_loss

results_5p = process_train_log('train_log_-4.65.txt')
results_10p = process_train_log('train_log_-4.55.txt')
results_20p = process_train_log('train_log_-4.47.txt')
results_40p = process_train_log('train_log_-4.39.txt')

colors = ['C0', 'C1', 'C2', 'C3']
data = [results_5p, results_10p, results_20p, results_40p]
labels = ['threshold 5%', 'threshold 10%', 'threshold 20%', 'threshold 40%']
for c,d, l in zip(colors,data,labels):
    plt.plot(d[0]/10000.0, d[1], c=c, linestyle='--')
    plt.plot(d[0]/10000.0, d[2], c=c, linestyle='-', label=l)
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Image loss')
plt.tight_layout()
plt.show()

