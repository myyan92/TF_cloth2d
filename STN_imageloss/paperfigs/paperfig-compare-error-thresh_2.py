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

final_eval_consist = [results_5p[2][-1], results_10p[2][-1], results_20p[2][-1], results_40p[2][-1]]

results_5p = process_train_log('train_log_-4.61_noconsistency.txt')
results_10p = process_train_log('train_log_-4.55_noconsistency.txt')
results_20p = process_train_log('train_log_-4.47_noconsistency.txt')
results_40p = process_train_log('train_log_-4.39_noconsistency.txt')

final_eval_noconsist = [results_5p[2][-1], results_10p[2][-1], results_20p[2][-1], results_40p[2][-1]]

plt.plot([5,10,20,40], final_eval_consist, c='C0', marker='.', markersize=10, label='with consistency')
plt.plot([0,50], [-4.867, -4.867], c='C0', linestyle='--')
plt.plot([5,10,20,40], final_eval_noconsist, c='C1', marker='.', markersize=10, label='without consistency')
plt.plot([0,50], [-4.794, -4.794], c='C1', linestyle='--')
plt.xlim([0,50])
plt.ylim([-5.1, -4.6])
plt.legend()
plt.xlabel('Error threshold quantile')
plt.ylabel('Image loss')
plt.tight_layout()
plt.show()

