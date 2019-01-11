import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

files = glob.glob('rollout_node_smooth_experiments/rollout_node_*/log_loss.txt')
losses = []
for f in files:
    with open(f) as fin:
        lines = fin.readlines()
    for l in lines[1::2]:
        loss = l.strip().split()
        losses.append([float(l) for l in loss])
losses = np.array(losses)

count_2x = 0
count_3x = 0
count_5x = 0
count_10x = 0

for i,loss in enumerate(losses):
    if loss[-1] / loss[0] > 2:
        count_2x += 1
    if loss[-1] / loss[0] > 3:
        count_3x += 1
    if loss[-1] / loss[0] > 5:
        count_5x += 1
    if loss[-1] / loss[0] > 10:
        count_10x += 1
        print(files[i//20])
        print('experiment', i%20+1)

print("total number of test cases: ", losses.shape[0])
print("number of test cases that 10x error: ", count_10x)
print("number of test cases that 5x error: ", count_5x)
print("number of test cases that 3x error: ", count_3x)
print("number of test cases that 2x error: ", count_2x)

bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,50,100,1000]
histos = []
for i in range(100):
    h,_=np.histogram(losses[:,i],bins)
    histos.append(h)
histos=np.array(histos)
histos=0.2*(histos[:-5,:]+histos[1:-4,:]+histos[2:-3,:]+histos[3:-2,:]+histos[4:-1,:])

fig, ax = plt.subplots()
bar_hd = ax.bar(list(range(23)),histos[0])

def animate(i):
    for idx, b in enumerate(bar_hd):
        b.set_height(histos[i][idx])
    return bar_hd

ani = animation.FuncAnimation(fig, animate, np.arange(1, 95),
                              interval=25)
#plt.show()
Writer = animation.writers['ffmpeg']
writer = Writer(fps=24)
ani.save('rollout_node_smooth_experiments/loss_distribution_smoothed.mp4', writer=writer)
plt.close()

