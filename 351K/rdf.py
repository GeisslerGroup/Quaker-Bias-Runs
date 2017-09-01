import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=28)
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)

n_atoms = 22116
n_lines = n_atoms + 2

x = []
y = []
z = []
with open("solvent-ord-big.txt") as f:
# with open("solvent-0.7500.txt") as f:
    for n, l in enumerate(f):
        l_arr = l.split()
#         if (n > 22116):
#             exit()
#         else:
        x.append(float(l_arr[1]))
        y.append(float(l_arr[2]))
        z.append(float(l_arr[3]))

x = np.array(x)
y = np.array(y)
z = np.array(z)
# print min(y)

ybins = np.arange(10, 100, 0.5)
newbins = ybins[1:] * 0.5 + ybins[:-1] * 0.5
hist = np.zeros(len(newbins))
for i in range(len(ybins) - 1):
    indices = np.where( (y >= ybins[i]) * (y < ybins[i+1]) )
    hist[i] = len(y[indices])

Lx = 82.4293 * 0.1
Lz = 81.004 * 0.1
dy = (ybins[1] - ybins[0]) * 0.1
newhist = hist/np.sum(hist)
newhist = newhist / (Lx * Lz * dy)
plt.plot(newbins, newhist, 'bo')
plt.plot(newbins, newhist, color="#2020CC", linewidth=4, alpha=0.6)
plt.vlines(35, 0, 0.02, color='k')
plt.vlines(50, 0, 0.02, color='k')
plt.show()


