import numpy as np
import logging
import sys
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bias", type=str, help="bias to analyze")
parser.add_argument("-temp", type=str, help="temperature to analyze")
parser.add_argument("-i", action="store_true", help="run interactively")
parser.add_argument("-m_corr", action="store_true", help="use mean correction")
parser.add_argument("-func", choices=["sin", "cos", None], default=None, help="Function to apply to data")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
bias = args.bias
if args.i:
    save = None
else:
    if args.func == "sin":
        save = "fig_sin{}".format(bias)
    elif args.func == "cos":
        save = "figure{}".format(bias)
if not save is None:
    import matplotlib
    matplotlib.use("Agg")

def safeplot(suffix=""):
    if not save is None:
        plt.savefig(save + suffix + ".png")
        plt.clf()
    else:
        plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

data = np.genfromtxt('theta-' + args.temp + '.' + bias + '.txt', delimiter=' ')
size = len(data)

N = 240
T = 2000
X = 20
Z = 12

data_end = data[size - T*N:]
# print size, len(data_end), len(data_end)/N
data_all_txz = np.zeros((T, X, Z))
for t in xrange(T):
    time_off = t * N
    for col in xrange(10):
        off = time_off + col * 12
        data_all_txz[t, 2*col + 0, :] = data_end[off +   0 : off +       Z ]
        data_all_txz[t, 2*col + 1, :] = data_end[off + N/2 : off + N/2 + Z ]

# print data_all_txz.shape
# func_txz  = np.cos(data_all_txz)
if args.func is None:
    func_txz = data_all_txz
elif args.func == "sin":
    func_txz = np.sin(data_all_txz * np.pi / 180)
elif args.func == "cos":
    func_txz = np.cos(data_all_txz * np.pi / 180)

mean      = np.mean(func_txz[:,   :,:], axis=(0,1,2))
mean_even = np.mean(func_txz[:,0::2,:], axis=(0,1,2))
mean_oddz = np.mean(func_txz[:,1::2,:], axis=(0,1,2))
print mean, mean_even, mean_oddz

if args.m_corr:
    func_txz[:, 0::2, :] -= mean_even
    func_txz[:, 1::2, :] -= mean_oddz
else:
    func_txz[:, :, :] -= mean

if args.i:
    plt.hist(func_txz.flatten())
    safeplot()

plt.scatter(func_txz[0:500,:,0:5], func_txz[0:500,:,1:6], alpha = .2)
plt.title("Z-neighbors")
safeplot("z_pdist")
plt.scatter(func_txz[0:500,0:5,:], func_txz[0:500,1:6,:], alpha = .2)
plt.title("X-neighbors")
safeplot("x_pdist")

# Prepare to subset the data and plot in 3d for 
# each subset
all_corr_xz = []
samplez = 10
DT = T / samplez
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
for sample in xrange(samplez):
    To = DT*sample
    Tf = DT*(sample+1)
    sub_txz = func_txz[To:Tf, :, :]
#    print sub_txz.shape
    
    cov_xz = np.zeros((X/2, Z/2))
    for t in xrange(DT):
       for x in xrange(X/2):
            for z in xrange(Z/2):
                cov_xz += sub_txz[t, x, z] * sub_txz[t, x : x + X/2, z : z + Z/2]
    
    cov_xz /= (DT * X/2 * Z/2 )
#    logging.debug("variance is {}".format(cov_xz[0, 0]))
    corr_xz = cov_xz / cov_xz[0,0]

    
#    print corr_xz 
    x = range(X/2)
    z = range(Z/2)
    xv, zv = np.meshgrid(x, z)
    
#    print xv.shape
#    print zv.shape
#    print corr_xz.shape
    
    all_corr_xz.append(corr_xz)
    ax.plot_surface(xv, zv, corr_xz.T)

all_corr_xz = np.array(all_corr_xz)
# print all_corr_xz.shape
m_corr_xz = np.mean(all_corr_xz, axis=0)
d_corr_xz = np.std (all_corr_xz, axis=0) / np.sqrt(samplez)
safeplot("3d")
# print m_corr_xz[:,0]
# print m_corr_xz[0,:]
# for i in range(X/2):
#   for j in range(Z/2):
#     print i,  j,  m_corr_xz[i,j],  d_corr_xz[i,j]

# correlation plots with first data point removed
plt.errorbar(range(1, X/2), m_corr_xz[1:,0], d_corr_xz[1:,0], c='b')
plt.hlines(0, 1, X/2, linestyles="dashed")
plt.xlim([0.9,X/2])
# plt.ylim([-0.1,0.1])
# safeplot("x")
plt.errorbar(range(1, Z/2), m_corr_xz[0,1:], d_corr_xz[0,1:], c='g')
plt.hlines(0, 1, Z/2, linestyles="dashed")
plt.xlabel('x or z')
plt.ylabel('G(x, z)')
plt.legend(["X", "Z"])
safeplot("xz")


# plt.errorbar(range(X/2), m_corr_xz[:,0], d_corr_xz[:,0], c='b')
# plt.hlines(0, 0, X/2, linestyles="dashed")
# plt.xlim([0,X/2])
# #safeplot("x")
# plt.errorbar(range(Z/2), m_corr_xz[0,:], d_corr_xz[0,:], c='g')
#plt.hlines(0, 0, Z/2, linestyles="dashed")
#plt.xlim([0,Z/2])
#plt.xlabel('x or z')
#plt.ylabel('G(x, z)')
#safeplot("xz")
 
