import numpy as np
import sys 
import math
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument("-bias", type=str, help="bias value to analyse")
args = parser.parse_args()

save = "figure{}".format(args.bias)
# save = None

data = np.genfromtxt('theta' + args.bias + '.txt', delimiter=' ');
size = len(data)

# fig = plt.figure()
# ax  = fig.add_subplot(111, projection='3d')

N = 240 
X = 20
Z = 12

data = -1.0 * data
data = data.reshape((-1,20,12))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

data_txz = np.zeros(data.shape)
data_txz[:, ::2, :] = data[:, 0:10, :]
data_txz[:, 1::2, :] = data[:, 10:20, :]
data = data_txz

# Compute the mean for each site in time
m_xz = np.mean(data, axis = 0)
print m_xz

mean_row = np.mean(m_xz, axis=(1))
std_row = np.std(m_xz, axis=(1))

max_val = max([max(row) for row in m_xz])
min_val = min([min(row) for row in m_xz])

plt.figure()
plt.imshow(m_xz, aspect=1.0, cmap="seismic", origin="lower", vmin=min_val, vmax=max_val, interpolation="none")
plt.show()

# strs = ["{:4.2f} +/- {:4.2f}".format(m,s) for m,s in zip(mean_row, std_row)]
# for row in range(X):
#      print str(row).zfill(2) + " " + strs[row]
# 
# pnt = ""
# for row in range(X):
#     strs = ["{:+4.2f}".format(x).zfill(6) for x in m_xz[row, :]]
#     pnt = pnt + str(row).zfill(2) + " " + " ".join(strs)
#     pnt = pnt + "\n"
# print pnt

