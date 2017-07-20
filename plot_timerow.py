import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="")
parser.add_argument("-bias", type=str, help="bias value to analyse")
parser.add_argument("-clean", action='store_true', help="whether to clean data or not")
parser.add_argument("-row", choices=["x","z"], help="which direction to average out")
args = parser.parse_args()

data = np.genfromtxt('theta' + args.bias + '.txt', delimiter=' ')
data = -1.0 * data
data = data.reshape((-1,20,12))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

data_txz = np.zeros(data.shape)
data_txz[:, ::2, :] = data[:, 0:10, :]
data_txz[:, 1::2, :] = data[:, 10:20, :]
data = data_txz

if args.row == "x":
    mean_tz = np.mean(data, axis=1)
    max_val = max([max(row) for row in mean_tz])
    print max_val
    min_val = min([min(row) for row in mean_tz])
#     min_val = 0.25 
    mid = (max_val - min_val) * 0.5
#     mid = 0.65 
#     mid = 42.5 * np.pi / 180.0
    print mid
    print min_val
    if args.clean:
        mean_tz[mean_tz < (min_val + 0.1)] = mid	# set disordered angles to mid_val
    mean_tz = (mean_tz - min_val) / (max_val - min_val)
    mean_tz = np.transpose(mean_tz)
    
    plt.figure()
#     plt.imshow(mean_tz, aspect=1000, cmap="seismic", origin="lower", vmin=min_val, vmax=max_val, interpolation="none")
    plt.imshow(mean_tz, aspect=len(data), cmap="seismic", origin="lower", vmin=min_val, vmax=max_val, interpolation="none")
#     plt.imshow(mean_tz, aspect=1000, cmap="seismic", origin="lower", vmin=30.0, vmax=60.0)
    plt.yticks(np.arange(0, 13, 1))
    plt.ylim(-0.5,11.5)
    for i in np.arange(-0.5,12,0.5):
#         plt.hlines(i, 0, 10000, linestyle='solid', linewidth=2)
        plt.hlines(i, 0, len(data), linestyle='solid', linewidth=2)
    plt.colorbar()
    plt.show()

if args.row == "z":
    mean_tx = np.mean(data, axis=2)
    max_val = max([max(row) for row in mean_tx])
    print max_val
    min_val = min([min(row) for row in mean_tx])
    mid = (max_val - min_val) * 0.5
    print min_val
    if args.clean:
        mean_tx[mean_tx < (min_val + 0.1)*np.pi/180.0] = mid*np.pi/180.0	# set disordered angles to mid_val
    mean_tx = np.transpose(mean_tx)
    
    plt.figure()
#     plt.subplot(2,1,1)
#     plt.imshow(mean_tx, aspect=len(data)/20, cmap="plasma", origin="lower", vmin=30*np.pi/180, vmax=53*np.pi/180, interpolation="none")
    plt.imshow(mean_tx, aspect=len(data)/20, cmap="plasma", origin="lower", vmin=min_val, vmax=max_val, interpolation="none")
    plt.yticks(np.arange(0, 21, 1))
    # plt.ylim(-0.5,20.5)
    for i in np.arange(0,20,1):
        plt.hlines(i-0.5, 0, len(data), linestyle='solid', linewidth=2)
    plt.colorbar()
    plt.xlabel(r'$t$', fontsize=32)
    plt.ylabel(r'$\langle\theta_z\rangle_z$', fontsize=32)
    plt.xticks(fontsize=28, fontweight='bold')
    plt.yticks(fontsize=28, fontweight='bold')
#     plt.subplot(2,1,2)
#     mean_t = np.mean(mean_tx,axis=0)
#     plt.plot(mean_t)
#     plt.hlines(0.89, 0, len(data))
    plt.show()

    sub_mean = mean_tx[:,0:100]
    plt.clf()
    plt.imshow(sub_mean, aspect=5, cmap="plasma", origin="lower", vmin=30*np.pi/180, vmax=53*np.pi/180, interpolation="none")
    plt.yticks(np.arange(0, 21, 1))
    # plt.ylim(-0.5,20.5)
    for i in np.arange(0,20,1):
        plt.hlines(i-0.5, 0, 100, linestyle='solid', linewidth=2)
    plt.colorbar()
    plt.xlabel(r'$t$', fontsize=32)
    plt.ylabel(r'$\langle\theta_z\rangle_z$', fontsize=32)
    plt.xticks(fontsize=28, fontweight='bold')
    plt.yticks(fontsize=28, fontweight='bold')
    plt.show()

#     plt.figure(3)
#     plt.subplot(2,1,1)
#     plt.imshow(mean_tx, aspect=len(data)/20, cmap="plasma", origin="lower", vmin=5*np.pi/180, vmax=53*np.pi/180, interpolation="none")
#     plt.yticks(np.arange(0, 21, 1))
#     # plt.ylim(-0.5,20.5)
#     for i in np.arange(0,20,1):
#         plt.hlines(i-0.5, 0, len(data), linestyle='solid', linewidth=2)
#     plt.colorbar()
#     plt.xlabel(r'$t$', fontsize=32)
#     plt.ylabel(r'$\langle\theta_z\rangle_z$', fontsize=32)
#     plt.xticks(fontsize=28, fontweight='bold')
#     plt.yticks(fontsize=28, fontweight='bold')
#     plt.subplot(2,1,2)
#     mean_t = np.mean(mean_tx,axis=1)
#     plt.plot(mean_t)
#     plt.show()
# 
#     sub_mean = mean_tx[:,0:100]
#     plt.clf()
#     plt.imshow(sub_mean, aspect=5, cmap="plasma", origin="lower", vmin=5*np.pi/180, vmax=53*np.pi/180, interpolation="none")
#     plt.yticks(np.arange(0, 21, 1))
#     # plt.ylim(-0.5,20.5)
#     for i in np.arange(0,20,1):
#         plt.hlines(i-0.5, 0, 100, linestyle='solid', linewidth=2)
#     plt.colorbar()
#     plt.xlabel(r'$t$', fontsize=32)
#     plt.ylabel(r'$\langle\theta_z\rangle_z$', fontsize=32)
#     plt.xticks(fontsize=28, fontweight='bold')
#     plt.yticks(fontsize=28, fontweight='bold')
#     plt.show()

