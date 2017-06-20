import pymbar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

namelist_1 = np.arange(-0.8500, -0.0240, 0.0250)
namelist_2 = np.arange(0.0, 0.1040, 0.0250)
namelist = np.concatenate((namelist_1, namelist_2))
# namelist = np.concatenate((namelist_1, namelist_2, namelist_3))

repex = np.genfromtxt('repex-350K.txt', delimiter=' ')

# initialise list of spring constants
k_list = np.ones(len(namelist)) * 15000.0

N_sims = len(namelist)
T = 5000
theta_kn = np.zeros((N_sims, T))
theta_x_kn = np.zeros((N_sims, T))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for k, biasval in enumerate(namelist):
#     if ("{:1.4f}".format(biasval) == "-0.7500"):
#         continue
    data = np.genfromtxt('theta{:1.4f}.txt'.format(biasval))
    data = data.reshape((-1, 240))
    data_t = np.mean(data, axis=1)
    theta_kn[k, :] = data_t[:]

for k, biasval in enumerate(namelist):
#     if ("{:1.4f}".format(biasval) == "-0.7500"):
#         continue
    data_x = np.genfromtxt('theta_x{:1.4f}.txt'.format(biasval))
    data_x = data_x.reshape((-1, 240))
    data_xt = np.mean(data_x, axis=1)
    theta_x_kn[k, :] = data_xt[:]

theta_kn = theta_kn[:,::5]
theta_x_kn = theta_x_kn[:,::5]
T = theta_kn.shape[1]
bias_kln = namelist.astype(float)[np.newaxis, :, np.newaxis]
k_list_kln = k_list[np.newaxis, :, np.newaxis]

kB = 1.3806503 * 6.0221415 / 4184.0
beta = 1/(kB * 350.0)
dtheta_kln = theta_kn[:, np.newaxis, :] - bias_kln
u_kln = 0.5 * k_list_kln * np.square(dtheta_kln) * beta
u_lkn = np.swapaxes(u_kln,0,1)
# u_lkn.shape
u_kn = np.reshape(u_lkn, (N_sims, T*N_sims))
theta_n = np.reshape(theta_kn, T*N_sims)
N_k = np.ones(N_sims) * T
# my_mbar.getFreeEnergyDifferences()
# u_n = u_kn[0, :]

N_k = np.zeros([N_sims], np.int32)
N_k[:] = T
mask_kn = np.zeros([N_sims,T], dtype=np.bool)
for k in range(0,N_sims):
    mask_kn[k,0:N_k[k]] = True
indices = np.where(mask_kn)

nbins_per_angle = 500
angle_min = -np.pi
angle_max = +np.pi
dx = (angle_max - angle_min) / float(nbins_per_angle)
bin_kn = np.zeros([N_sims, T], np.int16)
nbins = 0
bin_counts = list()
bin_centers = list()            # bin_centers[i] is a theta_z value that gives the center of bin i
nanpos = []
n_empty_bins = 0

for i in range(nbins_per_angle):
    for j in range(nbins_per_angle):
        val = angle_min + dx * (i + 0.5)
        val_x = angle_min + dx * (j + 0.5)
        # Determine which configurations lie in this bin.
        in_bin = (val-dx/2 <= theta_kn[indices]) & (theta_kn[indices] < val+dx/2) & (val_x-dx/2 <= theta_x_kn[indices]) & (theta_x_kn[indices] < val_x+dx/2)
  
        # Count number of configurations in this bin.
        bin_count = in_bin.sum()
  
        # Generate list of indices in bin.
        indices_in_bin = (indices[0][in_bin], indices[1][in_bin])
  
        if (bin_count > 0):
            bin_centers.append( (val, val_x) )
            bin_counts.append( bin_count )
  
            # assign these conformations to the bin index
            bin_kn[indices_in_bin] = nbins
  
            # increment number of bins
            nbins += 1
        else:
            nanpos.append((i, j))
            n_empty_bins = n_empty_bins + 1

# do MBAR and plot PMF as function of (theta_z, theta_x)

my_mbar = pymbar.MBAR(u_kn, N_k)
u_n = np.zeros(T*N_sims)
[f_i, df_i] = my_mbar.computePMF(u_n, bin_kn, nbins)
bin_centers = np.array(bin_centers)
thz = bin_centers[:,0]
thx = bin_centers[:,1]

dthz = np.arange(angle_min, angle_max, dx)
dthx = np.arange(angle_min, angle_max, dx)

# # contruct 2d meshgrid array
# nan_thz, nan_thx = np.meshgrid(dthz, dthx)
# nan_f = np.ones(nan_thz.shape) * float('nan')
# 
# # populate nan_f matrix
# count = 0
# for (a, b, c) in zip(thz, thx, f_i):
#     pos = np.array( np.where( (np.abs(nan_thz[:,:] - a) <= 0.5*dx) * (np.abs(nan_thx[:,:] - b) <= 0.5*dx) ) )
#     i = int(pos[0])
#     j = int(pos[1])
#     nan_f[i, j] = c
#     count = count + 1

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(nan_thz, nan_thx, nan_f)
# plt.show()

thznew = np.linspace(-0.82, 0.1, 250)
thxnew = np.linspace(-0.085, 0.085, 250)
func = interpolate.bisplrep(thz, thx, f_i)
fnew = interpolate.bisplev(thznew, thxnew, func)
fnew = fnew.reshape(-1, 250)
fnew = fnew.T
Z, X = np.meshgrid(thznew, thxnew)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p1 = ax.plot_surface(Z, X, fnew, cmap=cm.RdYlGn, linewidth=0, antialiased=False, alpha=0.2)
p1.set_facecolor((0, 0, 1, 0.2))
ax.add_collection3d(p1)
ax.scatter(thz, thx, f_i, c='k', alpha=1.0)
plt.show()



# nbins=20
# N_tot = N_k.sum()
# theta_n_sorted = np.sort(theta_n)
# # reduce size of theta_n_sorted to only have size nbins -- take every N_tot/nbins values
# bins = np.append(theta_n_sorted[0::int(N_tot/nbins)], theta_n_sorted.max()+0.1)
# 
# bin_widths = bins[1:] - bins[0:-1]
# bin_n = np.zeros(theta_n.shape, np.int64)
# 
# # puts theta_n into the bins and returns the indices of the bin where each theta_n value lies
# bin_n = np.digitize(theta_n, bins) - 1
# [f_i, df_i] = my_mbar.computePMF(u_n, bin_n, nbins)
# f_i_corrected = f_i - np.log(bin_widths)
# theta_axis = bins[:-1] * .5 + bins[1:] * .5
# # plt.fill_between(theta_axis * 180.0 / np.pi, f_i_corrected - 2*df_i, f_i_corrected + 2*df_i, alpha=.4)
# # plt.figure(figsize=(24,18), dpi=300)
# plt.plot(x_axis * 180.0 / np.pi, f_i_corrected, color="#2020CC", linewidth=4, alpha=.4)
# plt.plot(repex[:,0] * 180.0 / np.pi, repex[:,1] + min(f_i_corrected), color='r', linewidth=2)
# plt.xlabel(r'$\langle\theta_z\rangle$', fontsize=32)
# plt.ylabel(r'-\beta\Delta F(\langle\theta_z\rangle)$', fontsize=32)
# plt.xticks(fontsize=32, fontweight='bold')
# plt.yticks(fontsize=32, fontweight='bold')
# # plt.savefig('mbar-free-en-340K.png', dpi='figure')
# plt.show()
