import pymbar
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import cm
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description="")
parser.add_argument("-dim", type=float, default=1, help="order parameter dimensionality [1d = th_z, 2d = th_z and th_x]")
args = parser.parse_args()

kB = 1.3806503 * 6.0221415 / 4184.0
beta = 1/(kB * 350.18)

namelist_1 = np.arange(-0.8500, -0.0240, 0.0250)
namelist_2 = np.arange(0.0, 0.1040, 0.0250)

namelist = np.concatenate((namelist_1, namelist_2))

# initialise list of spring constants
k_list = np.ones(len(namelist)) * 15000.0

N_sims = len(namelist)
T = 5000
theta_ik = []
VO_ik = []
UO_ik = []
 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for k, biasval in enumerate(namelist):
    data = np.genfromtxt('theta{:1.4f}.txt'.format(biasval))
    data = data.reshape((-1, 240))
    data_t = np.mean(data, axis=1)
    theta_ik.append(data_t)

for k, th in enumerate(namelist):
    lines = np.genfromtxt("pot-350.18.{:1.4f}".format(th))
    VO_ik.append(lines)
    dtheta_i = np.array(theta_ik[k]) - th
    UO_ik.append( lines - 0.5 * k_list[k] * np.square(dtheta_i) )

N_k = [ len(VO_i) for VO_i in VO_ik ]
N_k = np.array(N_k)
u_mbar = np.zeros((len(VO_ik), sum(N_k)))
K = u_mbar.shape[0]
N = u_mbar.shape[1]

# make numpy arrays from data
N_max = max(N_k)
th_ik = np.zeros([K, N_max])
uo_ik = np.zeros([K, N_max])
k = 0
for line1, line2 in zip(theta_ik, UO_ik):
    th_ik[k,:] = np.array(line1)
    uo_ik[k,:] = np.array(line2)
    k = k + 1

# go row by row to evaluate configuration energy in each umbrella
for k in range(K):
    # populate off-diagonal blocks in MBAR array; go column by column, i.e. config by config
    for i in range(N_k[k]):
        dtheta = th_ik[k, i] - namelist		# deviation of current configuration from each umbrella centre 
        print k, i 
#         u_mbar[ k, count:count+N_k[i] ] = beta * ( UO_ik[k] + 0.5 * k_list[k] * np.square(dtheta_k) ) - beta * ( UO_ik[i] + 0.5 * k_list[i] * np.square(dtheta_i) )
        u_mbar[ :, sum(N_k[:k]) + i ] = beta * ( uo_ik[k,i] + 0.5 * k_list * np.square(dtheta) )

my_mbar = pymbar.MBAR(u_mbar, N_k)
u_n = u_mbar[0, :] - u_mbar.min()
# u_n = np.zeros(N)

nbins = 100
theta_n = [val for row in theta_ik for val in row]
theta_n = np.array(theta_n)
theta_n_sorted = np.sort(theta_n)
bins = np.append(theta_n_sorted[0::int(N/nbins)], theta_n_sorted.max()+0.005)
bin_widths = bins[1:] - bins[0:-1]
bin_n = np.zeros(theta_n.shape, np.int64)
bin_n = np.digitize(theta_n, bins) - 1

# compute and plot PMF as function of theta_z

[f_i, df_i] = my_mbar.computePMF(u_n, bin_n, nbins)
f_i_corrected = f_i - np.log(bin_widths)
theta_axis = bins[:-1] * .5 + bins[1:] * .5

prob_i = np.exp(-f_i)
plt.figure()
plt.plot(theta_axis, prob_i)
plt.show()

ord_indices = np.where(theta_axis < -0.557)
disord_indices = np.where(theta_axis > -0.557)

area_ord = integrate.simps(prob_i[ord_indices], theta_axis[ord_indices])
area_disord = integrate.simps(prob_i[disord_indices], theta_axis[disord_indices])
print area_ord, area_disord

