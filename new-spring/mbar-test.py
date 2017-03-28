import pymbar
import numpy as np
import matplotlib.pyplot as plt

namelist_1 = np.arange(-0.9500, -0.2450, 0.0125)
namelist_2 = np.arange(-0.2250, -0.0200, 0.025)
namelist_3 = np.arange(0.0, 0.0110, 0.025)
namelist = np.concatenate((namelist_1, namelist_2, namelist_3))
N_sims = len(namelist)
T = 5000
x_kn = np.zeros((N_sims, T))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# initialise list of spring constants
k_list = np.ones(N_sims)
for i in range(N_sims):
    if (namelist[i] <= -0.25):
        k_list[i] = 37500.0
    else:
        k_list[i] = 15000.0

for k, biasval in enumerate(namelist):
    data = np.genfromtxt('theta{:1.4f}.txt'.format(biasval))
    data = data.reshape((-1, 240))
    data_t = np.mean(data, axis=1)
    x_kn[k, :] = data_t[:]

x_kn = x_kn[:,::5]
T = x_kn.shape[1]
bias_kln = namelist.astype(float)[np.newaxis, :, np.newaxis]
k_list_kln = k_list[np.newaxis, :, np.newaxis]

beta = 1/0.6756
dx_kln = x_kn[:, np.newaxis, :] - bias_kln
u_kln = 0.5 * k_list_kln * np.square(dx_kln) * beta
u_lkn = np.swapaxes(u_kln,0,1)
u_lkn.shape
u_kn = np.reshape(u_lkn, (N_sims, T*N_sims))
x_n = np.reshape(x_kn, T*N_sims)
N_k = np.ones(N_sims) * T
my_mbar = pymbar.MBAR(u_kn, N_k)
# my_mbar.getFreeEnergyDifferences()
# u_n = u_kn[0, :]
u_n = np.zeros(T*N_sims)
nbins=100
N_tot = N_k.sum()
x_n_sorted = np.sort(x_n)
bins = np.append(x_n_sorted[0::int(N_tot/nbins)], x_n_sorted.max()+0.1)
bin_widths = bins[1:] - bins[0:-1]
bin_n = np.zeros(x_n.shape, np.int64)
bin_n = np.digitize(x_n, bins) - 1
[f_i, df_i] = my_mbar.computePMF(u_n, bin_n, nbins)
f_i_corrected = f_i - np.log(bin_widths)
x_axis = bins[:-1] * .5 + bins[1:] * .5
# plt.fill_between(bins[:-1] * .5 + bins[1:] * .5, f_i_corrected - 2*df_i, f_i_corrected + 2*df_i, alpha=.4)
plt.figure(figsize=(24,18), dpi=300)
plt.plot(x_axis * 180.0 / np.pi, f_i_corrected, color="#2020CC", linewidth=4)
plt.xlabel(r'$\langle\theta_z\rangle$', fontsize=32)
plt.ylabel(r'\Delta F(\langle\theta_z\rangle)$', fontsize=32)
plt.xticks(fontsize=32, fontweight='bold')
plt.yticks(fontsize=32, fontweight='bold')
plt.savefig('mbar-free-en-340K.png', dpi='figure')
plt.show()
