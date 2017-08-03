import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("-bias", type=str, help="bias value to analyse")
args = parser.parse_args()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

hist_data = np.genfromtxt('theta' + args.bias + '.txt', delimiter=' ')
mean_data = np.mean(hist_data.reshape((-1, 240)), axis=1)[3500:]

# unbiased_data = np.genfromtxt('/home/pratima/Biased-PeriodicLigand/dump_files/zangle_distr_top.340', delimiter=' ')
# unbiased_data = -unbiased_data * np.pi / 180.0

# print np.mean(unbiased_data)
# print np.std(unbiased_data)
print np.mean(hist_data)
print np.std(hist_data)
print np.mean(mean_data)
print np.std(mean_data)
bins = np.linspace(-1.70, 0.0, 100)
meanbins = np.linspace(-0.9, -0.6, 100)
hist, bins = np.histogram(hist_data, bins = bins, density = True)
meanhist, meanbins = np.histogram(mean_data, bins = meanbins, density = True)
# unbiased_hist, bins = np.histogram(unbiased_data, bins = bins, density = True)
bin_centres = bins[1:] * 0.5 + bins[:-1] * 0.5
meanbin_centres = meanbins[1:] * 0.5 + meanbins[:-1] * 0.5
plt.figure()
plt.plot(bin_centres, hist, 'bo')
plt.plot(bin_centres, hist, 'b', alpha=0.7)
plt.plot(meanbin_centres, meanhist, 'rv')
plt.plot(meanbin_centres, meanhist, 'r', alpha=0.7)
#  plt.plot(180.0*bin_centres/np.pi, unbiased_hist, color='red')
plt.show()


