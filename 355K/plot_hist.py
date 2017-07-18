import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("-bias", type=str, help="bias value to analyse")
args = parser.parse_args()

hist_data = np.genfromtxt('theta' + args.bias + '.txt', delimiter=' ')
# hist_data = np.mean(hist_data.reshape((-1, 240)), axis=1)

# unbiased_data = np.genfromtxt('/home/pratima/Biased-PeriodicLigand/dump_files/zangle_distr_top.355', delimiter=' ')
# unbiased_data = -unbiased_data * np.pi / 180.0

# print np.mean(unbiased_data)
# print np.std(unbiased_data)
print np.mean(hist_data)
print np.std(hist_data)
bins = np.linspace(-1.70, 0.0, 100)
hist, bins = np.histogram(hist_data, bins = bins, density = True)
# unbiased_hist, bins = np.histogram(unbiased_data, bins = bins, density = True)
bin_centres = bins[1:] * 0.5 + bins[:-1] * 0.5
plt.figure()
plt.plot(bin_centres, hist, color='blue', marker="o")
#  plt.plot(180.0*bin_centres/np.pi, unbiased_hist, color='red')
plt.show()


