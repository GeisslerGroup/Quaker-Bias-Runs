import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

temp = 340.0

parser = argparse.ArgumentParser(description="")
parser.add_argument("-bias", type=str, help="bias value to analyse")
parser.add_argument("-log", action='store_true', help="plot log of probability")
args = parser.parse_args()

save = "hist" + args.bias + ".png"

data = np.genfromtxt('/home/pratima/Quaker-Bias-Runs/theta' + args.bias + '.txt', delimiter=' ')
# data = np.genfromtxt('/home/pratima/Biased-PeriodicLigand/dump_files/lig.txt', delimiter=' ')

unbiased_data = np.genfromtxt('/home/pratima/Biased-PeriodicLigand/dump_files/zangle_distr_top.340', delimiter=' ')
unbiased_data = -unbiased_data * np.pi / 180.0

print np.mean(unbiased_data)
print np.std(unbiased_data)
print np.mean(hist_data)
print np.std(hist_data)
bins = np.linspace(-1.70, 1.70, 400)
hist, bins = np.histogram(hist_data, bins = bins, density = True)
unbiased_hist, bins = np.histogram(unbiased_data, bins = bins, density = True)
bin_centres = bins[1:] * 0.5 + bins[:-1] * 0.5
plt.figure()
if args.log:
    bin_centres = bin_centres[hist != 0]
    hist = hist[hist != 0]
    plt.plot(bin_centres, -np.log(hist))
else:
    plt.plot(180.0*bin_centres/np.pi, hist, color='blue')
    plt.plot(180.0*bin_centres/np.pi, unbiased_hist, color='red')
plt.show()
# plt.savefig(save)

