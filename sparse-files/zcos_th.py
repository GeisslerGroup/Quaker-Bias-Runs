import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

temp = 340.0

parser = argparse.ArgumentParser(description="")
parser.add_argument("-bias", type=str, help="bias value to analyse")
parser.add_argument("-log", action='store_true', help="plot log of probability")
parser.add_argument("-show", action='store_true', help="whether to show plot or not")
args = parser.parse_args()

save = "hist" + args.bias + ".png"

data = np.genfromtxt('/home/pratima/Quaker-Bias-Runs/sparse-files/lig.' + args.bias, delimiter=' ')

size = len(data)
x0 = 0.0
y0 = 0.0
z0 = 0.0
x1 = 0.0
y1 = 0.0
z1 = 0.0
f = open("theta" + args.bias + ".txt","w")
xb0 = 0.0
yb0 = -54.7217
zb0 = 0.0
xb1 = 82.4293
yb1 = 54.7217
zb1 = 81.0040

start = 100 * 18 * 240
frames = (len(data) / (240 * 18))
start = (frames - 101) * 240 * 18
print start

for i in range (start,size,18):
  if (i + 17 < size):
    x0 = data[i,1]
    y0 = data[i,2]
    z0 = data[i,3]
    x1 = data[i+9,1]
    y1 = data[i+9,2]
    z1 = data[i+9,3]
    x_vec = x1 - x0
    y_vec = y1 - y0
    z_vec = (z1 - z0) / (zb1 - zb0)
    z_vec = (z_vec - round(z_vec)) * (zb1 - zb0)
    th = np.arctan(z_vec / y_vec)
    f.write("%4.5f\n" %(th))

f.close()

hist_data = np.genfromtxt('/home/pratima/Quaker-Bias-Runs/sparse-files/theta' + args.bias + '.txt', delimiter=' ')
# hist_data = np.mean(hist_data.reshape((-1, 240)), axis=1)

# unbiased_data = np.genfromtxt('/home/pratima/Biased-PeriodicLigand/dump_files/zangle_distr_top.340', delimiter=' ')
# unbiased_data = -unbiased_data * np.pi / 180.0

# print np.mean(unbiased_data)
# print np.std(unbiased_data)
print np.mean(hist_data)
print np.std(hist_data)
bins = np.linspace(-1.70, 1.70, 100)
hist, bins = np.histogram(hist_data, bins = bins, density = True)
# unbiased_hist, bins = np.histogram(unbiased_data, bins = bins, density = True)
bin_centres = bins[1:] * 0.5 + bins[:-1] * 0.5
plt.figure()
if args.log:
    bin_centres = bin_centres[hist != 0]
    hist = hist[hist != 0]
    if args.show:
        plt.plot(bin_centres, -np.log(hist))
        plt.show()
else:
    if args.show:
        plt.plot(180.0*bin_centres/np.pi, hist, color='blue')
    #     plt.plot(180.0*bin_centres/np.pi, unbiased_hist, color='red')
        plt.show()
# plt.savefig(save)

