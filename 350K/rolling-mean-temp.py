import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="")
parser.add_argument("-bias", type=str, help="bias value to analyse")
parser.add_argument("-step", type=int, default=500, help="step value for running mean")
args = parser.parse_args()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

data = np.genfromtxt('temp' + args.bias, delimiter=' ')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure()
plt.plot(running_mean(data, args.step))
plt.ylim(349.5, 350.5)
plt.show()

