import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-temp", type=float, help="temperature of simulation")
parser.add_argument("-strength", type=float, help="spring constant used in simulation")
parser.add_argument("-lower", type=float, default=-0.95, help="lowest bias value in simulation")
parser.add_argument("-upper", type=float, default=-0.5875, help="highest bias value in simulation")
parser.add_argument("-step", type=float, default=0.0125, help="steps between bias values in simulation")
args = parser.parse_args()

temp = args.temp
kBT = 0.593 * temp / 298
beta = 1 / (kBT)
# print beta * beta
strength = args.strength

namelist = np.arange(args.lower, args.upper+args.step, args.step)
namelist = np.arange(0.5875, 0.9510, 0.0125)
# namelist = np.array(list(np.arange(0.30, 1.15, 0.05)) + list(np.arange(1.20, 1.40, 0.05)))
# namelist = [-0.40]
N_sims = len(namelist)
bins = np.linspace(0.50, 1.00, 100)		# angles are between 60 and 90 degrees approximately
bins_OG = bins[1:] * 0.5 + bins[:-1] * 0.5 
N_theta = np.zeros(len(bins_OG))
M_alpha = np.zeros(N_sims)

color = iter(plt.cm.copper(np.linspace(0,1,N_sims)))

# pot_list = []
# populate M_alpha and N_theta
for count, i in enumerate(namelist):
    c = next(color)
    # decide whether to use ceil or int based on which one works (keeps changing :/)

    data = np.genfromtxt("theta{:1.4f}.txt".format(-1.0*i), delimiter=' ')
    data = -1.0 * data

    total_prob, bins = np.histogram(data, bins=bins)
    bin_centres = 0.5 * bins[1:] + 0.5 * bins[:-1]

#     M_alpha[count] = np.sum(total_prob)
    M_alpha[count] = len(data)
    for j in range(len(bin_centres)):
        N_theta[j] = N_theta[j] + total_prob[j]

#     bias_en = 0.5 * strength * (bins_OG - i) * (bins_OG - i) * beta
#     plt.plot(bin_centres, bias_en)
#     pot_list.append(bias_en)

# plt.show()

tol = 1.0e-6
en_diff = 1.0
en_list = np.ones(N_sims)
# pot_list = np.exp(-np.array(pot_list))
# print pot_list.shape
# print pot_list[0]
# print N_theta
# print M_alpha
# exit(0)
pot_list = np.zeros((N_sims, len(bins_OG)))
# print pot_list.shape
for count, i in enumerate(namelist):
#     pot_list[count] = np.exp(-beta * 0.5 * strength * (bins_OG - i) * (bins_OG - i))
    pot_list[count] = 0.5 * strength * (bins_OG - i) * (bins_OG - i) * beta
# print pot_list[0]
# exit(0)

while (en_diff > tol):
    en_diff = 0.0
    old_en = np.zeros(len(en_list))
    old_en[:] = en_list[:]
    denominator = np.zeros(len(N_theta))

    # calculate denominator
    for t_i in range(len(N_theta)):
#     	denominator[t_i] = np.sum(M_alpha * (pot_list[:,t_i] / old_en))
#         free_en = np.exp(beta * old_en)
#         if (np.sum(free_en) != np.sum(free_en)):
#             print "SHIT"
#             exit(0)
    	denominator[t_i] = np.sum( M_alpha * np.exp(-(pot_list[:,t_i] - beta * old_en)) )

    # now update free energies
    for s_i in range(N_sims):
#         en_list[s_i] =  np.sum(N_theta * (pot_list[s_i,:] / denominator))
        en_list[s_i] =  -kBT * np.log( np.sum(N_theta * (np.exp(-pot_list[s_i,:]) / denominator)) )

    difference = np.abs(en_list - old_en)
    en_diff = np.sum(difference)

#     print count
#     print "numerator", numerator
#     print "denominator", denominator
#     print en_list
    print en_diff
    count = count + 1

str_out = ""
for en in en_list:
    str_out = str_out + "{:2.6f},".format(en)
str_out = str_out.rstrip(",")
print str_out



