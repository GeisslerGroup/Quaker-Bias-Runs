import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

biaslist = [-0.7500, -0.7250, -0.6500, -0.5750, -0.5000, -0.4250, -0.2750, -0.2000]
biaslist = np.array(biaslist)
reshaped_dims = np.reshape(biaslist, (-1, 4)).shape

# first plot SURF
im_arr = []

for bias in biaslist:
    im_arr.append(plt.imread('a-g{:1.4f}-surf.tga'.format(bias)))

im_arr = np.array(im_arr)

fig = plt.figure(tight_layout=True, figsize=(16,9))
fig.subplots_adjust(hspace=0, wspace=0)
for k, bias in enumerate(biaslist):
    subplot = fig.add_subplot(reshaped_dims[0], reshaped_dims[1], k+1)
    subplot.imshow(im_arr[k])
    subplot.set_title(r'$\langle\theta_z\rangle = {:1.4f}$'.format(bias), fontsize=24)
    subplot.set_xlim([300, 900])
    subplot.set_ylim([750, 250])
    subplot.axis('off')

plt.savefig('surf-a-g.png')
# plt.show()
plt.close()

# now VDW
im_arr = []

for bias in biaslist:
    im_arr.append(plt.imread('a-g{:1.4f}-vdw.tga'.format(bias)))

im_arr = np.array(im_arr)

fig = plt.figure(tight_layout=True, figsize=(16,9))
fig.subplots_adjust(hspace=0, wspace=0)
for k, bias in enumerate(biaslist):
    subplot = fig.add_subplot(reshaped_dims[0], reshaped_dims[1], k+1)
    subplot.imshow(im_arr[k])
    subplot.set_title(r'$\langle\theta_z\rangle = {:1.4f}$'.format(bias), fontsize=24)
    subplot.set_xlim([300, 900])
    subplot.set_ylim([750, 250])
    subplot.axis('off')

plt.savefig('vdw-a-g.png')

