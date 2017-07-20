import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

im_arr = []

im_arr.append(plt.imread('cropped-disord-a-g-surf.tga'))
im_arr.append(plt.imread('cropped-ord-a-g-surf.tga'))
im_arr.append(plt.imread('cropped-disord-O-DO-surf.tga'))
im_arr.append(plt.imread('cropped-ord-O-DO-surf.tga'))

im_arr = np.array(im_arr)

fig = plt.figure(tight_layout=True, figsize=(10,9))
fig.subplots_adjust(hspace=0.0, wspace=0.0)

subplot = fig.add_subplot(2, 2, 1)
subplot.imshow(im_arr[0])
subplot.set_title(r'$\textrm{disordered phase, } \textit{anti-gauche}$', fontsize=20)
subplot.axis('off')

subplot = fig.add_subplot(2, 2, 2)
subplot.imshow(im_arr[1])
subplot.set_title(r'$\textrm{ordered phase, } \textit{anti-gauche}$', fontsize=20)
subplot.axis('off')

subplot = fig.add_subplot(2, 2, 3)
subplot.imshow(im_arr[2])
subplot.set_title(r'$\textrm{disordered phase, O-DO} $', fontsize=20)
subplot.axis('off')

subplot = fig.add_subplot(2, 2, 4)
subplot.imshow(im_arr[3])
subplot.set_title(r'$\textrm{disordered phase, O-DO}$', fontsize=20)
subplot.axis('off')

plt.savefig('compare-surf.png')
# plt.show()
plt.close()

im_arr = []

im_arr.append(plt.imread('cropped-disord-a-g-vdw.tga'))
im_arr.append(plt.imread('cropped-ord-a-g-vdw.tga'))
im_arr.append(plt.imread('cropped-disord-O-DO-vdw.tga'))
im_arr.append(plt.imread('cropped-ord-O-DO-vdw.tga'))

im_arr = np.array(im_arr)

fig = plt.figure(tight_layout=True, figsize=(10,9))
fig.subplots_adjust(hspace=0.0, wspace=0.0)

subplot = fig.add_subplot(2, 2, 1)
subplot.imshow(im_arr[0])
subplot.set_title(r'$\textrm{disordered phase, } \textit{anti-gauche}$', fontsize=20)
subplot.axis('off')

subplot = fig.add_subplot(2, 2, 2)
subplot.imshow(im_arr[1])
subplot.set_title(r'$\textrm{ordered phase, } \textit{anti-gauche}$', fontsize=20)
subplot.axis('off')

subplot = fig.add_subplot(2, 2, 3)
subplot.imshow(im_arr[2])
subplot.set_title(r'$\textrm{disordered phase, O-DO} $', fontsize=20)
subplot.axis('off')

subplot = fig.add_subplot(2, 2, 4)
subplot.imshow(im_arr[3])
subplot.set_title(r'$\textrm{disordered phase, O-DO}$', fontsize=20)
subplot.axis('off')

plt.savefig('compare-vdw.png')
# plt.show()
plt.close()

