import numpy as np
import matplotlib.pyplot as plt

train = np.load('data/ihdp_npci_1-100.train.npz')
syn_train = np.load('data/synthetic_train_100.npz')
syn_rtsy_train = np.load('data/synthetic_train_rtsy_100.npz')

fig,axes = plt.subplots(2,2)
ax0, ax1, ax2, ax3 = axes.flatten()

nbins  =50
ax0.hist(train['yf'].flatten(), nbins, color='y')
ax0.set_title('IHDP')
ax1.hist(syn_train['yf'].flatten(), nbins, color='g')
ax1.set_title('Syn')
ax2.hist(syn_rtsy_train['yf'].flatten(), nbins, color='b')
ax2.set_title('Syn RtSy')
fig.tight_layout()
plt.show()
