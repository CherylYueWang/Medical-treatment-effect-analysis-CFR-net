import numpy as np
import matplotlib.pyplot as plt

#syn_train = np.load('data/synthetic_train_100.npz')
train = np.load('data/ihdp_npci_1-100.train.npz')
syn_balance_train = np.load('data/synthetic_train_balance_100.npz')
syn_rtsy_train = np.load('data/synthetic_train_rtsy_100.npz')
syn_train = np.load('data/synthetic_train.npz')


colors = ['#fe8947ff', '#4eb95fff',  '#2babe2ff', '#ec008cff']
if True:
    fig,axes = plt.subplots(4,1)
    ax0, ax1, ax2, ax3 = axes.flatten()

    nbins  = 50
    ax0.hist(train['yf'].flatten(), nbins, color=colors[0])
    ax0.set_title('IHDP')
    ax1.hist(syn_train['yf'].flatten(), nbins, color=colors[1])
    ax1.set_title('Syn 1 Experiment')
    ax2.hist(syn_rtsy_train['yf'].flatten(), nbins, color=colors[2])
    ax2.set_title('Syn RtSy')
    ax3.hist(syn_balance_train['yf'].flatten(), nbins, color=colors[3])
    ax3.set_title('Syn Balance')
    fig.tight_layout()
    #plt.show()
    plt.savefig("yf_hist.png",bbox_inches='tight')


if True:
    fig1,axes1 = plt.subplots(4,1)
    ax0, ax1, ax2, ax3 = axes1.flatten()

    nbins  = 50
    ax0.hist(train['ycf'].flatten(), nbins, color=colors[0])
    ax0.set_title('IHDP')
    ax1.hist(syn_train['ycf'].flatten(), nbins, color=colors[1])
    ax1.set_title('Syn 1 Experiment')
    ax2.hist(syn_rtsy_train['ycf'].flatten(), nbins, color=colors[2])
    ax2.set_title('Syn RtSy')
    ax3.hist(syn_balance_train['ycf'].flatten(), nbins, color=colors[3])
    ax3.set_title('Syn Balance')
    fig1.tight_layout()
    plt.savefig("ycf_hist.png",bbox_inches='tight')
    #plt.show()
