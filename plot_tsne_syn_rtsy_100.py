import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

syn_train = np.load('data/synthetic_train_rtsy_100.npz')
#syn_train = np.load('data/ihdp_npci_1-100.train.npz')

t = syn_train['t']
x = syn_train['x']

num_exp = 100
num_x1 = 0
num_x0 = 0
X1 = np.empty((0,  x.shape[1]))
X0 = np.empty((0,  x.shape[1]))

for exp in range(num_exp):

    print 'experiment %d' %exp
    It = np.array(np.where(t[:,exp]==1))[0,:]
    Ic = np.array(np.where(t[:,exp]==0))[0,:]

    x1 = x[It, :, exp]
    x0 = x[Ic, :, exp]

    #import pdb;pdb.set_trace()
    X1 = np.vstack((X1, x1))
    X0 = np.vstack((X0, x0))

X = np.vstack((X1, X0))
print 'shape of X ', X.shape
print 'shape of X1 ', X1.shape
print 'shape of X0 ', X0.shape

X_emb = TSNE(n_components=2).fit_transform(X)
X_emb1 = X_emb[0:X1.shape[0], :]
X_emb0 = X_emb[X1.shape[0]:, :]
print 'shape of X_emb ', X_emb.shape
print 'shape of X_emb1 ', X_emb1.shape
print 'shape of X_emb0 ', X_emb0.shape

plt.scatter(X_emb1[:,0], X_emb1[:,1], color='b', label='X(t=1)')
plt.scatter(X_emb0[:,0], X_emb0[:,1], color='r', label='X(t=0)')
plt.legend()
plt.savefig("tsne_syn.png",bbox_inches='tight')
#plt.savefig("tsne_ihdp.png",bbox_inches='tight')
