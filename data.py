import numpy as np
np.random.seed(0)


def sigmoid(xvec):
    """ Compute the sigmoid function """
    # Cap -xvec, to avoid overflow
    # Undeflow is okay, since it get set to zero
    if isinstance(xvec, (list, np.ndarray)):
        xvec[xvec < -100] = -100
    elif xvec < -100:
        xvec = -100

    vecsig = 1.0 / (1.0 + np.exp(np.negative(xvec)))

    return vecsig


def gen1(length, feature_num=25, cts_A=True, cts_B=True, cts_C=True):
    mu = [1, 2, 3]
    sigma = [1, 1, 1]

    A = np.random.normal(mu[0], sigma[0], length[0] * feature_num)
    B = np.random.normal(mu[1], sigma[1], length[1] * feature_num)
    C = np.random.normal(mu[2], sigma[2], length[2] * feature_num)

    ############################################## 2
    # weights for X_i
    weight_mu = [0.2, 0.4, 0.4]
    weight_sigma = [0.2, 0.4, 0.4]
    sample = []
    for i in xrange(feature_num):
        sample.append(
            np.average([(np.average(i),np.var(i)) for i in [A[i * length[0]:(i + 1) * length[0]], B[i * length[1]:(i + 1) * length[1]], C[i * length[2]:(i + 1) * length[2]]]],axis=0))
    sample = np.reshape(sample, (25,2))
    X = []

    X.append( np.random.normal(sample[:,0], sample[:,1],25))
    X = np.reshape(X,(1,25))
    ############################################## 3
    # weights for T
    weights = [0.3, 0.7]

    # GAI
    p = sigmoid(np.dot(weights, mu[0:2]))
    t = np.random.binomial(1, p, 1)

    ############################################## 4

    if t == 0:
        weights_alpha = [0.3, 0.7]
        weights_beta = [0.2, 0.8]
        a = np.dot([np.average(B), np.average(C)], weights_alpha)
        b = np.dot([np.var(B), np.var(C)], weights_beta)
        yf = np.random.gamma(np.divide(np.square(a), b)-1, np.divide(b, a), 1)
        ycf = np.random.gamma(np.divide(np.square(a), b), np.divide(b, a), 1)
    else:
        weights_alpha = [0.3, 0.7]
        weights_beta = [0.2, 0.8]
        a = np.dot([np.average(B), np.average(C)], weights_alpha)
        b = np.dot([np.var(B), np.var(C)], weights_beta)
        yf = np.random.gamma(np.divide(np.square(a), b), np.divide(b, a), 1)
        ycf = np.random.gamma(np.divide(np.square(a), b)-1, np.divide(b, a), 1)
    return [X, t, yf, ycf]

def generate_full_data(sample_size):
    sample_x = []
    sample_t = []
    sample_yf = []
    sample_ycf = []
    mu0 = []
    mu1 = []
    for i in range(sample_size):
        [X, t, yf, ycf] = gen1([10,10,10])
        sample_x.append(X)
        sample_t.append(t)
        sample_yf.append(yf)
        sample_ycf.append(ycf)
        if t==1:
            mu1.append(yf)
            mu0.append(ycf)
        else:
            mu1.append(ycf)
            mu0.append(yf)

    sample_x = np.reshape(sample_x, (sample_size, 25, 1))
    sample_t = np.reshape(sample_t, (sample_size, 1))
    sample_yf = np.reshape(sample_yf, (sample_size, 1))
    sample_ycf = np.reshape(sample_ycf, (sample_size, 1))
    mu0 = np.reshape(mu0,(sample_size,1))
    mu1 = np.reshape(mu1,(sample_size,1))
    ate = np.array(4)
    yadd = np.array(0)
    ymul = np.array(1)
    return [sample_x, sample_t, sample_yf, sample_ycf, mu0, mu1, ate, yadd, ymul]



# q = gen1([10,10,10])
#print q
q = generate_full_data(1000)
np.savez('./synthetic_train.npz', x=q[0], t= q[1], yf=q[2], ycf=q[3], mu0=q[4],
        mu1=q[5], ate=q[6], yadd=q[7], ymul=q[8])
np.random.seed(1)
q = generate_full_data(1000)
np.savez('./synthetic_test.npz', x=q[0], t= q[1], yf=q[2], ycf=q[3], mu0=q[4],
        mu1=q[5], ate=q[6], yadd=q[7], ymul=q[8])
