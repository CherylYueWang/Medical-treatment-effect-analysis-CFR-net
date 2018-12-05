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


def gen1(length, realization, weight_mu, weight_sigma, feature_num=25,cts_A=True, cts_B=True, cts_C=True):
    '''

    :param length: of A,B,C
    :param feature_num: of X
    :param cts_A:
    :param cts_B:
    :param cts_C:
    :return:
    '''

    # mu and sigma for generating Ai,Bi,Ci
    mu_a = np.random.normal(5, 20, length[0])
    sigma_a = np.random.gamma(2, 0.5, length[0])

    mu_b = np.random.normal(7, 15, length[1])
    sigma_b = np.random.gamma(2, 0.5, length[1])

    mu_c = np.random.normal(3, 17, length[2])
    sigma_c = np.random.gamma(2, 0.5, length[2])

    mu = [mu_a, mu_b, mu_c]
    sigma = [sigma_a, sigma_b, sigma_c]

    # shape_A=(length[0],1)
    A = np.random.normal(mu[0], sigma[0], length[0])
    B = np.random.normal(mu[1], sigma[1], length[1])
    C = np.random.normal(mu[2], sigma[2], length[2])

    ############################################## 2


    # generate 25 different sets of weights for each X_i




    X_mu = []
    X_sigma = []

    # average Ai
    average = [np.average(A), np.average(B), np.average(C)]
    for i in xrange(feature_num):
        X_mu.append( np.dot(average,weight_mu[i]) )
        # try to make sigma small !!!
        X_sigma.append( np.dot(average,weight_sigma[i]))

    # normalize sigma
    X_sigma = np.divide(X_sigma,np.sqrt((np.sum(np.asarray(X_sigma)**2))))
    X_sigma = np.abs(X_sigma)
    X = []

    # for 100 realization
    for j in xrange(realization):
        X.append(np.random.normal(X_mu, X_sigma, feature_num))

    # X is a matrix 100row, 25 column


    ############################################## 3
    # weights for T
    weights = [0.3, 0.7]

    p = sigmoid(np.dot(weights, average[0:2]))
    t = np.random.binomial(1, p, 1)
    # for the same patient we have the same t
    t = [t.astype('int64')]*realization
    ############################################## 4

    yf = []
    ycf = []
    if t[0] == 0:
        weights_alpha = [0.3, 0.7]
        weights_beta = [0.2, 0.8]
        a = np.dot([np.average(B), np.average(C)], weights_alpha)
        b = np.dot([np.var(B), np.var(C)], weights_beta)
        alpha = np.abs(np.divide(np.square(a), b))
        beta = np.abs(np.divide(b, a))
        for i in range(realization):
            yf.append(np.random.gamma(alpha, beta, 1) - 1)
            ycf.append(np.random.gamma(alpha, beta, 1))

    else:
        weights_alpha = [0.3, 0.7]
        weights_beta = [0.2, 0.8]
        a = np.dot([np.average(B), np.average(C)], weights_alpha)
        b = np.dot([np.var(B), np.var(C)], weights_beta)
        alpha = np.abs(np.divide(np.square(a), b))
        beta = np.abs(np.divide(b, a))
        for i in range(realization):
            yf.append(np.random.gamma(alpha, beta, 1)-1)
            ycf.append(np.random.gamma(alpha, beta, 1))

    # X is 100 row * 25 colomn
    # t is 1*100, yf is 1*100, ycf is 1*100

    return [X, t, yf, ycf]

def generate_full_data(sample_size,realization):
    sample_x = []
    sample_t = []
    sample_yf = []
    sample_ycf = []
    mu0 = []
    mu1 = []

    # generate random weights
    weight_mu = []
    weight_sigma = []
    for j in range(25):# feature number
        v = np.random.uniform(-5, 5, 3)
        weight_mu.append(np.divide(v, np.sum(v)))
        v = np.random.uniform(0, 1, 3)
        weight_sigma.append(np.divide(v, np.sum(v)))

    for i in range(sample_size):
        [X, t, yf, ycf] = gen1([10,15,20],realization,weight_mu,weight_sigma )
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

    sample_x = np.reshape(sample_x, (sample_size, 25, realization))
    sample_t = np.reshape(sample_t, (sample_size, realization))
    sample_yf = np.reshape(sample_yf, (sample_size, realization))
    sample_ycf = np.reshape(sample_ycf, (sample_size, realization))
    mu0 = np.reshape(mu0,(sample_size,realization))
    mu1 = np.reshape(mu1,(sample_size,realization))
    ate = np.array(4)
    yadd = np.array(0)
    ymul = np.array(1)
    return [sample_x, sample_t, sample_yf, sample_ycf, mu0, mu1, ate, yadd, ymul]





realization = 100
#q = gen1([5,5,5],realization=2)
#print q
q = generate_full_data(1000,realization)
np.savez('./synthetic_train_%d.npz'%realization, x=q[0], t= q[1], yf=q[2], ycf=q[3], mu0=q[4],
        mu1=q[5], ate=q[6], yadd=q[7], ymul=q[8])
np.random.seed(1)
q = generate_full_data(1000,realization)
np.savez('./synthetic_test_%d.npz'%realization, x=q[0], t= q[1], yf=q[2], ycf=q[3], mu0=q[4],
        mu1=q[5], ate=q[6], yadd=q[7], ymul=q[8])
