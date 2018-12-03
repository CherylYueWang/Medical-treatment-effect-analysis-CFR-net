import tensorflow as tf
import numpy as np

from util import *

class ditou_net(object):
    """
    ditou_net implements the cfr_net with extra loss on the hidden variables in
    order to remove the selection bias

    This file contains the class ditou_net as well as helper functions.
    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """

    def __init__(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims):
        self.variables = {}
        self.wd_loss = 0 # weight decay loss

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out

        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]

        # weights and bias of the encoder layers
        weights_in = []; biases_in = []

        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            bn_biases = []
            bn_scales = []

        ''' Construct input/representation layers '''
        h_in = [x]
        for i in range(0, FLAGS.n_in):
            # n_in represents the number of encoder layers
            if i==0:
                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel:
                    weights_in.append(tf.Variable(1.0/dim_input*tf.ones([dim_input])))
                else:
                    weights_in.append(tf.Variable(tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_input))))
            else:
                weights_in.append(tf.Variable(tf.random_normal([dim_in,dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_in))))

            ''' If using variable selection, first layer is just rescaling'''
            if FLAGS.varsel and i==0:
                biases_in.append([])
                h_in.append(tf.mul(h_in[i],weights_in[i]))
            else:
                biases_in.append(tf.Variable(tf.zeros([1,dim_in])))
                z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                if FLAGS.batch_norm:
                    batch_mean, batch_var = tf.nn.moments(z, [0])

                    if FLAGS.normalization == 'bn_fixed':
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    else:
                        bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                        bn_scales.append(tf.Variable(tf.ones([dim_in])))
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

                h_in.append(self.nonlin(z))
                h_in[i+1] = tf.nn.dropout(h_in[i+1], do_in)

        h_rep = h_in[len(h_in)-1]

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keepdims=True))
        else:
            h_rep_norm = 1.0*h_rep


        ''' split the internal layer to AD and BC '''
        # h_rep dims [batch, features]
        # AD
        dims_AD = int(FLAGS.r_A*dim_in)
        bias_rep = tf.gather(h_rep_norm, tf.range(dims_AD), axis=1)
        # BC
        cfr_rep = tf.gather(h_rep_norm, tf.range(dims_AD, dim_in), axis=1)

        ''' Construct ouput layers '''
        y, weights_out, weights_pred, x_pred, weights_out2, weights_pred2 = self._build_output_graph(h_rep_norm, cfr_rep, t, dim_input, dim_in, dim_out, do_out, FLAGS)

        ''' Compute sample reweighting '''
        if FLAGS.reweight_sample:
            w_t = t/(2*p_t)
            #w_c = (1-t)/(2*1-p_t) # bug
            w_c = (1-t)/(2*(1-p_t))
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        if FLAGS.loss == 'l1':
            risk = tf.reduce_mean(sample_weight*tf.abs(y_-y))
            pred_error = -tf.reduce_mean(res)
        elif FLAGS.loss == 'log':
            y = 0.995/(1.0+tf.exp(-y)) + 0.0025
            res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)

            risk = -tf.reduce_mean(sample_weight*res)
            pred_error = -tf.reduce_mean(res)
        else:
            # loss == 'l2'
            risk = tf.reduce_mean(sample_weight*tf.square(y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        ''' Regularization '''
        if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])

        ''' Imbalance error '''
        if FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        if FLAGS.imb_fun == 'mmd2_rbf':
            imb_dist = mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma)
            imb_error = r_alpha*imb_dist
        elif FLAGS.imb_fun == 'mmd2_lin':
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = r_alpha*mmd2_lin(h_rep_norm,t,p_ipm) # why not just pass in imb_dist?
            # imb_error = r_alpha * imb_dist
        elif FLAGS.imb_fun == 'mmd_rbf':
            imb_dist = tf.abs(mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma))
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
        elif FLAGS.imb_fun == 'mmd_lin':
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
            # why not
            # imb_error = r_alpha * safe_sqrt(imb_dist)
        elif FLAGS.imb_fun == 'wass':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=False,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        elif FLAGS.imb_fun == 'wass2':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=True,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm,p_ipm,t)
            imb_error = r_alpha * imb_dist

        ''' Total error '''
        tot_error = risk

        if FLAGS.p_alpha>0:
            tot_error = tot_error + imb_error

        if FLAGS.p_lambda>0:
            tot_error = tot_error + r_lambda*self.wd_loss

        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        ''' Reconstruction error '''
        if FLAGS.p_recons > 0:
            recons_err = tf.sqrt(tf.reduce_mean(tf.square(x_pred - x)))
        else:
            recons_err = tf.constant(0, dtype=tf.float32)
        tot_error = tot_error + FLAGS.p_recons * recons_err

        ''' cross cov over bias_rep(A) and cfr_rep(BC) '''
        if FLAGS.p_xcov > 0:
            xcov_err = 0
            # reshape bias_rep: [batch, 100] -> [100, batch]
            bias_rep_new = tf.transpose(bias_rep, [1, 0])
            cfr_rep_new = tf.transpose(cfr_rep, [1, 0])
            # mean of BC and AD
            # dims = [100, 1]
            mean_BC = tf.reduce_mean(bias_rep_new, axis=1, keepdims=True)
            mean_AD = tf.reduce_mean(cfr_rep_new, axis=1, keepdims=True)
            # shape=[100, N]
            bias_rep_zero_mean = bias_rep_new - mean_AD # broadcasting
            cfr_rep_zero_mean = cfr_rep_new - mean_BC
            # the following is an attempt at a faster implementation of xcov
            xcov_err = 0.5 *tf.reduce_sum(tf.square(tf.diag_part(tf.matmul(bias_rep_zero_mean,
                tf.transpose(cfr_rep_zero_mean)))))
            tot_error = tot_error + FLAGS.p_xcov * xcov_err
        else:
            xcov_err = tf.constant(0, dtype=tf.float32)

        ## The following are the slow implementation
        #N = tf.shape(h_rep_norm)[0] # the number of examples
        #N = tf.to_float(N)
        #elems = (bias_rep_zero_mean, cfr_rep_zero_mean)
        # a[0], feature_AD_1 of all examples
        #fn = lambda a:1./N * tf.matmul(tf.reshape(a[0],[1,-1]),tf.reshape(a[1],[-1,1]))
        #res = tf.map_fn(fn, elems, dtype=tf.float32)
        #import pdb;pdb.set_trace()
        #xcov_err = 0.5 * tf.reduce_sum(tf.square(tf.map_fn(fn,elems,dtype=tf.float32, parallel_iterations=10)))
        #################

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.xcov_loss = FLAGS.p_xcov * xcov_err
        self.recons_loss = FLAGS.p_recons * recons_err
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep # internal rep before normalization
        self.h_rep_norm = h_rep_norm # internal rep after normalization, used in the loss and the decoder layers

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        """ build the decoder branch """
        h_out = [h_input]
        # [100, 100, ...]
        dims = [dim_in - int(FLAGS.r_A*dim_in)] + ([dim_out]*FLAGS.n_out)

        weights_out = []; biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                        stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            # the following is how the weights are added in the encoder
            # weights_in.append(tf.Variable(tf.random_normal([dim_in,dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_in))))
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out,1],
            stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred')
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred)+bias_pred

        return y, weights_out, weights_pred

    def _build_output_ae(self, h_input, dim_input, dim_in, dim_out, do_out, FLAGS):
        """ build the reconstruction branch """
        h_out = [h_input]
        # every layer has same num of nodes dim_in
        # 200, 200 ...
        dims = [dim_in] + ([dim_in]*FLAGS.n_out)

        weights_out = []; biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                        stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            # the following is how the weights are added in the encoder
            # weights_in.append(tf.Variable(tf.random_normal([dim_in,dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_in))))
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1,dim_in])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out)

        # last layer
        weights_pred = self._create_variable(tf.random_normal([dim_in,dim_input],
            stddev=FLAGS.weight_init/np.sqrt(dim_input)), 'w_pred')
        bias_pred = self._create_variable(tf.zeros([1,dim_input]), 'b_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_input-1,1])) #don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred)+bias_pred

        return y, weights_out, weights_pred


    def _build_output_graph(self, rep, rep_cfr, t, dim_input, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:
            # use branching network
            # rep: [subject, features]
            # Example:
            #   dim_input: 25
            #   dim_in: 100
            #   dim_out: 200
            i0 = tf.to_int32(tf.where(t < 1)[:,0])
            i1 = tf.to_int32(tf.where(t > 0)[:,0])

            rep0 = tf.gather(rep_cfr, i0)
            rep1 = tf.gather(rep_cfr, i1)

            y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)
            # build the branch for reconstruction
            x_pred, weights_out2, weights_pred2 = self._build_output_ae(rep, dim_input, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            # concatenation
            h_input = tf.concat(1,[rep, t])
            y, weights_out, weights_pred = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS)

        return y, weights_out, weights_pred, x_pred, weights_out2, weights_pred2
