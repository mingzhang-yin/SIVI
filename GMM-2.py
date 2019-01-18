#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:49:41 2019

@author: mingzhangyin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd 
import scipy.stats as stats
from scipy.io import loadmat
import tensorflow as tf

import warnings
plt.style.use("ggplot")
warnings.filterwarnings('ignore')

#%matplotlib inline
slim=tf.contrib.slim
Normal=tf.contrib.distributions.Normal(loc=0., scale=1.)
import cPickle

#%%

def sample_z(K, noise, reuse=False): 
    z_dim = 2
    with tf.variable_scope("q_dist", reuse=reuse):
        h2 = slim.stack(noise,slim.fully_connected,[40,60,40])
        z_sample = slim.fully_connected(h2,z_dim,activation_fn=None)     
    return z_sample


def T_fun(z, reuse=False): 
    with tf.variable_scope("T_transform", reuse=reuse):
        t2 = slim.stack(z,slim.fully_connected,[20,20])
        t1 = slim.fully_connected(t2,1,activation_fn=None)
    return t1
        
    
#%%
tf.reset_default_graph();# %reset -f

n_comp = 8.
angle = np.linspace(0, 2*np.pi, n_comp)
mu0 = np.array([np.cos(angle)*5, np.sin(angle)*5]).T
mu = tf.convert_to_tensor(mu0, tf.float32)
#plt.plot(x, y, 'o')

EPS =1e-8
lr = tf.constant(1e-5)

noise_dim = 5
K = 200  #number of samples from q(z) and  p(z)


noise = tf.random_normal(shape=[K,noise_dim])
z_sample = sample_z(K, noise)

noise2 = tf.random_normal(shape=[K,noise_dim])
z_sample2 = sample_z(K, noise2,reuse=True)


T_zz = T_fun(tf.concat([noise,z_sample],axis = 1))  #K*1
T_zz2 = T_fun(tf.concat([noise,z_sample2],axis = 1), reuse=True)  #K*1


Tzz_r = tf.reduce_mean(T_zz)
Tzz2_r = tf.reduce_mean(T_zz2)


H_hat = tf.log(tf.nn.sigmoid(T_zz)+EPS) + tf.log(1 - tf.nn.sigmoid(T_zz2)+EPS)

#log p(x,z)
#sigma = 1.
#kernel = -0.5 / (sigma**2) * tf.reduce_sum(tf.square(z_sample[:,None,:] - mu[None,::]), axis = 2)
#log_P = tf.reduce_logsumexp(kernel, axis=1, keep_dims = True) - tf.log(n_comp)

log_P = tf.log(0.5*tf.exp(-0.5*tf.reduce_sum(tf.square(z_sample-tf.constant([-2.,0.])),1,keep_dims=True)) + \
               0.5*tf.exp(-0.5*tf.reduce_sum(tf.square(z_sample-tf.constant([2.,0.])),1,keep_dims=True)))


#loss_phi = -tf.reduce_mean(log_P + 6*H_hat)  #maximize elbo
loss_phi = -tf.reduce_mean(log_P + T_zz)  #maximize elbo
loss_T = - tf.reduce_mean(H_hat)  #maximize entropy


inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_dist')
inf_opt = tf.train.AdamOptimizer(lr)
inf_gradvars = inf_opt.compute_gradients(loss_phi, var_list=inf_vars)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)


T_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='T_transform')
T_opt = tf.train.AdamOptimizer(lr)
T_gradvars = T_opt.compute_gradients(loss_T, var_list=T_vars)
T_train_op = inf_opt.apply_gradients(T_gradvars)


with tf.control_dependencies([inf_train_op, T_train_op]):
    train_op = tf.no_op()
    
init_op=tf.global_variables_initializer()

#%%

def plot(sess):
    X,Y = np.mgrid[-4:4:.1, -4:4:.01]
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv1 = stats.multivariate_normal(mean=[-2,0], cov=[[1, 0], [0, 1]])
    rv2 = stats.multivariate_normal(mean=[2,0], cov=[[1, 0], [0, 1]])
    Z = 0.5*rv1.pdf(pos)+0.5*rv2.pdf(pos)

    zs = np.empty([0,2])
    for i in range(5):
        z_ = sess.run(z_sample)
        zs = np.concatenate([zs,z_],axis = 0) 
    plt.figure() 
    plt.contour(X, Y, Z, 5,colors='b')
    plt.plot(zs[:,0],zs[:,1],'bo')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.show()
    



sess=tf.InteractiveSession()
sess.run(init_op)

J = 10 #numer of steps to train T

stepsize = 5e-4
record = [] ; Record = []
record_Tloss = []; Record_Tloss = []
Record_Tzz = []
Record_Tzz2 = []
idx = []
for i in range(10000):  #converge in around 200 iters
    _, cost, cost_T=sess.run([train_op,loss_phi, loss_T],{lr:stepsize})
    record.append(cost)
    record_Tloss.append(cost_T)
    for j in range(J):
        sess.run([T_train_op])
    if i%200 == 0:
        print("iter:", '%04d' % (i+1), "cost=", np.mean(record),np.std(record))
        Record.append(np.mean(record))
        Record_Tloss.append(np.mean(record_Tloss))
        T1 , T2 = sess.run([Tzz_r, Tzz2_r])
        Record_Tzz.append(T1)
        Record_Tzz2.append(T2)
        idx.append(i)
        
        record = []   
        record_Tloss = []
        print("True prob =", 1/(1+np.exp(-Record_Tzz[-1])))
    
    if i%500 == 0:   
        plot(sess)
    if i%2000 == 0:
        stepsize = stepsize / 1.4

#%%
plt.figure()
plt.subplot(2,2,1)
plt.plot(idx, Record,'o-') 
plt.title('loss_q') 
plt.subplot(2,2,2)
plt.plot(idx, Record_Tloss,'o-')
plt.title('loss_T') 
plt.ylim([0,max(Record_Tloss)]) 
plt.subplot(2,2,3)
plt.plot(idx, 1/(1+np.exp(-np.array(Record_Tzz))),'o-')
plt.ylim([0,1]) 
plt.title('T_pair') 
plt.subplot(2,2,4)
plt.plot(idx,1/(1+np.exp(-np.array(Record_Tzz2))) ,'o-') 
plt.title('T_indp')  
plt.ylim([0,1])      
plt.tight_layout()

print(Record_Tzz[-1])
print(1/(1+np.exp(-Record_Tzz[-1])))















