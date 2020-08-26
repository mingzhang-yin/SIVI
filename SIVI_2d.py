#Coded by Mingzhang Yin
#02/09/2018 latest version.

#Copyright (c) <2018> <Mingzhang Yin>

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
import tensorflow as tf
slim=tf.contrib.slim

#%%

def sample_n(mu,sigma):

    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu+eps*sigma
    return z


def sample_hyper(noise_dim,K,reuse=False): 
    z_dim = 2
    with tf.variable_scope("hyper_q") as scope:
        if reuse:
            scope.reuse_variables()
        e2 = tf.random_normal(shape=[K,noise_dim])
        h2 = slim.stack(e2,slim.fully_connected,[40,60,40])
        mu = tf.reshape(slim.fully_connected(h2,z_dim,activation_fn=None,scope='implicit_hyper_mu'),[-1,2])
    return mu
#%%
data_p = {"5":"normal2d","6":"gmm2d","7":"banana","8":"Xshape"}
data_number = "8"
target = data_p[data_number]   

    
#%%
noise_dim = 10
K = 40 
sigma = tf.constant(0.1) 
J = tf.placeholder(tf.int32, shape=()) 

psi_sample = sample_hyper(noise_dim,K) 
z_sample = sample_n(psi_sample,sigma) 

psi_star_0 = sample_hyper(noise_dim,J,reuse=True)
psi_star_1 = tf.expand_dims(psi_star_0,axis=2)   
psi_star_2 = tf.tile(psi_star_1,[1,1,K])

merge = tf.placeholder(tf.int32, shape=[])
psi_star = tf.cond(merge>0,lambda:tf.concat([psi_star_2, tf.expand_dims(tf.transpose(psi_sample),axis=0)],0),lambda:psi_star_2)

z_sample_0 = tf.expand_dims(tf.transpose(z_sample),axis=0)
z_sample_1 = tf.cond(merge>0,lambda:tf.tile(z_sample_0,[J+1,1,1]),lambda:tf.tile(z_sample_0,[J,1,1]))

ker = tf.exp(-tf.reduce_sum(tf.square(z_sample_1-psi_star),1)/(2*tf.square(sigma))) 
log_H = tf.log(tf.reduce_mean(tf.transpose(ker),axis=1,keep_dims=True))

log_Q = -tf.reduce_sum(tf.square(z_sample-psi_sample),axis=1,keep_dims=True)/(2*tf.square(sigma))

if target == "normal2d":
    log_P = -0.5*tf.reduce_sum(tf.square(z_sample),1,keep_dims=True)/(2**2)
elif target == "gmm2d":
    log_P = tf.log(0.5*tf.exp(-0.5*tf.reduce_sum(tf.square(z_sample-tf.constant([-2.,0.])),1,keep_dims=True)) + \
               0.5*tf.exp(-0.5*tf.reduce_sum(tf.square(z_sample-tf.constant([2.,0.])),1,keep_dims=True)))
elif target == "banana":
    z1 = tf.slice(z_sample, [0,0],[-1,1])
    z2 = tf.slice(z_sample, [0,1],[-1,1])
    sg1 = 2.0; sg2 = 1.0
    log_P = -0.5*tf.square(z1)/tf.square(sg1)-0.5*tf.square(z2-0.25*tf.square(z1))/tf.square(sg2)-\
        tf.log(sg1)-tf.log(sg2)
elif target == "Xshape":
    def bi_gs(z1,z2,v,c):
        a = tf.square(v)-tf.square(c)
        b = -0.5*(v*tf.square(z1)+v*tf.square(z2)-2*c*z1*z2)/a
        return -0.5*tf.log(a) + b
    z1 = tf.slice(z_sample, [0,0],[-1,1])
    z2 = tf.slice(z_sample, [0,1],[-1,1])
    log_P = tf.log(0.5*tf.exp(bi_gs(z1,z2,2.0,1.8))+0.5*tf.exp(bi_gs(z1,z2,2.0,-1.8)))
else:
    raise ValueError('No pre-defined target distribution, you can write your own log(PDF) ')

##########

regular = log_Q - log_H
loss = tf.reduce_mean(log_H - log_P)

nn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hyper_q')
lr=tf.constant(0.0001)
train_op1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=nn_var)


init_op=tf.global_variables_initializer()


#%%

sess=tf.InteractiveSession()
sess.run(init_op)

record = []
for i in range(5000):
    _,cost=sess.run([train_op1,loss],{lr:0.001*(0.75**(i/100)),J:100,merge:1})
    record.append(cost)
    if i%500 == 0:
        print("iter:", '%04d' % (i+1), "cost=", np.mean(record))
        record = []

#%%

r_sivi=np.array([])
for i in range(200):
    if i==0:
        r = sess.run(z_sample)
        r_sivi=r
    else:
        r = sess.run(z_sample)
        r_sivi = np.concatenate((r_sivi, r), axis=0)
        

X,Y = np.mgrid[-2:2:.01, -2:2:.01]
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

if target == 'normal2d':
    rv = stats.multivariate_normal(mean=[0,0], cov=[[4, 0], [0, 4]])
    Z = rv.pdf(pos)
    CS = plt.contour(X, Y, rv.pdf(pos),5,colors='b')

elif target == 'gmm2d':
    rv1 = stats.multivariate_normal(mean=[-2,0], cov=[[1, 0], [0, 1]])
    rv2 = stats.multivariate_normal(mean=[2,0], cov=[[1, 0], [0, 1]])
    Z = 0.5*rv1.pdf(pos)+0.5*rv2.pdf(pos)
    CS = plt.contour(X, Y, Z,5,colors='b')
    
elif target == 'sin':  
    Z = np.exp(-0.5 * np.square((Y-np.sin(X))/0.4))
    CS = plt.contour(X, Y, Z, 5,colors='b')

elif target == 'banana':  
    sg1=2; sg2=1
    Z = (1/sg1)*np.exp(-0.5*X**2/(sg1**2))*(1/sg2)*np.exp(-0.5*((Y-0.25*X*X)**2)/(sg2**2))
    CS = plt.contour(X, Y, Z, 5,colors='b')  

elif target == 'Xshape':
    rv1 = stats.multivariate_normal(mean=[0,0], cov=[[2, 1.8], [1.8, 2]])
    rv2 = stats.multivariate_normal(mean=[0,0], cov=[[2, -1.8], [-1.8, 2]])
    Z = 0.5*rv1.pdf(pos)+0.5*rv2.pdf(pos)
    CS = plt.contour(X, Y, Z,5,colors='b')


sns.kdeplot(r_sivi[:,0],r_sivi[:,1],cmap="Reds",n_levels=5,label='Q distribution')
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Q distribution')
blue_patch = mpatches.Patch(color='blue', label='P distribution')
plt.legend(handles=[blue_patch,red_patch],fontsize=13,loc=1)

#%%
latent=np.array([])
for i in range(200):
    if i==0:
        latent = sess.run(psi_sample)
    else:
        latent = np.concatenate((latent, sess.run(psi_sample)), axis=0)
sns.kdeplot(latent[:,0],latent[:,1],cmap="Blues")
blue_patch = mpatches.Patch(color='blue', label='Psi')
plt.legend(handles=[blue_patch],fontsize=13,loc=1)





