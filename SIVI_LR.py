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
import os
import sys
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd 
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.contrib.distributions import fill_triangular

slim=tf.contrib.slim
Exponential=tf.contrib.distributions.Exponential(rate=1.0)
Normal=tf.contrib.distributions.Normal(loc=0., scale=1.)
Mvn=tf.contrib.distributions.MultivariateNormalDiag
Bernoulli = tf.contrib.distributions.Bernoulli

#Include data path here
path = os.getcwd()+'/waveform.mat'
#%%
def sample_n(mu,sigma):

    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu+tf.matmul(eps,sigma)   
    return z


def sample_hyper(noise_dim,K,z_dim,reuse=False): 
    with tf.variable_scope("hyper_q") as scope:
        if reuse:
            scope.reuse_variables()
        e2 = tf.random_normal(shape=[K,noise_dim])
        h2 = slim.stack(e2,slim.fully_connected,[100,200,100])
        mu = tf.reshape(slim.fully_connected(h2,z_dim,activation_fn=None,scope='implicit_hyper_mu'),[-1,z_dim])
    return mu

#%% 
matdata = loadmat(path)
X_train = matdata['X_train']
X_test = matdata['X_test']
y_train = np.int_(np.squeeze(matdata['y_train']))
y_test = np.int_(np.squeeze(matdata['y_test']))

y_train[y_train==0]=-1
y_test[y_test==0]=-1 

#Stored Mean-field and MCMC results
theta_mf=np.transpose(matdata['Beta_VB_sample'])
theta_mcmc=np.transpose(matdata['BetaMCMC'])


#%%

N,P = np.shape(X_train)   
noise_dim = 20
alpha = 0.01 
J = tf.placeholder(tf.int32, shape=()) 


fff = tf.get_variable("z", dtype=tf.float32, 
                         initializer=tf.zeros([(P+1)*P/2])+0.2)  
chol_cov = fill_triangular(fff) 
covariance = tf.matmul(chol_cov, tf.transpose(chol_cov))

inv_cov = tf.matrix_inverse(covariance)
inv_cov_1 = tf.expand_dims(inv_cov,axis=0)
inv_cov_2 = tf.tile(inv_cov_1,[J+1,1,1])  

log_cov_det = tf.log(tf.matrix_determinant(covariance))

K = 50  
scale = tf.placeholder(tf.float32, shape=())

x = tf.placeholder(tf.float32,[N,P],name='data_x')
y = tf.placeholder(tf.float32,[N],name='data_y')
psi_sample = tf.squeeze(sample_hyper(noise_dim,K,P)) 
z_sample = sample_n(psi_sample,tf.transpose(chol_cov))  

psi_star_0 = sample_hyper(noise_dim,J,P,reuse=True) 
psi_star_1 = tf.expand_dims(psi_star_0,axis=1)  
psi_star_2 = tf.tile(psi_star_1,[1,K,1])

merge = tf.placeholder(tf.int32, shape=[])
psi_star = tf.cond(merge>0,lambda:tf.concat([psi_star_2, tf.expand_dims(psi_sample,axis=0)],0),lambda:psi_star_2)

z_sample_0 = tf.expand_dims(z_sample,axis=0) 
z_sample_1 = tf.cond(merge>0,lambda:tf.tile(z_sample_0,[J+1,1,1]),lambda:tf.tile(z_sample_0,[J,1,1]))

xvx = tf.matmul(z_sample_1-psi_star,inv_cov_2)*(z_sample_1-psi_star)
ker = tf.transpose(-0.5*tf.reduce_sum(xvx,2)) 
log_H = tf.reduce_logsumexp(ker,axis=1,keep_dims=True)-\
        tf.log(tf.cast(J,tf.float32)+1.0)-\
        0.5*log_cov_det         

log_P = scale*tf.reduce_sum(-tf.nn.softplus(-tf.matmul(z_sample,tf.transpose(x))*y),axis=1,keep_dims=True)+\
        (-0.5)*alpha*tf.reduce_sum(tf.square(z_sample),axis=1,keep_dims=True)


loss = tf.reduce_mean(log_H - log_P)

nn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hyper_q')
lr=tf.constant(0.0001)
train_op1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=nn_var)

lr2=tf.constant(0.01)
train_op2 = tf.train.GradientDescentOptimizer(learning_rate=lr2).minimize(loss,var_list=[fff])

init_op=tf.global_variables_initializer()

#%%

sess=tf.InteractiveSession()
sess.run(init_op)

record = []
for i in range(5000):
    _,cost=sess.run([train_op1,loss],{x:X_train,y:y_train,lr:0.01*(0.9**(i/100)),J:100,merge:1,scale:1.0})

    if i<2000:  
        _,cost=sess.run([train_op2,loss],{x:X_train,y:y_train,lr2:0.001*(0.9**(i/100)),J:100,merge:1,scale:1.0})
    
    record.append(cost)
    if i%100 == 0:
        print("iter:", '%04d' % (i+1), "cost=", np.mean(record),',', np.std(record))
        record = []

#%%

#sample from learned posterior
theta_hive=np.zeros([1000,P])
for i in range(1000):    
    r = sess.run(z_sample)
    theta_hive[i,:] = r[0,:]


#%%

def evaluate(theta,X_test,y_test,color='b',marker='o'):
    import numpy.matlib as nm
    M, n_test = theta.shape[0], len(y_test)    
    prob = np.zeros([n_test, M])
    blr = np.zeros([n_test, M])
    for t in range(M):
        coff = np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1)
        blr[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        coff1 = np.multiply(y_test,np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
        prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff1)))
    prob = np.mean(prob, axis=1)   
    plt.scatter(np.mean(blr,axis=1),np.std(blr,axis=1),alpha=.2, s=100, c=color,marker=marker)
    return np.mean(blr,axis=1), np.std(blr,axis=1)

_,_ = evaluate(theta_mcmc,X_test,y_test,'r','*')
_,_ = evaluate(theta_hive,X_test,y_test,'b','o')


#%%

def evaluate2(theta1,theta2,index):
    TH1 = pd.DataFrame(theta1[:,index])
    TH2 = pd.DataFrame(theta2[:,index])
    TH1['method']='MCMC'
    TH2['method']='SIVI'
    TH = pd.concat([TH1,TH2])
    var_name = ['b'+str(i) for i in index]
    TH.columns= var_name + ['method']
    TH = pd.melt(TH, value_vars=var_name, id_vars='method')
    sns.boxplot(x='variable', y='value', hue='method',data=TH, palette="PRGn",showfliers=True)

evaluate2(theta_mcmc,theta_hive,range(20))





