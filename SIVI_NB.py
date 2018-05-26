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
import scipy.stats as stats
import tensorflow as tf

slim=tf.contrib.slim
Exponential=tf.contrib.distributions.Exponential(rate=1.0)
Normal=tf.contrib.distributions.Normal(loc=0., scale=1.)
Mvn=tf.contrib.distributions.MultivariateNormalDiag
Bernoulli = tf.contrib.distributions.Bernoulli


#%%
def lognormal(z,mu,sigma):
    pdf = 1/(sigma*z)*tf.exp(-0.5*tf.square(tf.log(z)-mu)/tf.square(sigma))
    return pdf

def sample_ln(mu,sigma):
    eps = tf.random_normal(shape=tf.shape(mu))
    z=tf.exp(mu+eps*sigma)
    return z

def logitnormal(z,mu,sigma):
    logit = tf.log(z/(1-z))
    term1 = 1/(z*(1-z))
    term2 = 1/(sigma)*tf.exp(-0.5*tf.square(logit-mu)/tf.square(sigma))
    pdf = term1*term2
    return pdf
    
def sample_logitn(mu,sigma):
    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu+eps*sigma
    return tf.exp(z)/(1+tf.exp(z))

def sample_mu(noise_dim,K,reuse=False): 
    z_dim = 2
    with tf.variable_scope("hyper_q") as scope:
        if reuse:
            scope.reuse_variables()
        e2 = tf.random_normal(shape=[K,noise_dim])
        h2 = slim.stack(e2,slim.fully_connected,[40,60,40])
        mu = tf.reshape(slim.fully_connected(h2,z_dim,activation_fn=None),[-1,z_dim])
    return mu

    
#%%
tf.reset_default_graph();# %reset -f

N = 150
noise_dim = 10
a = b = c = d = 0.01 #fixed gamma prior parameters
K = 50 

sigma_r = tf.constant(0.1) 
sigma_p = tf.constant(0.1)

x = tf.placeholder(tf.float32,[N],name='data_x')

mu_sample = tf.squeeze(sample_mu(noise_dim,K)) 
mu_r = tf.slice(mu_sample,[0,0],[-1,1])   
mu_p = tf.slice(mu_sample,[0,1],[-1,1]) 

r_sample = sample_ln(mu_r,sigma_r) 
p_sample = sample_logitn(mu_p,sigma_p) 

####

J = tf.placeholder(tf.int32, shape=())  #estimate h
mu_star_0 = sample_mu(noise_dim,J,reuse=True)  
mu_star_1 = tf.expand_dims(mu_star_0,axis=1)   
mu_star_2 = tf.tile(mu_star_1,[1,K,1]) 
mu_star = tf.concat([mu_star_2, tf.expand_dims(mu_sample,axis=0)],0) 

mu_star_r = tf.slice(mu_star,[0,0,0],[-1,-1,1]) 
mu_star_p = tf.slice(mu_star,[0,0,1],[-1,-1,1])  

r_sample_0 = tf.expand_dims(r_sample,axis=0)
r_sample_1 = tf.tile(r_sample_0,[J+1,1,1]) 
p_sample_0 = tf.expand_dims(p_sample,axis=0)
p_sample_1 = tf.tile(p_sample_0,[J+1,1,1]) 

term1 = lognormal(r_sample_1,mu_star_r,sigma_r)
term2 = logitnormal(p_sample_1,mu_star_p,sigma_p)

log_H = tf.log(tf.reduce_mean(term1*term2,axis=0))

####

log_P = tf.reduce_sum(tf.lgamma(r_sample+x),1,keep_dims=True) - N*tf.lgamma(r_sample)+ \
        tf.reduce_sum(x) * tf.log(p_sample) + N*r_sample *tf.log(1-p_sample) + \
        (a-1)*tf.log(r_sample) - b*r_sample + \
        (c-1)*tf.log(p_sample) + (d-1)*tf.log(1-p_sample)
        
loss = tf.reduce_mean(log_H - log_P)

####

nn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hyper_q')
lr=tf.constant(0.0001)
train_op1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=nn_var)

init_op=tf.global_variables_initializer()

#%%
# Demo: Red mites on apple leaves
# Table 1 of Bliss, C. I. and Fisher, R. A. Fitting the negative binomial distribution to biological data. 
# Biometrics, 1953.
path = os.getcwd()
X = np.concatenate([np.zeros(70),np.ones(38),2*np.ones(17),3*np.ones(10),4*np.ones(9),\
              5*np.ones(3),6*np.ones(2),7*np.ones(1)])
sess=tf.InteractiveSession()
sess.run(init_op)

record = []
for i in range(2000):  #converge in around 200 iters
    _, cost=sess.run([train_op1,loss],{x:X, lr:0.01*(0.75**(i/100)), J:1000})
    record.append(cost)
    if i%100 ==0:
        print("iter:", '%04d' % (i+1), "cost=", np.mean(record),np.std(record))
        record = []
#%% Implicit 
mu_P=[]
mu_R=[]
for i in range(50):
    r,p = sess.run([mu_r,mu_p],{x:X})
    mu_R.extend(np.squeeze(r))
    mu_P.extend(np.squeeze(p))
    
gg = sns.kdeplot(np.array(mu_R),np.array(mu_P),cmap="Blues",n_levels=10)
plt.xlim([-0.7,0.7])
plt.ylim([-1,1])
plt.xlabel(r'$\mu_r$')
plt.ylabel(r'$\mu_p$')
plt.title('Implicit')

 
#%% (r,p) 
p_sivi=[]
r_sivi=[]
for i in range(50):
    r,p = sess.run([r_sample,p_sample],{x:X})
    r_sivi.extend(np.squeeze(r))
    p_sivi.extend(np.squeeze(p))

plt.figure()
samples = np.array([np.squeeze(r_sivi),np.squeeze(p_sivi)])
vb = pd.DataFrame(samples.T, columns=["r", "p"])         
g2 = sns.kdeplot(vb.r,vb.p,cmap="Blues", shade=False,n_levels=10) 
plt.xlim([0.4,1.8])
plt.ylim([0.3,0.8])
plt.title('Joint')

#results obtained from MFVI code
mf_ga_a = 1.148120480673975e+02
mf_ga_b = 1.149553294523141e+02 
mf_be_a = 1.720000010000000e+02
mf_be_b = 1.498130396138695e+02
r_mf = stats.gamma.rvs(mf_ga_a,scale=1/mf_ga_b,size=2000)
p_mf = stats.beta.rvs(mf_be_a, mf_be_b, size=2000)

plt.figure()
plt.subplot(1, 2, 1)
sns.distplot(np.array(r_sivi),label='SIVI')
sns.distplot(r_mf,label='Mean-field')
plt.legend(fontsize=13,loc=1)
plt.title('r,histogram')
plt.subplot(1, 2, 2)
sns.distplot(np.array(p_sivi),label='SIVI')
sns.distplot(p_mf,label='Mean-field')
plt.legend(fontsize=13,loc=2)
plt.title('p,histogram')
plt.tight_layout()
plt.show()


    


