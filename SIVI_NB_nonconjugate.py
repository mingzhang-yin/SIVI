

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd 
from scipy.io import loadmat
import tensorflow as tf

slim=tf.contrib.slim
Normal=tf.contrib.distributions.Normal(loc=0., scale=1.)
Bernoulli = tf.contrib.distributions.Bernoulli


#%%
def loggamma(z,a1,b1):
    #b1 is rate
    log_pdf = a1*tf.log(b1) - tf.lgamma(a1) + (a1-1)*tf.log(z) - b1*z
    return log_pdf


def logbeta(z,a2,b2):
    log_B = tf.lgamma(a2) + tf.lgamma(b2) - tf.lgamma(a2+b2)
    log_pdf = (a2-1)*tf.log(z) + (b2-1)*tf.log(1-z) - log_B
    return log_pdf


def sample_mu(noise_dim,K,w1,w2,b1,b2): 
    z_dim = 4      
    epsi = tf.random_normal(shape=[K,noise_dim])
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(epsi, w1),b1))
    final = tf.add(tf.matmul(layer_1, w2),b2)
    mu = tf.reshape(final,[-1,z_dim])
    return mu

def elbo_mf(a1,be1,a2,be2,a,b,c,d,x,crt,N):
    Elogr = tf.digamma(a1) - tf.log(be1)
    Elogp = tf.digamma(a2) - tf.digamma(a2+be2)
    Elog1_p = tf.digamma(be2) - tf.digamma(a2+be2)
    log_B = tf.lgamma(a2) + tf.lgamma(be2) - tf.lgamma(a2+be2)
    term1 = a1*tf.log(be1) - tf.lgamma(a1) - log_B
    term2 = (a2 - c - tf.reduce_sum(x)) * Elogp + (a1 - a - tf.reduce_sum(crt)) * Elogr
    term3 = -(be1 - b) * (a1/be1) + (be2 - d) * Elog1_p - N*(a1/be1) * Elog1_p
    return term1 + term2 +term3

    
#%%
tf.reset_default_graph();# %reset -f

N = 150
noise_dim = 10
a = b = c = d = 0.01 #fixed gamma prior parameters
K = 5  #number of samples from q(\psi)


initializer = tf.contrib.layers.xavier_initializer()
W1 = tf.Variable(initializer([noise_dim, 50]))
W2 = tf.Variable(initializer([50, 4]))

B1 = tf.Variable(initializer([50]))
B2 = tf.Variable(initializer([4]))



x = tf.placeholder(tf.float32,[N],name='data_x')
crt = tf.placeholder(tf.float32,[N],name='data_l')

mu_sample = sample_mu(noise_dim,K,W1,W2,B1,B2) #shape=K*4
a1 = tf.exp(tf.slice(mu_sample,[0,0],[-1,1]))   #K*1
be1 = tf.exp(tf.slice(mu_sample,[0,1],[-1,1]))   #K*1
a2 = tf.exp(tf.slice(mu_sample,[0,2],[-1,1]))   #K*1
be2 = tf.exp(tf.slice(mu_sample,[0,3],[-1,1]))   #K*1


r_sample = tf.random_gamma([], a1, be1) #shape=K*1, alpha is shape, beta is inverse scale
beta = tf.distributions.Beta(concentration1 = a2, concentration0 = be2)  #K*1,concentration1 = alpha,concentration0 = beta
p_sample = beta.sample([])


kl_mf = elbo_mf(a1,be1,a2,be2,a,b,c,d,x,crt,N)


####

J = tf.placeholder(tf.int32, shape=())  
mu_star_0 = sample_mu(noise_dim,J,W1,W2,B1,B2)  
mu_star_1 = tf.expand_dims(mu_star_0,axis=1)  
mu_star_2 = tf.tile(mu_star_1,[1,K,1]) 
mu_star = tf.concat([mu_star_2, tf.expand_dims(mu_sample,axis=0)],0)

a1_star = tf.exp(tf.slice(mu_star,[0,0,0],[-1,-1,1])) 
be1_star = tf.exp(tf.slice(mu_star,[0,0,1],[-1,-1,1])) 
a2_star = tf.exp(tf.slice(mu_star,[0,0,2],[-1,-1,1]))  
be2_star = tf.exp(tf.slice(mu_star,[0,0,3],[-1,-1,1]))  

r_sample_0 = tf.expand_dims(r_sample,axis=0)
r_sample_1 = tf.tile(r_sample_0,[J+1,1,1])
p_sample_0 = tf.expand_dims(p_sample,axis=0)
p_sample_1 = tf.tile(p_sample_0,[J+1,1,1]) 


cond_loglik = loggamma(r_sample_1,a1_star,be1_star)+logbeta(p_sample_1,a2_star,be2_star) #J*K*1
log_H = tf.reduce_logsumexp(cond_loglik,axis=0) - tf.log(tf.cast(J,tf.float32)+1.0)

log_P = tf.reduce_sum(crt)*tf.log(r_sample) + tf.reduce_sum(x)*tf.log(p_sample) + \
        N*r_sample*tf.log(1-p_sample) + loggamma(r_sample,a,b) + logbeta(p_sample,c,d)


log_Q = loggamma(r_sample,a1,be1)+logbeta(p_sample,a2,be2)

part1 = tf.squeeze(log_Q,axis=1)
part2 = tf.squeeze(log_H - log_Q + kl_mf,axis=1)

loss = tf.reduce_mean(log_H - log_P)


def jacobian(y, x):
    with tf.name_scope("jacob"):
        grads = tf.stack([tf.squeeze(tf.gradients(yi, x)) for yi in tf.unstack(y)])
        return grads

        
jw1 = tf.reduce_mean(tf.expand_dims(log_H - log_Q,axis=2)*jacobian(part1, W1)+\
        jacobian(part2, W1),axis=0)

jw2 = tf.reduce_mean(tf.expand_dims(log_H - log_Q,axis=2)*jacobian(part1, W2)+\
        jacobian(part2, W2),axis=0)


jb1 = tf.reduce_mean((log_H - log_Q)*jacobian(part1, B1)+\
        jacobian(part2, B1),axis=0)

jb2 = tf.reduce_mean((log_H - log_Q)*jacobian(part1, B2)+\
        jacobian(part2, B2),axis=0)



learning_rate = tf.placeholder(tf.float32,[])

new_W1 = W1.assign(W1 - learning_rate * jw1)
new_W2 = W2.assign(W2 - learning_rate * jw2)
new_B1 = B1.assign(B1 - learning_rate * jb1)
new_B2 = B2.assign(B2 - learning_rate * jb2)

init_op=tf.global_variables_initializer()


#%% Training
path = os.getcwd()
X = np.squeeze(loadmat(path+'/data/NB_fix.mat')['x'])
L = np.squeeze(loadmat(path+'/data/NB_fix.mat')['L'])
sess=tf.InteractiveSession()
sess.run(init_op)

record = []
RR=[]
err=[]
x_id=[]
for i in range(5000):
    if i<4000:
        lr = 0.0001
    else:
        lr = 0.0001*(0.75**((i-4000)/100))
    _,_,_,_,cost=sess.run([new_W1,new_W2,new_B1,new_B2,loss],\
                          {x:X, crt:L, learning_rate:lr, J:200})
    record.append(cost)
    if i%100 ==0:
        print("iter:", '%04d' % (i+1), "cost=", np.mean(record),np.std(record),np.max(record))
        RR.append(np.mean(record))
        err.append(np.std(record))
        x_id.append(i)
        record = []

plt.figure()
plt.errorbar(x_id[1:],RR[1:],yerr=err[1:])
  


#%% (r,p) joint
        
pp=[]
rr=[]
for i in range(1000):
    r,p = sess.run([r_sample,p_sample],{x:X})
    rr.extend(np.squeeze(r))
    pp.extend(np.squeeze(p))

samples = np.array([np.squeeze(rr),np.squeeze(pp)])
vb = pd.DataFrame(samples.T, columns=["r", "p"])         
S = np.squeeze(loadmat(path+'/data/NB_fix.mat')['samples'])
mcmc = pd.DataFrame(S.T, columns=["r", "p"])
g2=sns.kdeplot(vb.r,vb.p, xlim=(0, 2), ylim=(0, 1),cmap="Blues", shade=False,n_levels=8) 

g=sns.kdeplot(mcmc.r,mcmc.p,xlim=(0, 2), ylim=(0, 1),cmap="Reds",n_levels=8)

plt.xlim(0.6,1.8)
plt.ylim(0.2,0.8)
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='P distribution')
blue_patch = mpatches.Patch(color='blue', label='Q distribution')
plt.legend(handles=[blue_patch,red_patch],fontsize=13,loc=1)
plt.title('Joint Distribution')
plt.show()











