from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys
import seaborn as sns
import scipy.spatial.distance
from matplotlib import pyplot as plt
import pandas as pd 
import scipy.stats as stats
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

slim=tf.contrib.slim
Exponential=tf.contrib.distributions.Exponential(rate=1.0)
Normal=tf.contrib.distributions.Normal(loc=0., scale=1.)
Mvn=tf.contrib.distributions.MultivariateNormalDiag
Bernoulli = tf.contrib.distributions.Bernoulli
plt.ioff()

sys.path.append(os.getcwd())

#%%

def sample_psi(x,noise_dim,K,z_dim,reuse=False): 
    with tf.variable_scope("hyper_psi") as scope:
        if reuse:
            scope.reuse_variables()
        x_0 = tf.expand_dims(x,axis=1)
        x_1 = tf.tile(x_0,[1,K,1])   #N*K*784
        
        B3 = Bernoulli(0.5)
        e3 = tf.cast(B3.sample([tf.shape(x)[0],K,noise_dim[0]]),tf.float32)
        input_ = tf.concat([e3,x_1],axis=2)
        h3 = slim.stack(input_,slim.fully_connected,[500,500,noise_dim[0]])
        
        B2 = Bernoulli(0.5)
        e2 = tf.cast(B2.sample([tf.shape(x)[0],K,noise_dim[1]]),tf.float32)
        input_1 = tf.concat([h3,e2,x_1],axis=2)
        h2 = slim.stack(input_1,slim.fully_connected,[500,500,noise_dim[1]])
        
        B1 = Bernoulli(0.5)
        e1 = tf.cast(B1.sample([tf.shape(x)[0],K,noise_dim[2]]),tf.float32)
        h1 = slim.stack(tf.concat([h2,e1,x_1],axis=2),slim.fully_connected,[500,500,500])

        mu = tf.reshape(slim.fully_connected(h1,z_dim,activation_fn=None,scope='implicit_hyper_mu'),[-1,K,z_dim])
    return mu


def sample_logv(x,noise_dim,z_dim,reuse=False): 
    with tf.variable_scope("hyper_sigma") as scope:
        if reuse:
            scope.reuse_variables()
        net1 = slim.stack(x,slim.fully_connected,[500,500],scope='sigma')
        z_logv = tf.reshape(slim.fully_connected(net1,z_dim,activation_fn=None,scope='sigma2'),[-1,z_dim])
    return z_logv 

def sample_n(psi,sigma):
    eps = tf.random_normal(shape=tf.shape(psi))
    z=psi+eps*sigma
    return z

def decoder(z,x_dim,reuse=False):
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        net3 = slim.stack(z,slim.fully_connected,[500,500,500],scope='decoder_1')
        logits_x = slim.fully_connected(net3,x_dim,activation_fn=None,scope='decoder_2')
        return logits_x

#%%
tf.reset_default_graph() 

z_dim = 64
noise_dim = [150,100,50]
x_dim = 784
eps = 1e-10

WU = tf.placeholder(tf.float32, shape=()) #warm-up

x = tf.placeholder(tf.float32,[None,x_dim])
J = tf.placeholder(tf.int32, shape=())  #estimate h
merge = tf.placeholder(tf.int32, shape=[])
K = tf.placeholder(tf.int32, shape=())  #iwae

z_logv = sample_logv(x,noise_dim,z_dim)
z_logv_iw = tf.tile(tf.expand_dims(z_logv,axis=1),[1,K,1])
sigma_iw1 = tf.exp(z_logv_iw/2)
sigma_iw2 = tf.cond(merge>0,lambda:tf.tile(tf.expand_dims(sigma_iw1,axis=2),[1,1,J+1,1]),
                    lambda:tf.tile(tf.expand_dims(sigma_iw1,axis=2),[1,1,J,1]))

psi_iw = sample_psi(x,noise_dim,K,z_dim)
z_sample_iw = sample_n(psi_iw,sigma_iw1)

z_sample_iw1 = tf.expand_dims(z_sample_iw,axis=2)
z_sample_iw2 = tf.cond(merge>0,lambda:tf.tile(z_sample_iw1,[1,1,J+1,1]),
                       lambda:tf.tile(z_sample_iw1,[1,1,J,1]))


psi_iw_star = sample_psi(x,noise_dim,J,z_dim,reuse=True)
psi_iw_star0 = tf.expand_dims(psi_iw_star,axis=1)
psi_iw_star1 = tf.tile(psi_iw_star0,[1,K,1,1])
psi_iw_star2 = tf.cond(merge>0,lambda:tf.concat([psi_iw_star1, tf.expand_dims(psi_iw,axis=2)],2),
                       lambda:psi_iw_star1)



ker = tf.exp(-0.5*tf.reduce_sum(tf.square(z_sample_iw2-psi_iw_star2)/tf.square(sigma_iw2+eps),3))
log_H_iw = tf.log(tf.reduce_mean(ker,axis=2))-0.5*tf.reduce_sum(z_logv_iw,2) #change to tf.reduce_logsumexp if there is NA

log_prior_iw = -0.5*tf.reduce_sum(tf.square(z_sample_iw),2)

x_iw = tf.tile(tf.expand_dims(x,axis=1),[1,K,1])
logits_x_iw = decoder(z_sample_iw,x_dim)
p_x_iw = Bernoulli(logits=logits_x_iw) 
reconstruct_iw = p_x_iw.mean()
log_lik_iw = tf.reduce_sum( x_iw * tf.log(reconstruct_iw + eps)
            + (1-x_iw) * tf.log(1 - reconstruct_iw + eps),2)


loss_iw0 = -tf.reduce_logsumexp(log_lik_iw+(log_prior_iw-log_H_iw)*WU,1)+tf.log(tf.cast(K,tf.float32))
loss_iw = tf.reduce_mean(loss_iw0)

var_all = slim.get_model_variables()
lr=tf.constant(0.001)
g_step = tf.Variable(0, trainable=False)
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_iw,var_list=var_all,global_step=g_step)


init_op=tf.global_variables_initializer()



#%%
mnist = input_data.read_data_sets(os.getcwd()+'/MNIST', one_hot=True)
train_data=mnist.train
test_data=mnist.test

dat_train=[]
dat_test=[]

sess=tf.InteractiveSession()

sess.run(init_op)

print("This is SIVAE_MNIST test")

training_epochs=2000
batch_size = 200

total_points = mnist.train.num_examples
total_batch = int(total_points / batch_size)
display_step=1
total_test_batch = int(mnist.test.num_examples / batch_size)


J_value = 1
warm_up = 0
from time import sleep
for epoch in range(training_epochs):
    avg_cost = 0.
    avg_cost_test = 0.
    np_lr = 0.001 * 0.75**(epoch/100)
    warm_up = np.min([epoch/300,1])
    if epoch<1900:
        J_value = 1
    else:
        J_value = 50

    for i in range(total_batch):
        train_xs_0,_ = train_data.next_batch(batch_size)  
        train_xs = np.random.binomial(1,train_xs_0)
        _ = sess.run([train_op],{x:train_xs,lr:np_lr,merge:1,J:J_value,K:1,WU:warm_up})


    if epoch>1900:
        for k in range(total_batch):
            train_xs_0,_ = train_data.next_batch(batch_size)  
            train_xs = np.random.binomial(1,train_xs_0)
            cost=sess.run(loss_iw,{x:train_xs,J:J_value,merge:1,K:1,WU:1.0})
            avg_cost += cost / total_batch
    
        
        for j in range(total_test_batch):
            test_xs_0,_ = test_data.next_batch(batch_size)  
            test_xs = np.random.binomial(1,test_xs_0)   
            cost_test=sess.run(loss_iw,{x:test_xs,J:J_value,merge:1,K:1,WU:1.0})
            avg_cost_test += cost_test / total_test_batch
            
        
    
        dat_train.append([epoch,avg_cost])
        dat_test.append([epoch,avg_cost_test])
    
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % epoch, 
                  "cost_train=", "{:.9f}".format(avg_cost),
                  "cost_test=", "{:.9f}".format(avg_cost_test))
            

avg_evi_test = 0
for j in range(total_test_batch):
    test_xs_0,_ = test_data.next_batch(batch_size)  
    test_xs = np.random.binomial(1,test_xs_0)   
    evi_test=sess.run(loss_iw,{x:test_xs,J:J_value,merge:1,K:1000,WU:1.0})
    avg_evi_test += evi_test / total_test_batch

L_1000 = avg_evi_test
print("&&&&&&&& The final test evidence is", L_1000)


if not os.path.exists('out/'):
    os.makedirs('out/')

    
dat0 = np.array(dat_train)
dat1 = np.array(dat_test)

df0 = pd.DataFrame({'epoch':dat0[:,0],'train':dat0[:,1]}) 
df1 = pd.DataFrame({'epoch':dat1[:,0],'test':dat1[:,1]}) 

df = pd.concat([df0,df1], ignore_index=True, axis=1)
name_data1 = 'out/data_dim4_'+str(noise_dim)+'.csv' 
df.to_csv(name_data1,index=False)


name_fig1 = 'out/slim_ELBO_dim4_'+str(noise_dim)+'.png'    
if 1:      
    plt.figure()    
    dat0 = np.array(dat_train)
    dat1 = np.array(dat_test)
    plt.plot(dat0[:,0],dat0[:,1],'o-',label='train')
    plt.plot(dat1[:,0],dat1[:,1],'o-',label='test')
    plt.legend(fontsize = 'x-large')
    plt.title("Training performance",fontsize = 'x-large')
    plt.ylabel('nats',fontsize = 'x-large')
    plt.xlabel('epoch',fontsize = 'x-large')
    plt.savefig(name_fig1, bbox_inches='tight')
    plt.close('all')


