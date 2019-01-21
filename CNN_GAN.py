import os
import re
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import random
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import dropout
import matplotlib.pyplot as plt
from datetime import datetime
import sys

now = datetime.utcnow().strftime("%Y%m%d")
logdir = "/temp/run-{}/example".format(now)

with open("full_dictionary.pkl","rb") as f:
  full_batches = pickle.load(f)

def xyz_norm(batch_dictionary):
  key_list = list(batch_dictionary.keys())
  full_sample = np.zeros((1, 1100, 3))
  
  for key in key_list:
    cur_batch = batch_dictionary[key]
    input_max = np.array([60, 500, 300])
    input_min = np.array([20, 0, 0])
    
    input_norm = (cur_batch - input_min) / (input_max - input_min)
    input_reshape = np.reshape(input_norm, newshape = (1, 1100, 3))
    full_sample = np.concatenate([full_sample, input_reshape], axis = 0)
    
  return full_sample[1:]
  
def generator(real_init):  #None, 10, 3
  """ Given input rank 3 matrix, run it over rnn model and return generated model output """
  with tf.variable_scope('rnn_gen', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
    gen_init = tf.reshape(real_init, shape = (15, 30)) #15, 10
    
    mean_w1 = tf.get_variable('mean_w1', [30, 10])
    mean_b1 = tf.get_variable('mean_b1', [10])
    
    gamma_w1 = tf.get_variable('gamma_w1', [30, 10])
    gamma_b1 = tf.get_variable('gamma_b1', [10])
    
    rand_layer4_mean = tf.matmul(gen_init, mean_w1) + mean_b1
    rand_layer4_gamma = tf.matmul(gen_init, gamma_w1) + gamma_b1

    rand_layer4 = rand_layer4_mean + rand_layer4_gamma * tf.random_normal(tf.shape(rand_layer4_gamma), dtype = tf.float32)
    
    rand_layer4_reshape = tf.reshape(rand_layer4, shape = (15, 1, 1, 10))
    rand_w5 = tf.get_variable('rand_w5', [2, 11, 10, 10])
    rand_b5 = tf.get_variable('rand_b5', [10])
    rand_layer5 = tf.nn.elu(tf.nn.conv2d_transpose(rand_layer4_reshape, rand_w5, output_shape =[15, 2, 11, 10], strides = [1,1,10,1],  padding = 'VALID') + rand_b5)
    
    rand_w6 = tf.get_variable('rand_w6', [2, 5, 10, 10])
    rand_b6 = tf.get_variable('rand_b6', [10])
    rand_layer6 = tf.nn.elu(tf.nn.conv2d_transpose(rand_layer5, rand_w6, output_shape =[15, 3, 55, 10], strides = [1,1,5,1],  padding = 'VALID') + rand_b6)
    
    rand_w7 = tf.get_variable('rand_w7', [1, 2, 10, 10])
    rand_b7 = tf.get_variable('rand_b7', [10])
    rand_layer7 = tf.sigmoid(tf.nn.conv2d_transpose(rand_layer6, rand_w7, output_shape =[15, 3, 110, 10], strides = [1,1,2,1],  padding = 'VALID') + rand_b7)
    
    output = tf.reshape(rand_layer7, shape = (15, 3, 1100))
    
  return tf.transpose(output, perm = [0, 2, 1])
  
def discriminator(the_xyz):
  with tf.variable_scope('d', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
    xyz_input = tf.reshape(the_xyz, shape = (15, 1100, 3, 1))
    
    en_w1 = tf.get_variable('en_w1', [11, 1, 1, 5])
    en_b1 = tf.get_variable('en_b1', [5])
    en_layer1 = tf.nn.elu(tf.nn.conv2d(xyz_input, en_w1, strides = [1,11,1,1], padding = 'VALID') + en_b1)#1, 100, 3, 5
    en_layer1 = tf.layers.dropout(en_layer1, rate = 0.5)
    
    en_w2 = tf.get_variable('en_w2', [10, 1, 5, 10])
    en_b2 = tf.get_variable('en_b2', [10])
    en_layer2 = tf.nn.elu(tf.nn.conv2d(en_layer1, en_w2, strides = [1,10,1,1], padding = 'VALID') + en_b2)#1, 10, 2, 10
    en_layer2 = tf.layers.dropout(en_layer2, rate = 0.5)
    
    en_w3 = tf.get_variable('en_w3', [10, 1, 10, 20])
    en_b3 = tf.get_variable('en_b3', [20])
    en_layer3 = tf.nn.elu(tf.nn.conv2d(en_layer2, en_w3, strides = [1,10,1,1], padding = 'VALID') + en_b3)#1, 1, 1, 20
    en_layer3 = tf.layers.dropout(en_layer3, rate = 0.5)
    
    en_layer3_reshape = tf.reshape(en_layer3, shape = (-1, 3, 20))
    en_layer4 = tf.contrib.layers.fully_connected(en_layer3_reshape, 10, activation_fn = None)
    
  return en_layer4
 
real_xyz = tf.placeholder("float", shape = [None, 1100, 3])
fake_xyz = generator(real_xyz[:,:10])

real_score = discriminator(real_xyz)
fake_score = discriminator(fake_xyz)

fake_true_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.ones_like(fake_score)))
real_true_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(real_score)))
fake_false_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(fake_score)))

parameter_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'rnn_gen')
parameter_d = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'd')

init_loss = tf.reduce_sum(tf.square(real_xyz[:,:10] - fake_xyz[:,:10]))
all_loss = tf.reduce_mean(tf.square(real_xyz - fake_xyz))

g_loss = fake_true_loss + init_loss + 0.1 * all_loss
d_loss = real_true_loss + fake_false_loss

train_G = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(g_loss, var_list = parameter_g)
train_D = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(d_loss, var_list = parameter_d)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
num_epoch = 50000
loss_matrix = np.zeros((num_epoch, 2))

with tf.Session() as sess:
  init.run()
  full_xyz = xyz_norm(full_batches)
  
  for epoch in range(num_epoch):
    print(epoch)
    np.random.shuffle(full_xyz)
    
    sess.run(train_G, feed_dict = {real_xyz: full_xyz})
    sess.run(train_D, feed_dict = {real_xyz: full_xyz})
    generator_loss, discriminator_loss = sess.run((fake_true_loss, real_true_loss), feed_dict = {real_xyz: full_xyz})
    
    print([generator_loss, discriminator_loss])
    loss_matrix[epoch,:] = np.array([generator_loss, discriminator_loss])
    
    if (epoch + 1) % 500 == 0:
      true_traj = full_xyz
      fake_traj = sess.run(fake_xyz, feed_dict = {real_xyz: full_xyz})
      save_path = saver.save(sess,"/tmp/ash_particle/CNN_GAN.ckpt")
'''      
fake_traj[0,:,0] = scipy.signal.savgol_filter(fake_traj[0,:,0], 251, 2)
fake_traj[0,:,1] = scipy.signal.savgol_filter(fake_traj[0,:,1], 251, 2)
fake_traj[0,:,2] = scipy.signal.savgol_filter(fake_traj[0,:,2], 251, 2)
true_traj[0,:,0] = scipy.signal.savgol_filter(true_traj[0,:,0], 251, 2)
true_traj[0,:,1] = scipy.signal.savgol_filter(true_traj[0,:,1], 251, 2)
true_traj[0,:,2] = scipy.signal.savgol_filter(true_traj[0,:,2], 251, 2)
'''

np.savetxt('CNN_GAN_training_curve.txt', loss_matrix, delimiter = '\t')

input_max = np.array([60, 500, 300])
input_min = np.array([20, 0, 0])

fake_traj = fake_traj * (input_max - input_min) + input_min
true_traj = true_traj * (input_max - input_min) + input_min

BATCH_NUM = 5
plt.subplot(211)
plt.plot(np.linspace(1, 1100, 1100), true_traj[BATCH_NUM,:,0], 'r', np.linspace(1, 1100, 1100), true_traj[BATCH_NUM,:,1], 'b', np.linspace(1, 1100, 1100), true_traj[BATCH_NUM,:,2], 'g')
plt.subplot(212)
plt.plot(np.linspace(1, 1100, 1100), fake_traj[BATCH_NUM,:,0], 'r', np.linspace(1, 1100, 1100), fake_traj[BATCH_NUM,:,1], 'b', np.linspace(1, 1100, 1100), fake_traj[BATCH_NUM,:,2], 'g')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(fake_traj[:, 0], fake_traj[0,:, 1], fake_traj[0,:, 2], label='generated curve')
ax.plot(true_traj[:, 0], true_traj[0,:, 1], true_traj[0,:, 2], label='experiment curve')
ax.legend()
plt.show()
