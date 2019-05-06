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

with open("input_array.pkl","rb") as f:
  input_array = pickle.load(f)
  
with open("output_array.pkl","rb") as f:
  output_array = pickle.load(f)

def xyz_norm(cur_array):
  input_max = np.array([60, 500, 300])
  input_min = np.array([20, 0, 0])
  
  input_norm = (cur_array - input_min) / (input_max - input_min)
  
  return input_norm
  
def generator(real_init):  #-1, 5, 3
  """ Given input rank 3 matrix, run it over rnn model and return generated model output """
  with tf.variable_scope('rnn_gen', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
    x_init = real_init[:,:,0] #-1, 8
    y_init = real_init[:,:,1] #-1, 8
    z_init = real_init[:,:,2] #-1, 8
    
    with tf.variable_scope('x_rw', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
      x_w1 = tf.get_variable('x_w1', [6, 3])
      x_b1 = tf.get_variable('x_b1', [3])
      x_w2 = tf.get_variable('x_w2', [9, 3])
      x_b2 = tf.get_variable('x_b2', [3])
      x_w3 = tf.get_variable('x_w3', [3, 1])
      x_b3 = tf.get_variable('x_b3', [1])
      x_w4 = tf.get_variable('x_w4', [3, 1])
      x_b4 = tf.get_variable('x_b4', [1])
        
      x_full = x_init#-1, 8
      for i in range(104):
        x_layer1 = tf.nn.elu(tf.matmul(x_init, x_w1) + x_b1)#-1, 4
        x_layer2 = tf.concat([x_init, x_layer1], 1)#-1, 12
        x_layer3 = tf.nn.elu(tf.matmul(x_layer2, x_w2) + x_b2)#-1, 4
        
        x_mean = tf.sigmoid(tf.matmul(x_layer3, x_w3) + x_b3)#-1, 1
        x_gamma = tf.sigmoid(tf.matmul(x_layer3, x_w4) + x_b4)#-1, 1
        x_delta = x_mean + x_gamma * tf.random_normal(tf.shape(x_gamma), dtype = tf.float32)#-1, 1
        x_delta_denorm = (x_delta * 2 - 1) / 80
        x_output = tf.expand_dims(x_full[:,-1], 1) + x_delta_denorm
        x_full = tf.concat([x_full, x_output], axis = 1)#-1, 10 + 1
        x_init = x_full[:,-6:]
    
    with tf.variable_scope('y_rw', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
      y_w1 = tf.get_variable('y_w1', [6, 3])
      y_b1 = tf.get_variable('y_b1', [3])
      y_w2 = tf.get_variable('y_w2', [9, 3])
      y_b2 = tf.get_variable('y_b2', [3])
      y_w3 = tf.get_variable('y_w3', [3, 1])
      y_b3 = tf.get_variable('y_b3', [1])
      y_w4 = tf.get_variable('y_w4', [3, 1])
      y_b4 = tf.get_variable('y_b4', [1])
      
      y_full = y_init#-1, 8
      for i in range(104):
        y_layer1 = tf.nn.elu(tf.matmul(y_init, y_w1) + y_b1)
        y_layer2 = tf.concat([y_init, y_layer1], 1)#-1, 12
        y_layer3 = tf.nn.elu(tf.matmul(y_layer2, y_w2) + y_b2)#-1, 4
        
        y_mean = tf.sigmoid(tf.matmul(y_layer3, y_w3) + y_b3) #-1, 1
        y_gamma = tf.sigmoid(tf.matmul(y_layer3, y_w4) + y_b4) #-1, 1
        y_delta = y_mean + y_gamma * tf.random_normal(tf.shape(y_gamma), dtype = tf.float32)#-1, 1
        y_delta_denorm = (y_delta * 5 - 2.7) / 1700
        y_output = tf.expand_dims(y_full[:,-1], 1) + y_delta_denorm
        y_full = tf.concat([y_full, y_output], axis = 1)#-1, 10 + 1, 1
        y_init = y_full[:,-6:]
    
    with tf.variable_scope('z_rw', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
      z_w1 = tf.get_variable('z_w1', [6, 3])
      z_b1 = tf.get_variable('z_b1', [3])
      z_w2 = tf.get_variable('z_w2', [9, 3])
      z_b2 = tf.get_variable('z_b2', [3])
      z_w3 = tf.get_variable('z_w3', [3, 1])
      z_b3 = tf.get_variable('z_b3', [1])
      z_w4 = tf.get_variable('z_w4', [3, 1])
      z_b4 = tf.get_variable('z_b4', [1])
      
      z_full = z_init#-1, 8
      for i in range(104):
        z_layer1 = tf.nn.elu(tf.matmul(z_init, z_w1) + z_b1)
        z_layer2 = tf.concat([z_init, z_layer1], 1)#-1, 12
        z_layer3 = tf.nn.elu(tf.matmul(z_layer2, z_w2) + z_b2)#-1, 4
        
        z_mean = tf.sigmoid(tf.matmul(z_layer3, z_w3) + z_b3) #-1, 1
        z_gamma = tf.sigmoid(tf.matmul(z_layer3, z_w4) + z_b4) #-1, 1
        z_delta = z_mean + z_gamma * tf.random_normal(tf.shape(z_gamma), dtype = tf.float32)#-1, 1
        z_delta_denorm = (z_delta * 3 - 2.2) / 300
        z_output = tf.expand_dims(z_full[:,-1], 1) + z_delta_denorm
        z_full = tf.concat([z_full, z_output], axis = 1)#-1, 10 + 1, 1
        z_init = z_full[:,-6:]
        
    rand_layer1 = tf.concat([tf.expand_dims(x_full, 2), tf.expand_dims(y_full, 2), tf.expand_dims(z_full, 2)], axis = 2)#-1, 200, 3
    rand_layer1_reshape = tf.reshape(rand_layer1, shape = (-1, 110, 3, 1))
    output_filt = tf.ones(shape = (11, 1, 1, 1)) / 11
    
    rand_layer2 = tf.nn.conv2d(rand_layer1_reshape, output_filt, strides = [1,1,1,1], padding = 'VALID')
    
  return tf.reshape(rand_layer2, shape = (-1, 100, 3)) #-1, 110, 3
  
def discriminator(the_xyz):
  with tf.variable_scope('d', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
    xyz_input = tf.reshape(the_xyz, shape = (-1, 100, 3, 1))
    
    en_w1 = tf.get_variable('en_w1', [10, 1, 1, 4])
    en_b1 = tf.get_variable('en_b1', [4])
    en_layer1 = tf.nn.elu(tf.nn.conv2d(xyz_input, en_w1, strides = [1,10,1,1], padding = 'VALID') + en_b1)#1, 10, 3, 5
    en_layer1 = tf.layers.dropout(en_layer1, rate = 0.5)
    
    en_w2 = tf.get_variable('en_w2', [5, 2, 4, 8])
    en_b2 = tf.get_variable('en_b2', [8])
    en_layer2 = tf.nn.elu(tf.nn.conv2d(en_layer1, en_w2, strides = [1,5,1,1], padding = 'VALID') + en_b2)#1, 2, 2, 10
    en_layer2 = tf.layers.dropout(en_layer2, rate = 0.5)
    
    en_w3 = tf.get_variable('en_w3', [2, 2, 8, 16])
    en_b3 = tf.get_variable('en_b3', [16])
    en_layer3 = tf.nn.elu(tf.nn.conv2d(en_layer2, en_w3, strides = [1,2,1,1], padding = 'VALID') + en_b3)#1, 1, 1, 20
    en_layer3 = tf.layers.dropout(en_layer3, rate = 0.5)
    
    en_layer3_reshape = tf.reshape(en_layer3, shape = (-1, 16))
    en_layer4 = tf.contrib.layers.fully_connected(en_layer3_reshape, 8, activation_fn = None)
    
  return en_layer4
 
real_input = tf.placeholder("float", shape = [None, 6, 3])
real_output = tf.placeholder("float", shape = [None, 100, 3])

fake_xyz = generator(real_input)

real_score = discriminator(real_output)
fake_score = discriminator(fake_xyz)

fake_true_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.ones_like(fake_score)))
real_true_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(real_score)))
fake_false_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(fake_score)))

parameter_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'rnn_gen')
parameter_d = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'd')

def msd_time_scaling(TRUE, FAKE): #r,phi,z. -1, 100, 3
  total_true = tf.zeros((1, 3))
  total_fake = tf.zeros((1, 3))
  for i in range(99):
    true_e2 = (TRUE[:,:100 - i,:] - TRUE[:,i:,:])**2
    fake_e2 = (FAKE[:,:100 - i,:] - FAKE[:,i:,:])**2
    true_mean = tf.reshape(tf.reduce_mean(tf.reduce_mean(true_e2, axis = 0), axis = 0), shape = (1,3))
    fake_mean = tf.reshape(tf.reduce_mean(tf.reduce_mean(fake_e2, axis = 0), axis = 0), shape = (1,3))
    total_true = tf.concat([total_true, true_mean], axis = 0)
    total_fake = tf.concat([total_fake, fake_mean], axis = 0)
  
  error = tf.reduce_mean(total_true - total_fake)
  return error

msd_loss = msd_time_scaling(real_output, fake_xyz)
init_loss = tf.reduce_mean(tf.square(real_output[:,:20] - fake_xyz[:,:20]))
all_loss = tf.reduce_mean(tf.square(real_output - fake_xyz))

g_loss = fake_true_loss + all_loss
d_loss = real_true_loss + fake_false_loss

train_G = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(g_loss, var_list = parameter_g)
train_D = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(d_loss, var_list = parameter_d)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
num_epoch = 5000
loss_matrix = np.zeros((num_epoch, 2))

with tf.Session() as sess:
  init.run()
  full_input = xyz_norm(input_array)#-1, 6, 3
  full_output = xyz_norm(output_array)#-1, 100, 3
  full_array = np.concatenate([full_input, full_output], axis = 1)
  
  for epoch in range(num_epoch):
    print(epoch)
    np.random.shuffle(full_array)
    
    part_input = full_array[:1000, :6]
    part_output = full_array[:1000, 6:]
    
    sess.run(train_G, feed_dict = {real_input: part_input, real_output: part_output})
    sess.run(train_D, feed_dict = {real_input: part_input, real_output: part_output})
    generator_loss, discriminator_loss, fitting_loss = sess.run((fake_true_loss, real_true_loss, init_loss), feed_dict = {real_input: part_input, real_output: part_output})
  
    print([generator_loss, discriminator_loss, fitting_loss])
    loss_matrix[epoch,:] = np.array([generator_loss, discriminator_loss])
    
    if (epoch + 1) % 1000 == 0:
      true_traj = part_output
      fake_traj = sess.run(fake_xyz, feed_dict = {real_input: part_input, real_output: part_output})
      save_path = saver.save(sess,"/tmp/ash_particle/seq2seq_short_GAN.ckpt")

np.savetxt('seq2seq_short_training_curve.txt', loss_matrix, delimiter = '\t')

input_max = np.array([60, 500, 300])
input_min = np.array([20, 0, 0])

fake_traj = fake_traj * (input_max - input_min) + input_min
true_traj = true_traj * (input_max - input_min) + input_min

BATCH_NUM = 700
plt.subplot(211)
plt.plot(np.linspace(1, 200, 200), true_traj[BATCH_NUM,:,0], 'r', np.linspace(1, 200, 200), true_traj[BATCH_NUM,:,1], 'b', np.linspace(1, 200, 200), true_traj[BATCH_NUM,:,2], 'g')
plt.subplot(212)
plt.plot(np.linspace(1, 200, 200), fake_traj[BATCH_NUM,:,0], 'r', np.linspace(1, 200, 200), fake_traj[BATCH_NUM,:,1], 'b', np.linspace(1, 200, 200), fake_traj[BATCH_NUM,:,2], 'g')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(fake_traj[:, 0], fake_traj[0,:, 1], fake_traj[0,:, 2], label='generated curve')
ax.plot(true_traj[:, 0], true_traj[0,:, 1], true_traj[0,:, 2], label='experiment curve')
ax.legend()
plt.show()
