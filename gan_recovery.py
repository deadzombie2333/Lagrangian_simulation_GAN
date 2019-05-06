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
from scipy import interpolate

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
  
def generator(real_init):  #-1, 10, 3
  """ Given input rank 3 matrix, run it over rnn model and return generated model output """
  with tf.variable_scope('rnn_gen', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
    x_init = tf.log(real_init[:,:,0] / (1 - real_init[:,:,0])) #-1, 10
    y_init = tf.log(real_init[:,:,1] / (1 - real_init[:,:,0])) #-1, 10
    z_init = tf.log(real_init[:,:,2] / (1 - real_init[:,:,0])) #-1, 10
    batch_num = tf.shape(real_init)[0]
    
    with tf.variable_scope('x_rw', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
      x_w1 = tf.get_variable('x_w1', [10, 5])
      x_b1 = tf.get_variable('x_b1', [5])
      x_w2 = tf.get_variable('x_w2', [5, 1])
      x_b2 = tf.get_variable('x_b2', [1])
      x_w3 = tf.get_variable('x_w3', [5, 1])
      x_b3 = tf.get_variable('x_b3', [1])
        
      x_layer1 = tf.nn.elu(tf.matmul(x_init, x_w1) + x_b1)
      x_mean = tf.matmul(x_layer1, x_w2) + x_b2#-1, 1
      x_gamma = tf.matmul(x_layer1, x_w3) + x_b3#-1, 1
      x_mean_reshape = tf.reshape(tf.tile(x_mean, (1, 100)), shape = (-1, 100, 1)) #-1, 110, 1
      x_gamma_reshape = tf.reshape(tf.tile(x_gamma, (1, 100)), shape = (-1, 100, 1)) #-1, 110, 1
      x_output = x_mean_reshape + x_gamma_reshape * tf.random_normal(tf.shape(x_gamma_reshape), dtype = tf.float32)#-1, 110, 1
      
    with tf.variable_scope('y_rw', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
      y_w1 = tf.get_variable('y_w1', [10, 5])
      y_b1 = tf.get_variable('y_b1', [5])
      y_w2 = tf.get_variable('y_w2', [5, 1])
      y_b2 = tf.get_variable('y_b2', [1])
      y_w3 = tf.get_variable('y_w3', [5, 1])
      y_b3 = tf.get_variable('y_b3', [1])
      
      y_layer1 = tf.nn.elu(tf.matmul(y_init, y_w1) + y_b1)
      y_mean = tf.matmul(y_layer1, y_w2) + y_b2#-1, 1
      y_gamma = tf.matmul(y_layer1, y_w3) + y_b3#-1, 1
      y_mean_reshape = tf.reshape(tf.tile(y_mean, (1, 100)), shape = (-1, 100, 1)) #-1, 110, 1
      y_gamma_reshape = tf.reshape(tf.tile(y_gamma, (1, 100)), shape = (-1, 100, 1)) #-1, 110, 1
      y_output = y_mean_reshape + y_gamma_reshape * tf.random_normal(tf.shape(y_gamma_reshape), dtype = tf.float32)#-1, 110, 1
      
    with tf.variable_scope('z_rw', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
      z_w1 = tf.get_variable('z_w1', [10, 5])
      z_b1 = tf.get_variable('z_b1', [5])
      z_w2 = tf.get_variable('z_w2', [5, 1])
      z_b2 = tf.get_variable('z_b2', [1])
      z_w3 = tf.get_variable('z_w3', [5, 1])
      z_b3 = tf.get_variable('z_b3', [1])
      
      z_layer1 = tf.nn.elu(tf.matmul(z_init, z_w1) + z_b1)
      z_mean = tf.matmul(z_layer1, z_w2) + z_b2#-1, 1
      z_gamma = tf.matmul(z_layer1, z_w3) + z_b3#-1, 1
      z_mean_reshape = tf.reshape(tf.tile(z_mean, (1, 100)), shape = (-1, 100, 1)) #-1, 110, 1
      z_gamma_reshape = tf.reshape(tf.tile(z_gamma, (1, 100)), shape = (-1, 100, 1)) #-1, 110, 1
      z_output = z_mean_reshape + z_gamma_reshape * tf.random_normal(tf.shape(z_gamma_reshape), dtype = tf.float32)#-1, 110, 1
      
    rand_layer4 = tf.concat([x_output, y_output, z_output], axis = 2)#-1, 110, 3
    
    prev_input = tf.log(real_init[:,-1] / (1- real_init[:,-1]))#-1, 3
    full_output = tf.log(real_init / (1 - real_init))
    
    for j in range(100):
      prev_input = prev_input + rand_layer4[:, j] #-1, 3
      full_output = tf.concat([full_output, tf.expand_dims(prev_input, 1)], axis = 1)
      
  return tf.sigmoid(full_output) #-1, 110, 3
 
real_input = tf.placeholder("float", shape = [None, 10, 3])
fake_xyz = generator(real_input)

parameter_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'rnn_gen')
reuse_vars_dict = dict([(var.op.name, var) for var in parameter_g])
pre_vars_dict = {**reuse_vars_dict}
pre_saver = tf.train.Saver(pre_vars_dict)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
num_epoch = 50000
loss_matrix = np.zeros((num_epoch, 2))

with tf.Session() as sess:
  init.run()
  full_xyz = xyz_norm(full_batches)
  
  pre_saver.restore(sess,"/tmp/ash_particle/Newfolder/seq2seq_short_GAN.ckpt")
  np.random.shuffle(full_xyz)
  
  init_xyz = full_xyz[:,:10]#-1, 10, 3
  full_gen = init_xyz[:]#-1, 10, 3
  
  for j in range(11):
    fake_traj = sess.run(fake_xyz, feed_dict = {real_input: init_xyz}) #-1, 110, 3
    init_xyz = fake_traj[:,-10:]
    full_gen = np.concatenate([full_gen, fake_traj[:,10:]], axis = 1)
    
  fake_traj = full_gen[:,:1100] + 0.001 * np.random.normal(size = (15, 1100, 3))
  true_traj = full_xyz

def smoothing_fn(traj):
  for j in range(15):
    tck, u = interpolate.splprep([traj[j,10:,0], traj[j,10:,1], traj[j,10:,2]], s=1.5, ub = 10)
    u_fine = np.linspace(0,1,1090)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    
    traj[j,10:,0] = x_fine
    traj[j,10:,1] = y_fine
    traj[j,10:,2] = z_fine
  return traj

def rz2xyz(rz_traj):
  r_traj = rz_traj[:,:,0:1]
  phi_traj = rz_traj[:,:,1:2]
  z_traj = rz_traj[:,:,2:3]
  
  x_traj = r_traj * np.cos(phi_traj*np.pi/180)
  y_traj = r_traj * np.sin(phi_traj*np.pi/180)
  
  xyz_traj = np.concatenate([x_traj, y_traj, z_traj], axis = 2)
  
  return xyz_traj

fake_traj = smoothing_fn(fake_traj)

input_max = np.array([60, 500, 300])
input_min = np.array([20, 0, 0])

fake_traj = fake_traj * (input_max - input_min) + input_min
true_traj = true_traj * (input_max - input_min) + input_min

fake_xyz = rz2xyz(fake_traj)
true_xyz = rz2xyz(true_traj)

fake_xyz_save = np.reshape(np.transpose(fake_xyz, (0, 2, 1)), newshape = (45, 1100))
true_xyz_save = np.reshape(np.transpose(true_xyz, (0, 2, 1)), newshape = (45, 1100))

#np.savetxt('cnn_gan_fake_xyz.txt', fake_xyz_save, delimiter='\t')
#np.savetxt('cnn_gan_true_xyz.txt', true_xyz_save, delimiter='\t')

BATCH_NUM = 0
plt.subplot(211)
plt.plot(np.linspace(1, 1100, 1100), true_xyz[BATCH_NUM,:,0], 'r', np.linspace(1, 1100, 1100), true_xyz[BATCH_NUM,:,1], 'b', np.linspace(1, 1100, 1100), true_xyz[BATCH_NUM,:,2], 'g')
plt.subplot(212)
plt.plot(np.linspace(1, 1100, 1100), fake_xyz[BATCH_NUM,:,0], 'r', np.linspace(1, 1100, 1100), fake_xyz[BATCH_NUM,:,1], 'b', np.linspace(1, 1100, 1100), fake_xyz[BATCH_NUM,:,2], 'g')
plt.show()

plt.subplot(211)
for j in range(15):
  plt.plot(true_xyz[j,:,0], true_xyz[j,:,1])
plt.subplot(212)
for j in range(15):
  plt.plot(fake_xyz[j,:,0], fake_xyz[j,:,1])
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(fake_xyz[BATCH_NUM, :, 0], fake_xyz[BATCH_NUM,:, 1], fake_xyz[BATCH_NUM,:, 2], label='generated curve')
ax.plot(true_xyz[BATCH_NUM, :, 0], true_xyz[BATCH_NUM,:, 1], true_xyz[BATCH_NUM,:, 2], label='experiment curve')
plt.xlabel('x')
plt.ylabel('y')
ax.legend()
plt.show()
