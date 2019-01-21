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

real_xyz = tf.placeholder("float", shape = [None, 1100, 3])
fake_xyz = generator(real_xyz[:,:10])

parameter_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'rnn_gen')
reuse_vars_dict = dict([(var.op.name, var) for var in parameter_g])
pre_vars_dict = {**reuse_vars_dict}
pre_saver = tf.train.Saver(pre_vars_dict)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
  init.run()
  full_xyz = xyz_norm(full_batches)
  pre_saver.restore(sess,"/tmp/ash_particle/Newfolder/CNN_GAN.ckpt")
  np.random.shuffle(full_xyz)
  
  fake_traj = sess.run(fake_xyz, feed_dict = {real_xyz: full_xyz})
  true_traj = full_xyz
  
def smoothing_fn(traj):
  for j in range(15):
    tck, u = interpolate.splprep([traj[j,:,0], traj[j,:,1], traj[j,:,2]])
    u_fine = np.linspace(0,1,1100)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    
    traj[j,:,0] = x_fine
    traj[j,:,1] = y_fine
    traj[j,:,2] = z_fine
  return traj

def rz2xyz(rz_traj):
  r_traj = rz_traj[:,:,0:1]
  phi_traj = rz_traj[:,:,1:2]
  z_traj = rz_traj[:,:,2:3]
  
  x_traj = r_traj * np.cos(phi_traj*np.pi/180)
  y_traj = r_traj * np.sin(phi_traj*np.pi/180)
  
  xyz_traj = np.concatenate([x_traj, y_traj, z_traj], axis = 2)
  
  return xyz_traj

#fake_traj = smoothing_fn(fake_traj)

input_max = np.array([60, 500, 300])
input_min = np.array([20, 0, 0])

fake_traj = fake_traj * (input_max - input_min) + input_min
true_traj = true_traj * (input_max - input_min) + input_min

fake_xyz = rz2xyz(fake_traj)
true_xyz = rz2xyz(true_traj)

fake_xyz_save = np.reshape(np.transpose(fake_xyz, (0, 2, 1)), newshape = (45, 1100))
true_xyz_save = np.reshape(np.transpose(true_xyz, (0, 2, 1)), newshape = (45, 1100))

np.savetxt('cnn_gan_fake_xyz.txt', fake_xyz_save, delimiter='\t')
np.savetxt('cnn_gan_true_xyz.txt', true_xyz_save, delimiter='\t')

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
