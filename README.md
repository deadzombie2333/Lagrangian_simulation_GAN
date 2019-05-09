# Lagrangian_simulation_GAN

Simulate Lagrangian motion use generative adversarial network.

https://arxiv.org/abs/1901.03960

Files in repository:

full_dictionary: full trajectory coordinates in r-\theta-z; dictionary, 15 elements, [1100,3] dimension.

input_array: initial coordinates of each 100 coordinates trajectories; dictionary, 165 elements, [6, 3] dimension.

output_array: full coordinates of each 100 coordinates trajectories; dictionary, 165 elements, [100, 3] dimension.

seq2seq_short_GAN.ckpt: trained model parameter after 5000 iterations.

lagrangian_gan.py: training program. Use input_array to generate simulated Lagrangian motion trajectory for adversarial training (compare with output_array).

gan_recovery: model recovery program. Use trained model parameter and full_dictionary to generate full length Lagrangian motion simulations.

sqe2seq_short_training_curve: obtained training curve.![GitHub Logo](https://github.com/deadzombie2333/Lagrangian_simulation_GAN/blob/master/Figure_2.png)
