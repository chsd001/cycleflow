"""
The core Real-NVP model
"""

import tensorflow as tf
import nn
tf.set_random_seed(0)
layers = []
# the final dimension of the latent space is recorded here
# so that it can be used for constructing the inverse model
final_latent_dimension = []

def construct_model_spec(scale_init=2, no_of_layers=8, add_scaling=True):
  global layers
  num_scales = 2
  for scale in range(num_scales-1):    
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale))
    layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale))
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale))
    layers.append(nn.SqueezingLayer(name='Squeeze%d' % scale))
    layers.append(nn.CouplingLayer('channel0', name='Channel%d_1' % scale))
    layers.append(nn.CouplingLayer('channel1', name='Channel%d_2' % scale))
    layers.append(nn.CouplingLayer('channel0', name='Channel%d_3' % scale))
    layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))

  # final layer
  scale = num_scales-1
  layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale))
  layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale))
  layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale))
  layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_4' % scale))
  layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))


def model_spec(x, reuse=True, train=False, 
  alpha=1e-7, init_type="uniform", hidden_layers=1000, no_of_layers=1, batch_norm_adaptive=0):
  xs = nn.int_shape(x)
  sum_log_det_jacobians = tf.zeros(xs[0])   

  jac = 0  

  x = x*(1-2*alpha) + alpha
  jac = tf.reduce_sum(-tf.log(x) - tf.log(1-x)+tf.log(1-2*alpha), [1,2,3])
  x = tf.log(x) - tf.log(1-x)
  sum_log_det_jacobians += jac

  if len(layers) == 0:
      construct_model_spec(no_of_layers=no_of_layers, add_scaling=(batch_norm_adaptive != 0))
  
  # construct forward pass    
  z = None
  jac = sum_log_det_jacobians
  for layer in layers:
    x,jac,z = layer.forward_and_jacobian(x, jac, z, reuse=reuse, train=train) 
    


  x = tf.concat(axis=3, values=[z,x])

  # record dimension of the final variable
  global final_latent_dimension
  final_latent_dimension = nn.int_shape(z)

  return z,jac

def inv_model_spec(y, reuse=False, train=False, alpha=1e-7):
  # construct inverse pass for sampling
  shape = final_latent_dimension
  z = tf.reshape(y, [-1, shape[1], shape[2], shape[3]])
  y = None

  for layer in reversed(layers):
    y,z = layer.backward(y,z, reuse=reuse, train=train)

  # inverse logit
  x = tf.sigmoid(y)
  x = (x-alpha)/(1-2*alpha)
  return x
    
  
