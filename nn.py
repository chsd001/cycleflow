import numpy as np
import tensorflow as tf
tf.set_random_seed(0)
np.random.seed(0)
from tensorflow.python.framework import ops

def int_shape(x):
    return list(map(int, x.get_shape()))

# Abstract class that can propagate both forward/backward,
# along with jacobians.
class Layer():
  def __init__(self, mask_type, name='Coupling'):
    tf.set_random_seed(0)
    np.random.seed(0)
  
  def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
    raise NotImplementedError(str(type(self)))

  def backward(self, y, z):
    raise NotImplementedError(str(type(self)))

def batch_norm(input_,
                name,
                train=True,
                epsilon=1e-6, 
                decay=.1,
                axes=[0, 1],
                reuse=None,
                bn_lag=0.,
                dim=[],
                scaling = True):
  """Batch normalization with corresponding log determinant Jacobian."""
  if reuse is None:
      reuse = not train
  # create variables
  with tf.variable_scope(name) as scope:
      if reuse:
          scope.reuse_variables()
      var = tf.get_variable(
          "var", dim, tf.float32, tf.constant_initializer(1.), trainable=False)
      mean = tf.get_variable(
          "mean", dim, tf.float32, tf.constant_initializer(0.), trainable=False)
      step = tf.get_variable("step", [], tf.float32, tf.constant_initializer(0.), trainable=False)
      if scaling:
        scale_g = tf.get_variable("g_scale", dim, tf.float32, tf.constant_initializer(1.))
        shift_b = tf.get_variable("g_shift", dim, tf.float32, tf.constant_initializer(0.))
  # choose the appropriate moments
  if train:
      used_mean, used_var = tf.nn.moments(input_, axes, name="batch_norm")
      cur_mean, cur_var = used_mean, used_var
      if bn_lag > 0.:
          used_var = stable_var(input_=input_, mean=used_mean, axes=axes)
          cur_var = used_var
          used_mean -= (1 - bn_lag) * (used_mean - tf.stop_gradient(mean))
          used_mean /= (1. - bn_lag**(step + 1))
          used_var -= (1 - bn_lag) * (used_var - tf.stop_gradient(var))
          used_var /= (1. - bn_lag**(step + 1))
  else:
      used_mean, used_var = mean, var
      cur_mean, cur_var = used_mean, used_var

  # update variables
  if train:
      with tf.name_scope(name, "AssignMovingAvg", [mean, cur_mean, decay]):
          with ops.colocate_with(mean):
              new_mean = tf.assign_sub(
                  mean,
                  tf.check_numerics(
                      decay * (mean - cur_mean), "NaN in moving mean."))
      with tf.name_scope(name, "AssignMovingAvg", [var, cur_var, decay]):
          with ops.colocate_with(var):
              new_var = tf.assign_sub(
                  var,
                  tf.check_numerics(decay * (var - cur_var),
                                    "NaN in moving variance."))
      with tf.name_scope(name, "IncrementTime", [step]):
          with ops.colocate_with(step):
              new_step = tf.assign_add(step, 1.)
      used_var += 0. * new_mean * new_var * new_step
  used_var += epsilon
  if scaling:
    return ((input_- used_mean)/tf.sqrt(used_var)) * scale_g + shift_b
  else:
    return ((input_- used_mean)/tf.sqrt(used_var))

def simple_batch_norm(x):
    mu = tf.reduce_mean(x)
    sig2 = tf.reduce_mean(tf.square(x-mu))    
    x = (x-mu)/tf.sqrt(sig2 + 1.0e-6)
    return x

# The coupling layer.
# Contains code for both checkerboard and channelwise masking.
class CouplingLayer(Layer):

  # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
  def __init__(self, mask_type, name='Coupling', num_residual_blocks=8, scaling=True):
    self.mask_type = mask_type
    self.name = name
    self.num_residual_blocks = num_residual_blocks
    self.scaling = scaling

    tf.set_random_seed(0)
    np.random.seed(0)

  # Weight normalization technique
  def get_normalized_weights(self, name, weights_shape):
    weights = tf.get_variable(name, weights_shape, tf.float32,
                              tf.contrib.layers.xavier_initializer())
    scale = tf.get_variable(name + "_scale", [weights_shape[-1]], tf.float32, 
                              tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(5e-5))
    norm = tf.sqrt(tf.reduce_sum(tf.square(weights), [0, 1, 2]))
    return weights/norm * scale
    
  
  # corresponds to the function m and l in the RealNVP paper
  # (Function m and l became s and t in the new version of the paper)
  def function_l_m(self,x,mask,name='function_l_m', reuse=False, train=False):
    with tf.variable_scope(name, reuse=reuse):
      channel = 64
      padding = 'SAME'
      xs = int_shape(x)
      kernel_h = 3
      kernel_w = 3
      input_channel = xs[3]
      y = x

      # y = batch_norm(input_=y, name="g_bn_in1", train=train, scale=False)
      if not self.scaling:
        y = simple_batch_norm(y)
      else:
        y = batch_norm(input_=y, dim=input_channel, name="g_bn_in",
          train=train, epsilon=1e-4, axes=[0,1,2], reuse=reuse, scaling=False)
      weights_shape = [1, 1, input_channel, channel]
      weights = self.get_normalized_weights("g_weights_input", weights_shape)
      
      y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
      if not self.scaling:
        print("this")
        y = simple_batch_norm(y)
        # y = batch_norm(input_=y, dim=channel, name="g_bn_in2",
        #   train=train, epsilon=1e-4, axes=[0,1,2], reuse=reuse, scaling=False)
      else:
        biases = tf.get_variable('g_biases_input', [channel], initializer=tf.constant_initializer(0.0))
        y = tf.reshape(tf.nn.bias_add(y, biases), y.get_shape())
      y = tf.nn.relu(y)
      if self.scaling:
        y = batch_norm(input_=y, dim=channel, name="g_bn_in2",
          train=train, epsilon=1e-4, axes=[0,1,2], reuse=reuse)


      skip = y
      # Residual blocks
      num_residual_blocks = self.num_residual_blocks
      for r in range(num_residual_blocks):
        weights_shape = [kernel_h, kernel_w, channel, channel]
        weights = self.get_normalized_weights("g_weights%d_1" % r, weights_shape)
        y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
        if not self.scaling:
          y = simple_batch_norm(y)
          # y = batch_norm(input_=y, dim=channel, name="g_bn%d_1" % r,
          #   train=train, epsilon=1e-4, axes=[0,1,2], reuse=reuse, scaling=False)
        else:
          biases = tf.get_variable('g_biases_%d_1' % r, [channel], initializer=tf.constant_initializer(0.0))
          y = tf.reshape(tf.nn.bias_add(y, biases), y.get_shape())
        y = tf.nn.relu(y)
        if self.scaling:
          y = batch_norm(input_=y, dim=channel, name="g_bn%d_1" % r,
            train=train, epsilon=1e-4, axes=[0,1,2], reuse=reuse, scaling=False)
        
        weights_shape = [kernel_h, kernel_w, channel, channel]
        weights = self.get_normalized_weights("g_weights%d_2" % r, weights_shape)
        y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)

        if not self.scaling:
          y = simple_batch_norm(y)
          # y = batch_norm(input_=y, dim=channel, name="g_bn%d_2" % r,
          #   train=train, epsilon=1e-4, axes=[0,1,2], reuse=reuse, scaling=False)
        else:
          biases = tf.get_variable('g_biases_%d_2' % r, [channel], initializer=tf.constant_initializer(0.0))
          y = tf.reshape(tf.nn.bias_add(y, biases), y.get_shape())

        y += skip
        y = tf.nn.relu(y)
        if self.scaling:
          y = batch_norm(input_=y, dim=channel, name="g_bn%d_2" % r,
            train=train, epsilon=1e-4, axes=[0,1,2], reuse=reuse)
        skip = y

        
      # 1x1 convolution for reducing dimension
      weights = self.get_normalized_weights("g_weights_output", 
                                            [1, 1, channel, input_channel*2])
      y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)    
      biases = tf.get_variable('g_biases_output', [input_channel*2], initializer=tf.constant_initializer(0.0))
      y = tf.reshape(tf.nn.bias_add(y, biases), y.get_shape())
      # For numerical stability, apply tanh and then scale
      y = tf.tanh(y)
      
      if 'checkerboard' in self.mask_type:
        scale_factor = tf.get_variable("g_weights_tanh_scale", [1], tf.float32, \
          tf.constant_initializer(0.), regularizer=tf.contrib.layers.l2_regularizer(5e-5))
      else:
        scale_factor = tf.get_variable("g_weights_tanh_scale", [1], tf.float32, \
          tf.constant_initializer(1.))
      scale_shift = tf.get_variable("g_weights_scale_shift", [1], tf.float32, \
          tf.constant_initializer(0.))
      
      

      # The first half defines the l function
      # The second half defines the m function
      l = (y[:,:,:,:input_channel] * scale_factor + scale_shift) * (-mask+1)
      m = y[:,:,:,input_channel:] * (-mask+1)

      return l,m

  # returns constant tensor of masks
  # |xs| is the size of tensor
  # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
  # |b| has the dimension of |xs|
  def get_mask(self, xs, mask_type):

    if 'checkerboard' in mask_type:
      unit0 = tf.constant([[1.0, 0.0]])
      unit1 = -unit0 + 1.0
      unit = unit0 if mask_type == 'checkerboard0' else unit1
      unit = tf.reshape(unit, [1, 1, 2, 1])
      b = tf.tile(unit, [xs[0], xs[1], xs[2]//2, xs[3]])
    elif 'channel' in mask_type:
      white = tf.ones([xs[0], xs[1], xs[2], xs[3]//2])
      black = tf.zeros([xs[0], xs[1], xs[2], xs[3]//2])
      if mask_type == 'channel0':
        b = tf.concat(axis=3, values=[white, black])
      else:
        b = tf.concat(axis=3, values=[black, white])

    bs = int_shape(b)
    assert bs == xs

    return b

  # corresponds to the coupling layer of the RealNVP paper
  # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
  # log_det_jacobian is a 1D tensor of size (batch_size)
  def forward_and_jacobian(self, x, sum_log_det_jacobians, z, reuse=False, train=False):
    with tf.variable_scope(self.name, reuse=reuse):
      xs = int_shape(x)
      b = self.get_mask(xs, self.mask_type)

      # masked half of x
      x1 = x * b
      l,m = self.function_l_m(x1, b, reuse=reuse, train=train)
      y = x1 + tf.multiply(-b+1.0, x*tf.check_numerics(tf.exp(l), "exp has NaN") + m)
      log_det_jacobian = tf.reduce_sum(l, [1,2,3])
      sum_log_det_jacobians += log_det_jacobian

      return y,sum_log_det_jacobians, z

  def backward(self, y, z, reuse=False, train=False):    
    with tf.variable_scope(self.name, reuse=True):
      ys = int_shape(y)
      b = self.get_mask(ys, self.mask_type)

      y1 = y * b
      l,m = self.function_l_m(y1, b, reuse=reuse, train=train)
      x = y1 + tf.multiply( y*(-b+1.0) - m, tf.check_numerics(tf.exp(-l), "exp has NaN"))
      return x, z

# The layer that performs squeezing.
# Only changes the dimension.
# The Jacobian is untouched and just passed to the next layer
class SqueezingLayer(Layer):
  def __init__(self, name="Squeeze"):
    self.name = name

  def forward_and_jacobian(self, x, sum_log_det_jacobians, z, reuse=False, train=False):
    xs = int_shape(x)
    assert xs[2] % 2 == 0
    print(xs)
    y = tf.reshape(x, [xs[0], xs[1], xs[2]//2, xs[3]*2])
    print(y.get_shape())
    if z is not None:
      zs = int_shape(z)
      z = y = tf.reshape(z, [zs[0], zs[1], zs[2]//2, zs[3]*2])      

    return y,sum_log_det_jacobians, z

  def backward(self, y, z, reuse=False, train=False):
    ys = int_shape(y)
    assert ys[3] % 2 == 0
    x = tf.reshape(y, [ys[0], ys[1], ys[2]*2, ys[3]/2])

    if z is not None:
      zs = int_shape(z)
      z = tf.reshape(z, [zs[0], zs[1], zs[2]*2, zs[3]/2])

    return x, z

# The layer that factors out half of the variables
# directly to the latent space.  
class FactorOutLayer(Layer):
  def __init__(self, scale, name='FactorOut'):
    self.scale = scale
    self.name = name
  
  def forward_and_jacobian(self, x, sum_log_det_jacobians, z, reuse=False, train=False):

    xs = int_shape(x)
    split = xs[3]//2

    # The factoring out is done on the channel direction.
    # Haven't experimented with other ways of factoring out.
    new_z = x[:,:,:,:split]
    x = x[:,:,:,split:]

    if z is not None:
      z = tf.concat(axis=3, values=[z, new_z])
    else:
      z = new_z
    
    return x, sum_log_det_jacobians, z
  
  def backward(self, y, z, reuse=False, train=False):

    # At scale 0, 1/2 of the original dimensions are factored out
    # At scale 1, 1/4 of the original dimensions are factored out
    # ....
    # At scale s, (1/2)^(s+1) are factored out
    # Hence, at backward pass of scale s, (1/2)^(s) of z should be factored in
    
    zs = int_shape(z)
    if y is None:
      split = zs[3] // (2**self.scale)
    else:
      split = int_shape(y)[3]
    new_y = z[:,:,:,-split:]
    z = z[:,:,:,:-split]

    assert (int_shape(new_y)[3] == split)

    if y is not None:
      x = tf.concat(axis=3, values=[new_y, y])
    else:
      x = new_y

    return x, z


# Given the output of the network and all jacobians, 
# compute the log probability.
def compute_log_density_x(z, sum_log_det_jacobians, prior):

  zs = int_shape(z)
  if len(zs) == 4:
    K = zs[1]*zs[2]*zs[3] #dimension of the Gaussian distribution
    z = tf.reshape(z, (-1, K))
  else:
    K = zs[1]
  if prior == "gaussian":
    log_density_z = -0.5*tf.reduce_sum(tf.square(z), [1]) -0.5*K*np.log(2*np.pi)
  elif prior == "logistic":
    log_density_z = -tf.reduce_sum(-z + 2*tf.nn.softplus(z),[1])
  elif prior == "uniform":
    log_density_z = 0
  log_density_x = log_density_z + sum_log_det_jacobians

  return log_density_x


# Computes log_likelihood of the network
def log_likelihood(z, sum_log_det_jacobians, prior):
  return -tf.reduce_sum(compute_log_density_x(z, sum_log_det_jacobians, prior))