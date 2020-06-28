import numpy as np
from keras import backend as K
from keras.engine.topology import Layer

from keras import activations, regularizers, initializers, constraints
from keras.utils import conv_utils
from keras.engine.base_layer import InputSpec

class pruned_Dense(Layer):
    def __init__(self, n_neurons_out, **kwargs):
        super(pruned_Dense,self).__init__(**kwargs)
        self.n_neurons_out = n_neurons_out
        

    def build(self, input_shape):
        #define the variables of this layer in the build function:
        n_neurons_in = input_shape[-1]
        # print(n_neurons_in)
        # print(self.n_neurons_out)
        stdv = 1/np.sqrt(n_neurons_in)
        w = np.random.normal(size=[n_neurons_in, self.n_neurons_out], loc=0.0, scale=stdv).astype(np.float32)
        self.w = K.variable(w)
        b = np.zeros(self.n_neurons_out)
        self.b = K.variable(b)
        # w is the weight matrix, b is the bias. These are the trainable variables of this layer.
        self.trainable_weights = [self.w, self.b]
        # mask is a non-trainable weight that simulates pruning. the values of mask should be either 1 or 0, where 0 will prune a weight. We initialize mask to all ones:
        mask = np.ones((n_neurons_in, self.n_neurons_out))
        self.mask = K.variable(mask)

        super(pruned_Dense, self).build(input_shape)

    def call(self, x):
        # define the input-output relationship in this layer in this function
        pruned_w = self.w * self.mask
        out = K.dot(x, pruned_w)
        out = out + self.b
        return out

    def compute_output_shape(self, input_shape):
        #define the shape of this layer's output:
        return (input_shape[0], self.n_neurons_out)

    def get_mask(self):
        #get the mask values
        return K.get_value(self.mask)

    def set_mask(self, mask):
        #set new mask values to this layer
        K.set_value(self.mask, mask)
    
    def get_config(self):
        config = super(pruned_Dense, self).get_config()
        # config.pop('rank')
        return config

    def set_config(self, config):
        self.n_neurons_out = config['units']

class pruned_Conv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        # print(kwargs)
        super(pruned_Conv2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
    
    def get_config(self):
        config = super(pruned_Conv2D, self).get_config()
        # config.pop('rank')
        return config
        
    def set_config(self, config_in):
        self.rank = 2
        self.filters = config_in['filters']
        self.kernel_size = conv_utils.normalize_tuple(config_in['kernel_size'], self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(config_in['strides'], self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(config_in['padding'])
        self.data_format = K.normalize_data_format(config_in['data_format'])
        self.dilation_rate = conv_utils.normalize_tuple(config_in['dilation_rate'], self.rank, 'dilation_rate')
        self.activation = activations.get(config_in['activation'])
        self.use_bias = config_in['use_bias']
        self.kernel_initializer = initializers.get(config_in['kernel_initializer'])
        self.bias_initializer = initializers.get(config_in['bias_initializer'])
        self.kernel_regularizer = regularizers.get(config_in['kernel_regularizer'])
        self.bias_regularizer = regularizers.get(config_in['bias_regularizer'])
        self.activity_regularizer = regularizers.get(config_in['activity_regularizer'])
        self.kernel_constraint = constraints.get(config_in['kernel_constraint'])
        self.bias_constraint = constraints.get(config_in['bias_constraint'])
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if self.data_format == 'channels_first':
          input_dim = input_shape[1]

        kernel_shape = self.kernel_size + (input_dim, self.filters)
        mask = np.ones(kernel_shape)
        self.mask = K.variable(mask)
        
        self.kernel = self.add_weight(shape = kernel_shape,
                                      initializer = self.kernel_initializer,
                                      name = 'kernel',
                                      regularizer = self.kernel_regularizer,
                                      constraint = self.kernel_constraint)
        if self.use_bias:
          self.bias = self.add_weight(shape = (self.filters,),
                                      initializer = self.bias_initializer,
                                      name = 'bias',
                                      regularizer = self.bias_regularizer,
                                      constraint = self.bias_constraint)
        else:
          self.bias = None
        
        self.trainable_weights = [self.kernel, self.bias]

        self.non_trainable_weights = [self.mask]

        super(pruned_Conv2D, self).build(input_shape)

    def call(self, x):
        masked_kernel = self.kernel * self.mask

        output = K.conv2d(x, masked_kernel, strides = self.strides, padding = self.padding,
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate)
        if self.use_bias:
          output = K.bias_add(output,
                              self.bias,
                              data_format = self.data_format)
        if self.activation is not None:
          return self.activation(output)

        return output
        
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        if self.data_format == 'channels_last':
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == 'channels_first':
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_mask(self):
        return K.get_value(self.mask)

    def set_mask(self, mask):
        K.set_value(self.mask, mask)
