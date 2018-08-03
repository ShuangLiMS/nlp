
from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K
from keras.engine import InputSpec

import os

import logging_util
import tensorflow as tf

logger = logging_util.logger(__name__)

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)



class AttWeightLayer(Layer):
    def __init__(self, **kwargs):
        self.init_normal = initializers.get('normal')
        self.init_zeros = initializers.get('zeros')

        # self.input_spec = [InputSpec(ndim=3)]
        """ 
        Keras documenation on Recurrent layers.

        To introduce masks to data, use an Embedding layer
        with the mask_zero parameter set to True.
        """

        super(AttWeightLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_spec = [InputSpec(shape=input_shape)]

        nb_samples, nb_time, input_dim = input_shape

        self.W = self.add_weight(name=f"{self.name}_W",
                                 shape=(input_dim, input_dim),
                                 initializer="normal",
                                 trainable=True)
        self.b = self.add_weight(name=f"{self.name}_b",
                                 shape=[input_dim],
                                 initializer="zero",
                                 trainable=True)

        self.u = self.add_weight(name=f"{self.name}_u",
                                 shape=[input_dim, 1],
                                 initializer='normal',
                                 trainable=True
                                 )

        super(AttWeightLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        #u_it = tanh(W * h_it + b)
        uit = K.tanh(dot_product(x, self.W) + self.b)


        # a_it = u_it^T * u, # compute similarity score
        ait = K.dot(uit, self.u)
        ait_shape = K.shape(ait)

        # reduce the ait to 2 dimension tensor as the last dimension is of size 1
        ait = K.reshape(ait, [-1, ait_shape[-2] * ait_shape[-1]])

        # convert the similarity score to softmax
        ai = K.exp(ait)  # take the exponential

        if mask is not None:
            # by now ait needs to be 2 dimensional, since mask is of 2 dimensional. Otherwise
            # ait of shape [None, 100, 1] and mask of shape [None, 100], would results in a tensor of shape [None, 100, 100]
            # the mask is a 2-dimension mask for masking padding zeros.
            ai *= K.cast(mask, K.floatx())

        # the sum is taken over the axis 1 the timestep dimension, instead of the last dimension 2 the embedding vector dimension
        # add espislon to avoid dividing by zero error.
        weights = ai / K.cast(K.sum(ai, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        # now the weights needs to add the 3rd dimension back
        weights_expanded = K.expand_dims(weights)

        return weights_expanded

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1], 1)



class AttentionApplyLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionApplyLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.input_spec = [InputSpec(shape=item) for item in input_shape]

        super(AttentionApplyLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, inputs, mask=None):

        weights_expanded = inputs[0]
        x = inputs[1]

        # x is of shape [None, 100, 64], weights [None, 100, 1]
        # the next element-wise multiplication works by broadcasting, as defined here: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        weighted_input = x * weights_expanded

        return K.sum(weighted_input, axis=1)  # the sum is at the timestep dimension


    def compute_output_shape(self, input_shape):
        return (input_shape[-1][0], input_shape[-1][-1])

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            # this layer has two input tensors, hence comes with two masks
            # but there is only one output tensor, therefore we need remove one mask
            assert(len(inputs) == len(mask))
            return mask[0]
        else:
            return None







