from tensorflow.keras import activations, constraints, initializers, regularizers
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K




class Bilinear(Layer):

    def __init__(self,
                 dropout=0,
                 use_bias=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.dropout = dropout

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        self.kernel = self.add_weight(shape=(input_shapes[-1], input_shapes[-1]),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True
        super().build(input_shapes)

    def call(self, inputs, **kwargs):

        x = inputs
        x = tf.nn.dropout(x, self.dropout)
        h1 = tf.matmul(x, self.kernel)
        output = tf.matmul(h1, tf.transpose(x))

        if self.use_bias:
            output += self.bias

        return self.activation(output)
