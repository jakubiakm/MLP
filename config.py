import tensorflow as tf

flags = tf.app.flags


#####################
#command prompt flags
#####################
flags.DEFINE_integer('number_of_layers', 2, 'Number of layers')
flags.DEFINE_integer('number_of_neurons', 50, 'Number of neurons in layers')
flags.DEFINE_integer('iterations', 100, 'Number of iterations')

flags.DEFINE_boolean('use_biases', True, 'use_biases')

flags.DEFINE_string('activation_function', '1=1', 'Activation function')
flags.DEFINE_string('learning_type', 'online', 'Learning type: [online, batch]')
flags.DEFINE_string('problem_type', 'classification', 'Problem type: [classification, regression]')

cfg = tf.app.flags.FLAGS
