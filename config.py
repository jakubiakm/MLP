import tensorflow as tf

flags = tf.app.flags


#####################
#command prompt flags
#####################
flags.DEFINE_integer('number_of_layers', 2, 'Number of layers')
flags.DEFINE_integer('number_of_neurons', 50, 'Number of neurons in layers')
flags.DEFINE_integer('iterations', 100, 'Number of iterations')

flags.DEFINE_boolean('use_biases', True, 'Use biases')

flags.DEFINE_string('activation_function', '1=1', 'Activation function')
flags.DEFINE_string('learning_type', 'online', 'Learning type: [online, batch]')
flags.DEFINE_string('problem_type', 'classification', 'Problem type: [classification, regression]')
flags.DEFINE_string('training_path', r'.\data\classification\data.simple.train.100.csv', 'Training file path')
flags.DEFINE_string('test_path', r'.\data\classification\data.simple.test.100.csv', 'Test file path')


cfg = tf.app.flags.FLAGS
