import tensorflow as tf

flags = tf.app.flags


#####################
#command prompt flags
#####################
flags.DEFINE_integer('training_epochs', 100, 'Number of epochs')
flags.DEFINE_integer('training_iterations', 0, 'Number of iterations')
flags.DEFINE_integer('batch_size', 200, 'Batch size')
flags.DEFINE_integer('display_step', 5, 'Display step')

flags.DEFINE_boolean('use_biases', True, 'Use biases')
flags.DEFINE_boolean('use_gpu', True, 'Use gpu')

flags.DEFINE_string('neurons_in_layers', '[3, 3, 3]', 'Number of neurons in layers (array input like ''[1, 2, 3]''')
flags.DEFINE_string('activation_function', 'sigmoid', 'Activation function: [relu, relu6, crelu, elu, selu, softplus, softsign, dropout, sigmoid, tanh]')
flags.DEFINE_string('learning_type', 'batch', 'Learning type: [online, batch]')
flags.DEFINE_string('problem_type', 'classification', 'Problem type: [classification, regression]')
flags.DEFINE_string('training_path', r'.\data\classification\data.three_gauss.train.10000.csv', 'Training file path')
flags.DEFINE_string('test_path', r'.\data\classification\data.three_gauss.test.10000.csv', 'Test file path')
flags.DEFINE_string('save_file', r'.\output\save.model', 'File model is saved to')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('momentum', 0.9, 'Momentum parameter for SGD')


#######################
# Visualization Flags #
#######################
flags.DEFINE_float('points_drawing_sampling', 0.01, 'Point sampling')
flags.DEFINE_float('points_drawing_pixel_size_multiplier', 800, 'pixel size is equal to points_drawing_sampling * points_drawing_pixel_size_multiplier')

cfg = tf.app.flags.FLAGS
