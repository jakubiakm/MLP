import tensorflow as tf
import data as data
import mlp as mlp
from config import cfg
import os

def validate_arguments():
    if((cfg.learning_type == 'batch' or cfg.learning_type == 'online') == False):
        raise ValueError('Wrong learning_type value. Possible values are: [''batch'', ''online'']')
    activation_functions = ['relu', 'relu6', 'elu', 'selu', 'softplus', 'softsign', 'dropout', 'sigmoid', 'tanh']
    if (cfg.activation_function in activation_functions) == False:
        raise ValueError('Wrong learning_type value. Possible values are: [''relu'', ''relu6'', ''elu'', ''selu'', ''softplus'', ''softsign'', ''dropout'', ''sigmoid'', ''tanh'']')


def print_arguments():
    print(f'Neurons in layers = {cfg.neurons_in_layers}')
    print(f'Activation function = {cfg.activation_function}')
    print(f'Use biases = {cfg.use_biases}')
    print(f'Learning type = {cfg.learning_type}')
    print(f'Batch size = {cfg.batch_size}')
    if(cfg.training_iterations != 0):
        print(f'Number of iterations = {cfg.training_iterations}')
    else:
        print(f'Number of epochs = {cfg.training_epochs}')    
    print(f'Problem type = {cfg.problem_type}')
    print(f'Learning file path = {cfg.training_path}')
    print(f'Training file path = {cfg.test_path}')
    print(f'Use gpu = {cfg.use_gpu}')
    print(f'Display step = {cfg.display_step}')
    print(f'Momentum = {cfg.momentum}')

def main(_):
    validate_arguments()
    print_arguments()

    training_data = data.get_data(cfg.training_path, cfg.problem_type)
    test_data = data.get_data(cfg.test_path, cfg.problem_type)
    mlp.learn(training_data, test_data)

def one_iteration_main():
    validate_arguments()
    print_arguments()

    training_data = data.get_data(cfg.training_path, cfg.problem_type)
    test_data = data.get_data(cfg.test_path, cfg.problem_type)
    mlp.learn_one_epoch(training_data, test_data)

def all_iteration_main():
    validate_arguments()
    print_arguments()

    training_data = data.get_data(cfg.training_path, cfg.problem_type)
    test_data = data.get_data(cfg.test_path, cfg.problem_type)
    mlp.learn_all_epochs(training_data, test_data)

def destroy():
    mlp.destroy()
    
if __name__ == "__main__":
    if(cfg.use_gpu == False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
    tf.app.run()