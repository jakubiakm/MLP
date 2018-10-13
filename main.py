import tensorflow as tf
import data as data
import mlp as mlp
from config import cfg
import os
def main(_):
    print(f'Neurons in layers = {cfg.neurons_in_layers}')
    print(f'Activation function = {cfg.activation_function}')
    print(f'Use biases = {cfg.use_biases}')
    print(f'Learning type = {cfg.learning_type}')
    print(f'Batch size = {cfg.batch_size}')
    print(f'Number of epochs = {cfg.training_epochs}')
    print(f'Problem type = {cfg.problem_type}')
    print(f'Learning file path = {cfg.training_path}')
    print(f'Training file path = {cfg.test_path}')
    print(f'Use gpu = {cfg.use_gpu}')

    training_data = data.get_data(cfg.training_path, cfg.problem_type)
    test_data = data.get_data(cfg.test_path, cfg.problem_type)
    mlp.learn(training_data, test_data)
    
if __name__ == "__main__":
    if(cfg.use_gpu == False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
    tf.app.run()