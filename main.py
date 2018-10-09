import tensorflow as tf
import data as d
from config import cfg

def main(_):
    print(f'Number of layers = {cfg.number_of_layers}')
    print(f'Number of neurons in layers = {cfg.number_of_neurons}')
    print(f'Activation function = {cfg.activation_function}')
    print(f'Use biases = {cfg.use_biases}')
    print(f'Learning type = {cfg.learning_type}')
    print(f'Number of iterations = {cfg.iterations}')
    print(f'Problem type = {cfg.problem_type}')
    print(f'Learning file path = {cfg.training_path}')
    print(f'Training file path = {cfg.test_path}')

    training_data = d.get_data(cfg.training_path, cfg.problem_type)
    test_data = d.get_data(cfg.test_path, cfg.problem_type)

if __name__ == "__main__":
    tf.app.run()