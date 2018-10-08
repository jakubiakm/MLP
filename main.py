import tensorflow as tf
from config import cfg

def main(_):
    print(f'Number of layers = {cfg.number_of_layers}')
    print(f'Number of neurons in layers = {cfg.number_of_neurons}')
    print(f'Activation function = {cfg.activation_function}')
    print(f'Use biases = {cfg.use_biases}')
    print(f'Learning type = {cfg.learning_type}')
    print(f'Number of iterations = {cfg.iterations}')
    print(f'Problem type = {cfg.problem_type}')
    
if __name__ == "__main__":
    tf.app.run()