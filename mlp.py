import tensorflow as tf
import numpy as np
import math
from config import cfg

class CountingVariables:
    
    def __init__(self, **kwargs):
        self.learning_rate = None
        self.training_epochs = None
        self.batch_size = None
        self.display_step = None

        # wielkość wektora cech
        self.n_input = None
    
    # ilość możliwych klas do klasyfikacji
        self.n_classes = None

    # wektory krańcowe
        self.X = None
        self.Y = None

    # buduje model
        self.model = None

    # zdefiniowanie funkcji straty i optymalizatora
        self.loss_op = None
    
        self.optimizer = None
    
        self.train_op = None
    
    # inicjalizacja zmiennych
        self.init = None

        self.training_data = None

        self.test_data = None

        self.x_iterable = None
        self.y_iterable = None
        self.session = None
        self.epoch_number = None
        self.learning_results = None

    def initialize(self, training_data, test_data):
        tf.set_random_seed(666)
        if self.training_data != None:
            return
        self.training_data = training_data
        self.test_data = test_data
        self.learning_rate = cfg.learning_rate
        self.learning_results = []
        self.training_epochs = cfg.training_epochs
        self.batch_size = cfg.batch_size if cfg.learning_type == 'batch' else 1
        self.display_step = cfg.display_step
        self.epoch_number = 0

        # wielkość wektora cech
        self.n_input = len(vars(self.training_data[0])) - 1
    
        # ilość możliwych klas do klasyfikacji
        self.n_classes = len(set([item.cls for item in self.training_data])) if cfg.problem_type == 'classification' else 1

        # wektory krańcowe
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])

        # buduje model
        self.model = construct_multilayer_perceptron_model(self.X, self.n_input, self.n_classes)

        # zdefiniowanie funkcji straty i optymalizatora
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.model, labels=self.Y))
    
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=cfg.momentum)# ,use_nesterov=True)
        

        self.train_op = self.optimizer.minimize(self.loss_op)
    
        # inicjalizacja zmiennych
        self.init = tf.global_variables_initializer()
        self.session = tf.Session();
        self.session.run([self.init])
        self.session.__enter__()
        self.x_iterable = [[item.x, item.y] for item in self.training_data]
        self.y_iterable = convert_classification_labels_vector_to_tensorflow_output(
            [item.cls for item in self.training_data])

    def destroy(self):
        self.session.__exit__()


_counting_variables = CountingVariables()
# zwraca generator do pobierania danych - nie obciąża pamięci przy tworzeniu listy
def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def activation_function(x, feature):
    return {
        'relu': tf.nn.relu(feature),
        'relu6': tf.nn.relu6(feature),
        'crelu': tf.nn.crelu(feature),
        'elu': tf.nn.elu(feature),
        'selu': tf.nn.selu(feature),
        'softplus': tf.nn.softplus(feature),
        'softsign': tf.nn.softsign(feature),
        'dropout': tf.nn.dropout(feature, 1),
        'sigmoid': tf.nn.sigmoid(feature),
        'tanh': tf.nn.tanh(feature)
    }.get(x) 

# tworzy wielowarstwoych perceptron neuronowy
def construct_multilayer_perceptron_model(input_tensor, n_input, n_classes):
    layers_neurons = eval(cfg.neurons_in_layers)
    ind = 0
    previous_layer_neurons_count = 0
    weights = {}
    biases = {}
    
    # generowanie zmiennych na wagi i bias
    previous_layer_neurons_count = n_input
    for neurons in layers_neurons:
        weights['w' + str(ind)] = tf.Variable(tf.random_normal([previous_layer_neurons_count, neurons]))
        biases['b' + str(ind)] = tf.Variable(tf.random_normal([neurons]))
        previous_layer_neurons_count = neurons 
        ind += 1
    
    weights['out'] = tf.Variable(tf.random_normal([previous_layer_neurons_count, n_classes]))
    biases['out'] = tf.Variable(tf.random_normal([n_classes]))

    previous_layer = input_tensor
    #tworzenie w pełni połączonych warstw neurnowych
    for ind in range(len(layers_neurons)):
        if(cfg.use_biases == False):
            previous_layer = tf.matmul(previous_layer, weights['w' + str(ind)])
        else:
            previous_layer = tf.add(tf.matmul(previous_layer, weights['w' + str(ind)]), biases['b' + str(ind)])   
        previous_layer = activation_function(cfg.activation_function, previous_layer)

    if(cfg.use_biases == False):
        return tf.matmul(previous_layer, weights['out'])
    else:
        return tf.matmul(previous_layer, weights['out']) + biases['out']


# przekształca wektor wyjściowy dla problemu klasyfikacji (pobrany z excela z klas)
# na wektor wyjściowy odpowiedniego rozmiaru dla tensorflow 
def convert_classification_labels_vector_to_tensorflow_output(labels_vector):
    number_of_classes = len(set(labels_vector))
    labels_vector = np.asarray(labels_vector, np.int) - 1
    labels_vector = tf.one_hot(labels_vector, number_of_classes).eval()
    return labels_vector        


# testuje model i wyświetla wyniki na wyjściu
def test(model, test_data, X, printResult = True):
    length = len(test_data)
    test_features = [[item.x, item.y] for item in test_data]
    predictions_labels = tf.argmax(model, 1).eval(feed_dict={X: test_features})
    labels_vector = np.asarray([item.cls for item in test_data], np.int) - 1
    points = 0
    for ind in range(length):
        if predictions_labels[ind] == labels_vector[ind]:
            points = points + 1
    accuracy = points / length
    if printResult:
        print("Accuracy:", accuracy)
    return accuracy


# uczy model i wyświetla wyniki skuteczności
def learn(training_data, test_data):
    tf.set_random_seed(666)

    learning_rate = cfg.learning_rate
    training_epochs = cfg.training_epochs
    batch_size = cfg.batch_size if cfg.learning_type == 'batch' else 1
    display_step = cfg.display_step

    # wielkość wektora cech
    n_input = len(vars(training_data[0])) - 1
    
    # ilość możliwych klas do klasyfikacji
    n_classes = len(set([item.cls for item in training_data])) if cfg.problem_type == 'classification' else 1

    # wektory krańcowe
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    # buduje model
    model = construct_multilayer_perceptron_model(X, n_input, n_classes)

    # zdefiniowanie funkcji straty i optymalizatora
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=model, labels=Y))
    
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=cfg.momentum)# ,use_nesterov=True)
    
    train_op = optimizer.minimize(loss_op)
    
    # inicjalizacja zmiennych
    init = tf.global_variables_initializer()
 
    # uczenie
    with tf.Session() as sess:
        sess.run([init])
        
        x_iterable = [[item.x, item.y] for item in training_data]
        y_iterable = convert_classification_labels_vector_to_tensorflow_output(
            [item.cls for item in training_data])

        for epoch in range(training_epochs):      
            batch_x_generator = batch(x_iterable, batch_size)
            batch_y_generator = batch(y_iterable, batch_size)
            avg_cost = 0.
            total_batch = int(math.ceil(len(training_data) / batch_size))
            
            for _ in range(total_batch):
                batch_x, batch_y = next(batch_x_generator), next(batch_y_generator)
                batch_x = np.asarray(batch_x, np.float32)
                batch_y = np.asarray(batch_y, np.float32)
                
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                Y: batch_y})
                # obliczenie średniej straty
                avg_cost += c / total_batch

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
                test(model, test_data, X)

        print("Optimization Finished!")
        test(model, test_data, X)


def learn_all_epochs(training_data, test_data):
    global _counting_variables
    _counting_variables.initialize(training_data, test_data)

    for epoch in range(_counting_variables.training_epochs):
        train_one_iteration(_counting_variables.session, epoch)
    print("Optimization Finished!")
    test(_counting_variables.model, _counting_variables.test_data, _counting_variables.X)

def learn_one_epoch(training_data, test_data):
    global _counting_variables
    _counting_variables.initialize(training_data, test_data)
    train_one_iteration(_counting_variables.session, -1)
    print(f"One epoch (number {_counting_variables.epoch_number}) Optimization Finished!")
    test(_counting_variables.model, _counting_variables.test_data, _counting_variables.X)



def initialize_variables(training_data, test_data):
    global _counting_variables
    print("initializing variables")
    _counting_variables.initialize(training_data, test_data)
    return _counting_variables


def train_one_iteration(sess, epoch):
    global _counting_variables
    batch_x_generator = batch(_counting_variables.x_iterable, _counting_variables.batch_size)
    batch_y_generator = batch(_counting_variables.y_iterable, _counting_variables.batch_size)
    avg_cost = 0.0
    total_batch = int(math.ceil(len(_counting_variables.training_data) / _counting_variables.batch_size))
    _counting_variables.epoch_number += 1
    for _ in range(total_batch):
        batch_x, batch_y = next(batch_x_generator), next(batch_y_generator)
        batch_x = np.asarray(batch_x, np.float32)
        batch_y = np.asarray(batch_y, np.float32)
                
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([_counting_variables.train_op, _counting_variables.loss_op], feed_dict={_counting_variables.X: batch_x,
                                                                _counting_variables.Y: batch_y})
        # obliczenie średniej straty
        avg_cost += c / total_batch

    if epoch > -1 and epoch % _counting_variables.display_step == 0:
        print("Epoch:", '%04d' % (_counting_variables.epoch_number), "cost={:.9f}".format(avg_cost), "current loop epoch: ", '%04d' % (epoch + 1))
        _counting_variables.learning_results.append(test(_counting_variables.model, _counting_variables.test_data, _counting_variables.X, True))
    else:
        _counting_variables.learning_results.append(test(_counting_variables.model, _counting_variables.test_data, _counting_variables.X, False))

def destroy():
    if _counting_variables.session != None:
        _counting_variables.session.close()