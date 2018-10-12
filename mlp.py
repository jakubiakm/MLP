import tensorflow as tf
import numpy as np
import math
from config import cfg

# zwraca generator do pobierania danych - nie obciąża pamięci przy tworzeniu listy
def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# tworzy wielowarstwoych perceptron neuronowy
def construct_multilayer_perceptron_model(input_tensor, n_input, n_classes):
    layers_neurons = eval(cfg.neurons_in_layers)
    ind = 0
    previous_layer_neurons_count = 0
    weights = {}
    biases = {}
    
    # generowanie zmiennych na wagi i bias
    for neurons in layers_neurons:
        if(ind == 0):
            weights['w0'] = tf.Variable(tf.random_normal([n_input, neurons]))
        else:
            weights['w' + str(ind)] = tf.Variable(tf.random_normal([previous_layer_neurons_count, neurons]))
        biases['b' + str(ind)] = tf.Variable(tf.random_normal([neurons]))
        previous_layer_neurons_count = neurons 
        ind = ind + 1
    
    weights['out'] = tf.Variable(tf.random_normal([previous_layer_neurons_count, n_classes]))
    biases['out'] = tf.Variable(tf.random_normal([n_classes]))

    #tworzenie w pełni połączonych warstw neurnowych
    for ind in range(len(layers_neurons)):
        if(ind == 0):
            previous_layer = tf.add(tf.matmul(input_tensor, weights['w' + str(ind)]), biases['b' + str(ind)])
        else:
            previous_layer = tf.add(tf.matmul(previous_layer, weights['w' + str(ind)]), biases['b' + str(ind)])   
    
    return tf.matmul(previous_layer, weights['out']) + biases['out']

# przekształca wektor wyjściowy dla problemu klasyfikacji (pobrany z excela z klas)
# na wektor wyjściowy odpowiedniego rozmiaru dla tensorflow 
def convert_classification_labels_vector_to_tensorflow_output(labels_vector):
    labels_vector = np.asarray(labels_vector, np.int) - 1
    labels_vector = labels_vector.reshape(labels_vector.shape[0], 1)
    return np.concatenate((1 - labels_vector, labels_vector), axis=1)

# testuje model i wyświetla wyniki na wyjściu
def test(model, test_data, X, Y):
    pred = tf.nn.softmax(model)  # zastosowanie softmax do model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # obliczenie skuteczności
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    features = [[item.x, item.y] for item in test_data]
    labels = convert_classification_labels_vector_to_tensorflow_output(
        labels_vector=[item.cls for item in test_data])
    print("Accuracy:", accuracy.eval({X: features, Y: labels}))

# uczy model i wyświetla wyniki skuteczności
def learn(training_data, test_data):   
    learning_rate = cfg.learning_rate
    training_epochs = cfg.training_epochs
    batch_size = cfg.batch_size
    display_step = 1

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
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=model, labels=Y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    train_op = optimizer.minimize(loss_op)
    
    # inicjalizacja zmiennych
    init = tf.global_variables_initializer()

    # uczenie
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):      
            batch_x_generator = batch([[item.x, item.y] for item in training_data], batch_size)
            batch_y_generator = batch([item.cls for item in training_data], batch_size)
            avg_cost = 0.
            total_batch = int(math.ceil(len(training_data) / batch_size))
            
            for _ in range(total_batch):
                batch_x, batch_y = next(batch_x_generator), next(batch_y_generator)
                batch_x = np.asarray(batch_x, np.float32)
                batch_y = convert_classification_labels_vector_to_tensorflow_output(batch_y)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                Y: batch_y})
                # obliczenie średniej straty
                avg_cost += c / total_batch

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                # testowanie aktualnego modelu
                test(model, test_data, X, Y)

        print("Optimization Finished!")
        test(model, test_data, X, Y)