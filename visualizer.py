import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
from config import cfg
import data as data
import networkx as nx
from mlp import CountingVariables
from operator import attrgetter

_colors = ["#3763D0", "#32679D", "#1EAB98", "#63738A", "#6E85A5", "#CC3237", "#DD4479", "#974599", "#6633D0", "#A77073", "#8C6D8D", "#11951C", "#319261", "#65AB00", "#AAAC07", "#5E8C89", "#8A8853", "#D4AF00", "#EA8B00", "#DD5415", "#B2885B"]

class _Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cls = -1

def visualize_learning_rate(learning_rate, cost_function):
    if learning_rate == None:
        return
    epochs = len(learning_rate)
    fig, ax = plt.subplots()
    ax.plot(list(range(1, epochs+1)), learning_rate, label="uczenie")
    ax.plot(list(range(1, len(cost_function)+1)), cost_function, label="koszt")
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.show()

def change_multiplier(val, pos_mult):
    odd = 80
    even = 100
    if (pos_mult[1] == 0):
        pos_mult[1] = 1
        return even * val
    res = even
    if (pos_mult[0] == even):
        pos_mult[0] = odd
    else:
        pos_mult[0] = even
    pos_mult[1] = 0
    return (res * val)

def visualize_graph(_counting_variables):
    G = nx.Graph()
    position = 1
    pos_mult = [100, 0]
    fixed_positions = {}
    names_mapping = {}
    weights = []
    biases = []
    for w in _counting_variables.weights:
        weights.append(_counting_variables.session.run(_counting_variables.weights[w]))
    for b in _counting_variables.biases:
        biases.append(_counting_variables.session.run(_counting_variables.biases[b]))
   # _counting_variables = CountingVariables()
    for input in range(_counting_variables.n_input):
        nodeName = "in" + str(input)
        G.add_node(nodeName)
        fixed_positions[nodeName] = (change_multiplier(position, pos_mult), change_multiplier((input+1),pos_mult))
        names_mapping[nodeName] = "{0:.2f}".format(biases[0][input])
    level = 0
    position += 1
    for neurons in _counting_variables.layers_neurons:
        for neuron in range(neurons):
            nodeName = "l" + str(level)+"n"+str(neuron)
            G.add_node(nodeName)
            fixed_positions[nodeName] = (change_multiplier(position, pos_mult), change_multiplier((neuron+1),pos_mult))
            names_mapping[nodeName] = "{0:.2f}".format(biases[level][neuron])
        level += 1
        position += 1

    for neuron in range(len(biases[len(biases)-1])):
        nodeName = "l" + str(level)+"n"+str(neuron)
        G.add_node(nodeName)
        fixed_positions[nodeName] = (change_multiplier(position, pos_mult), change_multiplier((neuron+1),pos_mult))
        names_mapping[nodeName] = "{0:.2f}".format(biases[level][neuron])


    
    for input in range(_counting_variables.n_input):
        for neurons in _counting_variables.layers_neurons:
            for neuron in range(neurons):
                G.add_edge("in" + str(input), "l" + str(0)+"n"+str(neuron),weight="{0:.2f}".format(weights[0][input][neuron]))
            break
    level = 1
    br = True
    prev = 0
    for neurons in _counting_variables.layers_neurons:
        if br:
            br = False
            prev = neurons
            continue
        for neuron in range(neurons):
            for prevNeuron in range(prev):
                G.add_edge("l" + str(level-1)+"n"+str(prevNeuron), "l" + str(level)+"n"+str(neuron),weight="{0:.2f}".format(weights[level][prevNeuron][neuron]))
        prev = neurons
        level += 1
    for prevNeuron in range(prev):
        for neuron in range(len(biases[len(biases)-1])):
            G.add_edge("l" + str(level-1)+"n"+str(prevNeuron), "l" + str(level)+"n"+str(neuron),weight="{0:.2f}".format(weights[level][prevNeuron][neuron]))

    #print("Nodes of graph: ")
    #print(G.nodes())
    #print("Edges of graph: ")
    #print(G.edges())
    fig = plt.figure()
    fixed_nodes = fixed_positions.keys()
    pos = nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)
    new_labels = nx.get_edge_attributes(G,'weight')
    if cfg.graph_visualisation_show_numertic_weights:
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels = new_labels)
    nx.draw_networkx_edges(G,pos,width=4, edge_color='g', arrows=True)
    edge_size = [G[u][v]['weight'] for u,v in G.edges()] #the higher |weight| the wider is edge
    nx.draw_networkx(G, pos, labels = names_mapping, font_size=6, width=edge_size)
    #plt.savefig("path_graph1.png")
    plt.show()


def visualize_points(model, count_predictions_func, show_points, is_classification):
    #training_data = [{'x':1, 'y':1, 'cls':1}, {'x':2, 'y':1, 'cls':2} ] #data.get_data(cfg.training_path, cfg.problem_type)
    if (model == None):
        return
    if is_classification:
         _visualize_classification_problem(model, count_predictions_func, show_points)
    else:
        _visualize_regression_problem(model, count_predictions_func, show_points)


def _visualize_classification_problem(model, count_predictions_func, show_points):
    training_data = _get_points_separated_by_class(data.get_data(cfg.training_path, cfg.problem_type))
    test_data = _get_points_separated_by_class(data.get_data(cfg.test_path, cfg.problem_type))
    fig = plt.figure()
    ax=plt.subplot()
    extremums = _draw_background_points(ax, model, count_predictions_func)
    if show_points:
        for cls in training_data: 
            arr = training_data[cls]
            ax.scatter([[item.x] for item in arr], [[item.y] for item in arr] , color=_count_color(cls, 0, extremums['min'], extremums['max']), marker='.', s = 1)
            cls +=1
        for cls in test_data:
            arr = test_data[cls]
            ax.scatter([[item.x] for item in arr], [[item.y] for item in arr] , color=_count_color(cls, 0, extremums['min'], extremums['max']), marker=',', s = 1)
            cls +=1
    plt.show()

def _visualize_regression_problem(model, count_predictions_func, show_points):
    training_data = data.get_data(cfg.training_path, cfg.problem_type)
    test_data = data.get_data(cfg.test_path, cfg.problem_type)
    minX = minY = 999999.0
    maxX = maxY = -999999.0
    for i in range(len(training_data)):
        point = training_data[i]
        if float(point.x) > maxX:
            maxX = float(point.x)
        if float(point.x) < minX:
            minX = float(point.x)
        if float(point.cls) > maxY:
            maxY = float(point.cls)
        if float(point.cls) < minY:
            minY = float(point.cls)
    for i in range(len(test_data)):
        point = test_data[i]
        if float(point.x) > maxX:
            maxX = float(point.x)
        if float(point.x) < minX:
            minX = float(point.x)
        if float(point.cls) > maxY:
            maxY = float(point.cls)
        if float(point.cls) < minY:
            minY = float(point.cls)
    func_x = [x for x in frange(minX,maxX,(maxX-minX)/1000)]
    func_y = count_predictions_func(model, func_x)
    fig = plt.figure()
    ax=plt.subplot()
    ax.scatter(func_x, func_y , color="#0000FF", marker=',', s = 1, label = "predykcja")
    if show_points:
        ax.scatter([[item.x] for item in training_data], [[item.cls] for item in training_data], color="#FF0000", marker='^', s=1, label="train")
        ax.scatter([[item.x] for item in test_data], [[item.cls] for item in test_data], color="#00FF00", marker='.', s=1, label="test")
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.show()


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def _draw_background_points(ax, model, count_predictions_func):
    minX = minY = 999999.0
    maxX = maxY = -999999.0
    training_data = data.get_data(cfg.training_path, cfg.problem_type)
    test_data = data.get_data(cfg.test_path, cfg.problem_type)
    sampling = cfg.points_drawing_sampling
    for i in range(len(training_data)):
        point = training_data[i]
        if float(point.x) > maxX:
            maxX = float(point.x)
        if float(point.x) < minX:
            minX = float(point.x)
        if float(point.y) > maxY:
            maxY = float(point.y)
        if float(point.y) < minY:
            minY = float(point.y)
    for i in range(len(test_data)):
        point = test_data[i]
        if float(point.x) > maxX:
            maxX = float(point.x)
        if float(point.x) < minX:
            minX = float(point.x)
        if float(point.y) > maxY:
            maxY = float(point.y)
        if float(point.y) < minY:
            minY = float(point.y)
    test_points = []
    currentX = minX
    currentY = minY
    while currentX < maxX and currentY < maxY:
        test_points.append(_Point(currentX, currentY))
        currentX += sampling
        if currentX > maxX:
            currentX = minX
            currentY += sampling
    test_points_predictions = count_predictions_func(model, test_points)

    minVal = 99999.0
    maxVal = -99999.0
    if cfg.problem_type == "classification":
        test_points_predictions = count_predictions_func(model, test_points)
        for i in range(len(test_points)):
            test_points[i].cls = test_points_predictions[i] + 1
        #print([[item.x, item.y, item.cls] for item in test_points])
        separated_points = _get_points_separated_by_class(test_points)
        point_size = sampling * 700
        if point_size < 1:
            point_size = 1
        for cls in separated_points:
            arr = separated_points[cls]
            ax.scatter([[item.x] for item in arr], [[item.y] for item in arr] , color=_count_color(cls, 30, minVal, maxVal), marker=',', s = point_size)
    else:
        test_points_predictions = count_predictions_func(model, test_points)
        for i in range(len(test_points)):
            test_points[i].cls = test_points_predictions[i]
            if test_points_predictions[i] < minVal:
                minVal = test_points_predictions[i]
            if test_points_predictions[i] > maxVal:
                maxVal = test_points_predictions[i]
        point_size = sampling * 700
        if point_size < 1:
            point_size = 1
        colors = [_count_color(item.cls, 0, minVal, maxVal) for item in test_points]
        x = [[item.x] for item in test_points]
        y = [[item.y] for item in test_points]
        for i in range(len(x)):
            ax.scatter(x[i], y[i] , color=colors[i], marker=',', s = point_size)

    return {'min': minVal, 'max':maxVal}

def _count_color(cls, shift, minVal, maxVal):
    if cfg.problem_type == "classification":
        return _shift_color(_colors[cls], shift)
    cls = ((cls - minVal)*1.0)/(maxVal - minVal)
    return _shift_color("#000000", int(255.0*cls))

def _shift_color(color, shift):
    R = int(color[1:3], 16) + shift
    G = int(color[3:5], 16) + shift
    B = int(color[5:], 16) + shift
    if R > 255:
        R = 255
    if G > 255:
        G = 255
    if B > 255:
        B = 255
    if R < 0:
        R = 0
    if G < 0:
        G = 0
    if B < 0:
        B = 0
    return '#%02x%02x%02x' % (R, G, B)
    

def _get_points_separated_by_class(arr):
    result = {}
    for i in range(len(arr)):
        if not int(arr[i].cls) in result:
            result[int(arr[i].cls)] = []
        result[int(arr[i].cls)].append(arr[i])
    return result


def destroy():
    plt.close('all')
