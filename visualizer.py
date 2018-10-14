import matplotlib.pyplot as plt
import tensorflow as tf
from config import cfg
import data as data

_colors = ["#3763D0", "#32679D", "#1EAB98", "#63738A", "#6E85A5", "#CC3237", "#DD4479", "#974599", "#6633D0", "#A77073", "#8C6D8D", "#11951C", "#319261", "#65AB00", "#AAAC07", "#5E8C89", "#8A8853", "#D4AF00", "#EA8B00", "#DD5415", "#B2885B"]

class _Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cls = -1

def visualize_learning_rate(epochs, learning_rate):
    if learning_rate == None:
        return
    plt.plot(list(range(1, epochs+1)), learning_rate)
    plt.axis([0, epochs+1, 0, 1])
    plt.show()



def visualize_points(model, count_predictions_func):
    #training_data = [{'x':1, 'y':1, 'cls':1}, {'x':2, 'y':1, 'cls':2} ] #data.get_data(cfg.training_path, cfg.problem_type)
    if (model == None):
        return
    training_data = _get_points_separated_by_class(data.get_data(cfg.training_path, cfg.problem_type))
    test_data = _get_points_separated_by_class(data.get_data(cfg.test_path, cfg.problem_type))
    #train_predictions = count_predictions_func(model, training_data)
    #test_predictions = count_predictions_func(model, test_data)
    #print(test_data)

    fig = plt.figure()
    ax=plt.subplot()
    _draw_background_points(ax, model, count_predictions_func)
    if cfg.points_drawing_draw_points:
        for cls in training_data: 
            arr = training_data[cls]
            ax.scatter([[item.x] for item in arr], [[item.y] for item in arr] , color=_colors[cls], marker='.', s = 1)
            cls +=1
        for cls in test_data:
            arr = test_data[cls]
            ax.scatter([[item.x] for item in arr], [[item.y] for item in arr] , color=_colors[cls], marker=',', s = 1)
            cls +=1

    #for i in range(len(training_data)):
    #    ax.scatter(training_data[i]['x'],training_data[i]['y'], color=_colors[int(training_data[i]['cls'])])
    plt.show()


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

    for i in range(len(test_points)):
        test_points[i].cls = test_points_predictions[i] + 1
    separated_points = _get_points_separated_by_class(test_points)
    #print(separated_points)
    point_size = sampling * 700
    if point_size < 1:
        point_size = 1
    for cls in separated_points:
        arr = separated_points[cls]
        ax.scatter([[item.x] for item in arr], [[item.y] for item in arr] , color=_shift_color(_colors[cls], 30), marker=',', s = point_size)

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
