import matplotlib.pyplot as plt

def visualize_learning_rate(epochs, learning_rate):
    if learning_rate == None:
        return
    plt.plot(list(range(1, epochs+1)), learning_rate)
    plt.axis([0, epochs+1, 0, 1])
    plt.show()
