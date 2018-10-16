from tkinter import Tk, Label, Button, StringVar, Frame
from tkinter import LEFT, RIGHT, TOP, BOTTOM
from tkinter import W

from threading import Thread
from config import cfg

import mlp  as mlp
import main as main
import visualizer as visualizer
import tensorflow as tf
import tensorboard as tb

class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("Neural Network")

        self.create_iteration_panel(master)

        self.create_visualization_panel(master)

        self.close_button = Button(master, text="Close", command=self.quit)
        self.close_button.pack(side = BOTTOM)

#-------PANELS----------------------

    def create_iteration_panel(self, master):
        self.iterating_frame = Frame(master)
        self.iterating_frame.pack(side = TOP)

        self.label_iterations = Label(self.iterating_frame, text = "Wykonaj iteracje")
        self.label_iterations.pack(side = TOP)
        self.button_one_iteration = Button(self.iterating_frame, text = ">", command=self.one_iteration_action )
        self.button_one_iteration.pack(side = LEFT)
        self.button_all_iteration = Button(self.iterating_frame, text = ">>>", command=self.all_iteration_action )
        self.button_all_iteration.pack(side = LEFT)
        self.button_all_iteration = Button(self.iterating_frame, text = ">ORIG>", command=self.all_iteration_normal_action )
        self.button_all_iteration.pack(side = LEFT)

    def create_visualization_panel(self, master):
        self.visualization_frame = Frame(master)
        self.visualization_frame.pack(side = TOP)

        self.label_visualization = Label(self.visualization_frame, text = "Wizualizacja")
        self.label_visualization.pack(side = TOP)
        self.button_net_visualization = Button(self.visualization_frame, text = "Pokaż sieć", command=self.visualize_network_action )
        self.button_net_visualization.pack(side = TOP)

        self.button_prediction_visualization = Button(self.visualization_frame, text = "Pokaż punkty i tło", command=self.visualize_points_action )
        self.button_prediction_visualization.pack(side = TOP)

        self.button_prediction_no_points_visualization = Button(self.visualization_frame, text = "Pokaż tło bez punktów", command=self.visualize_points_only_bcg_action )
        self.button_prediction_no_points_visualization.pack(side = TOP)

        self.button_learning_results_visualization = Button(self.visualization_frame, text = "Pokaż historię uczenia", command=self.visualize_learning_results_action )
        self.button_learning_results_visualization.pack(side = TOP)


#-------ACTIONS-------------------------
    def one_iteration_action(self):
        print("One itearation")
        main.one_iteration_main()

    def all_iteration_action(self):
        print("All itearations")
        self._all_iterations_action()
        #self.calculation_thread = Thread(target = self._all_iterations_action)
        #self.calculation_thread.start()

    def all_iteration_normal_action(self):
        main.main(self)

    def visualize_network_action(self):
        print("visualize network")
        for w in mlp._counting_variables.biases:
            print(w)
            print(mlp._counting_variables.session.run(mlp._counting_variables.biases[w]))

        visualizer.visualize_graph(mlp._counting_variables)

        #Graph export to visualize in tensorboard.
        writer = tf.summary.FileWriter(cfg.save_file,  mlp._counting_variables.session.graph)
        writer.close()
        #py -m tensorboard.main --logdir=./output/save.model
       

    def visualize_points_action(self):
        print("visualize points and background")
        visualizer.visualize_points(mlp._counting_variables.model, mlp.count_predictions, True, mlp._counting_variables.is_classification_problem)

    def visualize_points_only_bcg_action(self):
        print("visualize background only")
        visualizer.visualize_points(mlp._counting_variables.model, mlp.count_predictions, False, mlp._counting_variables.is_classification_problem)

    def visualize_learning_results_action(self):
        print("visualize learning results")
        visualizer.visualize_learning_rate(mlp._counting_variables.learning_results, mlp._counting_variables.cost_function)

    def quit(self):
        #TODO thread stop
        print('exit')
        main.destroy()
        visualizer.destroy()
        self.master.quit()

    def greet(self):
        print("Greetings!")

    def cycle_label_text(self, event):
        self.label_index += 1
        self.label_index %= len(self.LABEL_TEXT) # wrap around
        self.label_text.set(self.LABEL_TEXT[self.label_index])

#-------PRIVATE METHODS--------------------------

    def _all_iterations_action(self):
        main.all_iteration_main()



root = Tk()
my_gui = MainWindow(root)
root.mainloop()