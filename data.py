from csv import reader

class RegressionData(object):
    def __init__(self, x, cls):
        self._x = x
        self._cls =cls

    @property
    def x(self):
        return self._x

    @property
    def cls(self):
        return self._cls

    def __repr__(self):
        return 'x: {0}, cls: {1}'.format(self.x, self.cls)


class ClassificationData(object):
    def __init__(self, x, y, cls):
        self._x = x
        self._y = y
        self._class = cls

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def cls(self):
        return self._class

    def __repr__(self):
        return 'x: {0}, y: {1}, cls: {2}'.format(self.x, self.y, self.cls)

def get_data(path, problem_type):
    csv_list = load_csv(1, path)  # ignore header
    data_list = list()
    if(problem_type == 'regression'):
        for entry in csv_list:      
            if(len(entry) != 2):
                raise ValueError('Wrong data in {0} for {1} problem'.format(path, problem_type))
            data_list.append(RegressionData(entry[0], entry[1]))
        return data_list
    if(problem_type == 'classification'):
        for entry in csv_list:      
            if(len(entry) != 3):
                raise ValueError('Wrong data in {0} for {1} problem'.format(path, problem_type))
            data_list.append(ClassificationData(entry[0], entry[1], entry[2]))
        return data_list
    raise ValueError('Wrong problem type: {0}. Possible values are: [classification, regression]'.format(problem_type))

def load_csv(start_index, path):
    dataset = list()
    row_number = -1
    counter = 0
    with open(path, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            row_number += 1
            if not row or start_index > row_number:
                continue
            else:
                counter += 1
                dataset.append(row)
    return dataset