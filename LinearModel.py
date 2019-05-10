import random
import numpy as np

class FM(object):

    #there is a potential problem that should be an w_0 in the model,
    #but not used it right now

    def __init__(self, data, labels):
        self._data = data if type(data) is np.matrix else np.matrix(data)
        self.data = np.c_[np.ones(len(self._data)), self._data]
        self.labels = labels
        self.length = len(data)
        self.learning_result = {}

    def n_perceptron_learning(self, alpha = 0.01, iterations=7000):
        """
        the perceptron-learning algorithm, y(sum_w)
        :return:
        """
        P_set, N_set = set([]), set([])
        for i, label in enumerate(self.labels):
            P_set.add(i) if i == +1 else N_set.add(i)
        w = np.matrix([random.random() for x in range(self.data[0].size)])
        its, accuracy = 0, 0
        while its < iterations:
            #the converge situation is that all the inputs are classified correctly
            random_index  = random.randint(0, self.length - 1)
            random_item = self.data[random_index]
            temp = w  * random_item.T
            if temp.item(0) >= 0 and random_index in N_set:
                 w = w - alpha * random_item
            if temp.item(0) < 0 and random_index in P_set:
                 w = w + alpha * random_item
            temp_labels = [self.sign_sample(w * item.T) for item in self.data]
            accuracy = self.count_accuracy(self.labels, temp_labels)
            if accuracy == 1:
                return w, 1
            its += 1
            self.learning_result.setdefault(accuracy, w)

        return w, accuracy

    def perceptron_learning(self, alpha = 0.01, iterations=7000):
        its = 0
        w = np.matrix([random.random() for x in range(self.data[0].size)])
        while its <= iterations:
              for i, item in enumerate(self.labels):
                  temp_item = w * self.data[i].T
                  #print(temp_item)
                  w = w - alpha * (self.sign_sample(temp_item.item(0)) - item) * self.data[i]
              temp_labels = [self.sign_sample(w * item.T) for item in self.data]
              accuracy = self.count_accuracy(self.labels, temp_labels)
              self.learning_result.setdefault(accuracy, w)
              its += 1

        return w, accuracy

    @staticmethod
    def count_accuracy(lst1, lst2):
        count = 0
        for item in range(len(lst1)):
            if lst1[item] == lst2[item]:
                count += 1
        return count / len(lst1)

    @staticmethod
    def sign_sample(x):
        return 1 if x >= 0 else -1

    @staticmethod
    def sigmoid(x):
        return 1 - 1 / (2.71828 ** x + 1)

    def Pocket_algorithm(self, iterations=7000):
        #pocket_algorithm is just return the max item of perceptron_learning#
        self.perceptron_learning(iterations=iterations)
        accuracy = max(self.learning_result)
        w  =  self.learning_result[accuracy]
        return w, accuracy

    def logistic_regression(self , alpha=0.01, iterations=7000):
        its = 0
        w = np.matrix([random.random() for x in range(self.data[0].size)])
        while its <= iterations:
              for i, item in enumerate(self.labels):
                  temp_item = w * self.data[i].T
                  #print(temp_item)
                  w = w - alpha * (self.sigmoid(temp_item.item(0)) - item) * self.data[i]

              its += 1
              print(w)
        temp_l = [w * item.T for item in self.data]
        temp_labels = [1 if self.sigmoid(item.item(0)) >=0.5 else -1 for item in temp_l]
        accuracy = self.count_accuracy(self.labels, temp_labels)
        return w, accuracy

    def linear_regression(self):

        return np.linalg.inv(self.data.T * self.data) * self.data.T * np.matrix(self.labels).T