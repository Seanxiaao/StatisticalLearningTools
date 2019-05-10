import numpy as np
import random, math

class DR(object):

    def __init__(self, data):
        self.data = data

    def PCA(self, k):
        """
        :param k: the dimension original data reduced to
        :return: the first k principal components
        """
        # data = X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        # result = DimensionReduction.DR(data)
        # print(result.PCA(2))

        meanVa = np.mean(self.data)
        Cdata = self.data - meanVa
        d_cov =  np.dot(Cdata.T, Cdata) #covarience matrix
        va, ve = np.linalg.eig(d_cov)
        eig_pairs = [(np.abs(va[i]), ve[:, i]) for i in range(len(self.data.values[0]))]
        result = sorted(eig_pairs, key=lambda x: x[0], reverse= True)
        return [item[1] for item in result[:k]]



def FastMap(objects, D_function, k):

    def choose_distant_objects(objects, D_function):

        temp = objects[random.randint(0, len(objects) - 1)]
        first_pivot_ob, second_pivot_ob = 0, 0
        for i in range(4):
            value1 = [D_function(temp, x) for x in objects]
            first_pivot_ob = value1.index(max(value1)) + 1
            value2 = [D_function(first_pivot_ob, x) for x in objects]
            second_pivot_ob =  value2.index(max(value2)) + 1

            temp = second_pivot_ob

        return first_pivot_ob, second_pivot_ob

    def FastMap_helper(objects, D_function, i, result):

        if i == k:
            return
        else:
            temp = choose_distant_objects(objects, D_function)
            O_a, O_b = min(temp), max(temp)

            for item in objects:
                result[item - 1][i] = (D_function(O_a, item) ** 2 + D_function(O_a, O_b) ** 2 - D_function(O_b, item) ** 2) / (2 * D_function(O_a, O_b))

        #can be optimized
        NewD_function = lambda x, y: math.sqrt( D_function(x, y) ** 2 - (result[x - 1][i] - result[y - 1][i]) ** 2 )

        return FastMap_helper(objects, NewD_function, i + 1 , result)

    result = [[0 for x in range(k)] for y in range(len(objects))]

    FastMap_helper(objects, D_function, 0 , result)

    return result
