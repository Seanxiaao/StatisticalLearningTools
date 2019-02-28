# Python version: 3.4.2
#
#
import pandas as pd
import numpy as np
import DimensionReduction

if __name__ == '__main__':
    data = pd.read_csv('./pca-data.txt',delimiter='\t')
    result = DimensionReduction.DR(data)
    """test"""
    #data = X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #result = DimensionReduction.DR(data)
    #print(result.PCA(2))


    fastdata = pd.read_csv('./fastmap-data.txt',delimiter='\t')
    DisMatrix = [[0 for x in range(10)] for y in range(10)]
    DisMatrix[0][1], DisMatrix[1][0]= 4, 4
    for item in fastdata.values:
        DisMatrix[item[0] - 1][item[1] - 1] = item[2]
        DisMatrix[item[1] - 1][item[0] - 1] = item[2]

    D_func = lambda x, y: DisMatrix[x - 1][y - 1]
    ob = [x + 1 for x in range(10)]
    print(DimensionReduction.FastMap(ob, D_func, 2))
