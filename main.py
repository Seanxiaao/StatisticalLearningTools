# Python version: 3.4.2
#
#

import clustering
import pandas as pd

if __name__ == '__main__':
    dt_data = pd.read_csv("clusters.txt", header = None)
    data = clustering.clustering(dt_data.values, dt_data.values)
    print(data.k_means(3))


