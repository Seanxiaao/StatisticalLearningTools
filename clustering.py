import random
import decoraters
import numpy as np
from functools import wraps

class clustering(object):

    """
    making clustering for the unsupervised learning

    >>> X = [[1.3, 1.8, 4.8, 7.1, 5.0, 5.2, 8.0], \
         [1.5, 6.9, 3.9, -5.5, -8.5, -3.9, -5.5], \
         [6.5, 1.6, 8.2, -7.2, -8.7, -7.9, -5.2], \
         [3.8, 8.3, 4.7, 6.4, 7.5, 3.2, 7.4], \
         [-7.3, -1.8, -2.1, 2.7, 6.8, 4.8, 6.2] \
        ]
    >>> X = np.array(X).T
    >>> x_cluster = clustering(X, [[0, 1, 2, 3, 4, 5, 6]])
    >>> x_cluster.k_means(2)
    [0,0,0,1,1,1,1]

    """

    def __init__(self, val, sample):

        self.val = np.array(val) #should be sample * attribute arrarys
        self.sample = sample
        self.error = None


    @decoraters.register
    def k_means(self, k):

        """k-means method for supervised learning"""

        def calculate_distance(x, y, norm):
            return norm(x,y)

        #random initialize labels#
        labels = np.array([(index, random.randint(0, k - 1)) for index in range(len(self.sample))],\
                          dtype = [('index', int), ('cluster', int)])
        #print(labels)
        #clusters = np.array([labels[labels['cluster'] == i]['index'] for i in range(k)])
        clusters, centroids = [] , []
        epsilon0 = 9999999
        while True:

            #get clusters
            for i in range(k):
                sub_cluster = labels[labels['cluster'] == i]['index']
                clusters.append(sub_cluster)
                #clusters = np.concatenate((clusters, sub_cluster), axis=0 ) #concatenate the two lst
                centroids.append(np.average(self.val[sub_cluster], axis=0))

            #reassigned sample
            temp = 0
            for item, sample in enumerate(self.val):
                dis = [calculate_distance(sample, centroid, self.euclidean) for centroid in centroids]
                labels[item][-1] = dis.index(min(dis))
                temp += min(dis)

            clusters, centroids = [], []
            #print(epsilon0 - temp)
            if epsilon0 - temp <= 10 ** (-4):
                return labels['cluster']
            else:
                epsilon0 = temp


    @decoraters.register
    def soft_clustering(self):
        pass

    def fit(self):
        pass


    @staticmethod
    def euclidean(x, y):
        assert type(x) and type(y) is np.ndarray
        return sum((x - y) ** 2)



