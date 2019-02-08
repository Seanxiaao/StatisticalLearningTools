import random
class clustering(object):

    def __init__(self, val, sample):

        self.val = val
        self.sample =sample

    def k_means(self, k):

        """k-means method for supervised learning"""

        labels = [random.randint(0, k - 1) for item in range(self.sample)]


        def caculate_distance(x, y, norm):
            return norm(x,y)


    def soft_k_means(self):
        pass

    def fit(self):
        pass

    @staticmethod
    def euclidean(x, y):
        return abs(x - y) ** 2