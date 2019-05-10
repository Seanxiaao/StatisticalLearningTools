from sklearn import svm
import random
import numpy as np


class SVM(object):
    """
    W(α) = 􏰀\sum αi − 0.5 \sum(i) \sum(j) 􏰀􏰀y(i)y(j)αiαj⟨x(i),x(j)⟩
    """

    def __init__(self, data, target):
        self._X = data if type(data) is np.matrix else np.matrix(data)
        self.X = np.c_[np.ones(len(self._X)), self._X]
        self.y = list(target)

    def simplified_SMO(self, max_passes, tol=10 ** -5, kernel='linear', pen_constant=0.0000):

        def iter_a(a, L, H):
            if a > H:
                return H
            elif a < L:
                return L
            else:
                return a

        def computer(yi, yj, ai, aj, pen_constant):
            if yi == yj:
                L, H = max(0, aj - ai), min(pen_constant, pen_constant + aj - ai)
            else:
                L, H = max(0, ai + aj - pen_constant), min(pen_constant, ai + aj)
            return L, H

        def objective_function(alphas, y, kernel, X):
            temp = 0
            for i in range(len(y)):
                for j in range(len(y)):
                    temp += y[i] * y[j] * alphas[i] * alphas[j] * kernel(X[i], X[j])

            return np.sum(alphas) - 0.5 * temp

        alphas, b = [0.00 for x in range(len(self._X))], 0
        E = [0 for x in range(len(self._X))]

        if kernel == 'linear':
            k = self.linear
        if kernel == 'poly':
            k = self.poly

        num_changed_alpha, exm = 0, 1
        m = 0
        # while num_changed_alpha > 0 or exm:
        while m < max_passes:
            num_changed_alpha = 0

            # if exm:
            #    for i in range(len(self._X)):
            # f(x, X, y ,kernel, alpha, b):
            for i in range(len(self._X)):
                E[i] = (self.f(self._X[i], self._X, self.y, k, alphas, b) - self.y[i]).item(0)
                if (self.y[i] * E[i] < -tol and alphas[i] < pen_constant) or (self.y[i] * E[i] > tol and alphas[i] > 0):
                    j = random.randint(0, len(self.y) - 1)
                    if j == i:
                        j = random.randint(0, len(self.y) - 1)

                    E[j] = (self.f(self._X[j], self._X, self.y, k, alphas, b) - self.y[j]).item(0)
                    a_i_old, a_j_old = alphas[i], alphas[j]
                    L, H = computer(self.y[i], self.y[j], a_i_old, a_j_old, pen_constant)
                    if L == H:
                        continue
                    else:
                        ita = 2 * k(self._X[i], self._X[j]) - k(self._X[i], self._X[i]) - k(self._X[j], self._X[j])

                        if ita >= 0:

                            alphas_adj = alphas.copy()
                            alphas_adj[j] = L
                            Lobj = objective_function(alphas_adj, self.y, k, self._X)
                            alphas_adj[j] = H
                            Hobj = objective_function(alphas_adj, self.y, k, self._X)
                            if Lobj > Hobj + tol:
                                alphas[j] = L
                            elif Lobj < Hobj - tol:
                                alphas[j] = H
                            else:
                                alphas[j] = a_j_old


                        else:
                            temp = a_j_old - (self.y[j] * (E[i] - E[j]) / ita)  # ?
                            alphas[j] = iter_a(temp, L, H)

                        if alphas[j] < 10 ** -8:
                            alphas[j] = 0
                        elif alphas[j] > pen_constant - 10 ** -8:
                            alphas[j] = pen_constant
                        #print("difference:{}".format(abs(alphas[j] - a_j_old)))
                        if abs(alphas[j] - a_j_old) < tol * (alphas[j] + a_j_old + tol):
                            continue

                        alphas[i] = a_i_old + self.y[i] * self.y[j] * (a_j_old - alphas[j])
                        b_1 = b - E[i] - self.y[i] * (alphas[i] - a_i_old) * k(self._X[i], self._X[i]) - \
                              self.y[j] * (alphas[j] - a_j_old) * k(self._X[i], self._X[j])
                        b_2 = b - E[j] - self.y[i] * (alphas[i] - a_i_old) * k(self._X[i], self._X[j]) - \
                              self.y[j] * (alphas[j] - a_j_old) * k(self._X[j], self._X[j])

                        if 0 < alphas[i] < pen_constant:
                            b = b_1
                        elif 0 < alphas[j] < pen_constant:
                            b = b_2
                        else:
                            b = (b_1 + b_2) / 2

                        for index, alpha in zip([i, j], [alphas[i], alphas[j]]):
                            if 0.0 < alpha < pen_constant:
                                E[index] = 0.0

                        num_changed_alpha += 1

            if num_changed_alpha == 0:
                m += 1
            else:
                m = 0
        return alphas, b

    def svm(self):
        """
        max_alpha \sum_i^m \alpha_i - 1/2* sum(i to m) sum(j to m) aiajyiyjXi^t Xj
        w_i = a_iy_ix_i^T
        """
        alphas, b = self.simplified_SMO(100, pen_constant=0.001)
        print('alpha:{}'.format(alphas))
        w = np.matrix([0 for x in range(self._X[0].size)])
        for i in range(len(self._X)):
            w = w + alphas[i] * self.y[i] * self._X[i]
        return w, b

    @staticmethod
    def linear(x, y):
        """x = np,matrix([...])"""
        assert type(x) is np.matrix
        assert type(y) is np.matrix
        assert x.shape == y.shape
        return (x * y.T).item(0)

    @staticmethod
    def poly(x, y):
        """x = np,matrix([...])"""
        assert type(x) is np.matrix
        assert type(y) is np.matrix
        assert x.shape == y.shape
        return ((1 + x * y.T) ** 2).item(0)

    @staticmethod
    def f(x, X, y, kernel, alpha, b):
        result = 0
        for i in range(len(y)):
            result += alpha[i] * kernel(X[i], x) * y[i]
        return result + b

    def test_svm(self):
        clf = svm.SVC(kernel='linear')
        clf.fit(self.data, self.target)
