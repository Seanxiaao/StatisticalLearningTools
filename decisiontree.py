from math import log, log2
import pandas as pd
import numpy as np
import stringtree

class DT(object):

    def __init__(self, val, attributes, sample):
        """val should be a value matrix, and attribute should be a list with attribute
        if there is no attribute with some column, it should be filled with None
        """
        self.val = val
        self.attr = attributes
        self.sample = sample
        self.result_list = self.transpose(self.val)[-1]
        self.eigens = attributes[:-1]

    def __repr__(self):
        tree = self.tree
        result = stringtree.ST(tree[0], tree[1])
        return repr(result)

    @property
    def tree(self):
        "return the struction of decision tree"
        #tree = self.construct_tree(len(self.eigens), self.entropy)
        tree = self.new_construct_tree(len(self.eigens), self.entropy)
        return tree

    @property
    def attributes_dict(self):
        dict, vice_dict = {}, {}
        for i, item in enumerate(self.attr):
            dict.setdefault(item, i)
            vice_dict.setdefault(i, item)
        return dict, vice_dict

    @property
    def sample_dict(self):
        dict, vice_dict = {}, {}
        for i, item in enumerate(self.sample):
            dict.setdefault(item, i)
            vice_dict.setdefault(i, item)
        return dict, vice_dict


    def construct_tree(self, depth, loss_function ):
        """greedily return the most information gain tree, the result should consist
        the index of every sample"""
        if depth == 1 or len(self.sample) == 1:
            return self.sample

        temp = self.construct_tree_helper(self.attributes_dict[0], loss_function)
        new_attribute = np.delete(self.attr, self.attributes_dict[0][temp[0]])
        tree = []
        for i, item in enumerate(temp[1]):
            tree.append( DT(self.get_sub_lst(item, temp[0]), new_attribute, item) )

        return temp[0], [tr.construct_tree(depth - 1, loss_function) for tr in tree]

    def new_construct_tree(self, depth ,loss_function ):
        """greedily return the most information gain tree, the result should consist
        the attributes"""

        if len(self.sample) == 1: #perfect terminal condition
            return ['result: ', (self.result_list[0], [])]

        if depth == 1: #attribute run out terminal condition
            dict = {}
            for x in set(self.result_list):
                dict.setdefault(x,list(self.result_list).count(x))
            mode = max(dict, key=dict.get)    #if attribute was runned out, the tree cannot still be splited
                                              #return the most possible result of the set
            return ['result: ', ("most possiblly" + mode, [])] #here should be result


        temp = self.construct_tree_helper(self.attributes_dict[0], loss_function)
        new_attribute = np.delete(self.attr, self.attributes_dict[0][temp[0]])
        tree = []
        for i, item in enumerate(temp[1]):
            tree.append( DT(self.get_sub_lst(item, temp[0]), new_attribute, item) )

        return [temp[0], [(self.item_attribute(temp[1][i], temp[0]), tr.new_construct_tree(depth - 1, loss_function)) for i, tr in enumerate(tree)]]

    def construct_tree_helper(self, attributes, loss_function):
        """find the next best attribute for decision tree
            where attributes should be a dict"""

        if len(self.eigens) == 1:
            return self.sample

        res_lst, cmp_lst = [], []
        for item in self.eigens:
            index = attributes[item]
            characteristic = self.get_char_kind(index)
            _val = self.transpose(self.val)[index]  # get char value, here is a bug that result was caculated in,
            temp, var_temp = [], []                               # but do not interfere with the correct answer
            for char in characteristic:
                 prob, var_index = [], []
                 for i, character in enumerate(_val):
                     if character == char: #charact for lst, char from set
                         temp_result = self.result_list[i] #TODO: optimize the speed by remove the selected item
                         prob.append(temp_result)
                         var_index.append(self.sample[i])
                 var_temp.append(var_index)
                 loss = loss_function(prob) * len(prob) / len(_val)
                 temp.append(loss)  #group
            cmp_lst.append(sum(temp))
            res_lst.append([self.get_attribute(index), var_temp])

        #print(cmp_lst)

        result =  res_lst[cmp_lst.index(min(cmp_lst))]
        if min(cmp_lst) == 0: #threshold for entropy
            while i < (len(result[1])) - 1:
                result[1][i] = result[1][i][0]
                i += 1

        return result

    @staticmethod
    def predict(tree, x, eigens):
        """predict function, to predict the value of sample
        x"""
        #TODO:check the value of eigens and availability for prediction of Xs
        try:
            assert type(x) is list
        except:
            raise Exception("the input predict sample should be list")

        tr = tree
        dicts, i = {}, 0

        while i < len(eigens):
            dicts.setdefault(eigens[i], x[i])
            i += 1

        #brutal search, could be optimized if data set is large
        def loop_tree(tree, dic):

            if tree[0] == 'result: ':
                return tree[1][0]
            target = dic[tree[0]]

            for child in tree[1]:
                if child[0] == target:
                    return loop_tree(child[1], dic)

        return loop_tree(tr, dicts)


    def get_sub_lst(self, x, attr):
        """get the sublist of the val, where the input should be the name list"""
        result = []
        index = self.attributes_dict[0][attr]
        for name in x:
            i = self.sample_dict[0][name]
            temp = np.delete(self.val[i], index)
            result.append(temp)
        return np.array(result)

    def get_attribute(self, index):
        return self.attributes_dict[1][index]

    def get_char_kind(self, index):
        return set(self.transpose(self.val)[index])

    @staticmethod
    def entropy(x):
        """x should be a probability list, and it should maintain the property of
        probability"""
        set_x, length = set(x), len(x)
        item_lst = []
        for item in set_x:
            item_lst.append(len(list(filter(lambda y: y == item, x))) / length)
        return -sum([item * log2(item) for item in item_lst])

    @staticmethod
    def transpose(x):
        """transpose a matrix"""
        x_Data = pd.DataFrame(x)
        x = x_Data.T
        return x.values

    def item_attribute(self, lst, attr):
        """temp should be the result of construct tree helper"""
        row = self.sample_dict[0][lst[0]]
        column = self.attributes_dict[0][attr]

        return self.val[row][column]
