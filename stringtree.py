BLANK = "    "

class ST(object):

    def __init__(self, root, children, depth=1):
        """this should be a string tree"""

        self.root = root
        self.children = children
        self.depth = depth


    def __repr__(self):
        if "result: " in self.root :
            return BLANK * (self.depth + 1) + "the result is {}".format(self.children[0])

        attribute = BLANK * (self.depth - 1) + "> the {} attribute is {}".format(self.depth, self.root)
        res , i = "", 0
        #for i, child in enumerate(self.children):
        while i < len(self.children):
            child = self.children[i] #a[1][0]
            res += "\n" + BLANK * (self.depth + 1) + ">>> the No.{} value is {}".format(i + 1, child[0])

            if "result: " in child[1][0]:
                 #print(child)
                 res += "\n" + (BLANK * (self.depth + 2) + "the enjoyed result is {}".format(child[1][1][0]))
                 pass
            else:
                new_child = ST(child[1][0], child[1][1], self.depth + 1)
                res += "\n" + repr(new_child)
            i += 1


        return attribute + "\n" + res

