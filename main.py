import decisiontree
import pandas as pd

dt_data = pd.read_csv("dt-data.txt", header = None)
header, value, names = dt_data.T[0].values, dt_data[:][1:].values, []

for row in value:
    names.append(row[0][:2])
    row[0] = row[0][4:]
    row[-1] = row[-1][:-1]


if __name__ == '__main__':
    eigens = header[:-1]
    dt = decisiontree.DT(value, header, names)
    result = dt.construct_tree_helper(dt.attributes_dict[0], dt.entropy)
    print("the raw data of tree is {}:".format(dt.tree))
    print("\n"*2 + "-" * 80 + "\n"*2)
    print(dt)
    test = ["Moderate", " Cheap",  " Loud", " City-Center", " No", " No"]
    print("\n" + "*" * 80)
    predicted = dt.predict(dt.tree, test, dt.eigens)
    print("the predicted value is {}".format(predicted))

#  —————— result -------
#  old_tree ('(Occupied', [(' Location', [(' Price', [['22'], ['07']]), (' Price', [['10'],
# (' Music', [(' VIP', [['18', '21']])])]), ['12'], ['19']]), (' Location', [['01'], (' Price',
# [(' Music', [(' VIP', [['02', '09']])]), ['17']]), ['14'], ['16']]), (' Location', [(' Price',
#  [(' Music', [['15'], ['06']])]), ['03'], (' Price', [['13'], ['08']]), (' Price', [['11'], ['20']]),
#  (' VIP', [['05'], ['04']])])])

# new_tree
# ['(Occupied', [('Moderate', [' Location', [(' Talpiot', [' Price', [(' Cheap', ['result: ', (' No', [])]),
# (' Normal', ['result: ', (' Yes', [])])]]), (' German-Colony', [' VIP', [(' No', ['result: ', (' No', [])]),
# (' Yes', ['result: ', (' Yes', [])])]]), (' City-Center', ['result: ', (' Yes', [])]), (' Ein-Karem', [' Price',
# [(' Normal', [' Music', [(' Loud', ['result: ', (' Yes', [])]), (' Quiet', ['result: ', (' Yes', [])])]])]]),
#  (' Mahane-Yehuda', [' Price', [(' Expensive', ['result: ', (' Yes', [])]), (' Cheap', ['result: ', (' Yes', [])])]])]]),
# ('Low', [' Location', [(' Talpiot', ['result: ', (' No', [])]), (' Ein-Karem', [' Price', [(' Normal',
#  ['result: ', (' No', [])]), (' Cheap', ['result: ', (' Yes', [])])]]), (' City-Center',
# [' Price', [(' Cheap', ['result: ', (' No', [])]),
# (' Normal', [' Music', [(' Quiet', [' VIP', [(' No', ['result: ', (' No', [])])]])]])]]),
#  (' Mahane-Yehuda', ['result: ', (' No', [])])]]), ('High', [' Location', [(' Talpiot', ['result: ', (' No', [])]),
# (' German-Colony', ['result: ', (' No', [])]), (' City-Center', [' Price', [(' Expensive', [' Music', [(' Loud', [' VIP',
#  [(' Yes', ['result: ', (' Yes', [])])]])]]), (' Cheap', ['result: ', (' Yes', [])])]]), (' Mahane-Yehuda',
# ['result: ', (' Yes', [])])]])]]

# predicted_result : Yes
