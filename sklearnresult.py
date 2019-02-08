from sklearn import tree
from subprocess import call
import pandas as pd
from sklearn import preprocessing

# Load the data into pandas data frame
df = pd.read_csv("dt-data.txt", engine='python', sep= ", |: ", header = None
                 , names = ["Occupied", "Price", "Music", "Location", "VIP", "Favorite Beer", "Enjoy"], index_col = 0,
                 skiprows = [0], skip_blank_lines = True)
df["Enjoy"] = df["Enjoy"].apply(lambda x: x[:-1])

train_df = df
le = preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        train_df[column_name] = le.fit_transform(df[column_name])
    else:
        pass
# print(train_df)
model = tree.DecisionTreeClassifier(criterion="entropy")
x = train_df.drop('Enjoy', axis=1)
clf = model.fit(x, train_df['Enjoy'])

tree.export_graphviz(clf, out_file='tree.dot', feature_names=x.columns)
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])