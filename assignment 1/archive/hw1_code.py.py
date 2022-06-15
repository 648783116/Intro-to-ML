import numpy as np
import urllib
import sklearn
from sklearn.tree import DecisionTreeClassifier


def load_data():
    # this section of the code is to try to seprately load the datasets and get their features
    with urllib.request.urlopen(
            "http://www.cs.toronto.edu/~rgrosse/courses/csc2515_2019/homeworks/hw1/clean_real.txt") as url1:
        real_raw = url1.readlines()

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer1 = CountVectorizer()
    real_data = vectorizer1.fit_transform(real_raw)
    real_data = real_data.toarray()

    with urllib.request.urlopen(
            "http://www.cs.toronto.edu/~rgrosse/courses/csc2515_2019/homeworks/hw1/clean_fake.txt") as url2:
        fake_raw = url2.readlines()

    vectorizer2 = CountVectorizer()
    fake_data = vectorizer2.fit_transform(fake_raw)
    fake_data = fake_data.toarray()

    # print(real_data.shape)
    # print(fake_data.shape)

    # create a target
    real = 1
    fake = 0
    target_real = [real] * real_data.shape[0]
    target_fake = [fake] * fake_data.shape[0]
    target = target_real + target_fake
    # print(target)

    # this section of the code is to create one combined list of data
    combined_raw = real_raw + fake_raw
    vectorizer3 = CountVectorizer()
    combined_data = vectorizer3.fit_transform(combined_raw)
    combined_data = combined_data.toarray()

    # print(combined_data.shape)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(combined_data, target, test_size=0.3, random_state=1)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    return X_train, X_val, X_test, y_train, y_val, y_test


# calling the function load_data
X_train, X_val, X_test, y_train, y_val, y_test = load_data()


def select_model(X_train, X_val, X_test, y_train, y_val, y_test):
    from sklearn.metrics import accuracy_score

    max_depth = [3, 5, 10, 15, 20]
    criterion = ['gini', 'entropy']
    scores = {}
    for depths in max_depth:
        for values in criterion:
            clf_tree = DecisionTreeClassifier(max_depth=depths,criterion=values)
            clf_tree.fit(X_train,y_train)
            y_pred = clf_tree.predict(X_val)
            scores[depths,values]= accuracy_score(y_val, y_pred)
    print(scores)

    return
#callng the select_model function
select_model(X_train, X_val, X_test, y_train, y_val, y_test)

# #question 2c print tree with highest validation score : depth =20, criterion='gini'
clf_final_tree = DecisionTreeClassifier(max_depth=20,criterion='gini')
clf_final_tree.fit(X_train,y_train)

def printTree(clf_final_tree):
    from sklearn import tree
    tree.export_graphviz(clf_final_tree,out_file='tree.dot')
    from sklearn.externals.six import StringIO
    import pydot
    dot_data = StringIO()
    tree.export_graphviz(clf_final_tree, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    a=graph.write_png("tree.png")
    from IPython.display import Image
    import os
    return Image(filename=os.getcwd()+'/tree.png')

printTree(clf_final_tree)