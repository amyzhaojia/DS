# 画决策树

#-*- coding: utf-8 -*-
from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from IPython.display import Image
from sklearn import tree
import pydotplus
import os

os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'

# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'
# # os.environ["PATH"] += os.pathsep + 'G:/program_files/graphviz/bin'


# data_for_alert = pd.read_csv('data_232.csv')
data_for_alert = pd.read_csv("ASA_24hr_386.csv")
column_name = ['CallAnswered', 'TotalAbandoned', 'TotalTimetoAnswer', 'ASA_Alert']

data = data_for_alert[column_name]

X = data.iloc[:, 0:3]
y = data.iloc[:, 3]

feature_names = column_name[0:3]
target_names = ['ISnt_Alert', 'Is_Alert']
# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4)
fig, ax = plt.subplots(figsize=(50, 15))
# 拟合模型
clf.fit(X, y)
tree.plot_tree(clf, feature_names=feature_names, class_names=target_names, ax=ax, fontsize=20)
plt.savefig('DecisionTree_ASA_24hr_3_Alert_386.jpg')
