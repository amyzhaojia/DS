import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import os

# pie_figure, label
def pie_figure(num_data, label):
    """
    plot pie figures
    :param num_data: list
    :param label: list
    :return: figures
    """
    def make_autopct(num_data):
        def my_autopct(pct):
            total = sum(num_data)
            val = int(round(pct * total / 100.0))
            # 同时显示数值和占比的饼图
            return '{p:.2f}%({v:d})'.format(p=pct, v=val)
        return my_autopct
    plt.figure(figsize=(6, 6))  # 将画布设定为正方形，则绘制的饼图是正圆
    explode = 0.01*np.ones(len(label))  # 设定各项距离圆心n个半径
    plt.pie(num_data, explode=explode, labels=label, autopct=make_autopct(num_data),
            textprops={'fontsize': 15, 'color': 'w'}, pctdistance=0.5)  # 绘制饼图
    plt.legend(loc='lower right')
    # plt.show()


# venn_figure
def venn_figure(data, label):
    """
    plot venn figures
    :param data: list
    :param label: tuple
    :return: figures
    """
    g = venn2(subsets=[set(data[0]), set(data[1])], set_labels=label, set_colors=('r', 'b'))  #
    return g
    # plt.show()


# correlation_ship_figure
def cor_fig(data):
    """
    Calculate the relationship of data and plot a heatmap to show it.
    :param data: DataFrame
    :return: heatmap
    """
    cor_data = data.corr()
    plt.figure(figsize=(40, 40))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    corr_pic = sns.heatmap(cor_data, annot=True)
    # plt.show()


# double axes
def plot_double_axes(data,data1):
    """
    Plot two data use double axes.
    :param data: DataFrame or Series
    :param data1: DataFrame or Series
    :return: figure
    """
    ax1 = data.plot()
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    data1.plot(x='datetime', ax=ax2, c='k')
    plt.legend(loc='upper right')
    # plt.show()




