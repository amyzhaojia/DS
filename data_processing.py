import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import os
import graphviz
import copy
import datetime
from fbprophet import Prophet
from pytz import timezone, utc
pst_tz = timezone('America/Los_Angeles')


# Alert——defination
def alert_fun(measures, threshold):
    return 1 if measures > threshold else 0


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


# corelation_ship_figure
def cor_fig(data):
    cor_data = data.corr()
    plt.figure(figsize=(40, 40))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    corr_pic = sns.heatmap(cor_data, annot=True)
    # plt.show()


# decision_tree_figure
#-*- coding: utf-8 -*-
def decision_tree(X, y, feature_names, target_names):
    os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'
    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(max_depth=4)
    fig, ax = plt.subplots(figsize=(50, 15))
    # 拟合模型
    clf.fit(X, y)
    plot_tree(clf, feature_names=feature_names, class_names=target_names, ax=ax, fontsize=20)
    # plt.show()


if __name__ == '__main__':
    df = pd.read_csv('./file/CallOffered_202106.csv')
    df = df[df['QueueGroupId'] == 1286] 
    plt.plot(df['CallOffered'])
    plt.show()
    df = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueGroupData202010.csv")
    # df = pd.read_excel(r"E:\Data\QueueData_232_234_368_386_08\data_386_new.xlsx", sheet_name='Sheet1')
    # data_new = df[['QueueGroupId', 'CallAnswered', 'CallOffered', 'ServiceLevel60', 'PstTimeStamp', 'AverageSpeedAnswer']]
    # data_new.to_csv('../Data/QueueData_202009_CO.csv')
    num = 24 * 60  # one day
    df = df[df['QueueGroupId'] == 368]   # [-num:]
    df.to_csv('10_368.csv')
    df = pd.read_csv('10_368.csv')
    Time = df['Timestamp'].values
    time_232 = [datetime.datetime.strptime(utc.localize(datetime.datetime.strptime(Time[i].replace("T", " ")
            [:Time[1].replace("T", " ").rfind(".")], "%Y-%m-%d %H:%M:%S")).astimezone(tz=pst_tz).strftime(
            "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S") for i in range(len(Time))]  #
    data_new = pd.DataFrame()
    data_new['ds'] = time_232
    data_new['y'] = df['CallOffered'].values
    # plt.plot(time_232, data_new['y'])
    # plt.show()
    data_group = data_new.groupby(data_new['ds'].apply(lambda x: x.day), as_index=False).diff()
    CO_5min = pd.DataFrame()
    for i in time_232:
        j = i + datetime.timedelta(minutes=-5)
        num = np.where((data_new['ds'] > j) & (data_new['ds'] <= i))
        CO_5min = CO_5min.append(data_group.iloc[num[0]].sum(), ignore_index=True)
    CO_5min['ds'] = time_232
    data_new_5min = CO_5min
    CO_10min = pd.DataFrame()
    for i in time_232:
        j = i + datetime.timedelta(minutes=-10)
        num = np.where((data_new['ds'] > j) & (data_new['ds'] <= i))
        CO_10min = CO_10min.append(data_group.iloc[num[0]].sum(), ignore_index=True)
    CO_10min['ds'] = time_232
    data_new_10min = CO_10min
    CO_30min = pd.DataFrame()
    for i in time_232:
        j = i + datetime.timedelta(minutes=-30)
        num = np.where((data_new['ds'] > j) & (data_new['ds'] <= i))
        CO_30min = CO_30min.append(data_group.iloc[num[0]].sum(), ignore_index=True)
    CO_30min['ds'] = time_232
    data_new_30min = CO_30min
    CO_1hr = pd.DataFrame()
    for i in time_232:
        j = i + datetime.timedelta(hours=-1)
        num = np.where((data_new['ds'] > j) & (data_new['ds'] <= i))
        CO_1hr = CO_1hr.append(data_group.iloc[num[0]].sum(), ignore_index=True)
    CO_1hr['ds'] = time_232
    data_new_1hr = CO_1hr
    data_new['y'] = data_group['y'].values
    # data_new.drop_duplicates(['ds'], keep="last")  # 去重
    # plt.figure()
    # plt.plot(data_new['y'].values)
    # plt.figure()
    # plt.plot(data_new_1hr['y'].values)
    # plt.show()

    # outlier_threshold = int(input("threshold:"))
    # outlier_threshold_1hr = int(input("threshold:"))
    outlier_threshold = 40
    outlier_threshold_1hr = 400
    # 去异常
    data_new['y'][data_new['y'] > outlier_threshold] = None
    data_new['y'][data_new['y'] < 0] = None


    # 增长率百分比
    data_rate = pd.DataFrame()
    data_rate['timestamp'] = data_new['ds']

    data_new_rate_or = pd.DataFrame()
    data_rate['y_or'] = df['CallOffered']
    data_rate['y_or_diff'] = data_new['y']
    data_new_rate_or['y2'] = df['CallOffered']
    data_new_rate_or['y2'][1:] = df['CallOffered'][:-1]
    data_new_rate_or['y2'][0] = 0
    data_rate['y_rate_or'] = np.where(data_new_rate_or['y2'] == 0, data_new['y'], data_new['y'] / data_new_rate_or['y2'])

    data_new_rate = pd.DataFrame()
    data_rate['y_diff_1min_diff'] = data_new['y'].diff()
    data_new_rate['y2'] = data_new['y']
    data_new_rate['y2'][1:] = data_new['y'][:-1]
    data_new_rate['y2'][0] = 0
    data_rate['y_rate_diff_1min'] = np.where(data_new_rate['y2'] == 0, data_new['y'].diff(), data_new['y'].diff() / data_new_rate['y2'])
    # data_new_rate['timestamp'] = data_new[['ds']]

    data_new_rate_5min = pd.DataFrame()
    data_rate['y_diff_5min'] = data_new_5min['y']
    data_rate['y_diff_5min_diff'] = data_new_5min['y'].diff()
    data_new_rate_5min['y2'] = data_new_5min['y']
    data_new_rate_5min['y2'][1:] = data_new_5min['y'][:-1]
    data_new_rate_5min['y2'][0] = 0
    data_rate['y_rate_diff_5min'] = np.where(data_new_rate_5min['y2'] == 0, data_new_5min['y'].diff(), data_new_5min['y'].diff() / data_new_rate_5min['y2'])

    data_new_rate_10min = pd.DataFrame()
    data_rate['y_diff_10min'] = data_new_10min['y']
    data_rate['y_diff_10min_diff'] = data_new_10min['y'].diff()
    data_new_rate_10min['y2'] = data_new_10min['y']
    data_new_rate_10min['y2'][1:] = data_new_10min['y'][:-1]
    data_new_rate_10min['y2'][0] = 0
    data_rate['y_rate_diff_10min'] = np.where(data_new_rate_10min['y2'] == 0, data_new_10min['y'].diff(), data_new_10min['y'].diff() / data_new_rate_10min['y2'])

    data_new_rate_30min = pd.DataFrame()
    data_rate['y_diff_30min'] = data_new_30min['y']
    data_rate['y_diff_30min_diff'] = data_new_30min['y'].diff()
    data_new_rate_30min['y2'] = data_new_30min['y']
    data_new_rate_30min['y2'][1:] = data_new_30min['y'][:-1]
    data_new_rate_30min['y2'][0] = 0
    data_rate['y_rate_diff_30min'] = np.where(data_new_rate_30min['y2'] == 0, data_new_30min['y'].diff(), data_new_30min['y'].diff() / data_new_rate_30min['y2'])

    data_new_rate_1hr = pd.DataFrame()
    data_rate['y_diff_1hr'] = data_new_1hr['y']
    data_rate['y_diff_1hr_diff'] = data_new_1hr['y'].diff()
    data_new_rate_1hr['y2'] = data_new_1hr['y']
    data_new_rate_1hr['y2'][1:] = data_new_1hr['y'][:-1]
    data_new_rate_1hr['y2'][0] = 0
    data_rate['y_rate_diff_1hr'] = np.where(data_new_rate_1hr['y2'] == 0, data_new_1hr['y'].diff(), data_new_1hr['y'].diff() / data_new_rate_1hr['y2'])
    # data_new_rate_1hr['timestamp'] = data_new[['ds']]

    data_rate.fillna(0)
    data_rate.to_csv('data_rate_10_20210111.csv')
    # data_new_rate_or.to_csv('data_new_rate_or_11_368.csv')
    # # data_new_rate_or_1hr.to_csv('data_new_rate_or_1hr_11_386.csv')
    # data_new_rate.to_csv('data_new_rate_11_368.csv')
    # data_new_rate_1hr.to_csv('data_new_rate_1hr_11_368.csv')


    # # show(df, settings={'block': True})
    # data = pd.read_excel(r"E:\Data\QueueData_232_234_368_386\DATA_232_PST.xlsx")
    # data_new = data[0:12000].append(data[13000:21963])
    # data['ASA_Alert_24hr_new'] = 0
    # data['ASA_Alert_24hr_new'][np.where(data['ASA_24hr'].values > 4)[0]] = 1
    # correlationship
    # cor_fig(data_new)
    # plt.show()
    # plt.savefig('../Data_analysis_figures/SL_232_diff_cor.jpg')
    # qg = '234'  # '232' '234' '368'
    # s = '_1hr'  # '' '_30min'
    # # data = pd.read_excel('E:/Data/ASA_' + qg + '.xlsx')
    # data = pd.read_excel(r"E:\Data\ASA_1hr_234_PST.xlsx", sheet_name='Sheet1')
    # data = data[data['hour_a'] == 'business']
    # data = data[data['month'] == 8]
    # # # data = data[3000:10000].append(data[14000:])
    # cor_fig(data.iloc[:, 0:-2])
    # plt.savefig('../Data_analysis_figures/SL_234_1hr_cor.jpg')
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    # ax[0].plot(data['ASA_24hr'].values[1450:], '-o')
    # ax[0].set_ylabel('ASA')
    # ax[1].plot(data['SL_60_24hr'].values[1450:], '-o')
    # ax[1].set_ylabel('SL')
    # plt.title('ASA_SL_232_09_new')

    # # bar
    # ASA
    # q = '232'
    # s = '_24hr'
    # data_08 = pd.read_excel('E:/Data/ASA_' + q + '.xlsx')
    # data_09 = pd.read_excel('E:/Data/ASA_' + q + '_09.xlsx')
    # num_data_08 = [np.where(data_08['ASA_Alert'+s].values == 0)[0].shape[0],
    #                np.where(data_08['ASA_Alert'+s].values == 1)[0].shape[0]]
    # num_data_09 = [np.where(data_09['ASA_Alert'+s].values == 0)[0].shape[0],
    #                np.where(data_09['ASA_Alert'+s].values == 1)[0].shape[0]]
    # label_pie = ['NonAlert', 'IsAlert']
    # x = ['232', '234', '368', '386']
    # bar_width = 0.3
    # data_nonAlert = [num_data_08[0], num_data_09[0]]
    # data_Alert = [num_data_08[1], num_data_09[1]]
    # plt.bar(x, data_nonAlert, bar_width)
    # plt.bar(x, data_Alert, bar_width, bottom=data_nonAlert)
    # plt.show()

    # # pie
    # SL
    # num_data_pie = [np.where(data_new['Outlier_n'].values == 0)[0].shape[0],
    #                 np.where(data_new['Outlier_n'].values == 1)[0].shape[0]]
    # label_pie = ['NonOutlier', 'IsOutlier']
    # pie_figure(num_data_pie, label_pie)
    # plt.savefig('../Data_analysis_figures/CO_Outlier_1min_new' + '.jpg')
    # plt.cla()
    # ASA
    # num_data_pie = [np.where(data['ASA_Alert'].values == 0)[0].shape[0],
    #                 np.where(data['ASA_Alert'].values == 1)[0].shape[0]]
    # label_pie = ['NonAlert', 'IsAlert']
    # pie_figure(num_data_pie, label_pie)
    # plt.savefig('../Data_analysis_figures/ASA_' + qg + s + '.jpg')
    # plt.cla()
    # # LQT
    # num_data_pie = [np.where(data['IsAlert'].values == 0)[0].shape[0],
    #                 np.where(data['IsAlert'].values == 1)[0].shape[0]]
    # label_pie = ['NonAlert', 'IsAlert']
    # pie_figure(num_data_pie, label_pie)
    # plt.savefig('../Data_analysis_figures/08/LQT_' + qg + s + '.jpg')
    # plt.cla()
    #
    # # venn
    # ASA-SL
    # num_data_venn = [list(np.where(data['ASA_Alert'] == 1)[0]), list(np.where(data['SL_Alert'] == 1)[0])]
    # label_venn = ('ASA_Alert', 'SL_Alert')
    # g = venn_figure(num_data_venn, label_venn)
    # g.get_label_by_id('10').set_fontsize(15)
    # g.get_label_by_id('01').set_fontsize(15)
    # g.get_label_by_id('110').set_fontsize(15)
    # plt.savefig('../Data_analysis_figures/SL_' + qg + s + '_venn_new.jpg')
    # plt.cla()
    # ASA-LQT
    # num_data_venn = [list(np.where(data['ASA_Alert'] == 1)[0]), list(np.where(data['Is_Alert'] == 1)[0])]
    # label_venn = ('ASA_Alert', 'Is_Alert')
    # g = venn_figure(num_data_venn, label_venn)
    # g.get_label_by_id('10').set_fontsize(15)
    # g.get_label_by_id('01').set_fontsize(15)
    # g.get_label_by_id('110').set_fontsize(15)
    # plt.savefig('../Data_analysis_figures/LQT_' + qg + s + '_venn.jpg')
    # plt.cla()

    # # cor
    # column_name = ['CallAnswered', 'CallAnswered60_SL60_NM', 'CallOffered', 'CallOffered60_SL60_DN',
    #                'CallsReleased_AHT_DN', 'LongestQueueTime',
    #                'TotalAbandoned', 'TotalHandleTime_AHT_NM', 'TotalTimetoAnswer', 'SL_Alert', 'ASA_Alert', 'IS_Alert']
    # cor_fig(data[column_name])
    # plt.savefig('../Data_analysis_figures/SL_' + s + '_cor.jpg')

    # decision_tree
    # 'AverageSpeedAnswer', 'PercentageAbandoned', 'TotalAbandoned_24hr', 'CallsReleased_AHT_DN_24hr',
    # 'TotalTimetoAnswer_24hr', 'ServiceLevel60',
    # column_name = ['TotalTimetoAnswer', 'CallAnswered60_SL60_NM',
    #                'CallOffered', 'TotalAbandoned', 'TotalHandleTime_AHT_NM',
    #                'CallAnswered', 'CallOffered60_SL60_DN', 'ASA', 'SL_Alert']
    # data_cor = data[column_name]
    # X = data_cor.iloc[:, 0:8]
    # y = data_cor.iloc[:, 8]
    # feature_names = column_name[0:8]
    # target_names = ['ISnt_Alert', 'Is_Alert']  # 'others',
    # decision_tree(X, y, feature_names, target_names)
    # plt.savefig('../Data_analysis_figures/DecisionTree_SL_' + qg + s + 'weekday.jpg')
    # plt.cla()

    # # Time series --232
    # if qg == '234':
    #     data_232_or = pd.read_excel(r"E:\Data\QueueData_232_234_368_386_short\data_234_new.xlsx", sheet_name=1)
    #     data_232 = data
    #     Time = data_232_or['Timestamp'].values
    #     time_232 = [datetime.datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")],
    #                 "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=-7) + datetime.timedelta(minutes=-1)
    #                 for i in range(len(Time))]
    #     # pd.DataFrame(time_232).to_csv('datetime_232.csv')
    #     plt.figure()
    #     plt.plot(time_232[4346:5363], data_232['ServiceLevel60'].values[4346:5363], '-o')
    #     plt.yticks(range(50, 130, 10))
    #     plt.xlabel('Time')
    #     plt.ylabel('SL_60')
    #     plt.savefig('../Data_analysis_figures/SL_' + s + '_time_series.jpg')
    #     plt.figure()
    #     plt.plot(time_232[4346:5363], data_232['CallOffered'].values[4346:5363], '-o')
    #     plt.yticks(range(0, 800, 100))
    #     plt.xlabel('Time')
    #     plt.ylabel('CallOffered')
    #     plt.savefig('../Data_analysis_figures/CO_' + s + '_time_series.jpg')
    #     plt.figure()
    #     plt.plot(time_232[4346:5363], data_232['CallAnswered'].values[4346:5363], '-o')
    #     plt.yticks(range(0, 800, 100))
    #     plt.xlabel('Time')
    #     plt.ylabel('CallAnswered')
    #     # plt.savefig('../Data_analysis_figures/CA_' + s + '_time_series.jpg')
    #     plt.show()







