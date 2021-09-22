import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import time
from sklearn.ensemble import IsolationForest
import seaborn as sns
# from fbprophet import Prophet
# from pandas_profiling import ProfileReport

data_232 = pd.read_csv(r"E:\Data\QueueData_232_234_368_386_09\QueueGroupData202009_232.csv")
data_232 = data_232.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)
data_234 = pd.read_csv(r"E:\Data\QueueData_232_234_368_386_09\QueueGroupData202009_234.csv")
data_234 = data_234.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)
# data_368 = pd.read_csv(r"E:\Data\QueueData_232_234_368_386_09\QueueGroupData202009_368.csv")
# data_368 = data_368.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)
# data_386 = pd.read_csv(r"E:\Data\QueueData_232_234_368_386_09\QueueGroupData202009_386.csv")
# data_386 = data_386.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)

# # data_quality_report
# profile = ProfileReport(data_232, title="Pandas Profiling Report")  # large number: (, minimal=True)
# profile.to_file("report_data_232.html")
# profile = ProfileReport(data_234, title="Pandas Profiling Report")  # large number: (, minimal=True)
# profile.to_file("report_data_234.html")
# profile = ProfileReport(data_368, title="Pandas Profiling Report")  # large number: (, minimal=True)
# profile.to_file("report_data_368.html")
# profile = ProfileReport(data_386, title="Pandas Profiling Report")  # large number: (, minimal=True)
# profile.to_file("report_data_386.html")

Time = data_232['PstTimeStamp'].values
time_232 = [datetime.datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")],
        "%Y-%m-%d %H:%M:%S") for i in range(len(Time))]
#  + datetime.timedelta(hours=-7) + datetime.timedelta(minutes=-1)
data_232['datetime'] = time_232
measures = ['CallOffered', 'CallAnswered', 'TotalAbandoned', 'CallAnswered60_SL60_NM', 'CallOffered60_SL60_DN',
            'TotalTimetoAnswer', 'TotalHandleTime_AHT_NM', 'CallsReleased_AHT_DN', 'AverageSpeedAnswer']
data_232_group = data_232[measures].groupby(data_232['datetime'].apply(lambda x: x.day), as_index=False).diff() # .fillna(0)
data_232_group['datatime'] = time_232
# data_232_group['PstTimeStamp'] = data_232['PstTimeStamp']
ASA_30min_232 = pd.DataFrame()
for i in time_232:
    j = i + datetime.timedelta(hours=-1)
    num = np.where((data_232_group['datatime'] > j) & (data_232_group['datatime'] <= i))
    ASA_30min_232 = ASA_30min_232.append(data_232_group.iloc[num[0]].sum(), ignore_index=True)
ASA_30min_232['datatime'] = time_232
    # print()

Time = data_234['PstTimeStamp'].values
time_234 = [datetime.datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")],
        "%Y-%m-%d %H:%M:%S") for i in range(len(Time))]
# + datetime.timedelta(hours=-7) + datetime.timedelta(minutes=-1)
data_234['datetime'] = time_234
data_234_group = data_234[measures].groupby(data_234['datetime'].apply(lambda x: x.day), as_index=False).diff()
data_234_group['datatime'] = time_234
# data_234_group['PstTimeStamp'] = data_234['PstTimeStamp']
ASA_30min_234 = pd.DataFrame()
for i in time_234:
    j = i + datetime.timedelta(hours=-1)
    num = np.where((data_234_group['datatime'] > j) & (data_234_group['datatime'] <= i))
    ASA_30min_234 = ASA_30min_234.append(data_234_group.iloc[num[0]].sum(), ignore_index=True)
ASA_30min_234['datatime'] = time_234
# Time = data_368['Timestamp'].values
# time_368 = [datetime.datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")],
#         "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=-7) + datetime.timedelta(minutes=-1) for i in range(len(Time))]
# data_368['datetime'] = time_368
# data_368_group = data_368[measures].groupby(data_368['datetime'].apply(lambda x: x.day), as_index=False).diff()
# data_368_group['datatime'] = time_368
# ASA_30min_368 = pd.DataFrame()
# for i in time_368:
#     j = i + datetime.timedelta(minutes=-30)
#     num = np.where((data_368_group['datatime'] > j) & (data_368_group['datatime'] <= i))
#     ASA_30min_368 = ASA_30min_368.append(data_368_group.iloc[num[0]].sum(), ignore_index=True)
#
# Time = data_386['Timestamp'].values
# time_386 = [datetime.datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")],
#         "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=-7) + datetime.timedelta(minutes=-1) for i in range(len(Time))]
# data_386['datetime'] = time_386
# data_386_group = data_386[measures].groupby(data_386['datetime'].apply(lambda x: x.day), as_index=False).diff()
# data_386_group['datatime'] = time_386
# ASA_30min_386 = pd.DataFrame()
# for i in time_386:
#     j = i + datetime.timedelta(minutes=-30)
#     num = np.where((data_386_group['datatime'] > j) & (data_386_group['datatime'] <= i))
#     ASA_30min_386 = ASA_30min_386.append(data_386_group.iloc[num[0]].sum(), ignore_index=True)

data_232_group.to_csv('DATA_232_09_PST.csv')
data_234_group.to_csv('DATA_234_09_PST.csv')
ASA_30min_232.to_csv('ASA_1hr_232_09_PST.csv')
ASA_30min_234.to_csv('ASA_1hr_234_09_PST.csv')
# ASA_30min_368.to_csv('ASA_30min_368_09.csv')
# ASA_30min_386.to_csv('ASA_30min_386_09.csv')

# cor_232 = ASA_30min_232.corr()
# cor_234 = ASA_30min_234.corr()
# cor_368 = ASA_30min_368.corr()
# cor_386 = ASA_30min_386.corr()
#
# plt.figure(figsize=(20, 20))
# corr_pic_232 = sns.heatmap(cor_232, annot=True)
# plt.savefig('corr_pic_232.jpg')
# plt.figure(figsize=(20, 20))
# corr_pic_234 = sns.heatmap(cor_234, annot=True)
# plt.savefig('corr_pic_234.jpg')
# plt.figure(figsize=(20, 20))
# corr_pic_368 = sns.heatmap(cor_368, annot=True)
# plt.savefig('corr_pic_368.jpg')
# plt.figure(figsize=(20, 20))
# corr_pic_386 = sns.heatmap(cor_386, annot=True)
# plt.savefig('corr_pic_386.jpg')

# plt.plot(time_232, ASA_30min_232['AverageSpeedAnswer'].values, '-o')
# plt.plot(time_1, data_232['CallAnswered'].values, '-ro')
# plt.plot(time_1, data_232['TotalTimetoAnswer'].values, '-*')
# plt.show()
# print()
