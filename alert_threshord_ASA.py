import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import time
from sklearn.ensemble import IsolationForest
import seaborn as sns
# from fbprophet import Prophet
from pandas_profiling import ProfileReport

# measures = ['CallOffered_diff_2', 'CallAnswered_diff_2', 'TotalAbandoned_diff_2', 'CallAnswered60_SL60_NM_diff_2',
#             'CallOffered60_SL60_DN_diff_2', 'ASA_Alert_bool', 'TotalTimetoAnswer_diff_2',
#             'TotalHandleTime_AHT_NM_diff_2', 'CallsReleased_AHT_DN_diff_2']
# data_diff_232 = pd.read_csv(r"E:\WX_file\WeChat Files\qq877315416\FileStorage\File\2020-10\QueueGroupData202009_386_diff(1).csv")
# cor_232 = data_diff_232[measures].corr()
data_diff_232_24hr = pd.read_csv("ASA_24hr_234.csv")
data_diff_232_30min = pd.read_csv("ASA_30min_234.csv")
data_232 = pd.read_excel(r"E:\Data\QueueData_232_234_368_386_short\data_234_new.xlsx", sheet_name=1)
Time = data_232['Timestamp'].values
time_232 = [datetime.datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")],
        "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=-7) + datetime.timedelta(minutes=-1) for i in range(len(Time))]
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
ax[0].plot(time_232, data_232['AverageSpeedAnswer'].values, '-o')
ax[0].set_ylabel('ASA')
ax[1].plot(time_232, data_diff_232_24hr['TotalTimetoAnswer'].values, '-o')
ax[1].set_ylabel('TotalTimetoAnswer')
ax[2].plot(time_232, data_diff_232_24hr['CallAnswered'].values, '-o')
ax[2].set_ylabel('CallAnswered')
plt.xlabel('Time')

# plot one day data
plt.figure()
plt.plot(time_232[4346:5363], data_diff_232_24hr['ASA_24hr'].values[4346:5363], '-o')
plt.yticks(range(0, 8, 1))
plt.xlabel('Time')
plt.ylabel('ASA_24hr')
plt.figure()
plt.plot(time_232[4346:5363], data_diff_232_30min['ASA_30min'].values[4346:5363], '-o')
plt.xlabel('Time')
plt.ylabel('ASA_30min')
plt.yticks(range(0, 8, 1))
plt.figure()
plt.plot(time_232[4346:5363], data_diff_232_24hr['TotalTimetoAnswer_24hr'].values[4346:5363], '-o')
plt.yticks(range(0, 13000, 2000))
plt.xlabel('Time')
plt.ylabel('TotalTimetoAnswer_24hr')
plt.figure()
plt.plot(time_232[4346:5363], data_diff_232_30min['TotalTimetoAnswer_30min'].values[4346:5363], '-o')
plt.yticks(range(0, 13000, 2000))
plt.xlabel('Time')
plt.ylabel('TotalTimetoAnswer_30min')
plt.figure()
plt.plot(time_232[4346:5363], data_diff_232_24hr['CallAnswered_24hr'].values[4346:5363], '-o')
plt.yticks(range(0, 640, 80))
plt.xlabel('Time')
plt.ylabel('CallAnswered_24hr')
plt.figure()
plt.plot(time_232[4346:5363], data_diff_232_30min['CallAnswered_30min'].values[4346:5363], '-o')
plt.yticks(range(0, 640, 80))
plt.xlabel('Time')
plt.ylabel('CallAnswered_30min')
plt.figure()
plt.plot(time_232[4346:5363], data_232['AverageSpeedAnswer'].values[4346:5363], '-o')
plt.yticks(range(0, 8, 1))
plt.xlabel('Time')
plt.ylabel('ASA')
plt.figure()
plt.plot(time_232[4346:5363], data_232['TotalTimetoAnswer'].values[4346:5363], '-o')
plt.yticks(range(0, 13000, 2000))
plt.xlabel('Time')
plt.ylabel('TotalTimetoAnswer')
plt.figure()
plt.plot(time_232[4346:5363], data_232['CallAnswered'].values[4346:5363], '-o')
plt.yticks(range(0, 640, 80))
plt.xlabel('Time')
plt.ylabel('CallAnswered')

ASA_Alert = np.where(data_diff_232_30min['ASA_Alert'] == 1)[0]
Is_Alert = np.where(data_diff_232_30min['IsAlert'] == 1)[0]
intersection = list(set(ASA_Alert).intersection(set(Is_Alert)))

# # correlationship
# data_232 = pd.read_excel(r"E:\Data\QueueData_232_234_368_386_short\data_232_new.xlsx", sheet_name=1)
# measures = ['CallOffered', 'CallAnswered', 'TotalAbandoned', 'CallAnswered60_SL60_NM', 'CallOffered60_SL60_DN',
#             'TotalTimetoAnswer', 'TotalHandleTime_AHT_NM', 'CallsReleased_AHT_DN', 'AverageSpeedAnswer', 'ground_truth']
# cor_232 = data_232[measures].corr()
# plt.figure(figsize=(20, 20))
# corr_pic_232 = sns.heatmap(cor_232, annot=True)
# plt.savefig('corr_232_orASA.jpg')

plt.figure()
plt.plot(data_diff_232_24hr['ASA_24hr'], 'o')
plt.plot(data_diff_232_24hr['ASA_24hr'][np.where(data_diff_232_30min['ASA_Alert'] == 1)[0]], 'ro')
plt.plot(data_diff_232_24hr['ASA_24hr'][np.where(data_diff_232_30min['IsAlert'] == 1)[0]], 'y*')

data_232 = pd.read_excel(r"E:\Data\QueueData_232_234_368_386_short\data_232_new.xlsx", sheet_name=1)
data_232 = data_232.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)
data_234 = pd.read_excel(r"E:\Data\QueueData_232_234_368_386_short\data_234_new.xlsx", sheet_name=1)
data_234 = data_234.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)
data_368 = pd.read_excel(r"E:\Data\QueueData_232_234_368_386_short\data_368_new.xlsx", sheet_name=1)
data_368 = data_368.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)
data_386 = pd.read_excel(r"E:\Data\QueueData_232_234_368_386_short\data_386_new.xlsx", sheet_name=1)
data_386 = data_386.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)

# # data_quality_report
# profile = ProfileReport(data_232, title="Pandas Profiling Report")  # large number: (, minimal=True)
# profile.to_file("report_data_232.html")
# profile = ProfileReport(data_234, title="Pandas Profiling Report")  # large number: (, minimal=True)
# profile.to_file("report_data_234.html")
# profile = ProfileReport(data_368, title="Pandas Profiling Report")  # large number: (, minimal=True)
# profile.to_file("report_data_368.html")
# profile = ProfileReport(data_386, title="Pandas Profiling Report")  # large number: (, minimal=True)
# profile.to_file("report_data_386.html")

Time = data_232['Timestamp'].values
time_1 = [datetime.datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")],
        "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=-7) + datetime.timedelta(minutes=-1) for i in range(len(Time))]

data_232['datetime'] = time_1
measures = ['CallOffered', 'CallAnswered', 'TotalAbandoned', 'CallAnswered60_SL60_NM', 'CallOffered60_SL60_DN',
            'TotalTimetoAnswer', 'TotalHandleTime_AHT_NM', 'CallsReleased_AHT_DN', 'AverageSpeedAnswer']
data_232_group = data_232[measures].groupby(data_232['datetime'].apply(lambda x: x.day), as_index=False).diff()
data_232_group['datatime'] = time_1

ASA_30min = pd.DataFrame()
for i in time_1:
    j = i + datetime.timedelta(minutes=+30)
    num = np.where((data_232_group['datatime'] >= i) & (data_232_group['datatime'] < j))
    ASA_30min.append(data_232_group.iloc[num[0]].sum(), ignore_index=True)
plt.plot(time_1, data_232_group['AverageSpeedAnswer'].values, '-o')
# plt.plot(time_1, data_232['CallAnswered'].values, '-ro')
# plt.plot(time_1, data_232['TotalTimetoAnswer'].values, '-*')
plt.show()
print()
