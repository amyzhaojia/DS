import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import time
from sklearn.ensemble import IsolationForest
import seaborn as sns
from fbprophet import Prophet
from pandas_profiling import ProfileReport



data = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueData_232_234_368_386.csv")
data_232 = data[data['QueueGroupId'] == 232]
# data_234 = data[data['QueueGroupId'] == 234]
# data_368 = data[data['QueueGroupId'] == 368]
# data_386 = data[data['QueueGroupId'] == 386]

Time = data_232['Timestamp'].values
time_1 = [datetime.datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")],
        "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=-7) + datetime.timedelta(minutes=-1) for i in range(len(Time))]

data_232['datetime'] = time_1
measures = ['CallOffered', 'CallAnswered', 'TotalAbandoned', 'CallAnswered60_SL60_NM', 'CallOffered60_SL60_DN',
            'TotalTimetoAnswer', 'TotalHandleTime_AHT_NM', 'CallsReleased_AHT_DN']
data_232_group = data_232[measures].groupby(data_232['datetime'].apply(lambda x: x.day), as_index=False).diff()
data_232_group['datatime'] = time_1

del data

ASA_24hr = pd.DataFrame()
for i in time_1:
    j = i + datetime.timedelta(hours=-24)
    num = np.where((data_232_group['datatime'] >= j) & (data_232_group['datatime'] < i))
    ASA_24hr.append(data_232_group.iloc[num[0]].sum(), ignore_index=True)
ASA_24hr.to_csv('ASA_24hr_year.csv')
ASA_24hr['ASA_24hr'] = 0 if ASA_24hr['CallAnswered'] == 0 else ASA_24hr['TotalTimetoAnswer']/(ASA_24hr['CallAnswered']*60)
# plt.plot(time_1, data_232_group['AverageSpeedAnswer'].values, '-o')
# plt.plot(time_1, data_232['CallAnswered'].values, '-ro')
# plt.plot(time_1, data_232['TotalTimetoAnswer'].values, '-*')
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
ax[0].plot(time_1, ASA_24hr['ASA_24hr'].values, '-o')
ax[0].set_ylabel('ASA_24hr')
ax[1].plot(time_1, ASA_24hr['TotalTimetoAnswer'].values, '-o')
ax[1].set_ylabel('TotalTimetoAnswer')
ax[2].plot(time_1, ASA_24hr['CallAnswered'].values, '-o')
ax[2].set_ylabel('CallAnswered')
plt.xlabel('Time')
plt.show()
