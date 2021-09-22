import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import IsolationForest
# from fbprophet import Prophet

data_or = pd.read_csv('QueueGroupData202008_231_to_234.csv')
data = data_or.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)
data_al = pd.read_csv('Alert_Data202008_231_to_234.csv')

data_231 = data[data_or['QueueGroupId'] == 231]
data_232 = data[data_or['QueueGroupId'] == 232]
data_233 = data[data_or['QueueGroupId'] == 233]
data_234 = data[data_or['QueueGroupId'] == 234]

data_al_231 = data_al[data_al['QueueGroupId'] == 231]

AHT = data['AverageHandelTime']
AHT_231 = data_231['AverageHandelTime']

# # iforest_anomaly_detection
x_AHT = AHT_231.values.reshape(-1, 1)
clf = IsolationForest(max_samples=100*2, contamination=0.02)
clf.fit(x_AHT[0:2000])
y_AHT = clf.predict(x_AHT[0:2000])

# plt.plot(AHT_231)
# plt.plot(np.where(y_AHT == -1)[0], AHT_231[np.where(y_AHT == -1)[0]], "ro")
plt.plot(x_AHT[0:2000])
plt.plot(y_AHT, 'y')
plt.show()

AHT_232 = data_232['AverageHandelTime']
AHT_233 = data_233['AverageHandelTime']
AHT_234 = data_234['AverageHandelTime']
LQT = data_231['LongestQueueTime']
Time = data_231['Timestamp'].values
time_1 = [datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")], "%Y-%m-%d %H:%M:%S")
          for i in range(len(Time))]

cor = AHT.corr(LQT)
# cor = np.corrcoef(AHT, LQT)
cor_all_231 = data_231.corrwith(AHT_231).sort_values()
cor_all_232 = data_232.corrwith(AHT_232).sort_values()
cor_all_233 = data_233.corrwith(AHT_233).sort_values()
cor_all_234 = data_234.corrwith(AHT_234).sort_values()

# # save files
# cor_all_231.to_csv('cor_all_231.csv')
# cor_all_232.to_csv('cor_all_232.csv')
# cor_all_233.to_csv('cor_all_233.csv')
# cor_all_234.to_csv('cor_all_234.csv')

# # plot AHT--time
# plt.figure()
# # ax1 = plt.subplots(211)
# plt.plot(AHT_231.values)
# ax2 = plt.subplots(212)
plt.figure()
# plt.plot(AHT_232.values)
plt.plot(time_1, AHT_231.values, '-o')
plt.plot(time_1, 24*np.ones(len(time_1)), '--r')

# # AHT--Alerts
# AHT_n_al = data_234['AverageHandelTime'][data_234['LongestQueueTime'] < 15]
# AHT_al_1 = data_234['AverageHandelTime'][(data_234['LongestQueueTime'] >= 15) & (data_234['LongestQueueTime'] < 25)]
# AHT_al_2 = data_234['AverageHandelTime'][(data_234['LongestQueueTime'] >= 25) & (data_234['LongestQueueTime'] < 45)]
# AHT_al_3 = data_234['AverageHandelTime'][data_234['LongestQueueTime'] >= 45]
# plt.figure()
# plt.plot(AHT_234.keys(), AHT_234.values)
# plt.plot(AHT_al_1, 'yo')
# plt.plot(AHT_al_2, 'go')
# plt.plot(AHT_al_3, 'ro')
# # plt.plot(AHT_n_al, 'go')
# # plt.xticks(time_1, time_1, rotation=45)
# plt.show()


# # threshord--sum
# time = data['Timestamp'][data['Timestamp'].str.contains('2020-08-05')]
#
# threshord_old_0 = []
# threshord_old_1 = []
# threshord_old_2 = []
# threshord_old_3 = []
# for i in range(24):
#     time = data['Timestamp'].str.contains('2020-08-05T')
#     time_time = data['Timestamp'][time]
#     feature_feature = data['LongestQueueTime'][time]
#     t_h = [datetime.datetime.strptime(time_time[i].replace("T", " ")[:time_time[1].replace("T", " ").rfind(".")],
#                                       "%Y-%m-%d %H:%M:%S").strftime("%H") for i in feature_feature.index.tolist()]
#     t_h_int = list(map(lambda x: int(x), t_h))
#     feature = data['LongestQueueTime'][np.where(np.array(t_h_int) == i)[0]]
#     threshord_old_0.append(sum(feature.values < 15))
#     threshord_old_1.append(sum((feature.values >= 15) & (feature.values < 25)))
#     threshord_old_2.append(sum((feature.values >= 15) & (feature.values < 45)))
#     threshord_old_3.append(sum(feature.values >= 45))
#
# ind = np.arange(24)
# width = 0.3
# fig, ax = plt.subplots()
# # rects0 = ax.bar(ind - 3*width/2, threshord_old_0, width, color='SkyBlue', align='center')
# rects1 = ax.bar(ind - width/2, threshord_old_1, width, color='IndianRed', align='center')
# rects2 = ax.bar(ind + width/2, threshord_old_2, width, color='b', align='center')
# rects3 = ax.bar(ind + 3*width/2, threshord_old_3, width, color='r', align='center')
# plt.show()
# # print(data)
