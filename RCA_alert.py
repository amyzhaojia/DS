import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time
import json
import seaborn as sns

data_or = pd.read_csv('QueueGroupData202008_231_to_234.csv')
data = data_or.drop('QueueGroupId', axis=1).drop('PartitionKey', axis=1).drop('RowKey', axis=1)
data_al = pd.read_csv('Alert_Data202008_231_to_234.csv')
data_rca = pd.read_csv('AllRCAData202009.csv')
data_rca_ori = data_rca[data_rca['RCA'] != '[]']
data_rca_1 = data_rca['RCA'][data_rca['RCA'] != '[]']
data_rca_232 = data_rca_1[data_rca['QueueGroupId'] == 232]
data_rca_234 = data_rca_1[data_rca['QueueGroupId'] == 234]
json_rca_232 = [json.loads(data_rca_232.values[i]) for i in range(len(data_rca_232))]
json_rca_234 = [json.loads(data_rca_234.values[i]) for i in range(len(data_rca_234))]

rca_AHT_232 = np.zeros(len(json_rca_232))
rca_Volume_232 = np.zeros(len(json_rca_232))
rca_Staff_232 = np.zeros(len(json_rca_232))
rca_Unknown_232 = np.zeros(len(json_rca_232))
rca_AHT_234 = np.zeros(len(json_rca_234))
rca_Volume_234 = np.zeros(len(json_rca_234))
rca_Staff_234 = np.zeros(len(json_rca_234))
rca_Unknown_234 = np.zeros(len(json_rca_234))
rca_AHT_232_value = np.zeros((len(json_rca_232), 4))
rca_AHT_for_232_value = np.zeros((len(json_rca_232), 4))
rca_AHT_234_value = np.zeros((len(json_rca_234), 4))
rca_AHT_for_234_value = np.zeros((len(json_rca_234), 4))

for i in range(len(json_rca_232)):
    if len(json_rca_232[i]) == 4:
        rca_AHT_232[i] = json_rca_232[i][0]['IsAHT'] == str(True) or json_rca_232[i][1]['IsAHT'] == str(True) \
                     or json_rca_232[i][2]['IsAHT'] == str(True) or json_rca_232[i][3]['IsAHT'] == str(True)
        rca_Volume_232[i] = json_rca_232[i][0]['IsVolume'] == str(True) or json_rca_232[i][1]['IsVolume'] == str(True) \
                     or json_rca_232[i][2]['IsVolume'] == str(True) or json_rca_232[i][3]['IsVolume'] == str(True)
        rca_Staff_232[i] = json_rca_232[i][0]['IsStaff'] == str(True) or json_rca_232[i][1]['IsStaff'] == str(True) \
                        or json_rca_232[i][2]['IsStaff'] == str(True) or json_rca_232[i][3]['IsStaff'] == str(True)
        rca_Unknown_232[i] = json_rca_232[i][0]['IsUnknown'] == str(True) or json_rca_232[i][1]['IsUnknown'] == str(True) \
                        or json_rca_232[i][2]['IsUnknown'] == str(True) or json_rca_232[i][3]['IsUnknown'] == str(True)
        rca_AHT_232_value[i] = [json_rca_232[i][0]['AHT'], json_rca_232[i][1]['AHT'], json_rca_232[i][2]['AHT'],
                             json_rca_232[i][3]['AHT']]
        rca_AHT_for_232_value[i] = [json_rca_232[i][0]['Forecast_AHT'], json_rca_232[i][1]['Forecast_AHT'], json_rca_232[i][2]['Forecast_AHT'],
                             json_rca_232[i][3]['Forecast_AHT']]
    else:
        rca_AHT_232[i] = json_rca_232[i][0]['IsAHT'] == str(True) or json_rca_232[i][1]['IsAHT'] == str(True)\
                     or json_rca_232[i][2]['IsAHT'] == str(True)
        rca_Volume_232[i] = json_rca_232[i][0]['IsVolume'] == str(True) or json_rca_232[i][1]['IsVolume'] == str(True)\
                     or json_rca_232[i][2]['IsVolume'] == str(True)
        rca_Staff_232[i] = json_rca_232[i][0]['IsStaff'] == str(True) or json_rca_232[i][1]['IsStaff'] == str(True)\
                     or json_rca_232[i][2]['IsStaff'] == str(True)
        rca_Unknown_232[i] = json_rca_232[i][0]['IsUnknown'] == str(True) or json_rca_232[i][1]['IsUnknown'] == str(True)\
                     or json_rca_232[i][2]['IsUnknown'] == str(True)
        rca_AHT_232_value[i] = [json_rca_232[i][0]['AHT'], json_rca_232[i][1]['AHT'], json_rca_232[i][2]['AHT'], 0]
        rca_AHT_for_232_value[i] = [json_rca_232[i][0]['Forecast_AHT'], json_rca_232[i][1]['Forecast_AHT'], json_rca_232[i][2]['Forecast_AHT'], 0]


for i in range(len(json_rca_234)):
    if len(json_rca_234[i]) == 4:
        rca_AHT_234[i] = json_rca_234[i][0]['IsAHT'] == str(True) or json_rca_234[i][1]['IsAHT'] == str(True) \
                     or json_rca_234[i][2]['IsAHT'] == str(True) or json_rca_234[i][3]['IsAHT'] == str(True)
        rca_Volume_234[i] = json_rca_234[i][0]['IsVolume'] == str(True) or json_rca_234[i][1]['IsVolume'] == str(True) \
                     or json_rca_234[i][2]['IsVolume'] == str(True) or json_rca_234[i][3]['IsVolume'] == str(True)
        rca_Staff_234[i] = json_rca_234[i][0]['IsStaff'] == str(True) or json_rca_234[i][1]['IsStaff'] == str(True) \
                        or json_rca_234[i][2]['IsStaff'] == str(True) or json_rca_234[i][3]['IsStaff'] == str(True)
        rca_Unknown_234[i] = json_rca_234[i][0]['IsUnknown'] == str(True) or json_rca_234[i][1]['IsUnknown'] == str(True) \
                        or json_rca_234[i][2]['IsUnknown'] == str(True) or json_rca_234[i][3]['IsUnknown'] == str(True)
        rca_AHT_234_value[i] = [json_rca_234[i][0]['AHT'], json_rca_234[i][1]['AHT'], json_rca_234[i][2]['AHT'],
                             json_rca_234[i][3]['AHT']]
        rca_AHT_for_234_value[i] = [json_rca_234[i][0]['Forecast_AHT'], json_rca_234[i][1]['Forecast_AHT'], json_rca_234[i][2]['Forecast_AHT'],
                             json_rca_234[i][3]['Forecast_AHT']]
    else:
        rca_AHT_234[i] = json_rca_234[i][0]['IsAHT'] == str(True) or json_rca_234[i][1]['IsAHT'] == str(True)\
                     or json_rca_234[i][2]['IsAHT'] == str(True)
        rca_Volume_234[i] = json_rca_234[i][0]['IsVolume'] == str(True) or json_rca_234[i][1]['IsVolume'] == str(True)\
                     or json_rca_234[i][2]['IsVolume'] == str(True)
        rca_Staff_234[i] = json_rca_234[i][0]['IsStaff'] == str(True) or json_rca_234[i][1]['IsStaff'] == str(True)\
                     or json_rca_234[i][2]['IsStaff'] == str(True)
        rca_Unknown_234[i] = json_rca_234[i][0]['IsUnknown'] == str(True) or json_rca_234[i][1]['IsUnknown'] == str(True)\
                     or json_rca_234[i][2]['IsUnknown'] == str(True)
        rca_AHT_234_value[i] = [json_rca_234[i][0]['AHT'], json_rca_234[i][1]['AHT'], json_rca_234[i][2]['AHT'],
                             json_rca_234[i][3]['AHT']]
        rca_AHT_for_234_value[i] = [json_rca_234[i][0]['Forecast_AHT'], json_rca_234[i][1]['Forecast_AHT'], json_rca_234[i][2]['Forecast_AHT'],
                             json_rca_234[i][3]['Forecast_AHT']]

new_data_232 = pd.DataFrame(columns=['IsAHT', 'IsStaff', 'IsVolume', 'IsUnknown', 'Timestamp', 'Alert_level'], index=range(len(rca_AHT_232)))
new_data_234 = pd.DataFrame(columns=['IsAHT', 'IsStaff', 'IsVolume', 'IsUnknown', 'Timestamp', 'Alert_level'], index=range(len(rca_AHT_234)))
new_data_232[['IsAHT', 'IsStaff', 'IsVolume', 'IsUnknown']] = np.array([rca_AHT_232, rca_Staff_232, rca_Volume_232, rca_Unknown_232]).transpose()
new_data_234[['IsAHT', 'IsStaff', 'IsVolume', 'IsUnknown']] = np.array([rca_AHT_234, rca_Staff_234, rca_Volume_234, rca_Unknown_234]).transpose()
new_data_232['PartitionKey'] = data_rca_ori['PartitionKey'][data_rca['QueueGroupId'] == 232].values
new_data_234['PartitionKey'] = data_rca_ori['PartitionKey'][data_rca['QueueGroupId'] == 234].values
new_data_232['RowKey'] = data_rca_ori['RowKey'][data_rca['QueueGroupId'] == 232].values
new_data_234['RowKey'] = data_rca_ori['RowKey'][data_rca['QueueGroupId'] == 234].values
new_data_232['Timestamp'] = data_rca_ori['Timestamp'][data_rca['QueueGroupId'] == 232].values
new_data_234['Timestamp'] = data_rca_ori['Timestamp'][data_rca['QueueGroupId'] == 234].values
# new_data_232['Alert_level'] = data_al['ThresholdLevel'][data_rca['QueueGroupId'] == 232].values
# new_data_234['Alert_level'] = data_al['ThresholdLevel'][data_rca['QueueGroupId'] == 234].values
a_232 = pd.DataFrame({'AHT_value_1': rca_AHT_232_value[:,0], 'AHT_value_2': rca_AHT_232_value[:,1],
                      'AHT_value_3': rca_AHT_232_value[:, 2], 'AHT_value_4': rca_AHT_232_value[:,3]})
b_232 = pd.DataFrame({'AHT_for_value_1': rca_AHT_for_232_value[:,0], 'AHT_for_value_2': rca_AHT_for_232_value[:,1],
                      'AHT_for_value_3': rca_AHT_for_232_value[:,2], 'AHT_for_value_4': rca_AHT_for_232_value[:,3]})
new_data_232 = pd.concat([new_data_232, a_232, b_232], axis=1)
a_234 = pd.DataFrame({'AHT_value_1': rca_AHT_234_value[:,0], 'AHT_value_2': rca_AHT_234_value[:,1],
                      'AHT_value_3': rca_AHT_234_value[:,2], 'AHT_value_4': rca_AHT_234_value[:,3]})
b_234 = pd.DataFrame({'AHT_for_value_1': rca_AHT_for_234_value[:,0], 'AHT_for_value_2': rca_AHT_for_234_value[:,1],
                      'AHT_for_value_3': rca_AHT_for_234_value[:,2], 'AHT_for_value_4': rca_AHT_for_234_value[:,3]})
new_data_234 = pd.concat([new_data_234, a_234, b_234], axis=1)
new_data_232.to_csv('new_data_232_202009.csv')
new_data_234.to_csv('new_data_234_202009.csv')
new_data_232 = pd.read_csv('new_data_232_202009.csv')
new_data_234 = pd.read_csv('new_data_234_202009.csv')

new_RCA_data_232 = pd.merge(new_data_232, data_or, on=['PartitionKey', 'RowKey'], how='left')
new_RCA_data_234 = pd.merge(new_data_234, data_or, on=['PartitionKey', 'RowKey'], how='left')

or_data_232 = data_or[data_or['QueueGroupId'] == 232]
or_data_234 = data_or[data_or['QueueGroupId'] == 234]

data_232 = pd.merge(new_RCA_data_232, or_data_232, on=['PartitionKey', 'RowKey'], how='outer')
data_232 = data_232.drop(data_232.columns[10:34], axis=1).drop(['Unnamed: 0', 'Timestamp_x', 'Timestamp_y'], axis=1)
data_234 = pd.merge(new_RCA_data_234, or_data_234, on=['PartitionKey', 'RowKey'], how='outer')
data_234 = data_234.drop(data_234.columns[10:34], axis=1).drop(['Unnamed: 0', 'Timestamp_x', 'Timestamp_y'], axis=1)
data_232 = data_232.fillna(0)
data_234 = data_234.fillna(0)
data_232 = data_232.eval("IsAHT_Alert = IsAHT+Alert_level").eval("IsStaff_Alert = IsStaff+Alert_level")\
    .eval("IsVolume_Alert = IsVolume+Alert_level")
data_234 = data_234.eval("IsAHT_Alert = IsAHT+Alert_level").eval("IsStaff_Alert = IsStaff+Alert_level")\
    .eval("IsVolume_Alert = IsVolume+Alert_level")

data_232 = data_232.drop(['IsAHT', 'IsStaff', 'IsVolume', 'IsUnknown', 'PartitionKey', 'RowKey', 'QueueGroupId_y'], axis=1)
data_234 = data_234.drop(['IsAHT', 'IsStaff', 'IsVolume', 'IsUnknown', 'PartitionKey', 'RowKey', 'QueueGroupId_y'], axis=1)
new_column = ['Alert_level', 'Timestamp', 'CallsInQueue', 'LongestQueueTime', 'CallOffered', 'CallAnswered',
              'CurrentWaitTime', 'CallsReleased_AHT_DN', 'TotalHandleTime_AHT_NM', 'TotalTimetoAnswer',
              'CallOffered60_SL60_DN', 'CallAnswered60_SL60_NM', 'AverageHandelTime', 'AverageSpeedAnswer',
              'AverageWaitTime', 'ServiceLevel60', 'PercentageAbandoned', 'TotalAbandoned', 'PstTimeStamp',
              'CallsInQueueDeviation', 'PercentDeviation', 'PreCallsInQueue', 'CallsInQueueInterval',
              'PreCallsInQueueInterval', 'PreCallsInQueueDeviation', 'IsAHT_Alert', 'IsStaff_Alert', 'IsVolume_Alert']
data_232.columns = new_column
data_234.columns = new_column

# data_232.to_csv('data_232.csv')
# data_234.to_csv('data_234.csv')

# cor_RCA_232 = new_RCA_data_232.corr()
# cor_RCA_234 = new_RCA_data_234.corr()
# # save files
# abs(cor_RCA_232.iloc[:, 1:5]).to_csv('abs_cor_RCA_232.csv')
# abs(cor_RCA_234.iloc[:, 1:5]).to_csv('abs_cor_RCA_234.csv')

cor_232 = data_232.corr()
cor_234 = data_234.corr()
# cor_232.iloc[:, -3::].to_csv('cor_232.csv')
# cor_234.iloc[:, -3::].to_csv('cor_234.csv')
# abs(cor_232.iloc[:, -3::]).to_csv('abs_cor_232.csv')
# abs(cor_234.iloc[:, -3::]).to_csv('abs_cor_234.csv')

plt.figure(figsize=(20, 20))
# corr_pic = sns.heatmap(cor_232, annot=True)
# plt.savefig('corr_pic_232.jpg')
corr_pic = sns.heatmap(cor_234, annot=True)
plt.savefig('corr_pic_234.jpg')
# plt.show()
print()
