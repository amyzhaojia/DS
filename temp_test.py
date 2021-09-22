import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from AD_ALL import _process_data
import pyodbc
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
import seaborn as sns
import pre_processing
import datetime
from fbprophet import Prophet
import json

filename = r'E:\Data\QueueGroupData202105_234.csv'
data = pd.read_csv(filename)
data_CO = data[['CallOffered','Timestamp']]
data_CO['Datetime'] = pre_processing.data_time_process(data_CO)['datetime']
# with open(r"C:\Users\admin\Downloads\SkillRTM.json",'r') as load_f:
#     load_dict = json.load(load_f)
# data1 = []
# for i in range(len(load_dict)):
#     data = load_dict[i]
#     data1 = data1+list(data.items())
# print()
# data = json.loads(r"C:\Users\admin\Downloads\SkillRTM.json")
# data = pd.read_csv('./file/test_agentlogin_data.csv')
# data['datetime'] = pd.to_datetime(data['datetime'])
# ## diff filter
# # data['data_diff'] = data['Staff'].diff().diff().diff()
# # data['Staff'][abs(data['data_diff'])>5] = None
# # data['Staff'] = data['Staff'].fillna(method='ffill')
# # data.plot(x='datetime',y='Staff')
# ## median filter
# import scipy.signal as signal
# data['Staff'] = signal.medfilt(data['Staff'],kernel_size=55)
# data.plot(x='datetime',y='Staff')
# data.plot.scatter(x='datetime',y='data_diff')
# plt.show()
# QueueGroupId = 345
# num_data=7*24*60
# interval_width=0.95
# df = pd.read_csv('./file/data_AgentLogedIn' + str(QueueGroupId) + '.csv')[['datetime', 'Staff']]
# df['datetime'] = pd.to_datetime(df['datetime']).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")[0:17] + '00')
# df = df.rename(columns={'datetime':'ds','Staff':'y'})
# m = Prophet(interval_width=interval_width)
# m.fit(df)
# future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
# forecast = m.predict(future)
# forecast_out = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
# forecast_out.to_csv('./file/prophet_staff_'+str(QueueGroupId)+'.csv')

# data_staff = pd.read_csv('./file/data_staff_timestamp.csv')
# data_staff.plot(x='datetime')

# # agentgroup--staff
# def agentgroup_staff(data, datetime_name='Timestamp'):
#     data['datetime'] = pre_processing.data_time_process(data, datetime_name)
#     def time_approximation(Timestamp):
#         if int(str(Timestamp)[17:19])>30:
#             datetime_1 = pd.to_datetime(str(Timestamp)[0:16] + ':00')+ datetime.timedelta(0,60,0)
#         else:
#             datetime_1 = pd.to_datetime(str(Timestamp)[0:16]+':00')
#         return datetime_1
#     data['datetime_n'] = data['datetime'].apply(lambda x:time_approximation(x))
#     data_staff_name = data[data['Measure'] == 'AgentsLoggedIn'].drop_duplicates()
#     data_AgentLogedIn = pd.DataFrame()
#     data_AgentLogedIn['datetime'] = data_staff_name.groupby(data_staff_name['datetime']).sum().index
#     data_AgentLogedIn['Staff'] = data_staff_name.groupby(data_staff_name['datetime']).sum()['Value'].values
#     # data_staff.to_csv('./file/data_staff_timestamp.csv')
#     return data_AgentLogedIn
#
#
# def replace_outlier(df, col_name):
#     std = df[col_name].std()
#     mean = df[col_name].mean()
#     upper = std + mean
#     lower = mean - std
# #     df.loc[(df[col_name]<lower) |(df[col_name]>upper),[col_name]] = None
#     df[col_name][(df[col_name]<lower)| (df[col_name]>upper)]=None
#     return df
#
# if __name__ == '__main__':
#     # data = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\AgentGroupData202012_368.csv")[
#     #     ['QueueGroupId', 'Timestamp', 'Measure', 'Value', 'AgentGroupName']]
#     # data_AgentLogedIn = agentgroup_staff(data)
#     # print()
#     a = [1,2,10,100000,1,2,1]
#     b = [0,100, 3,4,2,4,1]
#     df = pd.DataFrame({'A':a, 'B':b})
#     replace_outlier(df, 'A')
#     print(df)
#     df['A'] = df['A'].interpolate()
#
# # # data = pd.read_csv('./file/12_368.csv')
# # # data_1 = pd.read_csv('./file/12_368_SL60.csv')
# # # data_n = pd.DataFrame()
# # # data_n['AverageHandelTime'] = data['AverageHandelTime']
# # # data_n['timestamp'] = data_1['timestamp']
# # # data = data_n.rename(columns={'timestamp':'datetime'})
# # # data['datetime'] = pd.to_datetime(data['datetime'].apply(lambda x: str(x)[0:17]+':00'))
# # # staff_data = pd.read_excel(r"E:\Data\qg_AHT.xlsx")
# # # staff_data = staff_data.rename(columns={'Cal_Date':'datetime'})
# # # staff_data['datetime'] = pd.to_datetime(staff_data['datetime'])
# # # TotalStaff = staff_data['queuegroup_aht'].values.repeat(30)
# # # every_minute_array = np.arange(0,30,1)
# # # every_minute_array = every_minute_array*60
# # # staff_data_1 = pd.DataFrame()
# # # for i in every_minute_array:
# # #     staff_data_1['datetime_'+str(i)+'s'] = staff_data['datetime'].apply(lambda dt:(dt + datetime.timedelta(0,int(i),0)))
# # # staff_data_1 = staff_data_1.stack()
# # # staff_data_new = pd.DataFrame()
# # # staff_data_new['datetime'] = staff_data_1.values
# # # staff_data_new['queuegroup_aht'] = TotalStaff/60
# # # data_new = pd.merge(staff_data_new, data)
# # # data_new.plot(x='datetime')
# # # # sns.distplot(data['ServiceLevel60'])
# # # # plt.show()
# #
# #
# # # data_merge
# # # staff_data = pd.read_excel(r"E:\Data\368Staff.xlsx")
# # # forecasting_data_0 = pd.read_excel('./file/predict_5min_1day_new.xlsx', sheet_name='predict_30min_1day_new_1')
# # forecasting_data_1 = pd.read_csv('./file/predict_30min_1day_new_CO_CA_234_01.csv')
# # # forecasting_data = pd.concat([forecasting_data_0,forecasting_data_1,forecasting_data_2,forecasting_data_3], ignore_index=True)
# # forecasting_data = forecasting_data_1
# # original_data = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueGroupData202102.csv")[['QueueGroupId','PstTimeStamp','CallOffered','CallAnswered']]
# #
# # s = '300'
# # forecasting_data = forecasting_data[forecasting_data['QueueGroupId']==386][['datetime','CallOffered_diff'+ s +'s_yhat',
# #                                                                             'CallOffered_diff'+ s +'s_yhat_upper',
# #                                                                             'CallOffered_diff'+ s +'s_yhat_lower']]
# #                                                                            # 'CallAnswered_diff' + s + 's_yhat',
# #                                                                            # 'CallAnswered_diff' + s + 's_yhat_upper',
# #                                                                            #'CallAnswered_diff' + s + 's_yhat_lower'
# # original_data = original_data[original_data['QueueGroupId']==386]
# # df_diff = _process_data(original_data, 5*60)
# #
# # staff_data = staff_data.rename(columns={'cal_date':'datetime'})
# # staff_data['datetime'] = pd.to_datetime(staff_data['datetime'])
# # TotalStaff = staff_data['totalStaff'].values.repeat(30)
# # # 60代表60秒
# # every_minute_array = np.arange(0,30,1)
# # every_minute_array = every_minute_array*60
# # staff_data_1 = pd.DataFrame()
# # for i in every_minute_array:
# #     staff_data_1['datetime_'+str(i)+'s'] = staff_data['datetime'].apply(lambda dt:(dt + datetime.timedelta(0,int(i),0)))
# # staff_data_1 = staff_data_1.stack()
# #
# # staff_data_new = pd.DataFrame()
# # staff_data_new['datetime'] = staff_data_1.values
# # staff_data_new['totalStaff'] = TotalStaff
# # forecasting_data['datetime'] = forecasting_data['datetime'].apply(lambda x: str(x)[0:17]+'00')
# # df_diff['datetime'] = df_diff['datetime'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")[0:17]+'00')
# # data_diff = df_diff[['datetime', 'CallOffered_diff'+s+'s']].fillna(0) #, 'CallAnswered_diff'+s+'s'
# #
# # forecasting_data['datetime'] = pd.to_datetime(forecasting_data['datetime'])
# # data_diff['datetime'] = pd.to_datetime(data_diff['datetime'])
# # forecasting_data['datetime'] = forecasting_data['datetime'].apply(lambda x: str(x)[0:17]+'00')
# # df_diff['datetime'] = df_diff['datetime'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")[0:17]+'00')
# # data_diff = df_diff[['datetime', 'CallOffered_diff'+s+'s']].fillna(0) #, 'CallAnswered_diff'+s+'s'
# #
# # # data_new = pd.merge(forecasting_data, data_diff)[['datetime',
# # #                                               'CallOffered_diff'+ s +'s',
# # #                                               'CallOffered_diff'+ s +'s_yhat',
# # #                                               'CallOffered_diff'+ s +'s_yhat_upper',
# # #                                               'CallOffered_diff'+ s +'s_yhat_lower']]
# # # data_new.to_csv('./file/forecast_386_02_1day.csv')
# #
# # data_new = pd.merge(forecasting_data, staff_data_new)[['datetime','totalStaff','CallOffered_diff'+ s +'s_yhat',
# #                                                        'CallOffered_diff'+ s +'s_yhat_upper',
# #                                                        'CallOffered_diff'+ s +'s_yhat_lower',
# #                                                        'CallAnswered_diff' + s + 's_yhat',
# #                                                        'CallAnswered_diff' + s + 's_yhat_upper',
# #                                                        'CallAnswered_diff' + s + 's_yhat_lower']]
# # data_new_new = pd.merge(data_new, data_diff)[['datetime','totalStaff',
# #                                               'CallOffered_diff'+ s +'s','CallAnswered_diff'+ s +'s',
# #                                               'CallOffered_diff'+ s +'s_yhat',
# #                                               'CallOffered_diff'+ s +'s_yhat_upper',
# #                                               'CallOffered_diff'+ s +'s_yhat_lower',
# #                                               'CallAnswered_diff' + s + 's_yhat',
# #                                               'CallAnswered_diff' + s + 's_yhat_upper',
# #                                               'CallAnswered_diff' + s + 's_yhat_lower']]
# #
# # data_new_new.to_csv('./file/forecast_staff_30min_1day_new_CO_CA.csv')
# #
# # # # rolling_corr
# # # def rolling_spearman(seqa, seqb, window):
# # #     stridea = seqa.strides[0]
# # #     ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
# # #     strideb = seqa.strides[0]
# # #     ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
# # #     ar = pd.DataFrame(ssa)
# # #     br = pd.DataFrame(ssb)
# # #     ar = ar.rank(1)
# # #     br = br.rank(1)
# # #     corrs = ar.corrwith(br, 1)
# # #     return pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)
# # # data_new = pd.read_csv('./file/forecast_staff_30min_1day_new.csv')
# # # df = data_new[['datetime','totalStaff','CallOffered_diff1800s_yhat','CallOffered_diff1800s_yhat_upper','CallOffered_diff1800s_yhat_lower','CallOffered_diff1800s']] #
# # #
# # # # Set window size to compute moving window synchrony.
# # # r_window_size = 12*60
# # # # Interpolate missing data.
# # # df_interpolated = df.interpolate()
# # # # Compute rolling window synchrony
# # # rolling_r = pd.DataFrame()
# # # rolling_r_s = pd.DataFrame()
# # # rolling_r['y'] = df_interpolated['totalStaff'].rolling(window=r_window_size, center=True).corr(df_interpolated['CallOffered_diff1800s'])
# # # rolling_r_s['y'] = rolling_spearman(df_interpolated.totalStaff.values, df_interpolated.CallOffered_diff1800s.values, window=r_window_size)
# # # rolling_r['datetime'] = pd.to_datetime(data_new['datetime'])
# # # rolling_r_s['datetime'] = pd.to_datetime(data_new['datetime'])
# # # df['datetime'] = pd.to_datetime(df['datetime'])
# # # # print(type(rolling_r['datetime'].iloc[0]))
# # # rolling_r.set_index('datetime')
# # # f,ax = plt.subplots(3,1,figsize=(14,6),sharex='all')
# # # # df.rolling(window=30,center=True).median().plot(ax=ax[0])
# # # df.plot(x='datetime',ax=ax[0]) #.scatter,y='CallOffered_diff1800s_yhat'
# # # ax[0].set(xlabel='Frame',ylabel='Smiling Evidence')
# # # rolling_r.plot.scatter(x='datetime',y='y',ax=ax[1])
# # # ax[1].set(xlabel='Frame',ylabel='Pearson r')
# # # rolling_r_s.plot.scatter(x='datetime',y='y',ax=ax[2])
# # # ax[2].set(xlabel='Frame',ylabel='Spearman r')
# # # plt.suptitle("Smiling data and rolling window correlation")
# # # plt.show()
# # # # print('pearson:\n', rolling_r.value_counts().sort_values(ascending=False))
# #
# # # # plot data_original
# # # data_or = pd.read_csv('./file/11_368.csv')[['QueueGroupId','PstTimeStamp','CallOffered','CallAnswered']]
# # # df = data_or[(data_or['QueueGroupId'] == 368)]
# # # df_diff = _process_data(df)
# # # df_diff['datetime'] = df_diff['datetime'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")[0:17]+'00')
# # # data_diff = df_diff[['datetime', 'CallOffered_diff1800s']].fillna(0)
# # # data_diff['datetime'] = pd.to_datetime(data_diff['datetime'])
# # # data_diff.plot(x='datetime')
# # # plt.show()
# #
# # # plot rate
# # def detectoutliers(list1, list):
# #     outlier_indices = []
# #     # iterate over features(columns)
# #     # 1st quartile (25%)
# #     Q1 = np.percentile(list, 1)
# #     # 3rd quartile (75%)
# #     Q3 = np.percentile(list, 95)
# #     # Interquartile range (IQR)
# #     IQR = Q3 - Q1
# #     # outlier step
# #     outlier_step = 1.5 * IQR
# #     # Determine a list of indices of outliers for feature col
# #     # outlier_list_col = list[(list < Q1 - outlier_step) | (list > Q3 + outlier_step)]
# #     outlier_list_col = list1[(list1 < Q1) | (list1 > Q3)]
# #     return outlier_list_col
# #
# # # data = pd.read_csv('./file/outlier_with_first_detect.csv')[['datetime','CallOffered_diff1800s','CallOffered_diff1800s_yhat','totalStaff']].iloc[0:4080]
# # data = pd.read_csv('./file/forecast_staff_30min_1day_new_CO_CA.csv') #.iloc[0:653]
# # s = '1800'
# # s1 = '30'
# # data_rate = data.rename(columns={'CallOffered_diff'+s+'s':'CO_'+s1+'min',
# #                                  'CallOffered_diff'+s+'s_yhat':'C0_'+s1+'min_hat',
# #                                  'CallOffered_diff'+s+'s_yhat_upper':'C0_'+s1+'min_hat_upper',
# #                                  'CallOffered_diff'+s+'s_yhat_lower':'C0_'+s1+'min_hat_lower',
# #                                  'CallAnswered_diff'+s+'s':'CA_'+s1+'min',
# #                                  'CallAnswered_diff'+s+'s_yhat':'CA_'+s1+'min_hat',
# #                                  'CallAnswered_diff'+s+'s_yhat_upper':'CA_'+s1+'min_hat_upper',
# #                                  'CallAnswered_diff'+s+'s_yhat_lower':'CA_'+s1+'min_hat_lower',
# #                                  'totalStaff':'Staff'})
# # data_rate['rate'] = data_rate['CA_'+s1+'min']/data_rate['Staff']
# # data_rate['rate_hat'] = data_rate['C0_'+s1+'min_hat']*0.8/data_rate['Staff']
# # data_rate['Outlier'] = detectoutliers(data_rate['rate_hat'], data_rate['rate'])
# # data_rate['datetime'] = pd.to_datetime(data_rate['datetime'])
# # data_rate_new = data_rate[data_rate['Outlier'].isna()]
# # Staff_forecast = pd.DataFrame()
# # Staff_forecast['date'] = data_rate_new['datetime']
# # Staff_forecast['data'] = data_rate_new['C0_'+s1+'min_hat']/data_rate_new['rate_hat']
# # Staff_forecast['date'] = pd.to_datetime(Staff_forecast['date'])
# # # outlier_rate = len(outlier['data'].values)/len(data_rate['data'].values)
# # sns.distplot(data_rate['data'])
# # ax = data_rate[['datetime','CO_'+s1+'min','C0_'+s1+'min_hat','Staff']].iloc[0:653].plot(x='datetime')
# # data_rate.iloc[0:653].plot(x='datetime',y='rate_hat',c='k',ax=ax,label='rate_hat')
# # data_rate.iloc[0:653].plot(x='datetime',y='Outlier',c='r',kind='scatter',ax=ax,label='Outlier')
# # ax.fill_between(data_rate["datetime"][0:653].values, data_rate['C0_'+s1+'min_hat_lower'][0:653], data_rate['C0_'+s1+'min_hat_upper'][0:653],color='b',alpha=.2)
# # plt.show()
# # # data_rate.plot(kind='box',ax=axes[2])
# # plt.figure()
# # Staff_forecast.plot(x='date')
# #
# # ax1 = data_rate[['datetime','CO_'+s1+'min']].iloc[0:653].plot(x='datetime')
# # plt.legend(loc='upper left')
# # ax2 = ax1.twinx()
# # data_rate.iloc[0:653].plot(x='datetime',y='Staff',c='k',ax=ax2,label='Staff')
# # plt.legend(loc='upper right')
# #
# # ax1 = data_rate[['datetime','CO_'+s1+'min','C0_'+s1+'min_hat']].iloc[0:653].plot(x='datetime')
# # ax1.fill_between(data_rate["datetime"][0:653].values, data_rate['C0_'+s1+'min_hat_lower'][0:653], data_rate['C0_'+s1+'min_hat_upper'][0:653],color='b',alpha=.2)
# # plt.legend(loc='upper left')
# # ax2 = ax1.twinx()
# # data_rate[['datetime','Staff']].iloc[0:653].plot(x='datetime',ax=ax2, c='k')
# # plt.legend(loc='upper right')
# # plt.show()
# #
# # data_rate['rate_CO'] = data_rate['CO_'+s1+'min']/data_rate['Staff']
# # data_rate['rate_CO_hat'] = data_rate['C0_'+s1+'min_hat']/data_rate['Staff']
# # data_rate['Outlier_CO'] = detectoutliers(data_rate['rate_CO_hat'], data_rate['rate_CO'])
# # ax = data_rate[['datetime','rate_CO_hat']].iloc[0:653].plot(x='datetime', c='k')
# # data_rate[['datetime','Outlier_CO']].iloc[0:653].plot(x='datetime',y='Outlier_CO',c='r',kind='scatter',ax=ax,label='Outlier')
# # Q1 = np.percentile(data_rate['rate_CO'], 1)
# # Q3 = np.percentile(data_rate['rate_CO'], 95)
# # plt.plot(data_rate['datetime'].iloc[0:653],np.repeat(Q1,653),'y', ls='--')
# # plt.plot(data_rate['datetime'].iloc[0:653],np.repeat(Q3,653),'y', ls='--')
# # # plt.legend(loc='upper right')
# # plt.show()
# #
# # # data_rate_new = data_rate.iloc[0:653][data_rate['Outlier_CO'].isna()]
# # Staff_forecast = pd.DataFrame()
# # Staff_forecast['date'] = data_rate.iloc[0:653]['datetime']
# # Staff_forecast['data'] = data_rate.iloc[0:653]['Staff']
# # Staff_forecast['data'][data_rate.iloc[0:653]['Outlier_CO'].notna()] = data_rate.iloc[0:653]['C0_'+s1+'min_hat']/Q3
# # Staff_forecast['date'] = pd.to_datetime(Staff_forecast['date'])
# # plt.figure()
# # plt.plot(Staff_forecast['date'],Staff_forecast['data'],label='Staff_forecast')
# # plt.plot(Staff_forecast['date'],data_rate.iloc[0:653]['Staff'],label='Staff')
# # plt.legend(loc='upper right')
# #
# # Q1 = np.percentile(data_rate['rate'], 0)
# # Q3 = np.percentile(data_rate['rate'], 98)
# # # data_rate_new = data_rate.iloc[0:653][data_rate['Outlier_CO'].isna()]
# # Staff_forecast = pd.DataFrame()
# # Staff_forecast['date'] = data_rate.iloc[0:653]['datetime']
# # Staff_forecast['data'] = data_rate.iloc[0:653]['Staff']
# # Staff_forecast['data'][data_rate.iloc[0:653]['Outlier_CO'].notna()] = (data_rate.iloc[0:653]['CA_'+s1+'min_hat']*0.8)/Q3
# # # Staff_forecast['data'] = (data_rate.iloc[0:653]['CA_'+s1+'min_hat']*0.8)/(Q3*data_rate.iloc[0:653]['Staff'])
# # Staff_forecast['date'] = pd.to_datetime(Staff_forecast['date'])
# # plt.figure()
# # plt.plot(Staff_forecast['date'],Staff_forecast['data'],label='forecast')
# # plt.plot(Staff_forecast['date'],data_rate.iloc[0:653]['Staff'],label='original')
# # # plt.plot(Staff_forecast['date'],(data_rate.iloc[0:653]['CA_'+s1+'min_hat']*0.8)/(data_rate.iloc[0:653]['rate']*data_rate.iloc[0:653]['Staff']),label='original')
# # plt.legend(loc='upper right')