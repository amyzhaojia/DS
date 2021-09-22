import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from pytz import timezone, utc
pst_tz = timezone('America/Los_Angeles')

# df = pd.read_excel(r"E:\Data\QueueData_232_234_368_386_08\data_386_new.xlsx", sheet_name='Sheet1')
df = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueGroupData202012.csv")
q = '368'
num_data = 24*60*4
df[df['QueueGroupId'] == int(q)].to_csv('12_368.csv')
df = pd.read_csv("12_368.csv")
# df = df[df['QueueGroupId'] == int(q)]
Time = df['Timestamp'].values
time_232 = [datetime.datetime.strptime(utc.localize(datetime.datetime.strptime(Time[i].replace("T", " ")[:Time[1].replace("T", " ").rfind(".")],
            "%Y-%m-%d %H:%M:%S")).astimezone(tz=pst_tz).strftime("%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S")
            for i in range(len(Time))]   #
# df['pst_timestamp'] = pd.to_datetime(df['pst_timestamp'])
# time_232 = df['pst_timestamp']
df['pst_timestamp'] = time_232
data_new = pd.DataFrame()
data_group = df[['CallOffered', 'CallAnswered', 'TotalTimetoAnswer']].groupby(df['pst_timestamp'].apply(lambda x: x.day), as_index=False).diff()
data_group['CallOffered'][np.where(data_group['CallOffered']<0)[0]] = 0
data_new['y'] = data_group['CallOffered'].values
data_new['ds'] = time_232

#去异常值
# data_new['y'][data_new['y']>40] = None
# data_new['y'][data_new['y']<0] = None

# data_new['y'][(data_new['ds'] > '2020-09-07') & (data_new['ds'] < '2020-09-15')] = None
# data_new['y'] = np.log(df['CallOffered'].values)
df_or = df[['CallOffered','pst_timestamp']]
CO_5min = pd.DataFrame()
for i in time_232:
    j = i + datetime.timedelta(minutes=-5)
    num = np.where((data_new['ds'] > j) & (data_new['ds'] <= i))
    CO_5min = CO_5min.append(data_group.iloc[num[0]].sum(), ignore_index=True)
CO_5min['ds'] = time_232
data_new_5min = CO_5min
data_new_5min['y'] = CO_5min['CallOffered'].values
CO_10min = pd.DataFrame()
for i in time_232:
    j = i + datetime.timedelta(minutes=-10)
    num = np.where((data_new['ds'] > j) & (data_new['ds'] <= i))
    CO_10min = CO_10min.append(data_group.iloc[num[0]].sum(), ignore_index=True)
CO_10min['ds'] = time_232
data_new_10min = CO_10min
data_new_10min['y'] = CO_10min['CallOffered'].values
CO_232_30min = pd.DataFrame()
for i in time_232:
    j = i + datetime.timedelta(minutes=-30)
    num = np.where((df['pst_timestamp'] > j) & (df['pst_timestamp'] <= i))
    CO_232_30min = CO_232_30min.append(data_group.iloc[num[0]].sum(), ignore_index=True)
CO_232_30min['ds'] = time_232
data_new_30min = CO_232_30min
data_new_30min['y'] = CO_232_30min['CallOffered'].values
CO_232_1hr = pd.DataFrame()
for i in time_232:
    j = i + datetime.timedelta(hours=-1)
    num = np.where((df['pst_timestamp'] > j) & (df['pst_timestamp'] <= i))
    CO_232_1hr = CO_232_1hr.append(data_group.iloc[num[0]].sum(), ignore_index=True)
CO_232_1hr['ds'] = time_232
data_new_1hr = CO_232_1hr
data_new_1hr['y'] = CO_232_1hr['CallOffered'].values
# def ASA_a(CA, TTA):
#     ASA = 0 if CA==0 else TTA/(CA*60)
#     # ASA_alert = 0 if ASA<=2 else 1
#     return ASA
# data_group['ASA'] = data_group.apply(lambda x: ASA_a(x.CallAnswered,x.TotalTimetoAnswer), axis=1)
# def SL_a(CO, CA):
#     SL = 100 if CO==0 else CA*100/CO
#     # SL_alert = 1 if SL<80 else 0
#     return SL
# data_group['SL'] = data_group.apply(lambda x: SL_a(x.CallOffered,x.CallAnswered), axis=1)
# # data_new['ASA_a'] = data_group['ASA_a'].values
# # data_new['SL_a'] = data_group['SL_a'].values
# df['SL_60_1HR'] = data_group['SL'].values
# df['ASA_1HR'] = data_group['ASA'].values
# df.to_csv('1128_368_new.csv')

# plt.plot(data_new['ASA_a'].values)

# Prophet
holidays = pd.DataFrame({
  'holiday': 'Thanksgiving',
  'ds': pd.to_datetime(['2020-11-26', '2020-11-27'])})

interval_width = 0.95

from fbprophet import Prophet
m = Prophet(interval_width=interval_width , holidays=holidays) #
# forecasting
# or
data_or = df_or.rename(columns={'pst_timestamp':'ds', 'CallOffered':'y'})
m.fit(data_or[['ds','y']][:-num_data])
# future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
future = data_or[['ds']][-num_data:]
forecast_or = m.predict(future)
forecast_or['y'] = data_or['y'][-num_data:].values
# forecast_or.to_csv('forecast_or_new_'+ str(interval_width)+'.csv')
del m

# diff
m = Prophet(interval_width=interval_width , holidays=holidays) #
m.fit(data_new[['ds','y']][:-num_data])
# future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
future = data_or[['ds']][-num_data:]
forecast = m.predict(future)
forecast['y'] = data_new['y'][-num_data:].values
# forecast.to_csv('forecast_new_'+ str(interval_width)+'.csv')
del m

# m.plot(forecast)
m = Prophet(interval_width=interval_width , holidays=holidays)
m.fit(data_new_5min[['ds','y']][:-num_data])
# future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
future = data_or[['ds']][-num_data:]
forecast_5min = m.predict(future)
forecast_5min['y'] = data_new_5min['y'][-num_data:].values
# forecast_5min.to_csv('forecast_5min_new_'+ str(interval_width)+'.csv')
del m

# m.plot(forecast)
m = Prophet(interval_width=interval_width , holidays=holidays)
m.fit(data_new_10min[['ds','y']][:-num_data])
# future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
future = data_or[['ds']][-num_data:]
forecast_10min = m.predict(future)
forecast_10min['y'] = data_new_10min['y'][-num_data:].values
# forecast_30min.to_csv('forecast_1hr_new_'+ str(interval_width)+'.csv')
del m

# m.plot(forecast)
m = Prophet(interval_width=interval_width , holidays=holidays)
m.fit(data_new_30min[['ds','y']][:-num_data])
# future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
future = data_or[['ds']][-num_data:]
forecast_30min = m.predict(future)
forecast_30min['y'] = data_new_30min['y'][-num_data:].values
# forecast_30min.to_csv('forecast_1hr_new_'+ str(interval_width)+'.csv')
del m

# m.plot(forecast)
m = Prophet(interval_width=interval_width , holidays=holidays)
m.fit(data_new_1hr[['ds','y']][:-num_data])
# future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
future = data_or[['ds']][-num_data:]
forecast_1hr = m.predict(future)
forecast_1hr['y'] = data_new_1hr['y'][-num_data:].values
# forecast_1hr.to_csv('forecast_1hr_new_'+ str(interval_width)+'.csv')
del m

forecast_original = pd.DataFrame()
forecast_original['timestamp'] = forecast_or['ds']
forecast_original[['y_or','yhat_or','yhat_upper_or','yhat_lower_or']] = forecast_or[['y','yhat','yhat_upper','yhat_lower']]
forecast_original[['y_1min','yhat_1min','yhat_upper_1min','yhat_lower_1min']] = forecast[['y','yhat','yhat_upper','yhat_lower']]
forecast_original[['y_5min','yhat_5min','yhat_upper_5min','yhat_lower_5min']] = forecast_5min[['y','yhat','yhat_upper','yhat_lower']]
forecast_original[['y_10min','yhat_10min','yhat_upper_10min','yhat_lower_10min']] = forecast_10min[['y','yhat','yhat_upper','yhat_lower']]
forecast_original[['y_30min','yhat_30min','yhat_upper_30min','yhat_lower_30min']] = forecast_30min[['y','yhat','yhat_upper','yhat_lower']]
forecast_original[['y_1hr','yhat_1hr','yhat_upper_1hr','yhat_lower_1hr']] = forecast_1hr[['y','yhat','yhat_upper','yhat_lower']]
forecast_original.to_csv('forecast_original_new_'+ str(interval_width)+'_10_20210112.csv')


# # orginal
# m.fit(data_new[['ds','y']])
# forecast = m.predict(data_new)
# m.plot(forecast)
# data_test = data_new.reset_index()
# data_test['Outlier'] = (data_test['y'] < forecast['yhat_lower']) | (forecast['yhat_upper'] < data_test['y'])

# data_test_or['Outlier'] = (data_test_or['y'] < forecast_or['yhat']*0.9) | (forecast_or['yhat']*1.1 < data_test_or['y'])
# data_test_or['Outlier_n'] = [1 if data_test_or['Outlier'].values[i] == True else 0 for i in
#                          range(len(data_test_or['Outlier'].values))]
# data_test_or['yhat'] = forecast_or['yhat']
# data_test_or_1hr['Outlier'] = (data_test_or_1hr['y'] < forecast_or_1hr['yhat']*0.9) | (forecast_or_1hr['yhat']*1.1 < data_test_or_1hr['y'])
# data_test_or_1hr['Outlier_n'] = [1 if data_test_or_1hr['Outlier'].values[i] == True else 0 for i in
#                          range(len(data_test_or_1hr['Outlier'].values))]
# data_test_or_1hr['yhat'] = forecast_or_1hr['yhat']
# data_test['Outlier'] = (data_test['y'] < forecast['yhat']*0.9) | (forecast['yhat']*1.1 < data_test['y'])
# data_test['Outlier_n'] = [1 if data_test['Outlier'].values[i] == True else 0 for i in
#                          range(len(data_test['Outlier'].values))]
# data_test['yhat'] = forecast['yhat']
# data_test_1hr['Outlier'] = (data_test_1hr['y'] < forecast_1hr['yhat']*0.9) | (forecast_1hr['yhat']*1.1 < data_test_1hr['y'])
# data_test_1hr['Outlier_n'] = [1 if data_test_1hr['Outlier'].values[i] == True else 0 for i in
#                          range(len(data_test_1hr['Outlier'].values))]
# data_test_1hr['yhat'] = forecast_1hr['yhat']
#
# data_test_or.to_csv('or_for_or_1min.csv')
# data_test_or_1hr.to_csv('or_for_or_1hr.csv')
# data_test.to_csv('or_for_diff_1min.csv')
# data_test_1hr.to_csv('or_for_diff_1hr.csv')

from data_processing import venn_figure
# # venn
# # ASA-CO
# num_data_venn = [list(np.where(data_new['ASA_a'] == 1)[0]), list(np.where(data_new['Outlier_n'] == 1)[0])]
# label_venn = ('ASA_a', 'Outlier_n')
# g = venn_figure(num_data_venn, label_venn)
# g.get_label_by_id('10').set_fontsize(15)
# g.get_label_by_id('01').set_fontsize(15)
# g.get_label_by_id('110').set_fontsize(15)
# # plt.show()
# plt.savefig('../Data_analysis_figures/ASA_CO_1hr' + q + '_venn.jpg')
# plt.cla()
# # SL-CO
# num_data_venn = [list(np.where(data_test['SL_a'] == 1)[0]), list(np.where(data_test['Outlier_n'] == 1)[0])]
# label_venn = ('SL_a', 'Outlier_n')
# g = venn_figure(num_data_venn, label_venn)
# g.get_label_by_id('10').set_fontsize(15)
# g.get_label_by_id('01').set_fontsize(15)
# g.get_label_by_id('110').set_fontsize(15)
# plt.savefig('../Data_analysis_figures/SL_CO_1hr' + q+ '_venn.jpg')
# plt.cla()

# def outlier_detection(forecast):
#     index = np.where(((forecast["y"] > forecast["yhat_lower"])&
#                      (forecast["y"] < forecast["yhat_upper"]))|(forecast["y"] == 0),False,True)
#     return index

# forecast['y'] = data_test['y']
# forecast['y_upper_10'] = forecast['yhat'] * 1.1
# forecast['y_lower_10'] = forecast['yhat'] * 0.9
# def outlier_detection(forecast):
#     index = np.where(((forecast["y"] > forecast["y_lower_10"])&
#                      (forecast["y"] < forecast["y_upper_10"]))|(forecast["y"] == 0),False,True)
#     return index
# outlier_index = outlier_detection(forecast)
# outlier_df = forecast[outlier_index]
# print("The number of outliers:",np.sum(outlier_index))
# print("The rate of outliers:",np.sum(outlier_index)/num_data)
# ## 可视化异常值的结果
# fig, ax = plt.subplots()
# ## 可视化预测值
# forecast.plot(x = "ds",y = "yhat",style = "b-",figsize=(14,7),
#               label = "forcast",ax=ax)
# ## 可视化出置信区间
# ax.fill_between(forecast["ds"].values, forecast["y_lower_10"],
#                 forecast["y_upper_10"],color='b',alpha=.2)          # yhat_lower, yhat_upper
# forecast.plot(kind = "scatter",x = "ds",y = "y",c = "k",
#               s = 20,label = "original_data",ax = ax)
# ## 可视化出异常值的点
# outlier_df.plot(x = "ds",y = "y",style = "rs",ax = ax,
#                 label = "outliers")
# plt.legend(loc = 2)
# plt.grid()
# plt.title("Anomaly detection")
# plt.show()

# from fbprophet.diagnostics import cross_validation
# df_cv = cross_validation(m, initial='15 days', period='1 days', horizon='2 days')
# from fbprophet.diagnostics import performance_metrics
# df_p = performance_metrics(df_cv)
# print(df_p)

# twitter_AD
# from pyculiarity import detect_ts
# # twitter_example_data = pd.read_csv('raw_data.csv',
# #                                     usecols=['timestamp', 'count'])
# twitter_example_data = data_new.rename(columns={'ds':'timestamp','y': 'count'})
# results = detect_ts(twitter_example_data, max_anoms=0.05, alpha=0.001, direction='both', only_last=None)
# # make a nice plot
# f, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(twitter_example_data['timestamp'], twitter_example_data['value'], 'b')
# ax[0].plot(results['anoms'].index, results['anoms']['anoms'], 'ro')
# ax[0].set_title('Detected Anomalies')
# ax[1].set_xlabel('Time Stamp')
# ax[0].set_ylabel('Count')
# ax[1].plot(results['anoms'].index, results['anoms']['anoms'], 'b')
# ax[1].set_ylabel('Anomaly Magnitude')
# plt.show()