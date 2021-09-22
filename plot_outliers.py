import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close()
# # rate
# data_rate = pd.read_csv('data_rate.csv')
# data_new_rate = data_rate[['timestamp','y_rate_or']][34960:36288].fillna(0).rename(columns={'timestamp':'timestamp', 'y_rate_or':'y'})
# # rate
# def outlier_detection(forecast):
#     index = np.where((forecast["y"] > 0.1)|(forecast["y"] < -0.1), True, False)
#     return index
# outlier_index = outlier_detection(data_new_rate)
# outlier_df = data_new_rate[outlier_index]
#
# data_new_rate['timestamp'] = pd.to_datetime(data_new_rate['timestamp'])
# data_new_rate.set_index('timestamp')
# outlier_df['timestamp'] = pd.to_datetime(outlier_df['timestamp'])
# outlier_df.set_index('timestamp')
# ## 可视化异常值的结果
# import matplotlib.dates as mdates
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 显示时间坐标的格式
# from matplotlib.dates import AutoDateLocator
# autodates = AutoDateLocator()                # 时间间隔自动选取
# plt.gca().xaxis.set_major_locator(autodates)
# ax = data_new_rate.plot.scatter(x="timestamp", y="y",color='k',figsize=(14,7),
#               label = "original_data")
# ## 可视化出置信区间
# ax.fill_between(data_new_rate["timestamp"].values, -0.1,
#                 0.1, color='b', alpha=.2)  # yhat_lower, yhat_upper
# ## 可视化出异常值的点
# outlier_df.plot.scatter(x="timestamp", y="y",color='r', label="outliers",ax=ax)
# plt.legend(loc=2)
# plt.grid()
# plt.title("Anomaly detection")
# plt.show()

# # forecast_orginal
# forecast_all = pd.read_excel('data_approach2_new_low_volume_10.xlsx', sheet_name='Sheet1')
# for_or = pd.read_csv('forecast_original_new_0.95_10_20210112.csv')
# forecast = forecast_all[['Date_Key','CallOffered']].rename(columns={'Date_Key':'ds', 'CallOffered':'y'})
# forecast['yhat'] = for_or['yhat_or'].values[1057:2305]
# forecast['y_upper_10'] = np.where(forecast_all['CallOffered_upper']!=0, forecast_all['CallOffered_upper'],0.5)
# forecast['y_lower_10'] = np.where(forecast_all['CallOffered_lower']!=0, forecast_all['CallOffered_lower'],forecast_all['CallOffered_lower'])
# outlier_df = forecast_all[['Date_Key','CallOffered_outliers']].rename(columns={'Date_Key':'ds', 'CallOffered_outliers':'y'})
# outlier_df = forecast[np.where((outlier_df['y']==1),True,False)]
# # forecast_all = pd.read_csv('forecast_original_new_0.95_10_20210112.csv')
# # forecast = forecast_all[['timestamp','y_1hr','yhat_1hr']][1057:2304].rename(columns={'timestamp':'ds', 'y_1hr':'y', 'yhat_1hr':'yhat'}) #[2006:3372]_10min  1030:[1057:3492]
# # forecast['y_upper_10'] = forecast['yhat']+0.1*abs(forecast['yhat'])
# # forecast['y_lower_10'] = forecast['yhat']-0.1*abs(forecast['yhat'])
# # def outlier_detection(forecast):
# #     index = np.where(((forecast["y"] < forecast["y_lower_10"])|
# #                      (forecast["y"] > forecast["y_upper_10"])),True,False)
# #     return index
# # outlier_index = outlier_detection(forecast)
# # outlier_df = forecast[outlier_index]
# # # print("The number of outliers:",np.sum(outlier_index))
# # # print("The rate of outliers:",np.sum(outlier_index)/num_data)
#
# forecast['ds'] = pd.to_datetime(forecast['ds'])
# forecast.set_index('ds')
# outlier_df['ds'] = pd.to_datetime(outlier_df['ds'])
# outlier_df.set_index('ds')
# ## 可视化异常值的结果
# import matplotlib.dates as mdates
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 显示时间坐标的格式
# from matplotlib.dates import AutoDateLocator
# autodates = AutoDateLocator()                # 时间间隔自动选取
# plt.gca().xaxis.set_major_locator(autodates)
# ## 可视化异常值的结果
# ## 可视化预测值
# ax = forecast.plot.line(x = "ds",y = "yhat",c = "b",figsize=(14,7), label = "forecast")
# ## 可视化出置信区间
# ax.fill_between(forecast["ds"].values, forecast["y_lower_10"], forecast["y_upper_10"],color='b',alpha=.2)  # yhat_lower,yhat_upper
# forecast.plot.scatter(x = "ds",y = "y",c = "k", s = 20,label = "original_data",ax=ax)
# ## 可视化出异常值的点forecast['ds']
# outlier_df.plot.scatter(x = 'ds',y = "y",c = "r", label = "outliers",ax=ax)
# plt.legend(loc = 2)
# plt.grid()
# plt.title("Anomaly detection")
# # plt.savefig('forecast_original_1hr')
# plt.show()

# forecast
# forecast_all = pd.read_csv('./file/forecast_original_new_0.95_10_20210112.csv')
forecast_all = pd.read_csv('./file/forecast_386_02_1day.csv')
# # 1128:[678:2006]  1029:[865:2169]
# forecast = forecast_all[['timestamp','y_or','yhat_or','yhat_upper_or','yhat_lower_or']][1057:2305]\
#     .rename(columns={'timestamp':'ds', 'y_or':'y', 'yhat_or':'yhat', 'yhat_upper_or':'yhat_upper', 'yhat_lower_or':'yhat_lower'})

forecast = forecast_all[['datetime','CallOffered_diff300s','CallOffered_diff300s_yhat','CallOffered_diff300s_yhat_upper','CallOffered_diff300s_yhat_lower']]\
    .rename(columns={'datetime':'ds','CallOffered_diff300s':'y', 'CallOffered_diff300s_yhat':'yhat', 'CallOffered_diff300s_yhat_upper':'yhat_upper', 'CallOffered_diff300s_yhat_lower':'yhat_lower'})

# forecast["yhat_upper"] = forecast["yhat_upper"]+0.1*abs(forecast["yhat_upper"])
# forecast["yhat_lower"] = forecast["yhat_lower"]-0.1*abs(forecast["yhat_lower"])
def outlier_detection(forecast):
    index = np.where(((forecast["y"] < forecast["yhat_lower"])|
                     (forecast["y"] > forecast["yhat_upper"])),True,False)
    return index
outlier_index = outlier_detection(forecast)
outlier_df = forecast[outlier_index]
# print("The number of outliers:",np.sum(outlier_index))
# print("The rate of outliers:",np.sum(outlier_index)/num_data)

forecast['ds'] = pd.to_datetime(forecast['ds'])
forecast.set_index('ds')
# outlier_df['ds'] = pd.to_datetime(outlier_df['ds'])
# outlier_df.set_index('ds')
## 可视化异常值的结果
# import matplotlib.dates as mdates
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 显示时间坐标的格式
# from matplotlib.dates import AutoDateLocator
# autodates = AutoDateLocator()                # 时间间隔自动选取
# plt.gca().xaxis.set_major_locator(autodates)
## 可视化异常值的结果
## 可视化预测值
ax = forecast.plot.line(x = "ds",y = "yhat",c = "b",figsize=(14,7), label = "forecast")
## 可视化出置信区间
ax.fill_between(forecast["ds"].values, forecast["yhat_lower"], forecast["yhat_upper"],color='b',alpha=.2)
forecast.plot(x = "ds",y = "y",c = "k", s = 20,kind='scatter',label = "original_data",ax=ax)
## 可视化出异常值的点
outlier_df.plot.scatter(x = "ds",y = "y",c = "r", label = "outliers",ax=ax)
plt.legend(loc = 2)
plt.grid()
plt.title("Anomaly detection")
plt.show()


