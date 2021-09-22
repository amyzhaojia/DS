import numpy as np
import pandas as pd
from fbprophet import Prophet
from pre_processing import data_time_process, data_diff_process
from low_volumn import low_volumn
import matplotlib.pyplot as plt

def forecasting(df_diff, num_data=7*24*60, interval_width=0.95):
    """
    ML forecast
    :param df_diff: DataFrame: ['ds','y']
    :param num_data: the length of forecasting data (minute)
    :param interval_width: confidence interval
    :return: forecast_out: DataFrame: ['ds', 'yhat', 'yhat_upper', 'yhat_lower']
    """
    holidays = pd.DataFrame({
        'holiday': 'Thanksgiving',
        'ds': pd.to_datetime(['2020-11-26', '2020-11-27'])})
    m = Prophet(interval_width=interval_width, holidays=holidays)
    m.fit(df_diff)
    future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
    forecast = m.predict(future)
    forecast_out = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
    return forecast_out

def forecast_orginal(df_diff, num_data=7*24*60, interval_width=0.95, low_volume=True, interval_seconds=0):
    """
    forcast + 10% threshold
    :param df_diff: DataFrame: ['ds','y']
    :param num_data: the length of forecasting data (minute)
    :param interval_width: confidence interval
    :param low_volume: whether use low volume threshold
    :param diff_period_hours: the lasting hours of diff data.(time goes back use negative values)
    :param diff_period_minutes:the lasting minutes of diff data.(time goes back use negative values)
    :return:forecast_original_out: DataFrame: ['ds', 'yhat', 'y_upper_10', 'y_lower_10']
    """
    holidays = pd.DataFrame({
        'holiday': 'Thanksgiving',
        'ds': pd.to_datetime(['2020-11-26', '2020-11-27'])})
    m = Prophet(interval_width=interval_width, holidays=holidays)
    m.fit(df_diff)
    future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
    forecast = m.predict(future)
    forecast['yhat_upper'] = forecast['yhat'] + 0.1 * abs(forecast['yhat'])
    forecast['yhat_lower'] = forecast['yhat'] - 0.1 * abs(forecast['yhat'])
    forecast_original_out = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
    if low_volume == True:
        forecast_original_out = low_volumn(forecast_original_out, interval_seconds)
    return forecast_original_out

def deviation(df_or, diff_period_seconds=0):
    """
    Percentage growth rate
    :param df_or:  DataFrame ['ds', 'y'] ...[时间,值]
    :param diff_period_hours: or--None, diff_1min--diff_period_minutes=-1...
    :return: data_rate: DataFrame ['ds', 'y', 'y_diff', 'y_rate']
    """
    # 增长率百分比
    data_rate = pd.DataFrame()
    data_rate['ds'] = df_or['ds']
    data_rate['y'] = df_or['y']
    data_rate['y_diff'] = data_diff_process(df_or, diff_period_seconds)['y']

    # diff_rate
    if diff_period_seconds==None:
        data_new_rate = pd.DataFrame()
        data_new_rate['y2'] = data_rate['y']
        data_new_rate['y2'][1:] = data_rate['y'][:-1]
        data_new_rate['y2'][0] = 0
        data_rate['y_rate'] = np.where(data_new_rate['y2'] == 0, data_rate['y_diff'],
                                       data_rate['y_diff'] / data_new_rate['y2'])
    else:
        data_new_rate = pd.DataFrame()
        data_new_rate['y2'] = data_rate['y'].diff()
        data_new_rate['y2'][1:] = data_rate['y'][:-1]
        data_new_rate['y2'][0] = 0
        data_rate['y_rate'] = np.where(data_new_rate['y2'] == 0, data_rate['y_diff'].diff(),
                                       data_rate['y_diff'].diff() / data_new_rate['y2'])
    data_rate.fillna(0)
    return data_rate


if __name__ == '__main__':
    # forecasting
    q = '368'
    # num_data = 24 * 60 * 1
    df = pd.read_csv("./file/11_368.csv")[['QueueGroupId','Timestamp','CallOffered']]
    df = df[df['QueueGroupId'] == int(q)][0:6000]
    df_time = data_time_process(df)
    df_or = pd.DataFrame()
    df_or['ds'] = df_time['datetime'].values
    df_or['y'] = df['CallOffered'].values
    df_diff = data_diff_process(df_or, interval_seconds=60)

    # forecast_095_1000 = forecasting(df_diff[0:1000], 1000, 0.95)
    forecast_090 = forecasting(df_diff[1000:5000], 1000, 0.9)
    forecast_090_5000 = forecasting(df_diff[0:5000], 1000, 0.9)
    # forecast_098 = forecasting(df_diff[0:1000], 1000, 0.98)

    ## 可视化异常值的结果
    fig, ax = plt.subplots()
    ## 可视化预测值
    forecast_090.plot(x = "ds",y = "yhat",style = "b-",figsize=(14,7),
                label = "forcast_090_4000",ax=ax)
    ## 可视化出置信区间
    ax.fill_between(forecast_090["ds"].values, forecast_090["yhat_lower"],
                    forecast_090["yhat_upper"],color='b',alpha=.2)          # yhat_lower, yhat_upper
    df_diff[5000:6000].plot(kind = "scatter",x = "ds",y = "y",c = "k",
                s = 20,label = "original_data",ax = ax)
    # ## 可视化出异常值的点
    # outlier_df.plot(x = "ds",y = "y",style = "rs",ax = ax,
    #                 label = "outliers")

    forecast_090_5000.plot(x = "ds",y = "yhat",style = "g-",figsize=(14,7),
                label = "forecast_090_5000",ax=ax)
    ax.fill_between(forecast_090_5000["ds"].values, forecast_090_5000["yhat_lower"],
                    forecast_090_5000["yhat_upper"],color='b',alpha=.2)          # yhat_lower, yhat_upper

    plt.legend(loc = 2)
    plt.grid()
    plt.title("Anomaly detection")
    plt.show()


    # import seaborn as sns
    # sns.distplot(df_diff['y'][1000:5000],label='train_2_5')
    # sns.distplot(df_diff['y'][0:5000],label='train_0_5')
    # plt.legend()
    # plt.show()

    # forecast.to_csv('forecast_new_' + '20210113.csv')
    # forecast_original = forecast_orginal(df_diff)
    # forecast_original.to_csv('forecast_or_new_' + '20210113.csv')
    # devation = deviation(df_or, diff_period_seconds=1)
    # devation.to_csv('devation_new_' + '20210113.csv')



