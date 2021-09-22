import pandas as pd
import numpy as np
import datetime
from fbprophet import Prophet


# time  Timestamp-->PstTimeStamp
def data_time_process(df, datetime_name='Timestamp'):
    """
        Time format conversion
    :param df: DataFrame: ['Timestamp',...]
    :return: data: DataFrame: ['datetime']
    """
    data = pd.DataFrame()
    data['datetime'] = pd.to_datetime(df[datetime_name], utc=True)  # , infer_datetime_format=True'Timestamp'
    data['datetime'] = data['datetime'].dt.tz_convert('America/Los_Angeles')
    data['datetime'] = data['datetime'].apply(
        lambda x: datetime.datetime.strptime(x.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"))
    return data


def time_approximation(Timestamp):
    if int(pd.to_datetime(Timestamp).second)>30:
        datetime_1 = pd.to_datetime(Timestamp)+datetime.timedelta(0,60-pd.to_datetime(Timestamp).second,0)
    else:
        datetime_1 = pd.to_datetime(Timestamp)-datetime.timedelta(0,pd.to_datetime(Timestamp).second,0)
    return datetime_1


def replace_outlier(df, col_name, multiple):
    std = df[col_name].std()
    mean = df[col_name].mean()
    upper = multiple*std + mean
    lower = mean - multiple*std
#     df.loc[(df[col_name]<lower) |(df[col_name]>upper),[col_name]] = None
    df[col_name][(df[col_name]<lower)| (df[col_name]>upper)]=None
    df[col_name] = df[col_name].fillna(df[col_name].mean())
    return df
# if __name__ == '__main__':
#     a = [1, 2, 10, 100000, 1, 2, 1]
#     b = [0, 100, 3, 4, 2, 4, 1]
#     df = pd.DataFrame({'A': a, 'B': b})
#     df = replace_outlier(df,'A')


def get_diff_data(df, QueueGroupId_name, datetime_name, measure_list):
    def my_date_parse(dt):
        return datetime.datetime(int(dt[0:4]), int(dt[5:7]), int(dt[8:10]))

    def my_datetime_parse(dt):
        dt_datetime = datetime.datetime(int(dt[0:4]), int(dt[5:7]), int(dt[8:10]), int(dt[11:13]), int(dt[14:16]),
                               int(dt[17:19]))
        return dt_datetime
    df['date'] = df[datetime_name].apply(lambda x:my_date_parse(str(x)))
    df['datetime'] = df[datetime_name].apply(lambda x:my_datetime_parse(str(x)))
    for m in measure_list:
        df[m+'_diff'] = df.groupby([QueueGroupId_name, 'date'])[m].diff()
    drop_col = measure_list+['date',]
    df.drop(drop_col, axis=1, inplace=True)
    return df


def back_interval_sum(QueueGroup_df, interval_seconds, dt_datetime, dt_datetime_name, measure_name_list):
    zero_time = pd.to_datetime(datetime.datetime.strptime((str(dt_datetime)[0:10]+' 00:00:00'),"%Y-%m-%d %H:%M:%S"))
    back_time = (dt_datetime-datetime.timedelta(0, interval_seconds, 0) if dt_datetime-datetime.timedelta(0, interval_seconds, 0)>zero_time else zero_time)
    QueueGroup_df_interval = QueueGroup_df[(QueueGroup_df[dt_datetime_name]>back_time) & (QueueGroup_df[dt_datetime_name]<=dt_datetime)]
    measure_sum_list = []
    for m in measure_name_list:
        measure_sum_list.append(QueueGroup_df_interval[m].sum())
    return measure_sum_list


# get diff data aggregated with specified time interval
def diff_interval_sum(df, QueueGroupId_name, datetime_name, measure_diff_list, interval_seconds=60*5):
    '''
    Get diff data aggregated with specified time interval
    '''
    QueueGroupId_array = df[QueueGroupId_name].unique()
    data = pd.DataFrame()
    for i in QueueGroupId_array:
        QueueGroup_df = df[df[QueueGroupId_name]==i]
        datetime_series = QueueGroup_df[datetime_name]
        measure_diff_interval_list = [i + str(interval_seconds) + 's' for i in measure_diff_list]
        measure_val_list = []
        for j,v in datetime_series.items():
            measure_val = back_interval_sum(QueueGroup_df, interval_seconds, v, datetime_name, measure_diff_list)
            measure_val_list.append(measure_val)
            measure_val_list_df = pd.DataFrame(measure_val_list, columns=measure_diff_interval_list)
        df1 = pd.concat(
            [pd.concat([df[QueueGroupId_name][df[QueueGroupId_name] == i], datetime_series], axis=1).reset_index(),
             measure_val_list_df], axis=1)
        data = pd.concat([data,df1])
    return data

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


def train_forecast(df_diff, QueueGroupId_name, datetime_name, measure_diff_list, num_data=7*24*60, interval_width=0.95):
    """
    :param df_diff: DataFrame: data diff value
    :param QueueGroupId_name: queue group id column name
    :param datetime_name: datetime column name
    :measure_diff_list: name list of measures aggregated with specified inteverl
    :param num_data: int: the length of data to forecasting, default=7*24*60(7 days)
    :param interval_width: float: interval width, default=0.95
    :return: df: DataFrame:
    """
    forecast_val = pd.DataFrame()
    queueGroupIds = df_diff[QueueGroupId_name].drop_duplicates()
    df_diff = df_diff.rename(columns={datetime_name:'ds'})
    for qid in queueGroupIds:
        temp_forecast_val = pd.DataFrame()
        qg_df_diff = df_diff[df_diff[QueueGroupId_name]==qid]
        for i in measure_diff_list:
            train_data = pd.concat([qg_df_diff['ds'], qg_df_diff[i]], axis=1).rename(columns={i:'y'})
            forecast_ds = forecasting(train_data, num_data, interval_width)
            temp_forecast_val[QueueGroupId_name] = np.repeat(qid, num_data)
            temp_forecast_val[datetime_name] = forecast_ds['ds'].values
            temp_forecast_val[i + '_yhat'] = forecast_ds['yhat'].values
            temp_forecast_val[i + '_yhat_upper'] = forecast_ds['yhat_upper'].values # np.around(, decimals=3)
            temp_forecast_val[i + '_yhat_lower'] = forecast_ds['yhat_lower'].values
        forecast_val = pd.concat([forecast_val, temp_forecast_val], axis=0)
    return forecast_val

