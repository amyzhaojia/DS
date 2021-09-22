import numpy as np
import pandas as pd
import datetime
# from pytz import timezone, utc
# pst_tz = timezone('America/Los_Angeles')
import matplotlib.pyplot as plt

# time
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

# diff/diff_30min/diff_1hr/diff_24hr
def data_diff_process(data_new, interval_seconds, datetime_name='ds'):
    """
        calculate the diff of data
    :param data_new: DataFrame [datetime_name, y] ...[时间,值]
    :param diff_period_hours: the lasting hours of diff data.(time goes back use negative values)
    :param diff_period_minutes: the lasting minutes of diff data.(time goes back use negative values)
    :return:data_diff: DataFrame [datetime_name, y] ...[时间,值]
    """

    data_group = data_new.groupby(data_new[datetime_name].apply(lambda x: x.day), as_index=False).diff()
    if interval_seconds == 60:
        data_diff = data_group
    else:
        data_diff = pd.DataFrame()
        for i in data_new[datetime_name]:
            j = i - datetime.timedelta(0,interval_seconds,0)
            num = np.where((data_new[datetime_name] > j) & (data_new[datetime_name] <= i))
            data_diff = data_diff.append(data_group.iloc[num[0]].sum(), ignore_index=True)
    data_diff[datetime_name] = data_new[datetime_name]
    return data_diff

if __name__ == '__main__':
    df = pd.read_csv('./file/12_368 - 副本.csv')
    data_time = data_time_process(df)
    data_new = pd.DataFrame()
    data_new['ds'] = data_time['datetime'].values
    data_new['y'] = df['CallOffered'].values
    data_diff = data_diff_process(data_new, interval_seconds=60)

    plt.plot(data_diff['y'])
    plt.show()


