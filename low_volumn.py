import numpy as np
import pandas as pd
from pre_processing import data_time_process, data_diff_process

def low_volumn(df, interval_seconds):
    """
    low volumn threshold
    :param df: DataFrame ['ds', 'yhat', 'y_upper_10', 'y_lower_10']
    :return:data_new: DataFrame ['ds', 'yhat', 'y_upper_10', 'y_lower_10']
    """
    if interval_seconds == 0:
        threshold = 27
    else:
        threshold = interval_seconds/60*0.5
        df['yhat_lower'][np.where(df['ds'].apply(lambda x: x.hour) < 3)[0]] = 0
        df['yhat_lower'][np.where(
            (pd.to_datetime(df['ds'], utc=True)
             > pd.to_datetime(df['ds'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '11:20'), utc=True)) &
            (pd.to_datetime(df['ds'], utc=True)
             <= pd.to_datetime(df['ds'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '23:59'), utc=True)))[
            0]] = 0
    df['yhat_upper'][np.where(df['ds'].apply(lambda x: x.hour) < 3)[0]] = threshold
    df['yhat_upper'][np.where(
        (pd.to_datetime(df['ds'], utc=True)
         > pd.to_datetime(df['ds'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '11:20'), utc=True)) &
        (pd.to_datetime(df['ds'], utc=True)
         <= pd.to_datetime(df['ds'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '23:59'), utc=True)))[0]] = 2
    return df

# def low_volumn(df):
#     """
#     low volumn threshold
#     :param df: DataFrame ['Date_Key','CallOffered','CallOffered_upper','CallOffered_lower',
#                             'CallOffered_1hr','CallOffered_1hr_upper','CallOffered_1hr_lower',
#                             'CallOffered_30min','CallOffered_30min_upper','CallOffered_30min_lower',
#                             'CallOffered_1min','CallOffered_1min_upper','CallOffered_1min_lower']
#     :return:data_new
#     """
#     data = df
#     data['datetime'] = pd.to_datetime(data['Date_Key'], utc=True)  # , infer_datetime_format=True
#     data['datetime'] = data['datetime'].dt.tz_convert('America/Los_Angeles')
#     data['CallOffered_upper'][np.where(data['datetime'].apply(lambda x: x.hour) < 3)[0]] = 27
#     data['CallOffered_1hr_upper'][np.where(data['datetime'].apply(lambda x: x.hour) < 3)[0]] = 23
#     data['CallOffered_30min_upper'][np.where(data['datetime'].apply(lambda x: x.hour) < 3)[0]] = 17
#     data['CallOffered_1min_upper'][np.where(data['datetime'].apply(lambda x: x.hour) < 3)[0]] = 5
#     data['CallOffered_1hr_upper'][np.where(
#         (pd.to_datetime(data['datetime'], utc=True)
#          > pd.to_datetime(data['datetime'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '11:37'), utc=True)) &
#         (pd.to_datetime(data['datetime'], utc=True)
#          < pd.to_datetime(data['datetime'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '23:59'), utc=True)))[0]] = 2
#     data['CallOffered_30min_upper'][np.where(
#         (pd.to_datetime(data['datetime'], utc=True)
#          > pd.to_datetime(data['datetime'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '11:20'), utc=True)) &
#         (pd.to_datetime(data['datetime'], utc=True)
#          < pd.to_datetime(data['datetime'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '23:59'), utc=True)))[0]] = 2
#     data['CallOffered_1min_upper'][np.where(
#         (pd.to_datetime(data['datetime'], utc=True)
#          > pd.to_datetime(data['datetime'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '10:53'), utc=True)) &
#         (pd.to_datetime(data['datetime'], utc=True)
#          < pd.to_datetime(data['datetime'].apply(lambda x: x.strftime('%Y-%m-%d') + ' ' + '23:59'), utc=True)))[0]] = 1
#     data_new = data
#     return data_new

if __name__ == '__main__':
    df = pd.read_csv('data_approach2_new_low_volume.csv')
    df.rename('')
    data_new = low_volumn(df)
    print()
