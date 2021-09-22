import numpy as np
import pandas as pd
import datetime

## data analysis
# data quality index

# anomaly analysis
def multiple_sigma(df, col_name, multiple):
    """
    multiple sigma anomaly detection algorithm
    :param df: DataFrame which you want to do anomaly detection;  type:DataFrame
    :param col_name: the column name of data you want to do anomaly detection;  type:string
    :param multiple: the multiple of sigma;  type:int
    :return: outliers in original df
    """
    std = df[col_name].std()
    mean = df[col_name].mean()
    upper = multiple*std + mean
    lower = mean - multiple*std
    outlier_col = df[col_name][(df[col_name] < lower) | (df[col_name]> upper)]
    return outlier_col


def box_quartile(list, list1):
    """
    quartile anomaly detection algorithm
    :param list: the data which is been used to determine upper and lower bounds; type:Series
    :param list1: the data which you want to do anomaly detection; type:Series
    :return: outliers in list1
    """
    Q1 = np.percentile(list, 25)  # 1st quartile (25%)
    Q3 = np.percentile(list, 75)  # 3rd quartile (75%)
    IQR = Q3 - Q1    # Interquartile range (IQR)
    outlier_step = 1.5 * IQR    # outlier step
    # Determine a list of indices of outliers for feature col
    t_upper = Q3 + outlier_step
    t_lower = Q1 - outlier_step
    outlier_col = list1[(list1 < t_lower) | (list1 > t_upper)]
    return outlier_col


def poly_fit_replace(sequence_data, multiple_sigma=1.0, order=20):
    """
    Fitting into a segment of time series data
    Args:
        sequence_data (pd.Series or np.ndarray): data to be fitted
        multiple_sigma (float): sigma multiples which require greater than zero
        order (int): the order of the fitting polynomial
    Returns:
        pred (np.ndarray): results of polynomial fitting
        col_replaced (list): the polynomial fitting results after the threshold value is replaced
    """
    sequence_data = pd.Series(sequence_data)
    sequence_data.dropna(inplace=True)
    sequence_lenth = len(sequence_data)
    if sequence_lenth == 0:
        return None, None

    mean = sequence_data.mean()
    std = sequence_data.std()
    upper = mean + multiple_sigma*std
    lower = mean - multiple_sigma*std

    # get pred value of sequence_data with polyfit
    xdata = np.linspace(0, sequence_lenth, num=sequence_lenth, endpoint=False)
    reg = np.polyfit(xdata, sequence_data, order)
    pred = np.polyval(reg, xdata)

    # replace sequence_data value which exceed the threshold
    def replace_with_threshold(origin, pred):
        if lower <= origin <= upper:
            return origin
        else:
            return pred

    col_replaced = list(map(replace_with_threshold, sequence_data, pred))

    return pred, col_replaced


def replace_outlier(df, col_name, multiple=2):
    """
    detect outliers(default:3 sigma) and replace with the mean(default) of the data.
    :param df: original DataFrame data; type:DataFrame
    :param col_name: the column name of data you want to do anomaly detection;  type:string
    :param multiple: the multiple of sigma(default:3);  type:int
    :return: new df data which outliers have been replaced
    """
    data_std = df[col_name].std()
    data_mean = df[col_name].mean()
    upper = multiple * data_std + data_mean
    lower = data_mean - multiple * data_std
    df[col_name][(df[col_name] < lower) | (df[col_name] > upper)] = None
    df[col_name] = df[col_name].fillna(data_mean)
    return df


def data_reconstruction(data, data_datetime_name='cal_date',data_col_name='totalStaff',number=30):
    """
    reconstruct data: expand half-hour(default:number=30) level data to minute level
    :param data: the data which you want to reconstruction; type:DataFrame
    :param data_datetime_name: the column name of datetime in data; type:string
    :param data_col_name: the column name of data you want to reconstruction in data; type:string
    :param number: Multiple of expansion; type:int
    :return: reconstruction data; type:DataFrame:['datetime',data_col_name]
    """
    data = data.rename(columns={data_datetime_name:'datetime'})
    data_new = data[data_col_name].values.repeat(number)
    every_minute_array = np.arange(0,number,1)*60
    data_r = pd.DataFrame()
    for i in every_minute_array:
        data_r['datetime_'+str(i)+'s'] = data['datetime'].apply(lambda dt:(pd.to_datetime(dt) + datetime.timedelta(0,int(i),0)))
    data_r = data_r.stack()
    data_r_new = pd.DataFrame()
    data_r_new['datetime'] = data_r.values
    data_r_new[data_col_name] = data_new
    return data_r_new


## Time format conversion
# utc_timestamp to pst_timestamp
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


# timestamp format: delete seconds
def time_approximation(Timestamp):
    """
    time approximation: if the second in datetime is lower than 30, the second turn to 00,
                        if the second in datetime is bigger than 30, the minute add 1,second turn to 00.
    :param Timestamp: datetime, type:Series
    :return: datetime which has been approximated.
    """
    if int(pd.to_datetime(Timestamp).second)>30:
        datetime_1 = pd.to_datetime(Timestamp)+datetime.timedelta(0,60-pd.to_datetime(Timestamp).second,0)
    else:
        datetime_1 = pd.to_datetime(Timestamp)-datetime.timedelta(0,pd.to_datetime(Timestamp).second,0)
    return datetime_1


## Diff
# diff/diff_30min/diff_1hr/diff_24hr
def data_diff_process(data_new, interval_seconds, datetime_name='ds'):
    """
        calculate the diff of data
    :param data_new: DataFrame [datetime_name, y] ...[时间,值]
    :param diff_period_hours: the lasting hours of diff data.(time goes back use negative values)
    :param diff_period_minutes: the lasting minutes of diff data.(time goes back use negative values)
    :return:data_diff: DataFrame [datetime_name, y] ...[时间,值]
    """
    data_new[datetime_name] = pd.to_datetime(data_new[datetime_name])
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
    rdm_datalist = [3,8,6,4,59,2,7,12,79,32]
    rdm_datetime = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start='2021-03-01', end='2021-03-10'))]
    rdm_dataframe = pd.DataFrame({'col_a':rdm_datalist, 'date_time':rdm_datetime})
    result = replace_outlier(rdm_dataframe, 'col_a', multiple=2)
    print(result)
