from azure.cosmosdb.table.tableservice import TableService
from datetime import datetime, timedelta
import logging, os, pytz
import pandas as pd
import numpy as np


measures = 'CallOffered' 
con_str = 'DefaultEndpointsProtocol=https;AccountName=atomadfstorageprod;AccountKey=MBGxvUQGoiKWzyj1bY7ZN6KEMjbwLgV0NQDToQao6BAH5hVie9dJ7fARTeUCJHMSnMQ1RC7Do9CwXYalXq7xSA==;EndpointSuffix=core.windows.net'
_interval_seconds = 300
# con_str = os.environ.get('Storage_Connection')
# measures = os.environ.get('MeasureList')
# _interval_seconds = int(os.environ.get('Interval_Seconds'))

_measure_list = measures.split(',')


def my_date_parse(dt):
    return datetime(int(dt[0:4]), int(dt[5:7]), int(dt[8:10]))


def my_datetime_parse(dt):
    dt_datetime = datetime(int(dt[0:4]), int(dt[5:7]), int(dt[8:10]), int(dt[11:13]), int(dt[14:16]), int(dt[17:19]))
    return dt_datetime

# get diff data with original time stamp
def get_diff_data(df, QueueGroupId_name, datetime_name, measure_list):
    df['date'] = df[datetime_name].apply(my_date_parse)
    df[datetime_name] = df[datetime_name].apply(my_datetime_parse)
    for m in measure_list:
        df[m+'_Diff'] = df.groupby([QueueGroupId_name, 'date'])[m].diff()
    drop_col = measure_list+['date',]
    df.drop(drop_col, axis=1, inplace=True)
    return df


def back_interval_sum(QueueGroup_df, interval_seconds, dt_datetime, dt_datetime_name, measure_name_list):
    zero_time = pd.to_datetime(datetime.strptime((str(dt_datetime)[0:10]+' 00:00:00'),"%Y-%m-%d %H:%M:%S"))
    back_time = (dt_datetime-timedelta(0, interval_seconds, 0) if dt_datetime-timedelta(0, interval_seconds, 0)>zero_time else zero_time)
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


def get_calculate(partionKey):
    logging.info('begin get and calculate: '+str(datetime.now()))
    datas = {'QueueGroupId':[],'PstTimeStamp':[]}
    for ms in _measure_list:
        datas[ms]=[]
    selectMeasures= 'QueueGroupId,Timestamp,' + measures
    filter_str = "PartitionKey eq '%s'" % partionKey
    tbName = "QueueGroupData"+partionKey[0:6]
    table_service = TableService(connection_string=con_str)
    logging.info('get data from storage begin: '+str(datetime.now()))
    tasks = table_service.query_entities(tbName,
        select =selectMeasures,
        filter = filter_str, timeout=3600)
    for ts in tasks:
        datas['QueueGroupId'].append(ts.QueueGroupId.value)
        datas['PstTimeStamp'].append(ts.Timestamp.astimezone(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d %H:%M:%S'))
        for ms in _measure_list:
            datas[ms].append(ts[ms].value)
    df = pd.DataFrame(datas).drop_duplicates()
    logging.info('data count: '+str(len(datas['QueueGroupId'])))
    logging.info('get data from storage end: '+str(datetime.now()))
    if len(datas['QueueGroupId'])==0:
        raise Exception('No data from storage for pk '+partionKey)
    df_diff = get_diff_data(df, 'QueueGroupId', 'PstTimeStamp', _measure_list)
    _measure_diff_list= [i+'_Diff' for i in _measure_list]
    df_diff = diff_interval_sum(df_diff, 'QueueGroupId', 'PstTimeStamp', _measure_diff_list, _interval_seconds)
    df_diff = df_diff.fillna(0)
    if len(df_diff['QueueGroupId']) ==0:
        raise Exception('No data after diff for pk '+partionKey)
    logging.info('save forecast data to db begin: '+str(datetime.now()))
    # if _save_diff_to_db(df_diff, partionKey[0:6]):
    #     logging.info("Get diff successfully for pk "+partionKey)
    logging.info('save forecast data to db end: '+str(datetime.now()))
    logging.info('end get and calculate: '+str(datetime.now()))

get_calculate('2021030118')