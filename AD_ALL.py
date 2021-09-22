# from azure.cosmosdb.table.tableservice import TableService
# from azure.cosmosdb.table.models import Entity
import datetime, os, pytz
import pandas as pd
import predict

print('start detect: ' + str(datetime.datetime.now()))
# measures = ['CallOffered', 'CallAnswered']
# qgids = [231, 232, 233, 234, 343, 344, 345, 368]
# con_str = 'DefaultEndpointsProtocol=https;AccountName=atomadfstorageprod;AccountKey=MBGxvUQGoiKWzyj1bY7ZN6KEMjbwLgV0NQDToQao6BAH5hVie9dJ7fARTeUCJHMSnMQ1RC7Do9CwXYalXq7xSA==;EndpointSuffix=core.windows.net'

# qgids = os.environ.get('QueueGroupIds')
# con_str = os.environ.get('Storage_Connection')
# measures = os.environ.get('MeasureList')
# _qids = os.environ.get('QueueGroupIds').split(',')
# for qid in _qids:
#     qgids.append((int)qid)


# _measureList = measures.split(',')
_measureList = ['CallAnswered','CallOffered']  # \,'CallDemand'
measure_diff_list = [i+'_diff' for i in _measureList]


def _save_result(df):
    '''
        df columns: QueueGroupId, PstTimeStamp, CallOffered_Diff, CallOffered_upper, CallOffered_lower
    '''
    try:
        table_service = TableService(connection_string=con_str)
        result_table_name = 'QGDiffTrend' + str(datetime.datetime.utcnow().year) + str(datetime.datetime.utcnow().month)
        if not table_service.exists(result_table_name):
            table_service.create_table(result_table_name)

        for row in df.iterrows():
            tempEntity = {'PartitionKey': str(row['QueueGroupId']),
                          'RowKey': str(row['PstTimeStamp'].strftime('%Y-%m-%d %H:%M:%S')),
                          'QueueGroupId': row['QueueGroupid']}
            for ms in _measureList:
                tempEntity[ms + '_diff_yhat'] = row[ms + '_diff_yhat']
            table_service.insert_or_replace_entity(result_table_name, tempEntity)

        print('save_result_to_redis complete successfully')
    except Exception as ex:
        print('save_result_to_redis met error: ' + str(ex))
        raise ex


def _process_data(df,interval_seconds,_measureList):
    try:
        df = predict.get_diff_data(df, 'QueueGroupId', 'PstTimeStamp', _measureList)
        measure_diff_list = [measures + '_diff' for measures in _measureList]
        return predict.diff_interval_sum(df, 'QueueGroupId', 'datetime', measure_diff_list,interval_seconds)
    except Exception as ex:
        print('process data met error: ' + str(ex))
        raise (ex)


def _train_predict(df_diff, QueueGroupId_name, datetime_name, measure_diff_list, num_data=24*60):
    try:
        return predict.train_forecast(df_diff, QueueGroupId_name, datetime_name, measure_diff_list, num_data)  # predict.train(df_diff)
    except Exception as ex:
        print('train predict met error: ' + str(ex))
        raise (ex)


def _get_data_from_storage(filter_str, tableNames):
    try:
        table_service = TableService(connection_string=con_str)
        datas = {'QueueGroupId': [], 'PstTimeStamp': []}
        for ms in _measureList:
            datas[ms] = []
        selectMeasures = 'QueueGroupId,Timestamp,' + measures
        for tbName in tableNames:
            tasks = table_service.query_entities(tbName,
                                                 select=selectMeasures,
                                                 filter=filter_str, timeout=600000)
            for ts in tasks:
                datas['QueueGroupId'].append(ts.QueueGroupId.value)
                datas['PstTimeStamp'].append(
                    ts.Timestamp.astimezone(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d %H:%M:%S'))
                for ms in _measureList:
                    datas[ms].append(ts[ms].value)

        return pd.DataFrame(datas).drop_duplicates()
    except Exception as ex:
        print('get data from storage met error: ' + str(ex))
        raise (ex)


# def detect():
    # print('begin detect: '+str(datetime.datetime.now()))
    # dtNow = datetime.datetime.now()
    # dtPre = dtNow+datetime.timedelta(days=-30)
    # preTBName = 'QueueGroupData%02d%02d' %(dtPre.year,dtPre.month)
    # nowTBName = 'QueueGroupData%02d%02d' %(dtNow.year,dtNow.month)
    # tbNames = [preTBName,nowTBName]
    # filter_str=None
    # if len(qgids) > 0:
    #     filter_str=''
    #     for qid in qgids:
    #         filter_str += 'QueueGroupId eq %d or ' % qid
    #     filter_str = filter_str[0:-3]
    # print(filter_str)
    # df =_get_data_from_storage(filter_str, tbNames)
    # df.to_csv('Data.csv')
if __name__ == '__main__':
    # df = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueGroupData202103.csv")[['QueueGroupId','PstTimeStamp','CallOffered','CallAnswered','CallDemand']]
    # df = df[(df['QueueGroupId'] == 386)]
    # interval_seconds = 1 * 60
    # df_diff = _process_data(df, interval_seconds,_measureList)
    QueueGroupId = 345
    df = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueGroupData202102.csv")[['QueueGroupId','PstTimeStamp','CallOffered','CallAnswered']]
    df = df[(df['QueueGroupId'] == QueueGroupId)]
    df_for = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueGroupData202103.csv")[['QueueGroupId','PstTimeStamp','CallOffered','CallAnswered']]
    df_for = df_for[(df_for['QueueGroupId'] == QueueGroupId)]
    for i in range(7):#[5,10,15,20,25,30]
        num =  i* 24 * 60
        df_new = pd.concat([df[num::], df_for.iloc[0:num]])
        interval_seconds = 30 * 60
        df_diff = _process_data(df_new, interval_seconds, _measureList)
        df_diff['datetime'] = df_diff['datetime'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")[0:17] + '00')
        measure_diff_interval_list = [i + str(interval_seconds) + 's' for i in measure_diff_list]
        # df_diff.to_csv('DataDiff.csv')
        QueueGroupId_name = 'QueueGroupId'
        datetime_name = 'datetime'
        res = _train_predict(df_diff, QueueGroupId_name, datetime_name, measure_diff_interval_list, num_data=1*24*60)
        res_new = pd.concat([res_new, res])
    res_new.to_csv('./file/predict_30min_1day_new_CO_CA_'+str(QueueGroupId)+'_03'+ '.csv', index=False)
    # _save_result(res)
    print('end detect: ' + str(datetime.datetime.now()))
