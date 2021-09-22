from datetime import datetime, timedelta
import os, pytz, logging
import numpy as np
import pandas as pd
from fbprophet import Prophet
import dbHelper

measures = 'CallOffered' 
_predict_num = 3*24*60 
_confidence_interval = 0.95
_data_days = 30
#measures = os.environ.get('MeasureList')
#_predict_num = int(os.environ.get('Predict_Num'))
#_confidence_interval = float(os.environ.get('Confidence_Interval'))
#_data_days = int(os.environ.get('Data_Days'))

_measure_list = measures.split(',')

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

def train_forecast(df_diff, QueueGroupId_name, datetime_name, measure_diff_list, num_data=7*24*60, confidence_interval=0.95):
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
        logging.info('Forecast for queue group '+str(qid))
        temp_forecast_val = pd.DataFrame()
        qg_df_diff = df_diff[df_diff[QueueGroupId_name]==qid]
        for i in measure_diff_list:
            train_data = pd.concat([qg_df_diff['ds'], qg_df_diff[i]], axis=1).rename(columns={i:'y'})
            forecast_ds = forecasting(train_data, num_data, confidence_interval)
            temp_forecast_val[QueueGroupId_name] = np.repeat(qid, num_data)
            temp_forecast_val[datetime_name] = forecast_ds['ds'].values
            temp_forecast_val[i + '_yhat'] = round(forecast_ds['yhat'], 2).values
            temp_forecast_val[i + '_yhat_upper'] = round(forecast_ds['yhat_upper'],2).values
            temp_forecast_val[i + '_yhat_lower'] = round(forecast_ds['yhat_lower'],2).values
        forecast_val = pd.concat([forecast_val, temp_forecast_val], axis=0)
    return forecast_val



def _save_result_to_db(df, qgid):
    '''
        DB columns: QueueGroupId, PstTimeStamp, CallOffered_Diff_yhat, CallOffered_Diff_yhat_upper, CallOffered_Diff_yhat_lower
    '''
    try:
        # insert_sql = 'INSERT INTO ml.ForecastedCallVolume(QueueGroupId, PstTimeStamp, CallOffered_Diff_yhat, CallOffered_Diff_yhat_upper, CallOffered_Diff_yhat_lower) VALUES(?,?,?,?,?)'#qgid, pst, y, yuper, lower,
        insert_sql = 'INSERT INTO ml.ForecastedCallVolume(QueueGroupId, PstTimeStamp, CallOffered_Diff_yhat, CallOffered_Diff_yhat_upper, CallOffered_Diff_yhat_lower) VALUES(?,?,?,?,?)'#qgid, pst, y, yuper, lower,
        datas = []
        for row in df.iterrows():
            datas.append([row[1]['QueueGroupId'], row[1]['PstTimeStamp'], row[1]['CallOffered_Diff_yhat'], row[1]['CallOffered_Diff_yhat_upper'],row[1]['CallOffered_Diff_yhat_lower']])
        predict_start_time = df['PstTimeStamp'].min()
        pre_sql = "DELETE FROM ml.ForecastedCallVolume WHERE QueueGroupId=%s AND PstTimeStamp >='%s'" %(qgid, str(predict_start_time))
        dbHelper.execute_batch_insert(pre_sql, insert_sql, datas)
        logging.info(qgid+' save_result_to_db complete successfully')
    except Exception as ex:
        logging.error('save_result_to_db met error: '+str(ex))
        raise ex


def forecast(qgid):
    logging.info(qgid+' forecast begin: '+str(datetime.utcnow()))
    nowPST = datetime.utcnow().astimezone(tz=pytz.timezone('US/Pacific'))
    preTime = (nowPST+timedelta(days=-_data_days)).strftime('%Y-%m-%d %H:%M:%S')
    df_diff = dbHelper.execute_query_df("SET NOCOUNT ON; EXEC sp_GetCallOfferedDiff5MinData %s,'%s','%s'" % (qgid, preTime, str(nowPST.date())))
    df_diff.to_csv('diff30days.csv')
    logging.info(qgid+' get data from db end: '+str(datetime.utcnow()))
    logging.info(qgid+' diff data count: '+ str(len(df_diff)))
    if len(df_diff)>0:
        _measure_diff_list= [i+'_Diff' for i in _measure_list]
        df_result = train_forecast(df_diff, 'QueueGroupId', 'PstTimeStamp', _measure_diff_list, _predict_num, _confidence_interval)
        logging.info(qgid+' train end'+str(datetime.utcnow()))
        #_save_result_to_db(df_result, qgid)
        df_result.to_csv('forecastresult.csv')
        logging.info(qgid+' save db end' + str(datetime.utcnow()))
    logging.info(qgid+' forecast end: '+str(datetime.utcnow()))
    return qgid+'successfully forecasted'

if __name__ == '__main__':
    forecast('386')