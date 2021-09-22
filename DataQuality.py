import os
import sys
import argparse
import configparser
import ast

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.realpath(os.path.join(dir_path, "../../../")))

import pandas as pd
import ML.utils.OutlierDetection as PMO
import datetime


# time
def data_time_process(df, datetime_name, datetime_name_out, time_zero,str_time_form):
    """
        Time format conversion
    :param df: DataFrame: ['Timestamp',...]
    :return: data: DataFrame: ['datetime']
    """
    data = pd.DataFrame()
    data[datetime_name_out] = pd.to_datetime(df[datetime_name], utc=True) 
    data[datetime_name_out] = data[datetime_name_out].dt.tz_convert(time_zero)
    data[datetime_name_out] = data[datetime_name_out].apply(
        lambda x: datetime.datetime.strptime(x.strftime(str_time_form), str_time_form))
    return data

class DataQuality:
    def missing_statistics(self, input_data, save_path=None):
        '''
        Display number of missing values/missing rate in each column,
        arranging in an order from largest to smallest;
        Also display number of rows in input_data

        input_data: Dataframe
        save_path: directory to store the display if necessary
        '''
        input_data_desc                 = pd.concat([input_data.dtypes, input_data.isnull().sum()], axis=1)
        input_data_desc.columns         = ["Data type", "Number of missing values"]
        input_data_desc["Missing rate"] = input_data_desc["Number of missing values"] * 1.0 / input_data.shape[0]
        input_data_desc                 = input_data_desc.sort_values(["Missing rate"], ascending=False)
        input_data_desc.index           = input_data_desc.index.rename("Features(" + str(input_data.shape[0]) + " rows)")
        
        if save_path is not None:
            input_data_desc.to_csv(save_path)
        return input_data_desc


    def zeros_statistics(self, input_data, save_path=None):
        """
            Display number of Zero values/Zeros rate in each column, arranging in an order from largest to smallest
        :param input_data: DataFrame:input data
        :param save_path: str: directory to store the display if necessary
        :return: input_data_desc: Display number of Zero values/Zeros rate in each column
        """

        input_data_desc                 = pd.concat([input_data.dtypes, (input_data == 0).astype(int).sum()], axis=1)
        input_data_desc.columns         = ["Data type", "Number of Zeros"]
        input_data_desc["Zeros rate"] = input_data_desc["Number of Zeros"] * 1.0 / input_data.shape[0]
        input_data_desc                 = input_data_desc.sort_values(["Zeros rate"], ascending=False)
        input_data_desc.index           = input_data_desc.index.rename("Features(" + str(input_data.shape[0]) + " rows)")
        
        if save_path is not None:
            input_data_desc.to_csv(save_path)
        return input_data_desc


    def valid_timeperiod_statistics(self, input_data_datetime, time_start, time_end, save_path=None):
        """
            Display number of Valid timeperiod values/Valid timeperiod rate in datetime
        :param input_data: Series:input data, dtype:Datetime
        :param save_path: str: directory to store the display if necessary
        :return: input_data_desc: Display number of Valid timeperiod values/Valid timeperiod rate in datetime
        """

        input_data_desc                 = pd.DataFrame(columns=["Data type", "Number of Valid timeperiod","Valid timeperiod rate"])
        input_data_desc["Data type"]    = pd.DataFrame(input_data_datetime).dtypes
        input_data_desc["Number of Valid timeperiod"] = input_data_datetime.apply(lambda x:(x>pd.to_datetime(str(x)[0:11]+time_start))
                                                        &(x<pd.to_datetime(str(x)[0:11]+time_end))).astype(int).sum()
        input_data_desc["Valid timeperiod rate"]      = input_data_desc["Number of Valid timeperiod"].values * 1.0 / input_data_datetime.shape[0]
        
        if save_path is not None:
            input_data_desc.to_csv(save_path)
        return input_data_desc
    

    def outlier_statistics(self, input_data, measure_name,save_path=None):
        """
            Display number of Outlier values/Outliers rate in datetime
        :param input_data: DataFrame:input data
        :param save_path: str: directory to store the display if necessary
        :return: input_data_desc: Display number of Outlier values/Outliers rate in data
        """

        input_data_desc                 = pd.DataFrame(columns=["Data type", "Number of Outliers","Outliers rate"])
        input_data_desc['Data type']    = pd.DataFrame(input_data).dtypes
        data_remove_outlier = PMO.outlierRemoval(input_data,[measure_name])
        # data_remove_outlier_rate = len(data_remove_outlier.STD())/len(data)
        input_data_desc['Number of Outliers'] = len(input_data)-len(data_remove_outlier.STD())
        input_data_desc["Outliers rate"]      = input_data_desc["Number of Outliers"].values * 1.0 / input_data.shape[0]
        
        if save_path is not None:
            input_data_desc.to_csv(save_path)
        return input_data_desc
    
class read_config():
    def __init__(self,cf,DFT):
        self.datetime_name_out = cf.get(DFT, 'datetime_name_out')
        self.time_zero = cf.get(DFT, 'time_zero')
        self.str_time_form = cf.get(DFT, 'str_time_form')
        self.save_path = ast.literal_eval(cf.get(DFT, 'save_path'))
        self.threshold_missing_rate = cf.getfloat(DFT, 'threshold_missing_rate')
        self.threshold_validtime_rate = cf.getfloat(DFT, 'threshold_validtime_rate')
        self.threshold_zero_rate = cf.getfloat(DFT, 'threshold_zero_rate')
        self.threshold_outlier_rate = cf.getfloat(DFT, 'threshold_outlier_rate')

def read_config_fun(filename):
    cf = configparser.RawConfigParser()
    cf.read(filename)
    DFT = "DEFAULT"
    return read_config(cf,DFT)


def read_parameter():
    ap = argparse.ArgumentParser()
    ap.add_argument('-ts', '--time_start', type=str, default='09:00:00',
                    help='#the valid timeperiod start point')
    ap.add_argument('-te', '--time_end', type=str, default='17:00:00',
                    help='#the valid timeperiod end point')
    ap.add_argument('-mn', '--measure_name', type=str, default='CallOffered',
                    help='#the measure which will do analysis')
    ap.add_argument('-tn', '--timestamp_name', type=str, default='Timestamp',
                    help='#the timestamp name')
    args = vars(ap.parse_args())
    time_start = args['time_start']
    time_end = args['time_end']
    measure_name = args['measure_name']
    datetime_name = args['timestamp_name']
    return time_start,time_end,measure_name,datetime_name
        

if __name__ == '__main__':
    # parameter
    time_start,time_end,measure_name,datetime_name = read_parameter()
    cf_filename = r'F:\BeyondLearning\Product\ML\core\Data_quality_analysis\config\config.ini'
    config = read_config_fun(cf_filename)
    # data_input
    data_path = r'F:\Data_storage\QueueGroupData202105_234.csv'
    data = pd.read_csv(data_path)[['CallOffered','Timestamp']]
    data['Datetime'] = data_time_process(data,datetime_name,config.datetime_name_out, config.time_zero, config.str_time_form)[config.datetime_name_out]
    # result
    DataQuality = DataQuality()
    data_missing = DataQuality.missing_statistics(data[[measure_name]], save_path=config.save_path)
    data_zero = DataQuality.zeros_statistics(data[[measure_name]], save_path=config.save_path)
    data_valid_time = DataQuality.valid_timeperiod_statistics(data['Datetime'], time_start=time_start, time_end=time_end, save_path=config.save_path)
    data_outlier = DataQuality.outlier_statistics(data[[measure_name]], measure_name=measure_name,save_path=config.save_path)

    if data_missing['Missing rate'].values[0]>config.threshold_missing_rate or data_valid_time['Valid timeperiod rate'].values[0]<config.threshold_validtime_rate:
        print('Poor data quality due to the number of data is too small.')
    if data_zero['Zeros rate'].values[0]>config.threshold_zero_rate:
        print('Poor data quality due to too many Zeros.')
    if data_outlier['Outliers rate'].values[0]>config.threshold_outlier_rate:
        print('The result may inaccurate due to too many Outliers.')
    print(data_missing, data_zero, data_valid_time, data_outlier)
