import pandas as pd
import datetime
from fbprophet import Prophet


def model(CO_diff=None, CO=None):
    """
    prophet model train
    :param CO_diff: DataFrame [Time, CO_diff]
    :param CO: DataFrame [Time, CO]
    :return: data_output:DataFrame [Time, yhat_lower, yhat_upper]
    """

    forecasting_num = 7*60*24
    if CO_diff is None:
        time = CO['PstTimeStamp'].values
        datatime = [datetime.datetime.strptime(time[i].replace("T", " ")[:time[1].replace("T", " ").rfind(".")],
                                               "%Y-%m-%d %H:%M:%S") for i in range(len(time))]
        CO['PstTimeStamp'] = datatime
        CO_diff = CO.groupby(CO['PstTimeStamp'].apply(lambda x: x.day), as_index=False).diff()

    time = CO_diff['PstTimeStamp'].values
    datatime = [datetime.datetime.strptime(time[i].replace("T", " ")[:time[i].replace("T", " ").rfind(".")],
                                           "%Y-%m-%d %H:%M:%S") for i in range(len(time))]
    CO_diff['PstTimeStamp'] = datatime
    data_input = CO_diff.rename(columns={'PstTimeStamp':'ds', 'CallOffered':'y'})
    m = Prophet()
    m.fit(data_input)
    future = m.make_future_dataframe(periods=forecasting_num, freq='min')
    forecast = m.predict(future)
    data_output = pd.merge(future,forecast)[['ds','yhat','yhat_lower','yhat_upper']]
    data_output.to_csv('forcasting_7days.csv')
    return data_output


def AD(CO_diff, forecast):
    """
    anomaly detection test
    :param CO_diff: DataFrame [Time, CO_diff]
    :param forecast: DataFrame [Time, yhat_lower, yhat_upper]
    :return: True/False
    """
    # time
    return (CO_diff['CallOffered'].values < forecast.iloc[[0]]['yhat_lower'].values)[0] | \
           (forecast.iloc[[0]]['yhat_upper'].values < CO_diff['CallOffered'].values)[0]


if __name__ == '__main__':
    df = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueGroupData202009.csv")
    q = '232'
    df = df[df['QueueGroupId'] == int(q)]
    # CO = df[['PstTimeStamp','CallOffered']][:-7*24*60]
    # data_output = model(CO)
    # # CO_diff = CO[-7*24*60:]
    # # AD(CO_diff, data_output[-7*24*60:])


    print()






