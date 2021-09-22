import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from fbprophet import Prophet
## feature selection
# correlation
# rolling_corr
def rolling_spearman(seqa, seqb, window):
    """
    calculate the correlation index use rolling windows.
    :param seqa: type:np.ndarray
    :param seqb: type:np.ndarray
    :param window: width of rolling window
    :return: correlation index
    """
    stridea = seqa.strides[0]
    ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
    strideb = seqa.strides[0]
    ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    ar = ar.rank(1)
    br = br.rank(1)
    corrs = ar.corrwith(br, 1, method="pearson")
    return pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)

# decision_tree_figure
def decision_tree(X, y, feature_name, target_name):
    """
    plot decision tree use X(features data) and y(target data).
    :param X: features data, type:DataFrame
    :param y: target data, type:Series
    :param feature_names: Names of features
    :param target_name: Name of target
    :return: figure of decision tree
    """
    os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'
    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(max_depth=4)
    fig, ax = plt.subplots(figsize=(50, 15))
    # 拟合模型
    clf.fit(X, y)
    plot_tree(clf, feature_names=feature_name, class_names=target_name, ax=ax, fontsize=20)
    # plt.show()

## anomaly detection
# prophet
def forecasting(df_diff, df_diff_col_name, df_diff_timestamp_col_name, num_data=7*24*60, interval_width=0.95):
    """
    ML forecast
    :param df_diff: DataFrame: ['ds','y']
    :param num_data: the length of forecasting data (minute)
    :param interval_width: confidence interval
    :return: forecast_out: DataFrame: ['ds', 'yhat', 'yhat_upper', 'yhat_lower']
    """
    df_diff = df_diff.rename(columns={df_diff_col_name:'y',df_diff_timestamp_col_name:'ds'})
    holidays = pd.DataFrame({
        'holiday': 'Thanksgiving',
        'ds': pd.to_datetime(['2020-11-26', '2020-11-27'])})
    m = Prophet(interval_width=interval_width, holidays=holidays)
    m.fit(df_diff)
    future = m.make_future_dataframe(periods=num_data, freq='min', include_history=False)
    forecast = m.predict(future)
    forecast_out = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
    return forecast_out

def box_quartile(list, list1):
    """
    quartile anomaly detection algorithm
    :param list: the data which is been used to determine upper and lower bounds; type:list
    :param list1: the data which you want to do anomaly detection; type:list
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

## Staff
# Volume/t_upper
def staff_for(CO_for, t_upper, CO_for_Name='CallOffered_diff1800s_yhat', alfa=1):
    """
    Use CallOffered forecast data and upper threshold to forecast staff numbers
    :param CO_for: CallOffered forecast data  type:DataFrame ['datetime,CO_for_Name]
    :param t_upper: upper threshold
    :param CO_for_Name: column name of CallOffered forecast data
    :param alfa: Call_back rate
    :return: Staff
    """
    Staff_forecast = pd.DataFrame()
    Staff_forecast['date'] = CO_for['datetime']
    # Staff_forecast['data'] = data_rate['totalStaff']
    Staff_forecast['data'] = CO_for[CO_for_Name]* 0.8 * alfa / t_upper
    Staff_forecast['date'] = pd.to_datetime(Staff_forecast['date'])
    return Staff_forecast

#prophet
# for_out = forecasting(Staff, num_data=7*24*60, interval_width=0.95)

if __name__ == '__main__':
    rdm_datalist = [3,8,6,4,59,2,7,12,79,32]
    rdm_datalist1 = [2,9,47,7,8,79,62,4,57,15]
    rdm_datetime = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start='2021-03-01', end='2021-03-10'))]
    rdm_label = [0,1,1,0,1,0,0,0,0,0]
    target_name = ['ISnt_Alert', 'Is_Alert']
    rdm_dataframe = pd.DataFrame({'col_a':rdm_datalist, 'date_time':rdm_datetime, 'col_b':rdm_datalist1})
    result = rolling_spearman(rdm_dataframe['col_a'].values, rdm_dataframe['col_a'].values, 2)
    print(result)