import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from AD_ALL import _process_data,_train_predict
import pyodbc
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
import seaborn as sns
import predict


# _measureList = ['CallOffered']  #, 'CallAnswered'
# measure_diff_list = [measures + '_diff' for measures in _measureList]
s = '1800'
s1 = '30'
QueueGroupId_name = 'QueueGroupId'
datetime_name = 'datetime'
QueueGroupId = 345


def staff_WFM_reconstruction(staff_WFM_data, staff_WFM_datetime_name='cal_date',staff_WFM_name='totalStaff'): # staff_data
    staff_WFM_data = staff_WFM_data.rename(columns={staff_WFM_datetime_name:'datetime'})
    staff_WFM_data['datetime'] = staff_WFM_data['datetime'].apply(lambda x:predict.time_approximation(x))
    TotalStaff = staff_WFM_data[staff_WFM_name].values.repeat(30)
    every_minute_array = np.arange(0,30,1)*60
    staff_data_r = pd.DataFrame()
    for i in every_minute_array:
        staff_data_r['datetime_'+str(i)+'s'] = staff_WFM_data['datetime'].apply(lambda dt:(dt + datetime.timedelta(0,int(i),0)))
    staff_data_r = staff_data_r.stack()
    staff_data_r_new = pd.DataFrame()
    staff_data_r_new['datetime'] = staff_data_r.values
    staff_data_r_new['totalStaff'] = TotalStaff
    return staff_data_r_new


# agentgroup--staff
def agentgroup_staff(data, data_datetime_name='Timestamp',data_name='AgentsLoggedIn'):
    data['datetime'] = predict.data_time_process(data, data_datetime_name)
    data['datetime'] = data['datetime'].apply(lambda x:predict.time_approximation(x))
    data_staff_name = data[data['Measure'] == data_name].drop_duplicates()
    data_AgentLogedIn = pd.DataFrame()
    data_AgentLogedIn['datetime'] = data_staff_name.groupby(data_staff_name['datetime']).sum().index  # transform()
    data_AgentLogedIn['Staff'] = data_staff_name.groupby(data_staff_name['datetime']).sum()['Value'].values
    # data_staff.to_csv('./file/data_staff_timestamp.csv')
    return data_AgentLogedIn


def detectoutliers(list, list1):
    Q1 = np.percentile(list, 25)  # 1st quartile (25%)
    Q3 = np.percentile(list, 75)  # 3rd quartile (75%)
    IQR = Q3 - Q1    # Interquartile range (IQR)
    outlier_step = 1.5 * IQR    # outlier step
    # Determine a list of indices of outliers for feature col
    t_upper = Q3 + outlier_step
    t_lower = Q1 - outlier_step
    outlier_list_col = list1[(list1 < t_lower) | (list1 > t_upper)]
    return outlier_list_col, t_upper, t_lower


# def data_forecast(df, measure_name='CallOffered', diff_datetime_name='PstTimeStamp',datetime_name='datetime'):
#     interval_seconds = 30 * 60
#     measure_list = [measure_name]
#     df_diff = predict.get_diff_data(df, QueueGroupId_name, diff_datetime_name, measure_list)
#     measure_diff_list= [i+'_diff' for i in measure_list]
#     df_diff = predict.diff_interval_sum(df_diff, QueueGroupId_name, diff_datetime_name, measure_diff_list, interval_seconds)
#     measure_diff_interval_list = [i + str(interval_seconds) + 's' for i in measure_diff_list]
#     CO_diff = df_diff.fillna(0)
#     CO_forecast = predict.train_forecast(CO_diff, QueueGroupId_name, datetime_name, measure_diff_interval_list, num_data=1*24*60)
#     return CO_forecast


def data_staff_rate(df, staff_WFM, CO_forecast, staff, alfa=1, measure_name='CallAnswered', staff_name='Staff',
                    staff_WFM_name='totalStaff', staff_WFM_datetime_name='cal_date',diff_datetime_name='Timestamp',
                    datetime_name='datetime',CO_forecast_datetime_name='datetime'):
    ## Calculate CA_diff_30min
    measure_time_list = [QueueGroupId_name,measure_name,diff_datetime_name]
    df = df[measure_time_list]
    interval_seconds = 30 * 60
    measure_list = [measure_name]
    df[diff_datetime_name] = predict.data_time_process(df,diff_datetime_name)
    df = df.rename(columns={diff_datetime_name:datetime_name})
    df_diff = predict.get_diff_data(df, QueueGroupId_name, datetime_name, measure_list)
    measure_diff_list= [i + '_diff' for i in measure_list]
    df_diff = predict.diff_interval_sum(df_diff, QueueGroupId_name, datetime_name, measure_diff_list, interval_seconds)
    CA_diff = df_diff.fillna(0)
    CA_diff[datetime_name] = CA_diff[datetime_name].apply(lambda x:predict.time_approximation(x))
    CA_diff.to_csv('./file/CA_diff'+str(QueueGroupId)+'.csv')
    ## Calculate AgentLogedIn
    data_AgentLogedIn = agentgroup_staff(staff)
    data_AgentLogedIn = predict.replace_outlier(data_AgentLogedIn,staff_name,6)   ## Outlier filling(remove the peaks) #####345-6,234-1
    data_AgentLogedIn.to_csv('./file/data_AgentLogedIn'+str(QueueGroupId)+'.csv')
    ## merge CA_diff_30min and AgentLogedIn
    data_m = pd.merge(CA_diff,data_AgentLogedIn,on=datetime_name).drop_duplicates() ##[[QueueGroupId_name,datetime_name,'CallAnswered_diff1800s',staff_name]]
    ## Calculate rate_CA(CA_diff_30min/AgentLogedIn)
    data_rate_CA = pd.DataFrame()
    data_rate_CA[datetime_name] = data_m[datetime_name]
    data_rate_CA['rate_CA'] = data_m['CallAnswered_diff'+str(interval_seconds)+'s'] / data_m[staff_name]
    data_rate_CA = data_rate_CA.replace(np.inf, 0)     ## CA!=0 & Staff(AgentLogedIn)=0
    data_rate_CA = data_rate_CA.fillna(0)              ## CA=0 & Staff(AgentLogedIn)=0
    ## Calculate staff_WFM_data(30min-->1min)
    staff_WFM = staff_WFM_reconstruction(staff_WFM,staff_WFM_name=staff_WFM_name,staff_WFM_datetime_name=staff_WFM_datetime_name)
    staff_WFM.to_csv('./file/staff_WFM'+str(QueueGroupId)+'.csv')
    ## merge CO_yhat_30min and Staff_WFM
    CO_forecast = CO_forecast.rename(columns={CO_forecast_datetime_name: datetime_name})
    CO_forecast[datetime_name] = CO_forecast[CO_forecast_datetime_name].apply(lambda x: predict.time_approximation(x))
    data_staff_CO_merge = pd.merge(staff_WFM, CO_forecast, on=datetime_name)
    data_staff_CO_merge.to_csv('./file/CO_staff_WFM'+str(QueueGroupId)+'.csv')
    ## Calculate rate_CO_SL(CO_yhat_30min*SL*alfa/staff_WFM)
    data_rate_CO_SL = pd.DataFrame()
    data_rate_CO_SL[datetime_name] = data_staff_CO_merge[datetime_name]
    data_rate_CO_SL['CallOffered_diff_yhat'] = data_staff_CO_merge['CallOffered_diff'+str(interval_seconds)+'s_yhat']
    data_rate_CO_SL['rate_CO_SL'] = data_staff_CO_merge['CallOffered_diff'+str(interval_seconds)+'s_yhat'] * 0.8 * alfa\
                                    /data_staff_CO_merge['totalStaff']
    data_rate_CO_SL = data_rate_CO_SL.replace(np.inf, 0)
    ## merge CA_diff_30min and staff_WFM
    data_m1 = pd.merge(CA_diff,staff_WFM, on=datetime_name)
    ## Calculate rate_CA_WFM
    data_rate_CA_WFM = pd.DataFrame()
    data_rate_CA_WFM[datetime_name] = data_m1[datetime_name]
    data_rate_CA_WFM['rate_CA_WFM'] = data_m1['CallAnswered_diff'+str(interval_seconds)+'s'] / data_m1['totalStaff']
    data_rate_CA_WFM = data_rate_CA_WFM.replace(np.inf, 0)     ## CA!=0 & Staff(AgentLogedIn)=0
    data_rate_CA_WFM = data_rate_CA_WFM.fillna(0)              ## CA=0 & Staff(AgentLogedIn)=0
    return CO_forecast, data_rate_CA, data_rate_CA_WFM, data_rate_CO_SL


def staff_for(data_rate, t_upper, alfa=1):
    Staff_forecast = pd.DataFrame()
    Staff_forecast['date'] = data_rate['datetime']
    # Staff_forecast['data'] = data_rate['totalStaff']
    Staff_forecast['data'] = data_rate['CallOffered_diff1800s_yhat']* 0.8 * alfa / t_upper
    Staff_forecast['date'] = pd.to_datetime(Staff_forecast['date'])
    return Staff_forecast
    # # plot
    # Q1 = np.percentile(data_rate['rate'], 0)
    # Q3 = np.percentile(data_rate['rate'], 98)
    # Staff_forecast = pd.DataFrame()
    # Staff_forecast['date'] = data_rate.iloc[0:653]['datetime']
    # Staff_forecast['data'] = data_rate.iloc[0:653]['Staff']
    # Staff_forecast['data'][data_rate.iloc[0:653]['Outlier_CO'].notna()] = (data_rate.iloc[0:653][
    #                                                                            'CA_' + s1 + 'min_hat'] * 0.8) / Q3
    # # Staff_forecast['data'] = (data_rate.iloc[0:653]['CA_'+s1+'min_hat']*0.8)/(Q3*data_rate.iloc[0:653]['Staff'])
    # Staff_forecast['date'] = pd.to_datetime(Staff_forecast['date'])
    # plt.figure()
    # plt.plot(Staff_forecast['date'], Staff_forecast['data'], label='forecast')
    # plt.plot(Staff_forecast['date'], data_rate.iloc[0:653]['Staff'], label='original')
    # # plt.plot(Staff_forecast['date'],(data_rate.iloc[0:653]['CA_'+s1+'min_hat']*0.8)/(data_rate.iloc[0:653]['rate']*data_rate.iloc[0:653]['Staff']),label='original')
    # plt.legend(loc='upper right')



# def plot_figure():
#     # distribution
#     sns.distplot(data_rate['data'])
#     # Two ylabels: CO, staff
#     ax1 = data_rate[['datetime', 'CO_' + s1 + 'min']].plot(x='datetime')
#     plt.legend(loc='upper left')
#     ax2 = ax1.twinx()
#     data_rate.plot(x='datetime', y='Staff', c='k', ax=ax2, label='Staff')
#     plt.legend(loc='upper right')
#     # Two ylabels: CO_forecast,upper,lower,staff
#     ax1 = data_rate[['datetime', 'CO_' + s1 + 'min', 'C0_' + s1 + 'min_hat']].plot(x='datetime')
#     ax1.fill_between(data_rate["datetime"].values, data_rate['C0_' + s1 + 'min_hat_lower'],
#                      data_rate['C0_' + s1 + 'min_hat_upper'], color='b', alpha=.2)
#     plt.legend(loc='upper left')
#     ax2 = ax1.twinx()
#     data_rate[['datetime', 'Staff']].plot(x='datetime', ax=ax2, c='k')
#     plt.legend(loc='upper right')
#     plt.show()
#     # outlier
#     ax = data_rate[['datetime', 'CO_' + s1 + 'min', 'C0_' + s1 + 'min_hat', 'Staff']].plot(x='datetime')
#     data_rate.plot(x='datetime', y='rate_hat', c='k', ax=ax, label='rate_hat')
#     data_rate.plot(x='datetime', y='Outlier', c='r', kind='scatter', ax=ax, label='Outlier')
#     ax.fill_between(data_rate["datetime"].values, data_rate['C0_' + s1 + 'min_hat_lower'],
#                     data_rate['C0_' + s1 + 'min_hat_upper'], color='b', alpha=.2)
#     # outlier,t_upper,t_lower
#     ax = data_rate[['datetime', 'rate_CO_hat']].plot(x='datetime', c='k')
#     data_rate[['datetime', 'Outlier_CO']].plot(x='datetime', y='Outlier_CO', c='r', kind='scatter', ax=ax,
#                                                            label='Outlier')
#     Q1 = np.percentile(data_rate['rate_CO'], 1)
#     Q3 = np.percentile(data_rate['rate_CO'], 95)
#     plt.plot(data_rate['datetime'], np.repeat(Q1, len(data_rate['datetime'])), 'y', ls='--')
#     plt.plot(data_rate['datetime'], np.repeat(Q3, len(data_rate['datetime'])), 'y', ls='--')
#     plt.show()


if __name__ == '__main__':
    # # QueueGroupId_name = 'QueueGroupId'
    # # datetime_name = 'datetime'
    # # QueueGroupId = 234
    # staff_WFM = pd.read_excel('E:/Data/QueueData_232_234_368_386/'+ str(QueueGroupId) +'.xlsx',sheet_name='Sheet1')
    # forecasting_data = pd.read_csv('./file/predict_30min_1day_new_CO_CA_'+ str(QueueGroupId) +'_03.csv')[
    #     ['QueueGroupId','datetime', 'CallOffered_diff' + s + 's_yhat',
    #      'CallOffered_diff' + s + 's_yhat_upper',
    #      'CallOffered_diff' + s + 's_yhat_lower']]
    # original_data = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueGroupData202102.csv")[
    #     ['QueueGroupId', 'Timestamp', 'CallOffered', 'CallAnswered']]
    # Staff = pd.read_csv('E:/Data/QueueData_232_234_368_386/AgentGroupData202102_234_345.csv')[
    #     ['QueueGroupId', 'Timestamp', 'Measure', 'Value', 'AgentGroupName']]
    # # staff_WFM = staff_WFM[staff_WFM[QueueGroupId_name]== QueueGroupId]
    # forecasting_data = forecasting_data[forecasting_data[QueueGroupId_name] == QueueGroupId]
    # original_data = original_data[original_data[QueueGroupId_name] == QueueGroupId]
    # Staff = Staff[Staff[QueueGroupId_name] == QueueGroupId]
    # forecasting_data.to_csv('./file/test_forecasting_data_'+ str(QueueGroupId) +'.csv',index=False)
    # original_data.to_csv('./file/test_original_data_'+ str(QueueGroupId) +'.csv', index=False)
    # Staff.to_csv('./file/test_Staff_data_'+ str(QueueGroupId) +'.csv', index=False)
    # staff_WFM.to_csv('./file/test_staff_WFM_'+ str(QueueGroupId) +'.csv', index=False)
    # forecasting_data = pd.read_csv('./file/test_forecasting_data_'+ str(QueueGroupId) +'.csv')
    # original_data = pd.read_csv('./file/test_original_data_'+ str(QueueGroupId) +'.csv')
    # Staff = pd.read_csv('./file/test_Staff_data_'+ str(QueueGroupId) +'.csv')
    # staff_WFM = pd.read_csv('./file/test_staff_WFM_'+ str(QueueGroupId) +'.csv')
    # CO_forecast = forecasting_data
    # CO_forecast_n, data_rate_CA, data_rate_CA_WFM, data_rate_CO_SL = data_staff_rate(original_data, staff_WFM, CO_forecast, Staff,
    #                                               staff_WFM_name='Forecast_Staff',staff_WFM_datetime_name='Cal_Date')
    # data_rate_CO_SL.to_csv('./file/test_data_rate_CO_SL_'+ str(QueueGroupId) +'.csv', index=False)
    # data_rate_CA.to_csv('./file/test_data_rate_CA_'+ str(QueueGroupId) +'.csv', index=False)
    # data_rate_CA_WFM.to_csv('./file/test_data_rate_CA_WFM_' + str(QueueGroupId) + '.csv', index=False)
    # data_rate_CO_SL = pd.read_csv('./file/test_data_rate_CO_SL_'+ str(QueueGroupId) +'.csv')
    # data_rate_CA = pd.read_csv('./file/test_data_rate_CA_'+ str(QueueGroupId) +'.csv')
    # data_rate_CA_WFM = pd.read_csv('./file/test_data_rate_CA_WFM_' + str(QueueGroupId) + '.csv')
    # outlier_list_col, t_upper, t_lower = detectoutliers(data_rate_CA[data_rate_CA['rate_CA'] != 0]['rate_CA'], data_rate_CO_SL['rate_CO_SL'])
    # # CO_forecast_n = CO_forecast.rename(columns={'datetime': 'datetime'})
    # CO_forecast_n[datetime_name] = CO_forecast_n['datetime'].apply(lambda x: predict.time_approximation(x))
    # Staff_forecast = staff_for(CO_forecast_n, t_upper)
    # outlier_list_col.to_csv('./outlier_staff_real_'+ str(QueueGroupId) +'_del0.csv', index=False)
    # Staff_forecast.to_csv('./Staff_forecast_'+ str(QueueGroupId) +'_del0.csv', index=False)
    # plt.show()
    ######## plot
    # ###PLOT_RATE_CA_WFM
    # data_rate_CA = pd.read_csv('./file/test_data_rate_CA_345.csv')
    # data_rate_CA_WFM = pd.read_csv('./file/test_data_rate_CA_WFM_345.csv')
    # data_rate_CA = predict.replace_outlier(data_rate_CA, 'rate_CA', 6)
    # data_rate_CA[data_rate_CA['rate_CA'] != 0].iloc[:, 1:3].plot(kind='box')
    # plt.figure()
    # sns.distplot(data_rate_CA[data_rate_CA['rate_CA'] != 0]['rate_CA'])
    # plt.show()
    # data_rate_CA_WFM = predict.replace_outlier(data_rate_CA_WFM, 'rate_CA_WFM', 6)
    # Q1 = np.percentile(data_rate_CA_WFM[data_rate_CA_WFM['rate_CA_WFM'] != 0]['rate_CA_WFM'].values, 25)
    # Q3 = np.percentile(data_rate_CA_WFM[data_rate_CA_WFM['rate_CA_WFM'] != 0]['rate_CA_WFM'].values, 75)
    # lower = 0.5 * Q1 - 1.5 * Q3
    # upper = 2.5 * Q3 + 1.5 * Q1
    # np.percentile(data_rate_CA_WFM[data_rate_CA_WFM['rate_CA_WFM'] != 0]['rate_CA_WFM'].values, 99)
    # sns.distplot(data_rate_CA['rate_CA'])
    # plt.show()
    # # sns.distplot(data_rate_CA_WFM['rate_CA_WFM'])
    # # plt.show()
    # ####PLOT_CO_FOR_STAFF_WFM
    # CO_for_original = pd.read_csv(r"E:\Data\QueueData_232_234_368_386\QueueGroupData202101.csv")[
    #     ['QueueGroupId', 'Timestamp', 'CallOffered', 'CallAnswered']]
    # CO_for_original = CO_for_original[CO_for_original[QueueGroupId_name]==QueueGroupId]
    # df = CO_for_original
    # ## Calculate CO_diff_30min
    # diff_datetime_name = 'Timestamp'
    # measure_name = 'CallOffered'
    # measure_time_list = [QueueGroupId_name,measure_name,diff_datetime_name]
    # df = df[measure_time_list]
    # interval_seconds = 30 * 60
    # measure_list = [measure_name]
    # df[diff_datetime_name] = predict.data_time_process(df,diff_datetime_name)
    # df = df.rename(columns={diff_datetime_name:datetime_name})
    # df_diff = predict.get_diff_data(df, QueueGroupId_name, datetime_name, measure_list)
    # measure_diff_list= [i + '_diff' for i in measure_list]
    # df_diff = predict.diff_interval_sum(df_diff, QueueGroupId_name, datetime_name, measure_diff_list, interval_seconds)
    # CO_diff = df_diff.fillna(0)
    # CO_diff[datetime_name] = CO_diff[datetime_name].apply(lambda x:predict.time_approximation(x))
    # CO_diff.to_csv('./file/CO_diff'+str(QueueGroupId)+'.csv', index=False)
    # CO_diff = pd.read_csv('./file/CO_diff'+str(QueueGroupId)+'.csv')
    # CO_staff_WFM = pd.read_csv('./file/CO_staff_WFM'+str(QueueGroupId)+'.csv')
    # CO_diff['datetime'] = pd.to_datetime(CO_diff['datetime'])
    # CO_staff_WFM['datetime'] = pd.to_datetime(CO_staff_WFM['datetime'])
    # CO_staff_WFM = pd.merge(CO_diff,CO_staff_WFM,on='datetime')
    # # Two ylabels: CO_forecast,upper,lower,staff
    # CO_staff_WFM = CO_staff_WFM.rename(columns={'CallOffered_diff'+s+'s':'CO_'+s1+'min',
    #                                  'CallOffered_diff'+s+'s_yhat':'C0_'+s1+'min_hat',
    #                                  'CallOffered_diff'+s+'s_yhat_upper':'C0_'+s1+'min_hat_upper',
    #                                  'CallOffered_diff'+s+'s_yhat_lower':'C0_'+s1+'min_hat_lower',
    #                                  # 'CallAnswered_diff'+s+'s':'CA_'+s1+'min',
    #                                  # 'CallAnswered_diff'+s+'s_yhat':'CA_'+s1+'min_hat',
    #                                  # 'CallAnswered_diff'+s+'s_yhat_upper':'CA_'+s1+'min_hat_upper',
    #                                  # 'CallAnswered_diff'+s+'s_yhat_lower':'CA_'+s1+'min_hat_lower',
    #                                  'totalStaff':'Staff'})
    # CO_staff_WFM_P = CO_staff_WFM  # .iloc[382:959,:]
    # ax1 = CO_staff_WFM_P[['datetime', 'CO_' + s1 + 'min', 'C0_' + s1 + 'min_hat']].plot(x='datetime')
    # ax1.fill_between(CO_staff_WFM_P['datetime'].values, CO_staff_WFM_P['C0_' + s1 + 'min_hat_lower'],
    #                  CO_staff_WFM_P['C0_' + s1 + 'min_hat_upper'], color='b', alpha=.2)
    # plt.legend(loc='upper left')
    # ax2 = ax1.twinx()
    # CO_staff_WFM_P[['datetime', 'Staff']].plot(x='datetime', ax=ax2, c='k')
    # plt.legend(loc='upper right')
    # # Staff_forecast.plot(x='date')
    # staff_for = pd.read_csv('Staff_forecast_234.csv')
    # staff_for[['date','data']].plot(x='date')
    # ####PLOT_STAFF
    # QueueGroupId = 234
    staff_yhat = pd.read_csv('./file/prophet_staff_'+str(QueueGroupId)+'.csv')
    staff_yhat['datetime'] = pd.to_datetime(staff_yhat['ds'])
    # staff_yhat_234 = staff_yhat[staff_yhat['QueueGroupId']==QueueGroupId]
    staff_orignal = pd.read_csv('./file/AgentGroupData202103_234_345.csv')
    staff_orignal = staff_orignal[staff_orignal['QueueGroupId']==QueueGroupId]
    data_AgentLogedIn = agentgroup_staff(staff_orignal)
    data_AgentLogedIn = predict.replace_outlier(data_AgentLogedIn, 'Staff',3)
    data_m = pd.merge(data_AgentLogedIn, staff_yhat, on='datetime',how='right')
    CO_staff_WFM = pd.read_csv('./file/CO_staff_WFM'+str(QueueGroupId)+'.csv')[['datetime', 'totalStaff']]
    CO_staff_WFM['datetime'] = pd.to_datetime(CO_staff_WFM['datetime'])
    Staff_forecast = pd.read_csv('Staff_forecast_'+str(QueueGroupId)+'_del0.csv')
    Staff_forecast['date'] = pd.to_datetime(Staff_forecast['date'])
    data_m_n = pd.merge(data_m, CO_staff_WFM, on='datetime',how='left')
    # data_m_n.drop('QueueGroupId', axis=1, inplace=True)
    data_m_n = data_m_n.rename(columns={'totalStaff': 'staff_WFM'})
    # data_m_n.to_csv('./file/data_plot.csv')
    data_m_n.sort_values(by='datetime', inplace=True)
    data_m_n['yhat'][data_m_n['yhat'] < 0] = 0
    Staff_forecast['data'][Staff_forecast['data'] < 0] = 0
    ax1 = data_m_n[['datetime', 'Staff']].plot(x='datetime', color='k')
    data_m_n[['datetime', 'yhat']].plot(x='datetime',y='yhat', ax=ax1, style='--',label='AgentLogedIn_for')
    ax1.fill_between(data_m_n['datetime'].values, data_m_n['yhat_lower'],
                     data_m_n['yhat_upper'], color='b', alpha=.2)
    plt.legend(loc='upper left')
    # ax2 = ax1.twinx()
    data_m_n[['datetime', 'staff_WFM']].plot(x='datetime', ax=ax1, c='r', style=':')
    Staff_forecast.plot(x='date',y='data',ax=ax1,c='y',label='CO*SL/t_upper')
    plt.show()
    # # ####PLOT_CO*SL/upper
    # Staff_forecast = pd.read_csv('Staff_forecast_234.csv')
    # Staff_forecast['date'] = pd.to_datetime(Staff_forecast['date'])
    # ALI_forecast = pd.read_csv('./file/prophet_staff_'+str(QueueGroupId)+'.csv')[['ds','yhat']]
    # ALI_forecast['ds'] = pd.to_datetime(ALI_forecast['ds'])
    # # Outlier = pd.read_csv('outlier_staff_real_234.csv')
    # Staff_WFM = pd.read_csv('./file/staff_WFM234.csv')
    # Staff_WFM['datetime'] = pd.to_datetime(Staff_WFM['datetime'])
    # data = pd.merge(ALI_forecast,Staff_forecast,left_on='ds',right_on='date',how='left')
    # data_m = pd.merge(data,Staff_WFM,left_on='ds',right_on='datetime',how='left')
    # data_m[['ds', 'yhat', 'data', 'totalStaff']].rename(columns={'yhat': 'AgentLogedIn_for', 'data': 'CO*SL/t_upper','totalStaff':'staff_WFM'}).plot(x='ds',color=['b','y','r'])
    # # plt.show()



