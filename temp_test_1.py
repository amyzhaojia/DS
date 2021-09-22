import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from AD_ALL import _process_data
import pyodbc
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
import seaborn as sns

# plot rate
def detectoutliers(list):
    outlier_indices = []
    # iterate over features(columns)
    # 1st quartile (25%)
    Q1 = np.percentile(list, 5)
    # 3rd quartile (75%)
    Q3 = np.percentile(list, 95)
    # Interquartile range (IQR)
    # IQR = Q3 - Q1
    # outlier step
    # outlier_step = 1.5 * IQR
    # Determine a list of indices of outliers for feature col
    # outlier_list_col = list[(list < Q1 - outlier_step) | (list > Q3 + outlier_step)]
    outlier_list_col = list[(list < Q1) | (list > Q3)]
    return outlier_list_col

# data = pd.read_csv('./file/outlier_with_first_detect.csv')[['datetime','CallOffered_diff1800s','CallOffered_diff1800s_yhat','totalStaff']].iloc[0:4080]
data = pd.read_csv('./file/forecast_staff_30min_1day_new_1.csv').iloc[0:653]
data_rate = data.rename(columns={'CallOffered_diff1800s':'CO_30min','CallOffered_diff1800s_yhat':'C0_30min_hat','CallOffered_diff1800s_yhat_upper':'C0_30min_hat_upper','CallOffered_diff1800s_yhat_lower':'C0_30min_hat_lower','totalStaff':'Staff'})
data_rate['rate'] = data_rate['CO_30min']/data_rate['Staff']
data_rate['rate_hat'] = data_rate['C0_30min_hat']/data_rate['Staff']
data_rate['Outlier'] = detectoutliers(data_rate['rate_hat'])
data_rate['datetime'] = pd.to_datetime(data_rate['datetime'])
data_rate_new = data_rate[data_rate['Outlier'].isna()]
Staff_forecast = pd.DataFrame()
Staff_forecast['date'] = data_rate_new['datetime']
Staff_forecast['data'] = data_rate_new['C0_30min_hat']/data_rate_new['rate_hat']
Staff_forecast['date'] = pd.to_datetime(Staff_forecast['date'])
# outlier_rate = len(outlier['data'].values)/len(data_rate['data'].values)
sns.distplot(data_rate['data'])
ax = data_rate[['datetime','CO_30min','C0_30min_hat','rate_hat','Staff']].plot(x='datetime')
data_rate.plot(x='datetime',y='Outlier',c='r',kind='scatter',ax=ax,label='Outlier')
plt.show()
# data_rate.plot(kind='box',ax=axes[2])
plt.figure()
Staff_forecast.plot(x='date')
