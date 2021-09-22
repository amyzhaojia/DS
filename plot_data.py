import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close()
data = pd.read_csv('forecast_original_new.csv')
data_x = data[['timestamp', 'y_1hr']][678:2006]
# data = pd.read_csv('data_rate.csv')
# data_x = data[['timestamp', 'y_or']][34960:36288]
data_x['timestamp'] = pd.to_datetime(data_x['timestamp'])
data_x.set_index('timestamp')
## 可视化异常值的结果
fig = plt.figure(figsize=(14,7))
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H')) # 显示时间坐标的格式
from matplotlib.dates import AutoDateLocator
plt.gca().xaxis.set_major_locator(AutoDateLocator()) # 时间间隔自动选取
plt.plot(data_x['timestamp'],data_x['y_1hr'], 'o')
plt.legend(loc = 2)
plt.grid()
plt.show()


## original series/1st diff/2nd diff + autocorrelation
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
# Import data
df = pd.read_csv('wwwusage.csv', names=['value'], header=0)
# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value)
axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])
# 1st Differencing
axes[1, 0].plot(df.value.diff())
axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])
# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff())
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])
plt.show()
