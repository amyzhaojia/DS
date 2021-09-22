import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读数据
# data_1min = pd.read_excel(r"E:\Data\increasing_rate\forcast_11_368.xlsx")
data_1min = pd.read_csv('data_rate.csv')
# 数值取绝对值
data_1min['y_abs'] = data_1min['y_rate_or'].abs()
# 分箱
data_1min['bin'] = pd.cut(data_1min['y_abs'], bins=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,30]) # -30,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,
# 统计每个箱子中数据量
rate_min = data_1min.groupby(['bin']).size().reset_index()
# 给统计后的值改列名
rate_min.rename(columns={0:'value'}, inplace=True)

fig = plt.figure(figsize=(10,4))
g = sns.barplot(rate_min['bin'], rate_min['value'])

plt.title('QueueGroup 368 Nov. CO or 1min rate')
plt.ylabel('Count #')
plt.savefig('QueueGroup 368 Nov. CO or 1min rate.png')