import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt

os.chdir(sys.path[0])

from detecta import detect_cusum

# with open('data_phase2/ya5.txt', 'r') as f:
#     data = f.read()
origin_data_path = 'data_phase2/004_UCR_Anomaly_2500.txt'
origin_data = pd.read_csv(origin_data_path, names=['origin_value'])

filename = 'data_phase2/ya5.txt'
data = pd.read_csv(filename,names=['value'])
data = data.join(origin_data)
data['index'] = data.index

dt = np.array(data["value"])
ta, tai, taf, amp = detect_cusum(dt, 26600, 0, True, False)
print('ta:', ta)
print('tai:', tai)
print('taf:', taf)


ta_df = pd.DataFrame(ta, columns=['ta'])
tai_df = pd.DataFrame(tai, columns=['tai'])
taf_df = pd.DataFrame(taf, columns=['taf'])
print('ta_df:\n', ta_df)
print('tai_df:\n', tai_df)
print('taf_df:\n', taf_df)


data = pd.merge(data, ta_df, left_on='index', right_on='ta', how='left')
data = pd.merge(data, tai_df, left_on='index', right_on='tai', how='left')
data = pd.merge(data, taf_df, left_on='index', right_on='taf', how='left')
print(data.loc[479:481])
print(data.loc[5567:5600])

fix, ax = plt.subplots(figsize=(200,10))
data.plot.line(x='index', y='value', c='lightgray', ax=ax, label='wavelet')
data.plot.line(x='index', y='origin_value', c='yellow', ax=ax, label='origin_value')
data.plot.scatter(x='ta', y='origin_value', c='r', ax=ax, label='alarm')
data.plot.scatter(x='tai', y='origin_value', c='g', marker='x', ax=ax, label='start')
data.plot.scatter(x='taf', y='origin_value', c='b', marker='*', ax=ax, label='finish')
plt.show()