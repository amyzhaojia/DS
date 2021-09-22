from operator import is_not
# from pandas.io.formats.style import no_mpl_message
import pywt
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import pre_processing

# data = np.loadtxt(r'E:\data_AD\data\data_phase2\004_UCR_Anomaly_2500.txt')
data = pd.read_csv(r'E:\Data\QueueGroupData202105_234.csv')['CallOffered'].values[420::]
data_CO = pd.read_csv(r'E:\Data\QueueGroupData202105_234.csv')[['CallOffered','Timestamp']][420::]
data_CO['Datetime'] = pre_processing.data_time_process(data_CO)['datetime']
# data = pd.read_csv(r'E:\Data\QueueGroup202105_234.csv')['CallOffered'].values[420::]
# data_CO = pd.read_csv(r'E:\Data\QueueGroup202105_234.csv')[['CallOffered','datetime']][420::]
# data_CO['Datetime'] = pd.to_datetime(data_CO['datetime'])
train_time = data_CO[['Datetime']].reset_index(drop=True)
data[10000:10100] = data[10000:10100]+1000
data[10100:10723] = data[10100:10723]+1000

# 小波（‘haar’）变换，5阶低频分量
coff = pywt.wavedec(data,'haar',level=5)
ya5 = pywt.waverec(np.multiply(coff,[1, 0, 0, 0, 0, 0]).tolist(),'haar')[1::]#第5层近似分量
# np.savetxt('./ya5.txt',ya5)
# yd5 = pywt.waverec(np.multiply(coff,[0, 0, 0, 0, 0, 1]).tolist(),'haar')#第1层细节分量

# 动态阈值计算
n_train = 7000
c = 1
win = 1000
outlier_n = []
predict_n = []
threshold_n = []
time_n = []
test_n = []
for i in range(round((len(data)-n_train)/win)-1):
    ya5_train = np.reshape(ya5[i*win:n_train+i*win],(round(n_train/win),win))  # N-n长度的数据作为历史数据训练得到动态阈值
    ya5_train_m = np.mean(ya5_train,axis=0)
    ya5_train_m2 = np.mean(ya5_train**2,axis=0)
    ya5_train_sigma = abs(ya5_train_m2-ya5_train_m)
    b = ya5_train_m+c*np.sqrt(ya5_train_sigma)  # 上限
    threshold_abnormal = np.max(b)+np.sqrt(np.mean(b**2))
    
    outlier_index = np.where(ya5[n_train+i*win:n_train+(i+1)*win]>threshold_abnormal)[0]
    if outlier_index is not None:
        outlier_index = outlier_index+n_train+i*win
    outlier = ya5[outlier_index]
    # if outlier is not None:
    #     ya5[outlier_index] = ya5[outlier_index-win]

## plot figure
    # plt.plot(np.arange(i*len(b)+n_train,(i+1)*len(b)+n_train),b,'g',label='predict_data')
    # plt.plot(np.arange(i*len(b)+n_train,(i+1)*len(b)+n_train),np.repeat(threshold_abnormal,len(b)),'r-.',label='threshold_abnormal')
    # plt.plot(np.arange(i*len(b)+n_train,(i+1)*len(b)+n_train),np.repeat(np.max(b),len(b)),'r--',label='predict_max')
# plt.plot(ya5[1::],'b',label='train_data')
# plt.plot(outlier_index, outlier,'ro',label='outliers')
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),loc='upper left',numpoints=1)
# plt.tick_params(labelsize=15)
# plt.show()
    plt.plot(train_time.astype(str)['Datetime'],ya5,'b',label='train_data')
    plt.plot(train_time.astype(str)['Datetime'][outlier_index], outlier,'ro',label='outliers')
    plt.plot(train_time.astype(str)['Datetime'][np.arange(i*len(b)+n_train,(i+1)*len(b)+n_train)],b,'g',label='predict_data')
    plt.plot(train_time.astype(str)['Datetime'][np.arange(i*len(b)+n_train,(i+1)*len(b)+n_train)],np.repeat(threshold_abnormal,len(b)),'r-.',label='threshold_abnormal')
    plt.plot(train_time.astype(str)['Datetime'][np.arange(i*len(b)+n_train,(i+1)*len(b)+n_train)],np.repeat(np.max(b),len(b)),'r--',label='predict_max')
    outlier_number = ya5[n_train+i*win:n_train+(i+1)*win]>threshold_abnormal
    outlier_n = outlier_n+list(outlier_number)
    predict_n = predict_n+list(b)
    threshold_n = threshold_n+list(np.repeat(threshold_abnormal,len(b)))
    time_n = time_n+list(train_time.astype(str)['Datetime'][np.arange(i*len(b)+n_train,(i+1)*len(b)+n_train)])
    test_n = test_n+list(ya5[n_train+i*win:n_train+(i+1)*win])
    
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),loc='upper left',numpoints=1)
plt.xticks(range(1,10723,1000),rotation=10)
plt.show()

## CUSUM
z = ya5[n_train::]-threshold_abnormal
T_time = np.array(train_time.astype(str)['Datetime'])[n_train::]
y = np.zeros(((len(ya5)-n_train-1),1))
t = 33
threshold_DOS = t*threshold_abnormal
for k in range(len(z)):
    if z[k]>0:
        y[k] = y[k-1]+z[k]
        # T_start = T_time[k]
        if y[k]>threshold_DOS:
            T_end = np.array(T_time)[k]
            T_start = np.array(T_time)[np.where(z>0)[0][1]]
            print(T_end,T_start)
            break

plt.plot(T_time,z,'b-')
plt.plot(T_time,ya5[n_train::],'g--')
plt.plot(np.array(T_time)[np.max([(k-win),0]):k],np.repeat(threshold_DOS/30,(k-np.max([k-win,0]))),'r--')
plt.plot(T_time,np.repeat(threshold_abnormal,len(z)),'r-.')
plt.plot(T_time,np.repeat(0,len(z)),'k-')
plt.xticks(range(1,len(z),500),rotation=10)
plt.plot(T_start,int(0),'ro')
plt.plot(T_end,int(0),'ro')
plt.plot(np.array(T_time)[np.where(z>0)[0][1]:k],(y/30)[np.where(z>0)[0][1]:k],'y.')
plt.show()

DWT_data = pd.DataFrame()
DWT_data['datetime'] = time_n
DWT_data['CallOffered'] = test_n
DWT_data['predict'] = predict_n
DWT_data['threshold'] = threshold_n
DWT_data['outlier'] = outlier_n
DWT_data['yn'] = None
DWT_data['yn'][np.where(z>0)[0][1]:k] = list((y/50)[np.where(z>0)[0][1]:k])
DWT_data['threshold_Dos'] = None
DWT_data['threshold_Dos'][np.max([(k-win),0]):k] = list(np.repeat(threshold_DOS/50,(k-np.max([k-win,0]))))
DWT_data.to_csv('./file/DWT_CallOffered_0610.csv')