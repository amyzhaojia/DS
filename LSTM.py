import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing   
from tensorflow.keras import backend as K #转换为张量



# def fill_zero(tmp_total_df):
#     new_df = pd.DataFrame(columns=['aggValue', 'recordTime'])
#     total_count = 0.0
#     fill_count = 0.0
#     for i in range(0, len(tmp_total_df) - 1):
#         total_count += 1.0
#         prev_record_time = np.long(tmp_total_df.iloc[i]['recordTime'])
#         next_record_time = np.long(tmp_total_df.iloc[i + 1]['recordTime'])
#         new_df = new_df.append(tmp_total_df.iloc[i])
#         #若两个相邻的时间列相差超过一分钟,则在缺失的时间上用0填充
#         if (next_record_time - prev_record_time) > 60000:
#             for j in range(prev_record_time, next_record_time, 60000):
#                 if j > prev_record_time & j < next_record_time:
#                     new_df = new_df.append({'aggValue': 0.0, 'recordTime': str(j)}, ignore_index=True)
#                     fill_count += 1
#                     total_count += 1.0
#         else:
#             total_count += 1.0
#     print("fill rate:" + str(fill_count / total_count))
#     return new_df



# test = 35175
data = pd.read_csv('periodism/QueueGroup234_5_6_time.csv')
data = data[['y']].astype('float64')
total_df = data
# 归一化
sc = preprocessing.MinMaxScaler(feature_range=(0, 1))

total_length = len(total_df)
print(total_length)
# 选取最后一天之前的作为训练数据集
train_df = total_df[0:35175].values
train_df = sc.fit_transform(train_df)
# 选取最后一天作为测试数据集
test_df = total_df[35175:]

x_train = []
y_train = []
predict_lag = 500
predict_interval = 1
# 使用前一段时间作为训练数据
for i in range(predict_lag, len(train_df)):
     x_train.append(train_df[i - predict_lag:i])
     y_train.append(train_df[i:i+predict_interval])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print('看最后一组的形状：')
print(type(y_train))
print(y_train.shape)
print(x_train.shape)

print('len x_train[-1]:', len(x_train[-1]))
print('len y_train[-1]:', len(y_train[-1]))
print('len x_train[-1]:', len(x_train[-2]))
print('len y_train[-1]:', len(y_train[-2]))
print('len x_train[-1]:', len(x_train[-3]))
print('len y_train[-1]:', len(y_train[-3]))
x_train = x_train[:-1]
y_train = y_train[:-1]
x_train = np.asarray(x_train).astype('float64')
y_train = np.asarray(y_train).astype('float64')
print('x_train:', x_train[0][0][0])
print('x_train type:', type(x_train[0][0][0]))
print('type(x_train):', type(x_train))
print('x_train size:', x_train.size)
print('x_train shape:', x_train.shape)
# x_train = K.cast_to_floatx(x_train)
# y_train = K.cast_to_floatx(y_train)


# 生成训练模型--lstm
def generate_lstm_mode(x_train, y_train, predict_interval):
    regressor = tf.keras.Sequential()
    # 第一层连接
    regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    regressor.add(tf.keras.layers.Dropout(0.2))
    # 第二层连接
    regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    regressor.add(tf.keras.layers.Dropout(0.2))
    # 第三层连接
    regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    regressor.add(tf.keras.layers.Dropout(0.2))
    # 第四层连接
    regressor.add(tf.keras.layers.LSTM(units=50))
    regressor.add(tf.keras.layers.Dropout(0.2))
    regressor.add(tf.keras.layers.Dense(units=predict_interval))
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    [print(i.shape, i.dtype) for i in regressor.inputs]
    [print(o.shape, o.dtype) for o in regressor.outputs]
    [print(l.name, l.input_shape, l.dtype) for l in regressor.layers]
    regressor.fit(x_train, y_train, epochs=1, batch_size=32)
    return regressor

# 训练模型
regressor = generate_lstm_mode(x_train, y_train, predict_interval)
# 真实值
real_business_value = test_df.values.reshape(-1)
inputs = total_df[len(total_df) - len(test_df) - predict_lag:].values
test_original = inputs[1:].reshape(-1)
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []
for i in range(predict_lag, len(test_df) + predict_lag):
    x_test.append(inputs[i - predict_lag:i])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# 预测
predicted_business_value = regressor.predict(x_test)

predicted_business_value = sc.inverse_transform(predicted_business_value)
x_test = x_test.reshape(-1)
predicted_business_value = predicted_business_value.reshape(-1)
print(type(x_test))
print(type(predicted_business_value))
print(x_test)
print('real_business_value')
print(real_business_value)
print('predicted_business_value')
print(predicted_business_value)
result = {"a":real_business_value, "b":predicted_business_value}
print('len1:', len(real_business_value))
print('len2:', len(predicted_business_value))
result = pd.DataFrame(result)
result.to_csv('periodism/QueueGroup234_5_6_time_lstm_predict_500lag.csv')
