import pandas as pd
import keras
from keras import layers, models
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
import tensorflow
tensorflow.random.set_seed(8)
import pre_processing
import preprocessing
import pywt

n_steps = 78

def normalizing_data(train):
    # for normalizing test data.
    training_mean = train.mean()
    training_std = train.std()
    training_value = (train - training_mean) / training_std
    # print("Number of training samples:", len(training_value))
    return training_value

def create_sequences(values, steps=n_steps):
    output = []
    for i in range(len(values) - steps):
        output.append(values[i : (i + steps)])
    return np.stack(output)

def model_data(x_train):
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(filters=16, kernel_size=4, padding="same", strides=2, activation="relu"),
            layers.Dropout(rate=0.2),
            layers.Conv1D(filters=8, kernel_size=4, padding="same", strides=2, activation="relu"),
            layers.Conv1DTranspose(filters=8, kernel_size=4, padding="same", strides=2, activation="relu"),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(filters=16, kernel_size=4, padding="same", strides=2, activation="relu"),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    # model.summary()
    return model

if __name__ == '__main__':
    filename = 'E:/data_AD/data/data_phase2/004_UCR_Anomaly_2500.txt'
    data = pd.read_csv(filename,names=['value'])
    data[8000:8300] = 1
    point = 2500
    dota = 1.00
    # filename = r'E:\Data\QueueGroupData202105_234.csv'
    # data_or = pd.read_csv(filename)
    # data_CO = data_or[['CallOffered','Timestamp']][420::]
    # data_CO['Datetime'] = pre_processing.data_time_process(data_CO)['datetime']
    # data_CO = pre_processing.data_diff_process(data_CO[['CallOffered','Datetime']], 30*60, datetime_name='Datetime')
    # data_CO.to_csv('co_05_diff_1min.csv')
    # data = data_CO[['CallOffered']]
    # point = 7632
    # dota = 1.10
    train = data.iloc[:point].reset_index(drop=True)
    # # # 小波（‘haar’）变换，5阶低频分量
    # # coff = pywt.wavedec(train['CallOffered'],'haar',level=5)
    # # ya5 = pywt.waverec(np.multiply(coff,[1, 0, 0, 0, 0, 0]).tolist(),'haar')#第5层近似分量
    # train = preprocessing.replace_outlier(train,'CallOffered')
    # train_new = pd.DataFrame(np.tile(train,(7,1)),columns=['CallOffered'])
    # train_time = data_CO[['Datetime']].iloc[:point].reset_index(drop=True)
    # fig, ax = plt.subplots()
    # plt.plot(train_time.astype(str)['Datetime'],ya5)
    # plt.xticks(range(1,7632,1000),rotation=10)
    # plt.show()
    # train = train_new
    test  = data.iloc[point:].reset_index(drop=True)
    # test_time = data_CO[['Datetime']].iloc[point:].reset_index(drop=True)
    # test[0:2100] = 2*test[0:2100]
    # fig, ax = plt.subplots()
    # train.plot(legend=False, ax=ax)
    # fig, ax = plt.subplots()
    # test.plot(legend=False, ax=ax)
    # plt.show()
    training_value = normalizing_data(train)
    x_train = create_sequences(training_value.values)
    # model = model_data(x_train)
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse")

    # filepath = "./work/model_{epoch:02d}-{val_loss:.5f}.h5"
    # checkpointer = ModelCheckpoint(filepath, 
    #                             monitor='val_loss',
    #                             verbose=1,
    #                             save_best_only='True',
    #                             mode='min',
    #                             period=1)
    # callback_list=[checkpointer]

    # dota_model = model.fit(
    #     x_train,
    #     x_train,
    #     epochs=2009,
    #     batch_size=512,
    #     validation_split=0.2,
    #     callbacks=callback_list)

    # model.save('./work/model01.h5')

    # plt.plot(dota_model.history["loss"], label="Training Loss")
    # plt.plot(dota_model.history["val_loss"], label="Validation Loss")
    # plt.legend()
    # plt.show()

    # # Get train MAE loss.
    model = models.load_model('./work/model01.h5')
    # model = models.load_model(r"E:\Data_analysis\work\model_374-0.02755.h5")
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.sum(np.abs(x_train_pred - x_train), axis=1)

    # plt.hist(train_mae_loss, bins=50)
    # plt.xlabel("Train MAE loss")
    # plt.ylabel("No of samples")
    # plt.show()

    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    # print("Reconstruction error threshold: ", threshold)

    # Checking how the first sequence is learnt
    # plt.plot(x_train[0])
    # plt.plot(x_train_pred[0])
    # plt.show()

    testing_value = normalizing_data(test)
    x_test = create_sequences(testing_value.values)
    # print("Test input shape: ", x_test.shape)

    x_test_pred = model.predict(x_test)
    test_mae_loss = np.sum(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))

    # plt.hist(test_mae_loss, bins=50)
    # plt.xlabel("test MAE loss")
    # plt.ylabel("No of samples")
    # plt.show()

    anomalies = test_mae_loss > threshold * dota
    print("Number of anomaly samples: ", np.sum(anomalies))
    print("Indices of anomaly samples: ", np.where(anomalies))

    test_time = np.arange(len(data)-point)
    plt.figure()
    plt.plot(test_time,test,'y-',label='CallOffered')
    plt.plot(test_time[np.where(anomalies)[0]],test['value'].values[np.where(anomalies)[0]],'ro',label='Anomaly')
    plt.plot(test_time[:-n_steps],test_mae_loss,color='lightblue',label='test_mae_loss')
    plt.plot(test_time,np.repeat(threshold * dota,len(test_time)),'g-',label='threshold_value')
    plt.show()
    
    data_out = pd.DataFrame()
    # data_out['CallOffered'] = test['CallOffered'][:-n_steps]
    # data_out['datetime'] = test_time['Datetime'][:-n_steps]
    data_out['Data'] = test['value']
    data_out['test_mae'] = np.append(test_mae_loss,np.zeros(n_steps))
    data_out['threshold'] = np.repeat(threshold * dota,len(test))
    data_out['Outlier'] = np.append(anomalies,np.repeat('False',n_steps))
    # data_out.to_csv('./file/Autoencoder_out_KDD_100_1.05.csv')

    # test_time = test
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)   
    # ax1.plot(test_time.values[:-n_steps],test[:-n_steps],'y-',label='CallOffered')
    # ax1.plot(test_time.values[:-n_steps][np.where(anomalies)[0]],test[:-n_steps].CallOffered[np.where(anomalies)[0]],'ro',label='Anomaly')
    # ax2 = ax1.twinx()
    # ax2.plot(test_time.values[:-n_steps],test_mae_loss,color='lightblue',label='test_mae_loss')
    # ax2.plot(test_time.values[:-n_steps],np.repeat(threshold * dota,len(test_time[:-n_steps])),'g-',label='threshold_value')
    # fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    # plt.show()