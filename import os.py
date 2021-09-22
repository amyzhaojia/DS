import os
import configparser
import ast
import pandas as pd
import time
import argparse
import ast
from configparser import ConfigParser
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib as mpl
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

style.use("seaborn")

cf = configparser.ConfigParser()
cf.read("./cfg/animation_cfg.ini")
DFT = "DEFAULT"
boundary_threshold = cf.getfloat(DFT, "BoundaryThreshold")
sleep_time = cf.getint(DFT, "SleepTime")
x_limit = cf.getint(DFT, "X_Limit")
data_path = ast.literal_eval(cf.get(DFT, "PlotDataPath"))
axes_cols = ast.literal_eval(cf.get(DFT, "AxesColumns"))
titles = ast.literal_eval(cf.get(DFT, "Titles"))
xlabel = cf.get(DFT, "XLabel")
ylabel = cf.get(DFT, "YLabel")
window_title = cf.get(DFT, "WindowTitle")


for f in data_path:
    if os.path.exists(f):
        os.remove(f)

show_dynamic_plot(axes_data_path=data_path,
                    axes_cols=axes_cols,
                    # titles[0]改成titles
                    titles=titles,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    window_title=window_title,
                    interval=sleep_time*1000,
                    boundary=boundary_threshold,
                    x_limit=x_limit,
)

# read parameters from command line
ap = argparse.ArgumentParser()
ap.add_argument('-ts', '--test_size', type=int, default=682,
                    help='#the length you want to predict')
ap.add_argument('-mh','--minute_or_half_hour', type=str, default='m',
                help='if you want to predict by minute please set "m" else set"h".')
# ap.add_argument('-c', '--config_file_path', type=str, default='AD_config.ini',
#                     help='#the path of the config file.')
args = vars(ap.parse_args())
test_data_size = args['test_size']
minute_or_half_hour = args['minute_or_half_hour']
print('type arg:', type(minute_or_half_hour))
print('arg:', minute_or_half_hour)
# config_file_path = args['config_file_path']
# get parameters from config file
# conf = ConfigParser()
# conf.read(config_file_path)

# col_name = ast.literal_eval(conf.get('DATA', 'LongestQueueTime'))
# data_path = conf.get('DATA', 'data_path')
# smoothing_window = conf.getint('DATA', 'smoothing_window')
# seq_len = conf.getint('DATA', 'seq_len')
# k = conf.getint('DATA', 'k_sigma')
print('minute_or_half_hour:', minute_or_half_hour)
if minute_or_half_hour == 'h':
    data_1 = pd.read_csv('longest_data_agg_halfhour_forProphet_result.csv')
    data_2 = pd.read_csv('calls_data_agg_halfhour_forProphet_result.csv')
else:
    data_1 = pd.read_csv('LongestQueueTime_all.csv')
    data_2 = pd.read_csv('CallsInQueue_all.csv')

file_name_1 = './LongestQueueTime.csv'
file_name_2 = './CallsInQueue.csv'
for i in range(test_data_size):
    value_1 = data_1.iloc[i:i + 1,:]
    value_1.loc[(value_1.y_hat_lower < 0),"y_hat_lower"] = 0
    value_2 = data_2.iloc[i:i + 1, :]
    value_2.loc[(value_2.y_hat_lower < 0),"y_hat_lower"] = 0
    if not os.path.exists(file_name_1):
        value_1.to_csv(file_name_1, header=True, mode="w",index=False)
        value_2.to_csv(file_name_2, header=True, mode="w", index=False)
    else:
        value_1.to_csv(file_name_1, header=False, mode="a",index=False)
        value_2.to_csv(file_name_2, header=False, mode="a", index=False)
    time.sleep(.1)

# cf = configparser.ConfigParser()
# cf.read("./animation_config.ini")
# boundary = cf.getfloat("DEFAULT", "BoundaryThreshold")


def axis_plot(axis, graph_data, dt_col, y_test_col, y_hat_col, y_upper, y_lower, outlier_col, multi_outlier_col, boundary=0.2, x_limit=100):
    # dt = pd.to_datetime(graph_data.loc[:, dt_col]).values
    dt = graph_data.loc[:, dt_col].values
    # dt = pd.to_datetime(graph_data.loc[:, dt_col]).values
    y_test = graph_data.loc[:, y_test_col].values
    y_hat = graph_data.loc[:, y_hat_col].values
    #新增
    y_upper = graph_data.loc[:, y_upper].values
    y_lower = graph_data.loc[:, y_lower].values
    outliers = graph_data.loc[:, outlier_col].values
    multi_outlier = graph_data.loc[:, multi_outlier_col].values


    # x_limit = 100
    if len(dt) > x_limit:
        dt = dt[-x_limit:]
        y_test = y_test[-x_limit:]
        y_hat = y_hat[-x_limit:]
        #新增
        y_upper = y_upper[-x_limit:]
        y_lower = y_lower[-x_limit:]
        outliers = outliers[-x_limit:]
        multi_outlier = multi_outlier[-x_limit:]

    # Deal with outliers
    dt_outliers = list()
    plot_outliers = list()

    dt_multi_outliers = list()
    plot_multi_outliers = list()
    # out_flag: indicates whether it is an abnormal point
    # c_flag: indicates it's a 1 sequence or 0 sequence
    out_flag, c_flag = 1, 1

    # for d, y, y_hat, y_upper, y_lower, out in zip(dt, y_test, y_hat, y_upper, y_lower, outliers):
    for d, y, out in zip(dt, y_test, outliers):
        if out == 1:
            dt_outliers.append(d)
            plot_outliers.append(y)

    for d, y, out in zip(dt, y_test, multi_outlier):
        if out == 1:
            dt_multi_outliers.append(d)
            plot_multi_outliers.append(y)

        # if out == 1:
        #     c_flag = 1
        #     if out_flag == 1 :#and c_flag == 1
        #         dt_outliers.append(d)
        #         plot_outliers.append(y)
        #         out_flag = 0
        # else:
        #     c_flag = 0
        #     out_flag = 1


    # axis.clear()
    #修改
    axis.clear()
    # axis[1].clear()

    axis.plot(dt, y_test, label='Actual Data', c='green')

    axis.plot(dt, y_hat, label='Forecasting Data',linestyle='--', c='blue')

    axis.plot(dt, y_upper, label='upper Threshold', linestyle='--', c='red')

    axis.plot(dt, y_lower, label='lower threshold', linestyle='--', c='red')

    # axis.plot(dt, y_hat*(1+boundary), c="lightgray", alpha=0.5, label='Forecasting Range')
    # axis.plot(dt, y_hat*(1-boundary), c="lightgray", alpha=0.5,)
    # axis.fill_between(dt, y_hat*(1-boundary), y_hat*(1+boundary), facecolor="lightgray", alpha=0.5)

    axis.scatter(dt_outliers, plot_outliers, label="Outliers", c='r', s=50)
    axis.scatter(dt_multi_outliers, plot_multi_outliers, label="Multi_Outliers", c='y', s=50)



def show_dynamic_plot(axes_data_path, axes_cols, titles, xlabel, ylabel, window_title, 
                      interval, boundary, x_limit):
    def animate(i):
        axes_data = []
        try:
            # Split date reading and plot for better animation performance
            for pth in axes_data_path:
                axes_data.append(pd.read_csv(pth))
            # for index in range(len(axes_data)):
            print('call axis_plot')
            axis_plot(axes[0], axes_data[0], *axes_cols[0], boundary=boundary, x_limit=x_limit)
            axis_plot(axes[1], axes_data[1], *axes_cols[1], boundary=boundary, x_limit=x_limit)

            # set title for each axis
            #修改
            axes[0].set_title(titles[0])
            axes[1].set_title(titles[1])

            # set xlabel and ylabel
            # axes[-1].set_xlabel(xlabel)
            # axes[math.ceil(len(axes)/2-1)].set_ylabel(ylabel)
            #修改
            axes[0].legend(loc='upper left', bbox_to_anchor=(0., 1.))
            axes[1].legend(loc='upper left', bbox_to_anchor=(0., 1.))

            formatter = mpl.dates.DateFormatter('%d/%m/%Y %H:%M:%S')
            # axes[0].xaxis.set_major_formatter(formatter)
            plt.gcf().autofmt_xdate()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass

    fig, axes = plt.subplots(len(axes_data_path), 1, sharex=True)

    ani = animation.FuncAnimation(fig, animate, interval=interval)
    fig.canvas.set_window_title(window_title)
    plt.show()


if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("./animation_config.ini")
    boundary_threshold = cf.getfloat("DEFAULT", "BoundaryThreshold")
    sleep_time = cf.getint("DEFAULT", "SleepTime")
    x_limit = cf.getint("DEFAULT", "X_Limit")

    data_path = ["./CallOffered.csv", "./CallAnswered.csv", "./LongestQueueTime.csv"]
    for f in data_path:
        if os.path.exists(f):
            os.remove(f)
    
    show_dynamic_plot(axes_data_path=data_path,
                      axes_cols= [
                          ["Date_Key", "CallOffered", "y_hat", "outlier"], 
                          ["Date_Key", "CallAnswered", "y_hat", "outlier"],
                          ["Date_Key", "LongestQueueTime", "y_hat", "outlier"],
                      ],
                      titles=["CallOffered", "CallAnswered", "LongestQueueTime"],
                      xlabel="Datetime",
                      ylabel="Values",
                      window_title="Realtime Anomaly Detection Monitoring",
                      interval=sleep_time,
                      boundary=boundary_threshold,
                      x_limit=x_limit,
    )
