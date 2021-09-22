import os
import configparser
import math

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib as mpl

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

style.use("seaborn")

# cf = configparser.ConfigParser()
# cf.read("./animation_config.ini")
# boundary = cf.getfloat("DEFAULT", "BoundaryThreshold")


def axis_plot(axis, graph_data, dt_col, y_test_col, y_hat_col, y_upper, y_lower, boundary=0.2, x_limit=100):
    print('graph_data:',type(graph_data))
    # dt = pd.to_datetime(graph_data.loc[:, dt_col]).values
    dt = graph_data.loc[:, dt_col].values
    # dt = pd.to_datetime(graph_data.loc[:, dt_col]).values
    y_test = graph_data.loc[:, y_test_col].values
    y_hat = graph_data.loc[:, y_hat_col].values

    #新增
    y_upper = graph_data.loc[:, y_upper].values
    y_lower = graph_data.loc[:, y_lower].values
    # outliers = graph_data.loc[:, outlier_col].values

    # x_limit = 100
    if len(dt) > x_limit:
        dt = dt[-x_limit:]
        y_test = y_test[-x_limit:]
        y_hat = y_hat[-x_limit:]
        #新增
        y_upper = y_upper[-x_limit:]
        y_lower = y_lower[-x_limit:]
        # outliers = outliers[-x_limit:]

    # Deal with outliers
    # dt_outliers = list()
    # plot_outliers = list()
    # out_flag: indicates whether it is an abnormal point
    # c_flag: indicates it's a 1 sequence or 0 sequence
    out_flag, c_flag = 1, 1

    # for d, y, y_hat, y_upper, y_lower, out in zip(dt, y_test, y_hat, y_upper, y_lower, outliers):
    # for d, y, out in zip(dt, y_test, outliers):
    #     if out == 1:
    #         dt_outliers.append(d)
    #         plot_outliers.append(y)
    #     # if out == 1:
    #     #     c_flag = 1
    #     #     if out_flag == 1 :#and c_flag == 1
    #     #         dt_outliers.append(d)
    #     #         plot_outliers.append(y)
    #     #         out_flag = 0
    #     # else:
    #     #     c_flag = 0
    #     #     out_flag = 1


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
    # axis.scatter(dt_outliers, plot_outliers, label="Outliers", c='r', s=50)



def show_dynamic_plot(axes_data_path, axes_cols, titles, xlabel, ylabel, window_title, 
                      interval, boundary, x_limit):
    def animate(i):
        # axes_data = []
        try:
            # Split date reading and plot for better animation performance
            for pth in axes_data_path:
                axes_data = pd.read_csv(pth)

            # for index in range(len(axes_data)):
            axis_plot(axes, axes_data, *axes_cols[0], boundary=boundary, x_limit=x_limit)
            # axis_plot(axis, graph_data, dt_col, y_test_col, y_hat_col, y_upper, y_lower, boundary=0.2, x_limit=100)
            # axis_plot(axes[1], axes_data[1], *axes_cols[1], boundary=boundary, x_limit=x_limit)
            # axis_plot(axes[2], axes_data[2], *axes_cols[2], boundary=boundary, x_limit=x_limit)
            # axis_plot(axes[3], axes_data[3], *axes_cols[3], boundary=boundary, x_limit=x_limit)

            # set title for each axis
            #修改
            axes.set_title(titles)
            # axes[1].set_title(titles[1])
            # axes[2].set_title(titles[2])
            # axes[3].set_title(titles[3])

            # set xlabel and ylabel
            # axes[-1].set_xlabel(xlabel)
            # axes[math.ceil(len(axes)/2-1)].set_ylabel(ylabel)
            #修改
            axes.legend(loc='upper left', bbox_to_anchor=(0., 1.))
            # axes[1].legend(loc='upper left', bbox_to_anchor=(0., 1.))
            # axes[2].legend(loc='upper left', bbox_to_anchor=(0., 1.))
            # axes[3].legend(loc='upper left', bbox_to_anchor=(0., 1.))

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
