import pandas as pd
from COAD import forecasting

def train_forecast(df_diff, num_data, interval_width):
    forecast_val = pd.DataFrame()
    for i in df_diff.keys()[2:]:
        train_data = pd.concat([df_diff[2], df_diff[i]], axis=1).rename(columns={'datetime':'ds', i:'y'})
        forecast_ds = forecasting(train_data, num_data, interval_width)
        forecast_val[i + '_ds'] = forecast_ds['ds']
        forecast_val[i + '_yhat'] = forecast_ds['yhat']
        forecast_val[i + 'yhat_upper'] = forecast_ds['yhat_upper']
        forecast_val[i + 'yhat_lower'] = forecast_ds['yhat_lower']
    df = pd.concat([df_diff[:2], forecast_val], axis=1)
    return df

