# %matplotlib inline
import plotly.offline as plyo
plyo.init_notebook_mode(connected=True)
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()

from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
# import plotly.io as pio

from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results

# Loads dataset into pandas DataFrame
dl = DataLoader()
df = dl.load_peyton_manning()

# specify dataset information
metadata = MetadataParam(
    time_col="ts",  # name of the time column ("date" in example above)
    value_col="y",  # name of the value column ("sessions" in example above)
    freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
            # Any format accepted by `pandas.date_range`
)

forecaster = Forecaster()  # Creates forecasts and stores the result
result = forecaster.run_forecast_config(  # result is also stored as `forecaster.forecast_result`.
    df=df,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=365,  # forecasts 365 steps ahead
        coverage=0.95,         # 95% prediction intervals
        metadata_param=metadata
    )
)

ts = result.timeseries
fig = ts.plot()
plyo.plot(fig)
fig.show()

grid_search = result.grid_search
cv_results = summarize_grid_search_results(
    grid_search=grid_search,
    decimals=2,
    # The below saves space in the printed output. Remove to show all available metrics and columns.
    cv_report_metrics=None,
    column_order=["rank", "mean_test", "split_test", "mean_train", "split_train", "mean_fit_time", "mean_score_time", "params"])
# Transposes to save space in the printed output
cv_results["params"] = cv_results["params"].astype(str)
cv_results.set_index("params", drop=True, inplace=True)
print(cv_results.transpose())

backtest = result.backtest
fig = backtest.plot()
plyo.plot(fig)
fig.show()

backtest_eval = defaultdict(list)
for metric, value in backtest.train_evaluation.items():
    backtest_eval[metric].append(value)
    backtest_eval[metric].append(backtest.test_evaluation[metric])
metrics = pd.DataFrame(backtest_eval, index=["train", "test"]).T
print(metrics)

forecast = result.forecast
fig = forecast.plot()
plyo.plot(fig)
fig.show()