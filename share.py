import pandas as pd
import numpy as np
import plotly_express as px
# px自带数据集(DataFrame数据集)
gapminder = px.data.gapminder()
print(gapminder.head())  # 取出前5条数据
# line 图
fig = px.line(gapminder, x="year",  y="lifeExp", color="continent",  line_group="continent",  hover_name="country",  line_shape="spline",  render_mode="svg" )
fig.show()
# area 图
fig = px.area(gapminder, x="year", y="pop", color="continent", line_group="country")
fig.show()
fig = px.scatter(gapminder, x="gdpPercap", y="lifeExp", color="continent", size="pop")
fig.show()
fig = px.choropleth(gapminder,locations="iso_alpha", color="lifeExp", hover_name="country", animation_frame="year", color_continuous_scale=px.colors.sequential.Plasma, projection="natural earth")
fig.show()