import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#bokeh
from bokeh.palettes import brewer
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, DataTable, TableColumn, CustomJS, Dropdown
from bokeh.models.tools import HoverTool
from bokeh.transform import linear_cmap
from bokeh.io import output_notebook, show

"""This will be used to display data from CTA matches."""

latest_games = pd.read_csv('https://raw.githubusercontent.com/ctaela/cta_dive/master/data/reduced_games_27467.csv')
latest_stats = pd.read_csv('https://raw.githubusercontent.com/ctaela/cta_dive/master/data/reduced_stats_27467.csv')

game_list = list(latest_stats['Game ID'].sort_values(ascending=False).unique())

# def show_table(df, game):
#     source = ColumnDataSource(df[df["Game ID"] == game].sort_values(by=['Outcome', 'Kills'], ascending=False))
#     columns = [
#             TableColumn(field="Player", title="Player"),
#             TableColumn(field="Outcome", title="Outcome"),
#         ]
#     data_table = DataTable(source=source, columns=columns, width=400, height=280)
#     show(data_table)

game_option = st.selectbox(
    'Select game to view stats.', game_list)

"""\nGame Stats"""

st.dataframe(data=latest_stats[latest_stats["Game ID"] == game_option].dropna(axis=1))
# show_table(latest_stats, game_option)


