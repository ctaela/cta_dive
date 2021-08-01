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

"""This will be used to display stats from game matches."""

latest_games = pd.read_csv('https://raw.githubusercontent.com/ctaela/cta_dive/master/data/reduced_games_27467.csv')
latest_stats = pd.read_csv('https://raw.githubusercontent.com/ctaela/cta_dive/master/data/reduced_stats_27467.csv')

game_list = list(latest_stats['Game ID'].sort_values(ascending=False).unique())

keep_list = ['Player', 'Team','Game ID', 'Player', 'Outcome', 'Team', 'Kills', 'Deaths', '+/-',
       'Flag Captures', 'Flag Attempts', 'Death Chart','Switch', 'Elected', 'Assassinations',
       'Score', 'Time Played', 'Damage Dealt', 'Damage Taken',
       'DR', 'DKR', 'DDF', 'TDD', 'Laser DD', 'Laser DT', 'Missile DD',
       'Missile DT', 'Grenade DD', 'Grenade DT', 'Bouncy DD', 'Bouncy DT']
main_tab = ['Player', 'Outcome', 'Team', 'Kills', 'Deaths', '+/-',
       'Flag Captures', 'Flag Attempts', 'Switch', 'Elected', 'Assassinations',
       'Score', 'Time Played', 'Game ID']
damage_tab = ['Player', 'Team', 'Damage Dealt', 'Damage Taken','DR', 'DKR', 'DDF', 'TDD', 'Game ID']
weapon_tab = ['Player', 'Team', 'Laser DD', 'Laser DT', 'Missile DD','Missile DT', 
              'Grenade DD', 'Grenade DT', 'Bouncy DD', 'Bouncy DT', 'Game ID']

game_option = st.selectbox(
    'Select game to view stats.', game_list)

damaged_cols = []
killed_cols = []

for col in latest_stats[latest_stats["Game ID"] == game_option].dropna(axis=1).columns:
    if col not in ['Player', 'Team','Game ID', 'Player', 'Outcome', 'Team', 'Kills', 'Deaths', '+/-',
       'Flag Captures', 'Flag Attempts', 'Death Chart','Switch', 'Elected', 'Assassinations',
       'Score', 'Time Played', 'Damage Dealt', 'Damage Taken',
       'DR', 'DKR', 'DDF', 'TDD', 'Laser DD', 'Laser DT', 'Missile DD',
       'Missile DT', 'Grenade DD', 'Grenade DT', 'Bouncy DD', 'Bouncy DT']:
        if col[0] == "K":
            killed_cols.append(col)
        else:
            damaged_cols.append(col)

"""
Overview
"""
st.dataframe(data=latest_stats[latest_stats[main_tab]["Game ID"] == game_option].dropna(axis=1))

"""
Damage
"""
st.dataframe(data=latest_stats[latest_stats[damage_tab]["Game ID"] == game_option].dropna(axis=1))

"""
Weapons
"""
st.dataframe(data=latest_stats[latest_stats[weapon_tab]["Game ID"] == game_option].dropna(axis=1))

"""
Matchups - Kills
"""
st.dataframe(data=latest_stats[latest_stats[killed_cols]["Game ID"] == game_option].dropna(axis=1))

"""
Matchups - Damage
"""
st.dataframe(data=latest_stats[latest_stats[damaged_cols]["Game ID"] == game_option].dropna(axis=1))


